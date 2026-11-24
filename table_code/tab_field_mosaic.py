r"""168 - The mosaic at two sizes, RN-50 and ViT-B/16, on the SAME pool.

experiment_name = 'mosaic_sizes'
dataset_name    = 'imagenet1k'

WHY
  The mosaic rows in circulation came from different protocols than the
  max-logit ones: the ViT row (transformer_generality) uses n_id=10000, and
  the RN-50 row (123_granularity) runs on the from-scratch backbones. Neither
  can be paired with table_code/tab_field_posemb.py, which measures max-logit on torchvision V1
  with n_id = n_ood = 1000. This script redoes the four cells on a single
  pool.

THE FOUR CELLS
  rn50  tile 224 -> mosaic 448   the standard construction (enlarges the field)
  rn50  tile 112 -> mosaic 224   shrinks the tile instead of enlarging the field
  vit   tile 224 -> mosaic 448   requires interpolating the positional embeddings
  vit   tile 112 -> mosaic 224   NATIVE size: no interpolation at all

  The fourth is the control that separates the two effects: if the ViT is
  still poor at 224 with no interpolation, interpolation is absolved.

WHY IT DOES NOT REUSE scripts/frozen_xai_ood/09
  That script does `from _abslib import ImageStore`, and _abslib.py is no
  longer on disk. What is reused from it is the ViT
  readout RECIPE, reimplemented here: Grad-CAM on the PENULTIMATE encoder
  block. After the last block only the CLS token reaches the head, the
  gradient on the spatial tokens is exactly zero and the map comes out all
  zeros.

REPRODUCIBILITY
  Pools from m07.query_pools (SEED=0), the same as 164 and 167. The manifest
  records the seed, the file list of each pool and the three prototypes chosen
  per class, BY PATH rather than by index, so that it does not depend on
  _abslib.

Output (results/mosaic_sizes/):
  168_auroc_<n>samples_<YYYYMMDD-HHMM>.csv
  168_manifest_<YYYYMMDD-HHMM>.json
  168_summary_<YYYYMMDD-HHMM>.text

Usage:
  $PYTHON_BIN table_code/tab_field_mosaic.py
  $PYTHON_BIN table_code/tab_field_mosaic.py --n 32 --n-proto-candidates 8
"""
import os

experiment_name = "mosaic_sizes"
dataset_name    = "imagenet1k"

import sys
import json
import time
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import (ResNet50_Weights, ViT_B_16_Weights,
                                resnet50, vit_b_16)
from torchvision.models.vision_transformer import interpolate_embeddings

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_code"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(ROOT / "external_xai_ood"))
m07 = importlib.import_module("rood_qualify_protocol")
m164 = importlib.import_module("transformer_quarter")
from src.mosaic import build_mosaic

OUT = ROOT / "results" / experiment_name
OUT.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M")
SEED = m07.SEED
NEAR = ["ssb_hard", "ninco"]
FAR = ["inaturalist", "openimage_o", "textures"]
EPS = 1e-12
W_OF = {"rn50": ResNet50_Weights.IMAGENET1K_V1,
        "vit": ViT_B_16_Weights.IMAGENET1K_V1}


class RN50:
    """CAM = <w_c, A> no layer4. Linear na feature: nenhum backward."""

    def __init__(self, dev):
        self.dev = dev
        m = resnet50(weights=W_OF["rn50"]).to(dev).eval()
        self.body = torch.nn.Sequential(*list(m.children())[:-2])
        self.W = m.fc.weight.detach()
        self.b = m.fc.bias.detach()
        self.m = m

    @torch.no_grad()
    def pred_pool(self, x, bs):
        p, z = [], []
        for i in range(0, len(x), bs):
            A = self.body(x[i:i + bs].to(self.dev))
            zz = A.mean(dim=(2, 3))
            z.append(zz.cpu())
            p.append((zz @ self.W.T + self.b).argmax(1).cpu())
        return torch.cat(p).numpy(), torch.cat(z)

    @torch.no_grad()
    def cam(self, x, cls, bs):
        out = []
        for i in range(0, len(x), bs):
            A = self.body(x[i:i + bs].to(self.dev))
            w = self.W[cls[i:i + bs].to(self.dev)]
            out.append(torch.einsum("bc,bchw->bhw", w, A).cpu())
        return torch.cat(out)


class ViT:
    """Grad-CAM no PENULTIMO block do encoder; CLS descartado."""

    def __init__(self, dev, size):
        self.dev, self.size = dev, size
        w = W_OF["vit"]
        if size == 224:
            self.m = vit_b_16(weights=w).to(dev).eval()
        else:
            sd = interpolate_embeddings(size, 16,
                                        vit_b_16(weights=w).state_dict())
            self.m = vit_b_16(image_size=size)
            self.m.load_state_dict(sd)
            self.m = self.m.to(dev).eval()

    @torch.no_grad()
    def pred_pool(self, x, bs):
        p, z = [], []
        for i in range(0, len(x), bs):
            st = {}
            h = self.m.heads.register_forward_pre_hook(
                lambda mm, inp: st.__setitem__("z", inp[0]))
            try:
                lg = self.m(x[i:i + bs].to(self.dev))
            finally:
                h.remove()
            p.append(lg.argmax(1).cpu())
            z.append(st["z"].cpu())
        return torch.cat(p).numpy(), torch.cat(z)

    def cam(self, x, cls, bs):
        out = []
        for i in range(0, len(x), bs):
            xb = x[i:i + bs].to(self.dev)
            cb = cls[i:i + bs].to(self.dev)
            st = {}
            h = self.m.encoder.layers[-2].register_forward_hook(
                lambda mm, ii, o: st.__setitem__("A", o))
            try:
                lg = self.m(xb)
                A = st["A"]
                sel = lg.gather(1, cb.view(-1, 1)).sum()
                g = torch.autograd.grad(sel, A)[0][:, 1:, :]
                A = A[:, 1:, :]
                alpha = g.mean(dim=1)
                cam = (A * alpha[:, None, :]).sum(-1)
                side = int(cam.shape[1] ** 0.5)
                out.append(cam.reshape(-1, side, side).detach().cpu())
            finally:
                h.remove()
        return torch.cat(out)


def wrap(name, dev, size):
    return RN50(dev) if name == "rn50" else ViT(dev, size)


def resize_tiles(x, px):
    if x.shape[-1] == px:
        return x
    return F.interpolate(x, size=px, mode="bicubic",
                         align_corners=False, antialias=True)


def prototypes(name, dev, tf, n_cand, bs, tr, log):
    """3 prototypes per class, in the feature space of the model itself."""
    rng = np.random.default_rng(SEED)
    w = wrap(name, dev, 224)
    chosen, t0 = {}, time.time()
    for c in range(len(tr)):
        candidates = [tr[c][i] for i in
                np.sort(rng.choice(len(tr[c]), min(n_cand, len(tr[c])),
                                   replace=False))]
        x = m164.load_official([str(p) for p in candidates], tf, pre=False)
        pred, z = w.pred_pool(x, bs)
        zn = z / (z.norm(dim=1, keepdim=True) + EPS)
        cen = zn.mean(0)
        cen = cen / (cen.norm() + EPS)
        order = torch.argsort(-(zn @ cen)).numpy()
        ok = [i for i in order if pred[i] == c] or list(order)
        chosen[c] = [str(candidates[i]) for i in ok[:3]]
        if c % 100 == 0:
            log(f"    prototypes {name}: class {c} [{time.time()-t0:.0f}s]")
    del w
    torch.cuda.empty_cache()
    return chosen


def s_for(name, dev, size_tile, pools, imgs, proto, tf, bs, log):
    """s (the legacy ratio) per sample, query in the top-left tile."""
    px_mos = size_tile * 2
    w = wrap(name, dev, px_mos)
    w224 = wrap(name, dev, 224) if px_mos != 224 else w
    cache, out = {}, {}
    t0 = time.time()
    for g, x in imgs.items():
        pred, z = w224.pred_pool(x, bs)
        mos, cls = [], []
        for i, c in enumerate(pred):
            c = int(c)
            if c not in cache:
                if len(cache) > 48:
                    cache.clear()
                r = m164.load_official(proto[c], tf, pre=False)
                cache[c] = resize_tiles(r, size_tile)
            q = resize_tiles(x[i:i + 1], size_tile)[0]
            mos.append(build_mosaic(q, cache[c], qpos=0))
            cls.append(c)
        L = w.cam(torch.stack(mos), torch.tensor(cls), max(1, bs // 4))
        mu = L.flatten(1).mean(1, keepdim=True)[:, :, None]
        E = torch.relu(L - mu)
        h = E.shape[-1] // 2
        eq = E[:, :h, :h].flatten(1).mean(1)
        er = ((E[:, :h, h:].flatten(1).mean(1)
               + E[:, h:, :h].flatten(1).mean(1)
               + E[:, h:, h:].flatten(1).mean(1)) / 3)
        out[g] = (eq / (eq + er + EPS)).numpy()
        log(f"    {name} tile{size_tile} {g:<12} [{time.time()-t0:.0f}s]")
    del w, w224
    torch.cuda.empty_cache()
    return out


def main(args):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    pools = m07.query_pools(args.n)
    log(f"pools: { {k: len(v) for k, v in pools.items()} }  (SEED={SEED})")
    tr, _ = m07.imagefolder_files(m07.P["imagenet1k"]["train"])

    rows, all_prototypes = [], {}
    for name in args.models:
        tf = W_OF[name].transforms()
        imgs = {g: m164.load_official(f, tf, pre=(g == "id"))
                for g, f in pools.items()}
        proto = prototypes(name, dev, tf, args.n_proto_cand, args.bs, tr, log)
        all_prototypes[name] = proto
        for size_tile in args.tiles:
            sc = s_for(name, dev, size_tile, pools, imgs, proto, tf,
                      args.bs, log)
            sid = sc["id"]
            per = {}
            for g in NEAR + FAR:
                y = np.r_[np.zeros(len(sid)), np.ones(len(sc[g]))]
                per[g] = 1 - roc_auc_score(y, np.r_[sid, sc[g]])
                rows.append(dict(model=name, tile=size_tile,
                                 mosaic=size_tile * 2, ood=g, auroc=per[g]))
            nm = float(np.mean([per[g] for g in NEAR]))
            fm = float(np.mean([per[g] for g in FAR]))
            rows += [dict(model=name, tile=size_tile, mosaic=size_tile * 2,
                          ood="_mean_near", auroc=nm),
                     dict(model=name, tile=size_tile, mosaic=size_tile * 2,
                          ood="_mean_far", auroc=fm)]
            log(f"  {name:5s} tile {size_tile} -> mosaico {size_tile*2}   "
                f"near {nm:.3f}  far {fm:.3f}")

    df = pd.DataFrame(rows)
    o = OUT / f"168_auroc_{args.n}samples_{STAMP}.csv"
    df.to_csv(o, index=False)
    (OUT / f"168_manifest_{STAMP}.json").write_text(json.dumps(
        {"seed": SEED, "n_per_set": args.n,
         "n_proto_cand": args.n_proto_cand,
         "backbones": {k: str(W_OF[k]) for k in args.models},
         "tiles": args.tiles,
         "pools": pools, "prototypes": all_prototypes}, indent=1))
    (OUT / f"168_summary_{STAMP}.txt").write_text(
        "168_mosaic_sizes.py | the mosaic at two sizes, same pool\n"
        f"SEED={SEED} | n={args.n}/set | prototypes: {args.n_proto_cand} "
        "candidates/class\n\n" + "\n".join(log_lines) + "\n")
    print(f"\n[ok] {o}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-proto-candidates", type=int, default=100)
    ap.add_argument("--tiles", type=int, nargs="+", default=[224, 112])
    ap.add_argument("--models", nargs="+", default=["rn50", "vit"])
    main(ap.parse_args())
