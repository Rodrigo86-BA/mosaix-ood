"""07 - the matched run: r_ood under the QUALIFY mosaic specification.

The question this settles. Scripts 57 and 62 of the frozen repo both say the
mosaic loses to the plain score, but neither is built the way the qualify /
draft mosaic is built (see docs/construcoes_de_mosaico_2026-08-24.md):

  script 57  tiles at 112 in a 224 canvas (the query is DOWNSCALED while the
             bar scores it at 224), random VALIDATION exemplars resampled per
             query, |A| energy, inexact 3x3 quadrants on a 7x7 grid
  script 62  native tiles, right rectification -- but a TORCHVISION backbone
             instead of the from-scratch cells, exemplars taken in
             alphabetical order rather than by centroid, n=200

This driver runs the qualify specification exactly, for all seven detectors:

  exemplars   PROTOTYPE: the 3 TRAIN images whose GAP features are closest to
              the class centroid (cosine), fixed per class
  tiles       native 224 (Resize 256 + CenterCrop 224)  ->  mosaic 448
  grid        layer4 14x14, each tile an EXACT 7x7 block, no discarded cells
  query       fixed TOP-LEFT (the qualify's primary protocol)
  energy      relu(A - spatial mean) -- `field=evidence`, `energy=pos`, the
              convention that produced the published dominance numbers
  backbones   the from-scratch RN-50 CE / LS / mixup cells
  n           1000 per group (ID + 5 OpenOOD sets)

BUILT-IN VALIDATION. For a GAP head the MLS attribution map IS the
predicted-class CAM, and r_ood is a monotone transform of dominance
(r_ood = 1 - r, d = r/(1+r), r = Ebar_q/Ebar_r), so they have IDENTICAL
AUROC. The `mls` row of this run must therefore reproduce the frozen
dominance row of Table 1 (CE: .682 .772 .897 .894 .921). If it does not, the
protocol is not matched and nothing else in the table can be trusted.

Why it does not use the frozen loaders: `src.backbones.load_model_fc` and
`ImageStore` bridge to a legacy repository that no longer exists on disk. Checkpoints load into a plain torchvision resnet50 and feature
row indices map back to files through ImageFolder order -- both verified by
shared_code/rood_qualify_protocol.pya_probe_assets.py.

Outputs (results/rood_qualify/):
  rood_qualify__<model>.csv     one row per (group, detector, sample)
  panel__<model>.json           the prototype exemplar files actually used
  meta__<model>.json            gate values, geometry, settings

Usage:
  smoke: $PYTHON_BIN shared_code/rood_qualify_protocol.py \
             --models ce --n-per-set 24
  full:  $PYTHON_BIN shared_code/rood_qualify_protocol.py \
             --models ce ls mixup --n-per-set 1000
"""
import os

experiment_name = "rood_qualify"
dataset_name    = "imagenet1k"

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[1]
XAI = ROOT / "external_xai_ood"
sys.path.insert(0, str(XAI))

from src.detector_maps import (KnnParams, fit_mahalanobis,
                               fit_react, fit_vim, pooled_score)
from src.exemplars import prototype_indices
from src.mosaic import build_mosaic
from src.pointing import explain_mosaics, rood, tile_contrast

P = yaml.safe_load((XAI / "configs" / "paths.yaml").read_text())
OUT = ROOT / "results" / experiment_name
CACHE = ROOT / "results" / experiment_name / "_cache"
DETECTORS = ["msp", "mls", "energy", "react", "maha", "knn", "vim"]
TOP1 = {"ce": 0.75624, "ls": 0.76382, "mixup": 0.75372}
if os.environ.get("CE_CKPT"):
    P["backbones"]["ce"]["ckpt"] = os.environ["CE_CKPT"]
if os.environ.get("CE_FEATDIR"):
    P["feature_dirs"]["ce"] = os.environ["CE_FEATDIR"]
if os.environ.get("CE_TOP1"):
    TOP1["ce"] = float(os.environ["CE_TOP1"])
IMNORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
IMG_EXT = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}
SEED = 0
TFM_RAW = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize(*IMNORM)])
TFM_PRE = T.Compose([T.ToTensor(), T.Normalize(*IMNORM)])


def strip(sd):
    sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
    return {k.replace("module.", "").replace(".shortcut.", ".downsample."): v
            for k, v in sd.items()}


def load_backbone(key, dev):
    m = resnet50(num_classes=1000)
    m.load_state_dict(strip(torch.load(P["backbones"][key]["ckpt"],
                                       map_location="cpu", weights_only=False)),
                      strict=True)
    m = m.to(dev).eval()
    return m, m.fc.weight.data.cpu(), m.fc.bias.data.cpu()


def npy(fd, *names):
    for n in names:
        if (fd / n).exists():
            return fd / n
    raise FileNotFoundError(f"none of {names} under {fd}")


def feats(key):
    fd = Path(P["feature_dirs"][key])
    return (np.load(npy(fd, "imagenet1k_train_emb.npy", "train_emb.npy"),
                    mmap_mode="r"),
            np.load(npy(fd, "imagenet1k_train_lbl.npy", "train_lbl.npy")
                    ).astype(np.int64),
            npy(fd, "imagenet1k_val_emb.npy", "val_emb.npy"),
            npy(fd, "imagenet1k_val_lbl.npy", "val_lbl.npy"))


def head_gate(key, W, b):
    _, _, ve, vl = feats(key)
    emb, lbl = np.load(ve, mmap_mode="r"), np.load(vl).astype(np.int64)
    hit = 0
    for i in range(0, len(emb), 8192):
        z = torch.from_numpy(np.array(emb[i:i + 8192], dtype=np.float32))
        hit += int(((z @ W.T + b).argmax(1).numpy() == lbl[i:i + 8192]).sum())
    acc = hit / len(emb)
    if abs(acc - TOP1[key]) > 2e-3:
        raise AssertionError(f"head/feature mismatch for {key}: {acc:.5f} vs "
                             f"{TOP1[key]:.5f}")
    return acc


def imagefolder_files(root):
    """Files in ImageFolder order: sorted class dirs, sorted files within.

    This is the ordering the curated feature caches were written in (verified
    by 07a: 1,281,167 files == rows, per-class counts identical), so it is
    what turns a feature row index back into an image path -- the job the
    missing ImageStore used to do.
    """
    root = Path(root)
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    per = [sorted(q for q in d.iterdir() if q.suffix in IMG_EXT) for d in dirs]
    offs = np.cumsum([0] + [len(x) for x in per])
    return per, offs


def prototype_panel(key, tr_files, offs, n_classes=1000):
    """{class -> 3 train file paths}, nearest the class centroid (cosine).

    Cached to .npy: the pass reads the whole 1.28M x 2048 train cache once.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = (Path(os.environ["PROTO_IDX_NPY"]) if os.environ.get("PROTO_IDX_NPY")
          else CACHE / f"proto_idx__{key}.npy")
    if cp.exists():
        idx = np.load(cp)
    else:
        gtr, ltr, _, _ = feats(key)
        idx = np.zeros((n_classes, 3), dtype=np.int64)
        t = time.time()
        for c in range(n_classes):
            idx[c] = prototype_indices(gtr, ltr, c, 3)
            if c % 200 == 0:
                print(f"    prototypes {c}/{n_classes} [{time.time()-t:.0f}s]",
                      flush=True)
        np.save(cp, idx)
    panel = {}
    for c in range(n_classes):
        panel[c] = [str(tr_files[c][int(g - offs[c])]) for g in idx[c]]
    return panel


def query_pools(n):
    rng = np.random.default_rng(SEED)
    pools = {}
    vper, _ = imagefolder_files(P["imagenet1k"]["val"])
    vfiles = [f for d in vper for f in d]
    pools["id"] = [str(vfiles[i]) for i in
                   np.sort(rng.choice(len(vfiles), min(n, len(vfiles)),
                                      replace=False))]
    for name, info in P["ood_sets"].items():
        f = sorted(str(q) for q in Path(info["path"]).rglob("*")
                   if q.suffix in IMG_EXT)
        pools[name] = [f[i] for i in
                       np.sort(rng.choice(len(f), min(n, len(f)),
                                          replace=False))]
    return pools


def load_batch(paths_, pre=False):
    """pre=True for the already-cropped clean/val tree; False elsewhere."""
    tfm = TFM_PRE if pre else TFM_RAW
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths_])


def body_of(model):
    return torch.nn.Sequential(*list(model.children())[:-2])


@torch.no_grad()
def layer4(body, x, dev, bs):
    return torch.cat([body(x[i:i + bs].to(dev)).cpu()
                      for i in range(0, len(x), bs)])


@torch.no_grad()
def predict(model, x, dev, bs):
    return torch.cat([model(x[i:i + bs].to(dev)).argmax(1).cpu()
                      for i in range(0, len(x), bs)]).numpy()


def run_model(key, args, dev):
    t0 = time.time()
    print(f"\n=== backbone {key} ===", flush=True)
    model, W, b = load_backbone(key, dev)
    acc = head_gate(key, W, b)
    print(f"  head gate OK: cached top-1 {acc:.5f} (ref {TOP1[key]:.5f})",
          flush=True)

    gtr, ltr, _, _ = feats(key)
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(gtr), min(args.fit_n, len(gtr)),
                             replace=False))
    z = torch.from_numpy(np.array(gtr[sel], dtype=np.float32))
    y = torch.from_numpy(ltr[sel])
    params = {"W": W.double(), "b": b.double(),
              "maha": fit_mahalanobis(z, y),
              "knn": KnnParams(bank=torch.nn.functional.normalize(z, dim=1)
                               .double().to(dev), k=args.k),
              "vim": fit_vim(z, W, b),
              "react_tau": fit_react(z, q=0.9)}
    print(f"  fits on {len(sel)} train rows [{time.time()-t0:.0f}s]", flush=True)

    tr_files, offs = imagefolder_files(P["imagenet1k"]["train"])
    panel = prototype_panel(key, tr_files, offs)
    print(f"  prototype panel: {len(panel)} classes x3 [{time.time()-t0:.0f}s]",
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"panel__{key}.json").write_text(json.dumps(panel, indent=1))

    body = body_of(model)
    ref_cache = {}
    rows = []
    for gname, files in query_pools(args.n_per_set).items():
        for s in range(0, len(files), args.chunk):
            batch = files[s:s + args.chunk]
            tiles = load_batch(batch, pre=(gname == "id"))
            pred = predict(model, tiles, dev, args.bs)
            Fq = layer4(body, tiles, dev, args.bs)
            zq = Fq.mean(dim=(2, 3)).double()
            single = {d: pooled_score(d, zq, **params).cpu().numpy()
                      for d in DETECTORS}
            mos = []
            for i, c in enumerate(pred):
                c = int(c)
                if c not in ref_cache:
                    if len(ref_cache) > 64:
                        ref_cache.clear()
                    ref_cache[c] = load_batch(panel[c])
                mos.append(build_mosaic(tiles[i], ref_cache[c], qpos=0))
            F = layer4(body, torch.stack(mos), dev, args.bs)
            maps = explain_mosaics(F, params, maps=DETECTORS, field="evidence")
            qpos = np.zeros(len(batch), dtype=np.int64)
            for det, a in maps.items():
                r = rood(a, qpos, denom="mean", energy="pos")
                c_ = tile_contrast(a, qpos)
                for i in range(len(r)):
                    rows.append({"model": key, "group": gname,
                                 "detector": det, "i": s + i,
                                 "pred": int(pred[i]),
                                 "rood": f"{r[i]:.6f}",
                                 "tile_contrast": f"{c_[i]:.6f}",
                                 "single": f"{single[det][i]:.6f}"})
        print(f"  {gname:<12} n={len(files):<5} [{time.time()-t0:.0f}s]",
              flush=True)

    p = OUT / f"rood_qualify__{key}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (OUT / f"meta__{key}.json").write_text(json.dumps({
        "model": key, "head_top1": acc, "top1_reference": TOP1[key],
        "exemplars": "prototype (3 nearest class centroid, cosine, train GAP)",
        "tfm_id": "ToTensor+Normalize (clean/val is already 224 center-crop)",
        "tfm_ood_and_refs": "Resize(256)+CenterCrop(224)+ToTensor+Normalize",
        "native": 224, "mosaic": 448, "layer4": list(F.shape[-2:]),
        "qpos": "fixed top-left (0)", "field": "evidence", "energy": "pos",
        "denom": "mean", "detectors": DETECTORS, "fit_n": int(len(sel)),
        "k": args.k, "n_per_set": args.n_per_set, "seed": SEED}, indent=1))
    print(f"  wrote {p} ({len(rows)} rows) [{time.time()-t0:.0f}s]", flush=True)
    del model, params
    if dev.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ce"],
                    choices=["ce", "ls", "mixup"])
    ap.add_argument("--n-per-set", type=int, default=1000)
    ap.add_argument("--fit-n", type=int, default=100000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()
    dv = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for m in a.models:
        run_model(m, a, dv)
    print("\ndone")
