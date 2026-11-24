r"""167 - Field of view and positional-embedding interpolation: RN-50 against ViT-B/16.

experiment_name = 'field_and_posemb'
dataset_name    = 'imagenet1k'

WHY
  The transformer table shows the framework transferring poorly to the ViT.
  Two alternative explanations have to be ruled out BEFORE the loss is
  attributed to the mosaic construction:

    (a) SCALE. In the 2x2 mosaic the query occupies 1/4 of the field. A
        transformer slices the input into fixed-size patches, so it sees the
        object with a quarter of the tokens. A CNN with GAP does not have
        this problem: it is equivariant to the input size.
    (b) INTERPOLATION. Building the mosaic at 448 on a ViT requires
        interpolating the positional embeddings, which were trained at 224.
        That alone may cost something.

  This script measures both, WITHOUT a mosaic, on the SAME queries as
  shared_code/transformer_quarter.py (same `m07.query_pools`), so the rows are paired with the ones
  there.

CONDITIONS
  orig      the whole 224 image, as the detector normally sees it
  quarter   the image reduced to 112x112 in the corner of a zeroed 224
            canvas, the field fraction the query tile occupies in the mosaic,
            without neighbours
  up448     the 224 image resampled to 448 and read by a ViT whose positional
            embeddings were interpolated from 224 to 448. ViT only: the RN-50
            has no positional embedding and accepts 448 unchanged.

  The RN-50 enters as the CONTROL for (a): if `quarter` hurts the ViT and does
  not hurt the RN-50, the loss of field is specific to the transformer.

TRANSFORM
  Each checkpoint uses its own transform (ViT-B/16 expects a bilinear resize
  to 256; the RN-50 also 256 bilinear, but the rule holds per checkpoint). A
  single transform for different architectures disqualifies one of them, which
  is what distorted the Swin result in shared_code/transformer_quarter.py.

Output (results/field_and_posemb/):
  167_auroc_field_im1k_<n>samples_<YYYYMMDD-HHMM>.csv
  167_summary_field_<YYYYMMDD-HHMM>.txt

Usage:
  $PYTHON_BIN table_code/tab_field_posemb.py
  $PYTHON_BIN table_code/tab_field_posemb.py --n 200
"""
import os

experiment_name = "field_and_posemb"
dataset_name    = "imagenet1k"

import sys
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
m07 = importlib.import_module("rood_qualify_protocol")
m164 = importlib.import_module("transformer_quarter")

OUT = ROOT / "results" / experiment_name
OUT.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M")
NEAR = ["ssb_hard", "ninco"]
FAR = ["inaturalist", "openimage_o", "textures"]

MODELS_W = {"rn50": ResNet50_Weights.IMAGENET1K_V1,
            "vit": ViT_B_16_Weights.IMAGENET1K_V1}


def quarter(x):
    """112x112 in the corner of a zeroed 224 canvas. Same as shared_code/transformer_quarter.py."""
    small = F.interpolate(x, size=112, mode="bicubic",
                          align_corners=False, antialias=True)
    canvas = torch.zeros_like(x)
    canvas[..., :112, :112] = small
    return canvas


def vit448(dev):
    """ViT-B/16 with the positional embeddings interpolated from 224 to 448."""
    w = ViT_B_16_Weights.IMAGENET1K_V1
    sd = interpolate_embeddings(image_size=448, patch_size=16,
                                model_state=w.get_state_dict(progress=False))
    m = vit_b_16(weights=None, image_size=448)
    m.load_state_dict(sd)
    return m.to(dev).eval()


@torch.no_grad()
def scores_cond(model, x, dev, bs, conds):
    out = {c: [] for c in conds}
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(dev, non_blocking=True)
        for c in conds:
            if c == "orig":
                xb = b
            elif c == "quarter":
                xb = quarter(b)
            elif c == "up448":
                xb = F.interpolate(b, size=448, mode="bicubic",
                                   align_corners=False, antialias=True)
            else:
                raise ValueError(c)
            out[c].append(model(xb).max(1).values.float().cpu())
    return {c: torch.cat(v).numpy() for c, v in out.items()}


def auroc_bloco(scores, cond, rows, model_nome, blocks):
    sid = scores["id"][cond]
    per = {}
    for g in NEAR + FAR:
        s = scores[g][cond]
        y = np.r_[np.zeros(len(sid)), np.ones(len(s))]
        per[g] = 1 - roc_auc_score(y, np.r_[sid, s])
        rows.append(dict(model=model_nome, cond=cond, ood=g, auroc=per[g]))
    nm = float(np.mean([per[g] for g in NEAR]))
    fm = float(np.mean([per[g] for g in FAR]))
    rows += [dict(model=model_nome, cond=cond, ood="_mean_near", auroc=nm),
             dict(model=model_nome, cond=cond, ood="_mean_far", auroc=fm)]
    blocks.append(f"  {model_nome:5s} {cond:8s} near {nm:.3f}  far {fm:.3f}")
    print(blocks[-1], flush=True)


def main(args):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pools = m07.query_pools(args.n)
    print(f"pools: { {k: len(v) for k, v in pools.items()} }", flush=True)

    rows, blocks = [], []
    for nome in args.models:
        t0 = time.time()
        tf = MODELS_W[nome].transforms()
        imgs = {g: m164.load_official(f, tf, pre=(g == "id"))
                for g, f in pools.items()}

        conds = ["orig", "quarter"]
        model = (resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                 if nome == "rn50"
                 else vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1))
        model = model.to(dev).eval()
        sc = {g: scores_cond(model, x, dev, args.bs, conds)
              for g, x in imgs.items()}
        print(f"  {nome} 224 pronto [{time.time()-t0:6.0f}s]", flush=True)
        del model
        torch.cuda.empty_cache()

        if nome == "vit":
            m448 = vit448(dev)
            for g, x in imgs.items():
                sc[g].update(scores_cond(m448, x, dev, max(1, args.bs // 4),
                                         ["up448"]))
            conds.append("up448")
            print(f"  {nome} 448 pronto [{time.time()-t0:6.0f}s]", flush=True)
            del m448
            torch.cuda.empty_cache()

        for c in conds:
            auroc_bloco(sc, c, rows, nome, blocks)

    df = pd.DataFrame(rows)
    o = OUT / f"167_auroc_field_im1k_{args.n}samples_{STAMP}.csv"
    df.to_csv(o, index=False)
    (OUT / f"167_summary_field_{STAMP}.txt").write_text(
        "167_field_and_posemb.py | max-logit under three conditions, no mosaic\n"
        "  orig    the whole 224 image\n"
        "  quarter 112x112 in the corner of a zeroed 224 canvas (tile field)\n"
        "  up448   image at 448, ViT with pos-embedding interpolated 224->448\n"
        f"n={args.n}/set | pools identical to those of shared_code/transformer_quarter.py\n\n"
        + "\n".join(blocks) + "\n")
    print(f"\n[ok] {o}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--models", nargs="+", default=["rn50", "vit"])
    main(ap.parse_args())
