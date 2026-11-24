r"""164 - Transformer sensitivity to shrinking the mosaic.

experiment_name = 'transformer_quarter'
dataset_name    = 'imagenet1k'

WHY
  Table 5 shows the framework transferring poorly to ViT and Swin. One
  possible explanation is purely geometric: in the 2x2 mosaic the query
  occupies a QUARTER of the field, and a transformer, which slices the input
  into fixed-size patches, sees the object with a quarter of the tokens.

  This script isolates that effect WITHOUT a mosaic. It measures max-logit
  under two conditions on the SAME queries:
    orig     the whole 224 image, as the detector normally sees it
    quarter  the image reduced to 112x112 and pasted in the top-left corner
             of a 224 canvas filled with zeros, the same field fraction the
             query tile occupies in the mosaic, without neighbours

  If `quarter` drives max-logit down to levels close to what the mosaic
  produces, the weakness on the transformer is one of scale and not of the
  construction. If it does not, scale does not explain it.

Output (results/transformer_quarter/):
  164_auroc_quarter_<model>_<n>samples_<YYYYMMDD-HHMM>.csv
  164_summary_quarter_<YYYYMMDD-HHMM>.txt

Usage:
  $PYTHON_BIN shared_code/transformer_quarter.py
  $PYTHON_BIN shared_code/transformer_quarter.py --n 200
"""
import os

experiment_name = "transformer_quarter"
dataset_name    = "imagenet1k"

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import (Swin_T_Weights, ViT_B_16_Weights, swin_t,
                                vit_b_16)

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_code"
sys.path.insert(0, str(SHARED))
m07 = importlib.import_module("rood_qualify_protocol")

OUT = ROOT / "results" / experiment_name
OUT.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d-%H%M")
NEAR = ["ssb_hard", "ninco"]
FAR = ["inaturalist", "openimage_o", "textures"]


MODELS_W = {"vit": ViT_B_16_Weights.IMAGENET1K_V1,
            "swin": Swin_T_Weights.IMAGENET1K_V1}


def load_official(files, tf, pre=False):
    """The transform of the CHECKPOINT ITSELF. Swin-T expects a bicubic resize to
    232; ViT-B/16, bilinear to 256. A single transform for both disqualifies one
    of them, which is what made the far-OOD score of Swin rise when the image was
    shrunk."""
    from PIL import Image
    import torchvision.transforms as T
    rsz, crop = int(tf.resize_size[0]), int(tf.crop_size[0])
    steps = ([] if pre else [T.Resize(rsz, interpolation=tf.interpolation),
                             T.CenterCrop(crop)])
    comp = T.Compose(steps + [T.ToTensor(),
                              T.Normalize(tf.mean, tf.std)])
    return torch.stack([comp(Image.open(p).convert("RGB")) for p in files])


def quarter(x):
    """The image at 112x112 in the corner of a zeroed 224 canvas.

    The same field fraction the query tile occupies in the 2x2 mosaic, without
    neighbours: it isolates scale from context.
    """
    small = F.interpolate(x, size=112, mode="bicubic",
                          align_corners=False, antialias=True)
    canvas = torch.zeros_like(x)
    canvas[..., :112, :112] = small
    return canvas


@torch.no_grad()
def mls(model, x, dev, bs):
    out = []
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(dev, non_blocking=True)
        for tag, xb in (("orig", b), ("quarter", quarter(b))):
            out.append((tag, model(xb).max(1).values.float().cpu()))
    o = {}
    for tag, v in out:
        o.setdefault(tag, []).append(v)
    return {k: torch.cat(v).numpy() for k, v in o.items()}


def main(args):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pools = m07.query_pools(args.n)
    print(f"pools: { {k: len(v) for k, v in pools.items()} }", flush=True)

    MODELS = {
        "vit": lambda: vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1),
        "swin": lambda: swin_t(weights=Swin_T_Weights.IMAGENET1K_V1),
    }
    rows, blocks = [], []
    for name, build in MODELS.items():
        if name not in args.models:
            continue
        t0 = time.time()
        model = build().to(dev).eval()
        scores = {}
        tf = MODELS_W[name].transforms()
        for g, files in pools.items():
            x = load_official(files, tf, pre=(g == "id"))
            scores[g] = mls(model, x, dev, args.bs)
            print(f"  {name} {g:<12} {len(files):5d}  "
                  f"[{time.time()-t0:6.0f}s]", flush=True)
        for cond in ("orig", "quarter"):
            sid = scores["id"][cond]
            per = {}
            for g in NEAR + FAR:
                s = scores[g][cond]
                y = np.r_[np.zeros(len(sid)), np.ones(len(s))]
                per[g] = 1 - roc_auc_score(y, np.r_[sid, s])
                rows.append(dict(model=name, cond=cond, ood=g, auroc=per[g]))
            nm = float(np.mean([per[g] for g in NEAR]))
            fm = float(np.mean([per[g] for g in FAR]))
            rows += [dict(model=name, cond=cond, ood="_mean_near", auroc=nm),
                     dict(model=name, cond=cond, ood="_mean_far", auroc=fm)]
            blocks.append(f"  {name:5s} {cond:8s} near {nm:.3f}  far {fm:.3f}")
            print(blocks[-1], flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    o = OUT / f"164_auroc_quarter_im1k_{args.n}samples_{STAMP}.csv"
    df.to_csv(o, index=False)
    (OUT / f"164_summary_quarter_{STAMP}.txt").write_text(
        "164_transformer_quarter.py | max-logit, the whole image against the\n"
        "same image at 112x112 in the corner of a zeroed 224 canvas (the field\n"
        f"fraction the query occupies in the 2x2 mosaic) | n={args.n}/set\n\n"
        + "\n".join(blocks) + "\n")
    print(f"\n[ok] {o}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--models", nargs="+", default=["vit", "swin"])
    main(ap.parse_args())
