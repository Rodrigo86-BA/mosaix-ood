"""Table 8 (AugDelete), evaluation runs on the own LS and mixup models and on V2.

experiment_name = 'camera_ready_augdelete'
dataset_name    = 'imagenet1k'

WHAT IT DOES
------------
Runs `run_arch` of scripts/64 (the code that measured the torchvision V2 row)
WITHOUT changing that file, swapping from outside only what depends on the
model:
  * backbone: RN-50 trained from scratch, seed 0, LS or mixup (scripts/07
    load_backbone);
  * head (second half of the run name):
      `ckpt`      the original head of the checkpoint;
      `refitsgd`  the head used in the paper: a linear 2048->1000 layer refit
                  with plain cross-entropy on the frozen training features
                  (SGD lr 0.1, momentum 0.9, wd 1e-4, 50 epochs, cosine,
                  batch 512, seed 42), written by tab8_refit_head_sgd.py as
                  head_ce_refitsgd__<key>.pt;
      `refit`     SUPERSEDED, not in the paper: AdamW 1e-3, wd 1e-4, 3 epochs,
                  seed 0 (scripts/68 refit_head). The LS head is the one of
                  Fig. 4; the mixup head is fit here with the same recipe;
      `refit20`   SUPERSEDED, not in the paper: the recipe stored in the
                  earlier V2 AugDelete head (AdamW 1e-3, wd 1e-4, 20 epochs,
                  batch 4096, cosine, seed 0), read from
                  head_ce_refit20__<key>.pt, which is written by the
                  working-repo script cr_220 (not shipped here).
    Both AdamW heads overfit and were replaced by the SGD `refitsgd` head;
    they are kept only so earlier result files can be regenerated;
  * training features for theta_c: those of the checkpoint itself;
  * ID: the clean/val tree (pre-cropped, pre=True), the protocol of the own
    models; V2 uses the raw tree with the torchvision transform.
OOD: the same FULL sets as the V2 row. Maha/kNN/ViM are not run (the table
does not use them). fp32, no AMP, no compile, as in scripts/64.

Runs:
  gate          V2 through this wrapper, n=1000 (gate 1)
  heads         fit the superseded AdamW `refit` heads of LS and mixup
  v2_refitsgd   V2 with the SGD head (reads head_ce_refitsgd__v2.pt)
  ls_ckpt ls_refitsgd mixup_ckpt mixup_refitsgd   the paper rows
  ls_refit mixup_refit ls_refit20 mixup_refit20   superseded heads

GATES (stop the run)
  1. --run gate: V2 through this wrapper, n=1000, reproduces
     64_auroc_tvV2_grad_rn50_1000samples_20260909-0702.csv (MLS and mosaic, tol 1e-4)
  2. ckpt head: top-1 on the full ID within 0.002 of m07.TOP1
  3. gate 3 of scripts/64 itself (|s without relu - 1.5| < 1e-6)

Outputs (results/camera_ready_augdelete/ of $MOSAIX_ROOT):
  head_ce_refit__<key>.pt, scores4__<run>_<n>.npz, auroc__<run>_<n>.csv

Usage:
  $PYTHON_BIN camera_ready_code/tab8_augdelete_eval.py --run gate
  $PYTHON_BIN camera_ready_code/tab8_augdelete_eval.py --run ls_refitsgd [--n 1000]
"""
import os

experiment_name = "camera_ready_augdelete"
dataset_name    = "imagenet1k"

import argparse
import importlib
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
sys.path.insert(0, str(ROOT / "scripts"))
m64 = importlib.import_module("64_depth_sweep")
m68 = importlib.import_module("68_head_refit_figure")
m07 = importlib.import_module("07_rood_qualify_protocol")
if os.environ.get("TV_FEAT_ROOT"):
    # cached torchvision training features; overrides the path hard-coded in scripts/64
    m64.TV_FEAT = Path(os.environ["TV_FEAT_ROOT"])

RES = ROOT / "results" / experiment_name
RES.mkdir(parents=True, exist_ok=True)
m64.RES = RES
m68.RES = RES
NEAR, FAR = m64.NEAR, m64.FAR
COLS = {"mls": "MLS", "s": "mosXai-OOD", "mos4": "mosXai-OOD-4x", "mos4tau": "mosXai-OOD-4x+tau"}


class FixedDir:
    """TV_FEAT / anything -> the feature directory of this checkpoint."""
    def __init__(self, p):
        self.p = Path(p)

    def __truediv__(self, _):
        return self.p


def refit(key, dev):
    ck = RES / f"head_ce_refit__{key}.pt"
    old = ROOT / "results/head_refit" / f"head_ce_refit__{key}.pt"
    if not ck.exists() and old.exists():
        shutil.copy(old, ck)
    a = SimpleNamespace(lr=1e-3, wd=1e-4, epochs=3, fit_bs=8192, force_refit=False)
    return m68.refit_head(key, dev, a)


def builder(key, head, dev):
    def build(arch, dev_, wver="V1", fc_path=None):
        model = m07.load_backbone(key, dev_)
        model = model[0] if isinstance(model, tuple) else model
        model = model.to(dev_).eval()
        fc = model.fc
        if head in ("refit", "refit20", "refitsgd"):
            if head == "refit":
                W, b, v = refit(key, dev_)
            else:
                h20 = torch.load(RES / f"head_ce_{head}__{key}.pt", map_location="cpu", weights_only=False)
                W, b, v = h20["W"], h20["b"], h20.get("val_top1", float("nan"))
            n0 = float(fc.weight.norm(dim=1).mean())
            fc.weight.data.copy_(W.to(dev_)); fc.bias.data.copy_(b.to(dev_))
            print(f"  head swapped: val top-1 {v:.4f} | mean norm of w_c "
                  f"{n0:.3f} (ckpt) -> {float(fc.weight.norm(dim=1).mean()):.3f}", flush=True)
        raw = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(*m64.IMNORM)])
        pre = T.Compose([T.ToTensor(), T.Normalize(*m64.IMNORM)])
        return (model, fc.weight.data.cpu(), fc.bias.data.cpu(),
                lambda ref, pre_cropped=False: (pre if pre_cropped else raw)(Image.open(ref).convert("RGB")))
    return build


def aurocs(tag, S, MLS, extra, G):
    sc = {"mls": MLS, "s": S, "mos4": extra["mos4"], "mos4tau": extra["mos4tau"]}
    rows = []
    for k, v in sc.items():
        for g in NEAR + FAR:
            rows.append(dict(run=tag, group=g, detector=COLS[k],
                             auroc=m64.auroc(v[G == "id"], v[G == g]),
                             fpr95=m64.fpr95(v[G == "id"], v[G == g]), n_ood=int((G == g).sum())))
    return pd.DataFrame(rows)


def main(a):
    dev = torch.device("cuda:0")
    if a.run == "heads":
        for key in ("ls", "mixup"):
            W, b, v = refit(key, dev)
            print(key, "refit val top-1", round(v, 4), "| ckpt ref", m07.TOP1[key])
        return
    m64.m62.build_params = lambda *x, **k: None
    args = SimpleNamespace(n_per_set=a.n, max_candidates=24, fit_n=200000, k=50, grad=False,
                           eigen=False, id_tree="clean", panel_batch=8, chunk=64, bs=32,
                           weights="V1", fc=None)
    if a.run == "gate":
        args.n_per_set, args.id_tree, args.weights, arch = 1000, "raw", "V2", "resnet50"
    elif a.run == "v2_refitsgd":
        args.id_tree, args.weights, arch = "raw", "V2", "resnet50"
        hd = torch.load(RES / "head_ce_refitsgd__v2.pt", map_location="cpu", weights_only=False)
        torch.save({"weight": hd["W"], "bias": hd["b"]}, RES / "fc_refitsgd__v2.pth")
        args.fc = str(RES / "fc_refitsgd__v2.pth")
    else:
        key, head = a.run.split("_")
        m64.build_tv = builder(key, head, dev)
        m64.TV_FEAT = FixedDir(m07.P["feature_dirs"][key])
        arch = a.run
    cand = m64.m62.class_candidates("imagenet1k", args)
    pools = m64.m62.query_pools("imagenet1k", args)
    if a.run in ("gate", "v2_refitsgd"):
        rv = Path(m64.paths()["imagenet200"]["raw_val"])
        f = [x for d in sorted(q for q in rv.iterdir() if q.is_dir()) for x in sorted(d.iterdir())]
        rng = np.random.default_rng(m64.m62.SEED)
        pools["id"] = [f[i] for i in np.sort(rng.choice(len(f), min(args.n_per_set, len(f)), replace=False))]
    print(f"tab8_augdelete_eval | {a.run} | AMP=off | compile=off | " +
          " ".join(f"{k}={len(v)}" for k, v in pools.items()), flush=True)
    tr = Path(m64.paths()["imagenet1k"]["train"])
    widx = {w.name: i for i, w in enumerate(sorted(x for x in tr.iterdir() if x.is_dir()))}
    S, MLS, _, _, extra, G, top1 = m64.run_arch(arch, args, dev, pools, cand, widx)
    df = aurocs(a.run, S, MLS, extra, G)
    df["top1"] = top1
    n = "fullsample" if a.n >= 10 ** 6 else f"{args.n_per_set}samples"
    np.savez_compressed(RES / f"scores4__{a.run}_{n}.npz", group=G, s=S, mls=MLS,
                        mos4=extra["mos4"], mos4tau=extra["mos4tau"])
    df.to_csv(RES / f"auroc__{a.run}_{n}.csv", index=False)
    if a.run == "gate":
        ref = pd.read_csv(ROOT / "results/depth_sweep/64_auroc_tvV2_grad_rn50_1000samples_20260909-0702.csv")
        m = df.merge(ref, on=["group", "detector"], suffixes=("", "_ref"))
        d = float((m.auroc - m.auroc_ref).abs().max())
        print(f"GATE 1: {len(m)} cells, max|dAUROC| = {d:.1e} -> {'PASSED' if d < 1e-4 else 'FAILED'}")
        assert len(m) == 20 and d < 1e-4
    elif a.run.endswith("ckpt") and a.n >= 10 ** 6:
        ref = m07.TOP1[a.run.split("_")[0]]
        print(f"GATE 2: top-1 {top1:.5f} vs {ref:.5f}")
        assert abs(top1 - ref) <= 0.002, (top1, ref)
    df["reg"] = np.where(df.group.isin(NEAR), "near", "far")
    print(df.groupby(["detector", "reg"]).auroc.mean().unstack()[["near", "far"]].round(4).to_string())
    print("[ok]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    choices=["gate", "heads", "ls_ckpt", "ls_refit", "mixup_ckpt", "mixup_refit",
                             "ls_refit20", "mixup_refit20", "v2_refitsgd", "ls_refitsgd", "mixup_refitsgd"])
    ap.add_argument("--n", type=int, default=10 ** 6)
    main(ap.parse_args())
