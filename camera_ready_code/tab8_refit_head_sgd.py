"""Table 8 (AugDelete), the head refit used in the paper, plus MLS from cached features.

experiment_name = 'camera_ready_augdelete'
dataset_name    = 'imagenet1k'

Refits the linear head of a frozen ResNet-50 (torchvision V2, own LS, own
mixup) with plain cross-entropy on the frozen ID training features: SGD lr 0.1,
momentum 0.9, wd 1e-4, 50 epochs, cosine schedule stepped per epoch, batch 512,
default nn.Linear init, seed 42, no augmentation. Training on ID features only
preserves the accuracy of the checkpoint (V2 80.86 -> 80.78 with the reference
head trained by this recipe).

It also measures, on the cached OOD features (list sizes of OpenOOD v1.5), the
MLS of every head. On V2 the comparison includes a reference CE-only head
trained with the same recipe in an independent codebase (V2_REF_CE_HEAD).

Heads compared (column `head` of the CSV):
  ckpt            the head of the checkpoint
  paper_adamw20   V2 only: trained_fc_ce_retrained.pth, the earlier AdamW head
  ref_file        V2 only: the reference CE-only head (V2_REF_CE_HEAD)
  refit3_adamw    LS/mixup only: the superseded 3-epoch AdamW head
                  (head_ce_refit__<key>.pt, from tab8_augdelete_eval.py --run heads)
  refit20_adamw   LS/mixup only: the superseded 20-epoch AdamW head
                  (head_ce_refit20__<key>.pt, from working-repo script cr_220)
  refit_sgd       the head fit here, the one used in the paper

Outputs (results/camera_ready_augdelete/ of $MOSAIX_ROOT):
  head_ce_refitsgd__<key>.pt, cr_222_mls__<key>.csv

Usage:
  $PYTHON_BIN camera_ready_code/tab8_refit_head_sgd.py --key v2
  $PYTHON_BIN camera_ready_code/tab8_refit_head_sgd.py --key ls
  $PYTHON_BIN camera_ready_code/tab8_refit_head_sgd.py --key mixup
"""
import os

experiment_name = "camera_ready_augdelete"
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
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
sys.path.insert(0, str(ROOT / "scripts"))
m07 = importlib.import_module("07_rood_qualify_protocol")
RES = ROOT / "results" / experiment_name
# cached GAP features of the torchvision V2 checkpoint; by default a sibling of
# the curated feature tree of the own models
V2 = (Path(os.environ["V2_FEATDIR"]) if os.environ.get("V2_FEATDIR")
      else Path(m07.P["feature_dirs"]["ls"]).parents[2] / "torchvision/resnet50_v2")
# reference CE-only V2 head (state dict with `weight` and `bias`), --key v2 only
REF_HEAD = Path(os.environ["V2_REF_CE_HEAD"]) if os.environ.get("V2_REF_CE_HEAD") else None
NEAR, FAR = ["ssb_hard", "ninco"], ["inaturalist", "openimage_o", "textures"]
dev = "cuda:0"


def arr(fd, n):
    return torch.from_numpy(np.array(np.load(fd / n))).to(dev)


def auroc(a, b):
    r = rankdata(np.r_[a, b]); k = len(a)
    return (r[:k].sum() - k * (k + 1) / 2) / (k * len(b))


def mls(X, W, b):
    o = []
    with torch.no_grad():
        for i in range(0, len(X), 16384):
            o.append((X[i:i + 16384].float() @ W.T + b).max(1).values.cpu().numpy())
    return np.concatenate(o)


def fit(fd):
    X, Y = arr(fd, "imagenet1k_train_emb.npy"), arr(fd, "imagenet1k_train_lbl.npy").long()
    torch.manual_seed(42)
    h = torch.nn.Linear(X.shape[1], 1000).to(dev)
    E, BS = 50, 512
    o = torch.optim.SGD(h.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=E)
    g = torch.Generator(device=dev).manual_seed(42)
    t0 = time.time()
    for ep in range(E):
        p = torch.randperm(len(X), generator=g, device=dev)
        hit = 0
        for i in range(0, len(X), BS):
            j = p[i:i + BS]
            out = h(X[j].float()); l = F.cross_entropy(out, Y[j])
            o.zero_grad(set_to_none=True); l.backward(); o.step()
            hit += int((out.argmax(1) == Y[j]).sum())
        s.step()
        if (ep + 1) in (1, 10, 25, 50):
            print(f"  epoch {ep + 1}/{E} train top-1 {hit / len(X):.4f} [{time.time() - t0:.0f}s]", flush=True)
    del X
    return h.weight.data.clone(), h.bias.data.clone()


def main(key):
    fd = V2 if key == "v2" else Path(m07.P["feature_dirs"][key])
    heads = {}
    if key == "v2":
        if REF_HEAD is None:
            raise SystemExit("--key v2 needs V2_REF_CE_HEAD (reference CE-only head)")
        import torchvision.models as tvm
        f = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).fc
        heads["ckpt"] = (f.weight.data.to(dev), f.bias.data.to(dev))
        d = torch.load(V2 / "trained_fc_ce_retrained.pth", map_location=dev, weights_only=False)
        heads["paper_adamw20"] = (d["weight"].float(), d["bias"].float())
        d = torch.load(REF_HEAD, map_location=dev, weights_only=False)
        heads["ref_file"] = (d["weight"].float(), d["bias"].float())
    else:
        _, W, b = m07.load_backbone(key, dev)
        heads["ckpt"] = (W.to(dev), b.to(dev))
        for n, fn in (("refit3_adamw", f"head_ce_refit__{key}.pt"), ("refit20_adamw", f"head_ce_refit20__{key}.pt")):
            d = torch.load(RES / fn, map_location=dev, weights_only=False)
            heads[n] = (d["W"].float(), d["b"].float())
    W, b = fit(fd)
    heads["refit_sgd"] = (W, b)
    torch.save({"W": W.cpu(), "b": b.cpu(), "recipe": "SGD0.1 m0.9 wd1e-4 50ep cos bs512 seed42"},
               RES / f"head_ce_refitsgd__{key}.pt")
    Xv, Yv = arr(fd, "imagenet1k_val_emb.npy").float(), arr(fd, "imagenet1k_val_lbl.npy").long()
    ood = {}
    for g in NEAR + FAR:
        fs = [f for f in (fd / "ood" / g).glob("*_emb.npy")] if (fd / "ood" / g).exists() else []
        if fs:
            ood[g] = arr(fs[0].parent, fs[0].name)
    rows = []
    for n, (W, b) in heads.items():
        idm = mls(Xv, W, b)
        acc = float(((Xv @ W.T + b).argmax(1) == Yv).float().mean())
        r = dict(model=key, head=n, top1=acc, wnorm=float(W.norm(dim=1).mean()))
        for g, X in ood.items():
            r[g] = auroc(idm, mls(X, W, b))
        r["near"] = float(np.mean([r[g] for g in NEAR if g in r])); r["far"] = float(np.mean([r[g] for g in FAR if g in r]))
        rows.append(r)
    D = pd.DataFrame(rows)
    p = RES / f"cr_222_mls__{key}.csv"; D.to_csv(p, index=False)
    print(D.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True, choices=["v2", "ls", "mixup"])
    main(ap.parse_args().key)
