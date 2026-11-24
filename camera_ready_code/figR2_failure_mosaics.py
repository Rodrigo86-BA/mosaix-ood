"""Response Figure R2: failure cases of the mosaic score (described in the Limitations section).

experiment_name = 'camera_ready_localization'
dataset_name    = 'imagenet1k (Table 1)'

WHAT IT SHOWS
-------------
A failure analysis: one mosaic with a misclassified ID query and low energy,
and the near-OOD case.

SELECTION (fixed rule, stated in the caption; extremes, not typical cases)
  (a) misclassified ID: among the ID queries whose predicted class differs from
      the true label (wnid folder of val), the one with the LOWEST GradCAM s.
  (b) near-OOD: among SSB-hard and NINCO, the query with the HIGHEST GradCAM s.
  s = E_q/(E_q + mean E_r), the s of the paper (null 1/2), read from the
  frozen CSV of scripts/76 (the GradCAM row of Table 1). The mosaic and the map
  are recomputed through the plumbing of cr_200 (same pools, panel and
  functions as scripts/76). GATE: recomputed s = frozen s (tol 1e-5) and live
  prediction = frozen prediction.

SUPPORTING STATISTICS (no network)
  threshold tau = 5th percentile of ID s (the FPR95 threshold); misclassification
  rate among the ID queries below tau against the overall rate; fraction of
  near-OOD above the ID median.

Outputs (results/camera_ready_localization/ of $MOSAIX_ROOT):
  cr_215_failure_mosaics.pdf/.png   figure (column width, 2.95 in; text at 7 pt)
  cr_215_failure_mosaics.csv        chosen cases
  cr_215_failure_mosaics.txt        companion

Usage:
  $PYTHON_BIN camera_ready_code/figR2_failure_mosaics.py
"""
import os

experiment_name = "camera_ready_localization"
dataset_name    = "imagenet1k"

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

import cr_200_loc_common as C
from cr_200_loc_common import m29, m73
import cr_201_loc_gate as G
from src.mosaic import build_mosaic                                # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
CATS = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
GN = {"id": "ID", "ssb_hard": "SSB-hard", "ninco": "NINCO"}


def share(L):
    return float(m29.d_from_quads(m73.quads_of(L, 14))["d"])


def short(c):
    return CATS[c].split(",")[0]


def main():
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    args76 = SimpleNamespace(n_per_set=1000, proto_sub=120, max_candidates=24,
                             fit_n=200000, k=50)
    ad = m73.ADAPTERS["imagenet1k"](args76, dev)
    net = SimpleNamespace(model=ad.model, W=ad.W, b=ad.b, body=ad.body, dev=dev)
    margs = C.map_args()
    gen = torch.Generator().manual_seed(m29.SEED)

    fz = pd.read_csv(G.FROZEN_CAM)
    wn = sorted(p.name for p in Path(ad.pools["id"][0]).parents[1].iterdir() if p.is_dir())
    idq = fz[fz.group == "id"].copy()
    idq["true"] = [wn.index(Path(ad.pools["id"][i]).parent.name) for i in idq.i]
    idq["wrong"] = idq.true != idq.pred
    tau = float(np.percentile(idq.cam, 5))
    err_all = float(idq.wrong.mean())
    err_low = float(idq[idq.cam < tau].wrong.mean())
    n_low = int((idq.cam < tau).sum())
    near = fz[fz.group.isin(["ssb_hard", "ninco"])]
    med = float(idq.cam.median())
    near_above = float((near.cam > med).mean())

    a = idq[idq.wrong].sort_values(["cam", "i"]).iloc[0]
    b = near.sort_values(["cam", "i"], ascending=[False, True]).iloc[0]
    cases = [("id", int(a.i), int(a.true)), (b.group, int(b.i), None)]

    print(f"fig8 | RN-50 CE | AMP=off | compile=off | tau={tau:.3f}", flush=True)
    rows, panels = [], []
    for g, i, true in cases:
        tile = ad.load_tiles([ad.pools[g][i]], g)
        c_live = int(ad.predict(tile, dev, 1)[0])
        c = int(fz[(fz.group == g) & (fz.i == i)].pred.iloc[0])
        mos = build_mosaic(tile[0], ad.panel_tiles(c), qpos=0)
        Am = C.layer4(net, mos)
        L = C.saliency("cam", net, mos, Am, c, 14, margs, gen).double().cpu()
        s = share(L)
        s_fz = float(fz[(fz.group == g) & (fz.i == i)].cam.iloc[0])
        r = dict(group=g, i=i, pred=c, pred_live=c_live, pred_name=short(c),
                 true=true, true_name=short(true) if true is not None else "",
                 s=s, s_frozen=s_fz, file=ad.pools[g][i])
        rows.append(r)
        img = (mos.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        panels.append((img, L, r))
        print("  " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(C.RES / "cr_215_failure_mosaics.csv", index=False)
    gate_s = float((D.s - D.s_frozen).abs().max())
    gate_p = bool((D.pred == D.pred_live).all())
    status = "PASSED" if gate_s < 1e-5 and gate_p else "FAILED"
    print(f"GATE s max|d|={gate_s:.1e}, live pred == frozen: {gate_p} -> {status}")

    fig, ax = plt.subplots(2, 2, figsize=(2.95, 3.05), constrained_layout=True)
    for k, (img, L, r) in enumerate(panels):
        U = F.interpolate(L[None, None].float(), size=img.shape[:2], mode="bilinear",
                          align_corners=False)[0, 0].numpy()
        U = np.maximum(U - U.mean(), 0)
        U = U / U.max() if U.max() > 0 else U
        if r["group"] == "id":
            t0 = f"(a) ID, true: {r['true_name']}\npredicted: {r['pred_name']}"
        else:
            t0 = f"(b) {GN[r['group']]} (near-OOD)\npredicted: {r['pred_name']}"
        ax[k, 0].imshow(img)
        ax[k, 0].set_title(t0, fontsize=7)
        ax[k, 1].imshow(img)
        ax[k, 1].imshow(U, cmap="magma", alpha=0.6, vmin=0, vmax=1)
        ax[k, 1].set_title(f"GradCAM\n$s = {r['s']:.3f}$", fontsize=7)
        for a_ in ax[k]:
            h = img.shape[0] // 2
            a_.add_patch(Rectangle((0, 0), h - 1, h - 1, fill=False, ec="#ff7f0e", lw=1.2))
            a_.axis("off")
    fig.savefig(C.RES / "cr_215_failure_mosaics.pdf")
    fig.savefig(C.RES / "cr_215_failure_mosaics.png", dpi=300)

    (C.RES / "cr_215_failure_mosaics.txt").write_text(f"""(1) Script: camera_ready_code/figR2_failure_mosaics.py
(2) What it shows: two failure cases of the score s (GradCAM, the GradCAM row of Table 1)
    on the real mosaic of the paper (query in the top-left corner, orange outline).
    (a) the misclassified ID query with the LOWEST s; (b) the near-OOD query (SSB-hard or
    NINCO) with the HIGHEST s. These are extremes chosen by a fixed rule, not typical cases.
(3) Files: cr_215_failure_mosaics.pdf/.png (column, 2.95 in); .csv (cases).
(4) Numbers:
{D.drop(columns=['file']).round(4).to_string(index=False)}
    GATE recomputed s vs frozen max|d| = {gate_s:.1e}; live prediction = frozen: {gate_p} -> {status}
    tau (5th percentile of ID s) = {tau:.3f}
    misclassification rate on ID: overall {err_all:.3f}; among the {n_low} ID with s < tau: {err_low:.3f}
    fraction of near-OOD (SSB-hard + NINCO) with s above the ID median ({med:.3f}): {near_above:.3f}
(5) Guide:
  1. (a) s ~ 0 on a misclassified ID query: the mosaic compares the query with prototypes
     of the WRONG class; the query lacks the evidence of that class and the score rejects it.
     For the detector this is an OOD false positive; for the user the map shows the
     reason (energy on the references, not on the query).
  2. Error rate among the ID below tau >> overall rate: ID rejections concentrate on
     misclassified queries.
  3. (b) near-OOD with high s: the query carries visual evidence of the predicted class
     (close semantics), so the map lights it up as it would an ID query. This is the main
     failure mode (FPR95 SSB-hard 0.85, cr_208).
""")
    print("[ok]")


if __name__ == "__main__":
    main()
