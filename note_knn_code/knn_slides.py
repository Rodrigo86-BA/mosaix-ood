"""170 - Two slides: why kNN loses to the prototype policy.

experiment_name = 'slides_knn'
dataset_name    = 'imagenet1k'

Slide 1 (example): one ID query and one far-OOD query, each built with the
references of both policies, with the CAM overlaid and the resulting s.

Slide 2 (measurement): the four quantities that explain the loss, read from
the per-sample output of note_knn_code/knn_vs_prototype.py.

Input:   results/knn_vs_prototype/169_per_sample_*.csv (the most recent)
Outputs: figures/slides_knn/170_s1_exemplo.pdf and .png
         figures/slides_knn/170_s2_medida.pdf and .png
         figures/slides_knn/170_slides_knn.text

Usage:
  $PYTHON_BIN note_knn_code/knn_slides.py
  $PYTHON_BIN note_knn_code/knn_slides.py --ood textures
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9

experiment_name = "slides_knn"
dataset_name = "imagenet1k"

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_code"
XAI = ROOT / "external_xai_ood"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(XAI))

m07 = importlib.import_module("rood_qualify_protocol")
m169 = importlib.import_module("knn_vs_prototype")
from src.exemplars import prototype_indices, query_adaptive_indices
from src.backbones import load_train_feats
from src.metrics import auroc
from torchvision.models import ResNet50_Weights

CATS = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

EPS = 1e-8
RES = ROOT / "results" / "knn_vs_prototype"
FIG = ROOT / "figures" / experiment_name
IMNORM = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
COLOR = {"prototype": "#0072B2", "knn": "#D55E00"}


def denorm(t):
    x = t.permute(1, 2, 0).numpy() * IMNORM[1] + IMNORM[0]
    return np.clip(x, 0, 1)


def panel(ax, mosaico, L, s, title_box):
    ax.imshow(denorm(mosaico))
    Lp = torch.nn.functional.interpolate(
        L[None, None], size=mosaico.shape[-2:], mode="bilinear",
        align_corners=False)[0, 0].numpy()
    ax.imshow(Lp, cmap="jet", alpha=0.45)
    n = mosaico.shape[-1]
    ax.axhline(n / 2, color="w", lw=1.2)
    ax.axvline(n / 2, color="w", lw=1.2)
    ax.add_patch(plt.Rectangle((0, 0), n / 2, n / 2, fill=False,
                               ec="#FFD700", lw=3))
    ax.set_title(title_box, fontsize=9)
    ax.text(0.02, 0.97, f"s = {s:.3f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=11, weight="bold", color="w",
            bbox=dict(fc="k", alpha=0.6, pad=2, ec="none"))
    ax.set_xticks([])
    ax.set_yticks([])


def slide1(df, args, dev, log):
    bb = m169.Backbone(dev)
    gtr, ltr = load_train_feats("ce")
    per, offs = m07.imagefolder_files(m07.P["imagenet1k"]["train"])

    piv = df.pivot_table(index=["group", "file", "pred"],
                         columns="policy", values="s").reset_index()
    piv["gap"] = piv["knn"] - piv["prototype"]

    choices = []
    d_ood = piv[piv.group == args.ood].sort_values("gap", ascending=False)
    choices.append(("far-OOD", d_ood.iloc[0]))
    d_id = piv[(piv.group == "id")]
    d_id = d_id.iloc[(d_id.gap.abs()).argsort()[:1]]
    choices.insert(0, ("in-distribution", d_id.iloc[0]))

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.8))
    fig.subplots_adjust(hspace=0.30)
    for r, (label, row) in enumerate(choices):
        img_path, c = row["file"], int(row["pred"])
        nome_cls = CATS[c] if c < len(CATS) else str(c)
        x = m07.load_batch([img_path], pre=(row["group"] == "id"))
        axes[r, 0].imshow(denorm(x[0]))
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].set_xlabel(f"{label} query\npredicted: {nome_cls}",
                              fontsize=9.5)
        for k in axes[r, 0].spines.values():
            k.set_edgecolor("#FFD700")
            k.set_linewidth(3)

        z, _ = bb.gap_e_pred(x, 1)
        for col, pol in enumerate(("prototype", "knn"), start=1):
            ridx = (prototype_indices(gtr, ltr, c, 3) if pol == "prototype"
                    else query_adaptive_indices(gtr, ltr, c, z[0], 3))
            ref_paths = [m169.idx_to_path(j, per, offs) for j in ridx]
            R = m07.load_batch(ref_paths, pre=False)
            mos = m169.build_mosaic(x[0], R)
            with torch.no_grad():
                A = bb.body(mos[None].to(dev))
                L = torch.einsum("c,chw->hw", bb.W[c].to(dev), A[0])
                mu = L.mean()
                E = torch.relu(L - mu).cpu()
            h = E.shape[-1] // 2
            eq = E[:h, :h].mean()
            er = (E[:h, h:].mean() + E[h:, :h].mean() + E[h:, h:].mean()) / 3
            s = float(eq / (eq + er + EPS))
            name = ("prototype references" if pol == "prototype"
                    else "kNN references")
            panel(axes[r, col], mos, E, s, name if r == 0 else "")
            log(f"  slide1 {label:18s} {pol:10s} s={s:.3f}")

    fig.suptitle("The query sits in the yellow tile. The other three tiles "
                 "are the references the policy chose.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.6)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"170_s1_exemplo.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


def slide2(df, log):
    NEAR = [k for k, v in m07.P["ood_sets"].items() if v["split"] == "near"]
    FAR = [k for k, v in m07.P["ood_sets"].items() if v["split"] == "far"]
    df = df.copy()
    df["reg"] = df.group.map(
        lambda g: "ID" if g == "id" else ("near" if g in NEAR else "far"))

    fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.5))
    width, pos = 0.35, np.arange(3)
    regs = ["ID", "near", "far"]

    def bars(a, values_, title_box, ylab):
        for k, pol in enumerate(("prototype", "knn")):
            a.bar(pos + (k - 0.5) * width, [values_[(pol, r)] for r in regs],
                  width, label=pol, color=COLOR[pol])
        a.set_xticks(pos)
        a.set_xticklabels(regs)
        a.set_title(title_box, fontsize=9)
        a.set_ylabel(ylab, fontsize=8)
        a.spines[["top", "right"]].set_visible(False)

    er = {(p, r): df[(df.policy == p) & (df.reg == r)].e_ref.mean()
          for p in COLOR for r in regs}
    bars(ax[0], er, "Reference energy, the bar the query\n"
           "is measured against", "$\\bar E_{ref}$")
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    for k, pol in enumerate(("prototype", "knn")):
        d = 100 * (er[(pol, "far")] - er[(pol, "ID")]) / er[(pol, "ID")]
        ax[0].annotate(f"{d:+.0f}%\nID to far", xy=(2 + (k - 0.5) * width,
                                                    er[(pol, "far")]),
                       xytext=(0, 6), textcoords="offset points",
                       ha="center", fontsize=8.5, weight="bold",
                       color=COLOR[pol])
    ax[0].set_ylim(0, max(er.values()) * 1.55)

    cc = {(p, r): df[(df.policy == p) & (df.reg == r)].cos_ref_cen.mean()
          for p in COLOR for r in regs}
    bars(ax[1], cc, "Reference typicality,\n"
           "cosine to the class centroid", "cos(ref, centroid)")

    cr = {}
    for p in COLOR:
        for r in regs:
            d = df[(df.policy == p) & (df.reg == r)]
            cr[(p, r)] = np.corrcoef(d.e_q, d.e_ref)[0, 1]
    bars(ax[2], cr, "Query and reference move together\n"
           "only under kNN", "corr($\\bar E_q$, $\\bar E_{ref}$)")
    ax[2].axhline(0, color="k", lw=0.8)

    for k, pol in enumerate(("prototype", "knn")):
        d = df[df.policy == pol]
        sid = d[d.reg == "ID"].s.values
        vals = [auroc(sid, d[d.reg == r].s.values) for r in ("near", "far")]
        ax[3].bar(np.arange(2) + (k - 0.5) * width, vals, width,
                  label=pol, color=COLOR[pol])
        for j, v in enumerate(vals):
            ax[3].text(j + (k - 0.5) * width, v + 0.004, f"{v:.3f}",
                       ha="center", fontsize=8)
    ax[3].set_xticks(np.arange(2))
    ax[3].set_xticklabels(["near", "far"])
    ax[3].set_ylim(0.6, 1.0)
    ax[3].set_title("What the moving bar costs,\nin AUROC", fontsize=9)
    ax[3].set_ylabel("AUROC", fontsize=8)
    ax[3].spines[["top", "right"]].set_visible(False)
    ax[3].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"170_s2_medida.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    log("")
    log("slide 2 values")
    for name, d in (("E_ref", er), ("cos(ref,centroid)", cc),
                    ("corr(Eq,Eref)", cr)):
        for p in ("prototype", "knn"):
            log(f"  {name:18s} {p:10s} " +
                "  ".join(f"{r} {d[(p, r)]:+.3f}" for r in regs))
    return er, cc, cr


def main(args):
    FIG.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    csvs = sorted(RES.glob("169_per_sample_*.csv"))
    if not csvs:
        sys.exit("run note_knn_code/knn_vs_prototype.py first")
    df = pd.read_csv(csvs[-1])
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"reading {csvs[-1].name} | {len(df)} rows")
    slide1(df, args, dev, log)
    er, cc, cr = slide2(df, log)

    (FIG / "170_slides_knn.txt").write_text("\n".join([
        "Producing script: note_knn_code/knn_slides.py",
        f"Data: results/knn_vs_prototype/{csvs[-1].name}", "",
        "What the slides measure, and why:",
        "  The kNN policy picks the references nearest THE QUERY. For an OOD",
        "  query that means picking atypical members of the predicted class.",
        "  The reference stops being a fixed bar and starts following the",
        "  query, and the ratio s = Eq/(Eq+Er) cancels part of the fall that",
        "  should signal OOD.", "",
        "Output files:",
        "  170_s1_exemplo.pdf/.png  one ID query and one far-OOD query, each",
        "     under both policies, with the CAM overlaid and s annotated",
        "  170_s2_medida.pdf/.png   four panels: E_ref, reference typicality,",
        "     correlation Eq against Er, and AUROC",
        "  170_slides_knn.txt       this companion", "",
        "Numeric summary:",
        *[f"  {l}" for l in lines], "",
        "Interpretation guide:",
        "  1. Panel 1: if E_ref falls from ID to far under kNN and not under",
        "     prototype, the bar is moving along with the query.",
        "  2. Panel 2: a lower cos(ref, centroid) under kNN for far queries",
        "     shows the selection drifting off the centre of the class.",
        "  3. Panel 3: a positive correlation under kNN is the cancellation in",
        "     the ratio. Under prototype it stays near zero or negative.",
        "  4. Panel 4: the final count, in AUROC.",
        "  5. In slide 1, s rises under kNN for the OOD query and barely moves",
        "     for the ID one. That is the whole effect in one example.",
    ]) + "\n")
    log(f"wrote {FIG.relative_to(ROOT)}/170_slides_knn.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", default="textures")
    main(ap.parse_args())
