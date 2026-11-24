r"""163_fig_ablations_col - the two ablations in a COLUMN-WIDTH 2x2.

experiment_name = 'ablations_col'
dataset_name    = 'imagenet1k'

WHY REPLOT
  The two original figures (`fig_recipe_barplot`, `fig_placement_bias`) were
  drawn with figsize=(5.5, ...) and 8 pt fonts, the TMLR column measure. In
  the SCITEPRESS column (3.35 in) that is a 0.61x scale and puts the ticks at
  about 4.9 pt. Shrinking the PDF does not help; the fonts have to be
  redefined for the target width. Same treatment given to the histogram in
  scripts/150.

  The four panels are the same data as the two figures, now in a single 2x2:
  top row the effect of the RECIPE, bottom row the effect of the POSITION.

WHAT CHANGES BESIDES THE SIZE
  The numeric labels on top of each bar are gone: at 1.55 in per panel they
  collide. The exact values are in the companion .txt.

Source: external_xai_ood/results/{stats,protocol_placement}/imagenet1k/
        (read only; never writes into the frozen repository).

Usage:
    $PYTHON_BIN fig5_code/fig5_placement.py
"""
import os

experiment_name = "ablations_col"
dataset_name    = "imagenet1k"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 6, "axes.titlesize": 6.2, "axes.labelsize": 6,
    "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "legend.fontsize": 5.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
XAI = HERE / "external_xai_ood"
OUTDIR = HERE / "figures" / experiment_name
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_R = OUTDIR / "163_ablations_recipe_col.pdf"
OUT_P = OUTDIR / "163_ablations_placement_col.pdf"

MODELS = ["ce", "ls", "mixup"]
MLAB = {"ce": "CE", "ls": "LS", "mixup": "Mixup"}
GCOL = {"near": "#E69F00", "far": "#009E73"}
MCOL = {"ce": "#0072B2", "ls": "#d55181", "mixup": "#4a3aa7"}
MS_DOT, MEW_DOT = 2.6, 0.9
SETS = ["ssb_hard", "ninco", "inaturalist", "openimage_o", "textures"]
SLAB = {"ssb_hard": "SSB", "ninco": "Ninco", "inaturalist": "iNat",
        "openimage_o": "OpenI-O", "textures": "Txtr"}
DEF = "legacy"


def _save(fig, out):
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"wrote {out}")


def fig_recipe(vals):
    """(a) AUROC and (b) FPR95 by training recipe."""
    fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.75),
                           gridspec_kw={"wspace": 0.42})
    x, wd = np.arange(len(MODELS)), 0.36
    for a_, metric, idx in ((ax[0], "AUROC (%)", 0), (ax[1], "FPR95 (%)", 1)):
        for k, g in enumerate(("near", "far")):
            a_.bar(x + (k - 0.5) * wd,
                   [vals[(m, g)][idx] for m in MODELS], wd * 0.94,
                   color=GCOL[g], label=f"{g}-OOD")
        a_.set_xticks(x); a_.set_xticklabels([MLAB[m] for m in MODELS])
        a_.set_ylabel(metric); a_.set_ylim(0, 100)
    ax[0].axhline(50, color="0.45", lw=0.7, ls=":", zorder=0)
    ax[0].set_title("(a) separation", pad=3)
    ax[1].set_title("(b) FPR95", pad=3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), handlelength=0.9,
               columnspacing=1.0, handletextpad=0.35)
    _save(fig, OUT_R)


def fig_placement(bias, aur, models):
    """(a) positional bias and (b) fixed corner against the mean of the 4 positions."""
    fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.42),
                           gridspec_kw={"wspace": 0.42})
    b = bias[(bias.definition == DEF) & (~bias.pos.isin(["all", "std"]))].copy()
    b["pos"] = b.pos.astype(int)
    w2 = 0.26
    ax[0].axhspan(-0.01, 0.01, color="0.85", alpha=0.5, zorder=0)
    for i, m in enumerate(models):
        dd = b[b.model == m].sort_values("pos")
        ax[0].bar(dd.pos + (i - 1) * w2, dd.signed_bias, width=w2 * 0.92,
                  color=MCOL[m], label=MLAB[m], zorder=3)
    ax[0].axhline(0, color="0.3", lw=0.7)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(["Top Left", "Top Right",
                           "Bottom Left", "Bottom Right"],
                          rotation=45, ha="right", rotation_mode="anchor")
    ax[0].set_ylabel("signed bias (ID)")
    ax[0].set_title("(a) positional bias", pad=3)
    ax[0].legend(frameon=False, loc="lower left", handlelength=0.8,
                 borderpad=0.15, labelspacing=0.2, handletextpad=0.3,
                 fontsize=5.4)

    a2 = aur[aur.definition == DEF]
    yc = {s: i for i, s in enumerate(SETS)}
    off = {m: (i - 1) * 0.22 for i, m in enumerate(models)}
    for m in models:
        for _, r in a2[a2.model == m].iterrows():
            y = yc[r.ood] + off[m]
            ax[1].plot([100 * r.auroc_pos0, 100 * r.auroc_avg4], [y, y],
                       color=MCOL[m], lw=1.0, zorder=2)
            ax[1].plot(100 * r.auroc_pos0, y, "o", mfc="white",
                       mec=MCOL[m], ms=MS_DOT, mew=MEW_DOT, zorder=3)
            ax[1].plot(100 * r.auroc_avg4, y, "o", mfc=MCOL[m],
                       mec=MCOL[m], ms=MS_DOT, zorder=3)
    ax[1].axhline(1.5, color="0.6", lw=0.6, ls="--")
    ax[1].set_yticks(range(len(SETS)))
    ax[1].set_yticklabels([SLAB[s] for s in SETS])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("AUROC (%)")
    ax[1].set_title("(b) placement protocol", pad=3)

    marca = dict(ls="none", marker="o", color="0.15", ms=MS_DOT)
    handles = [Line2D([], [], mfc="white", mec="0.15", mew=MEW_DOT,
                      label="fixed corner", **marca),
               Line2D([], [], mfc="0.15", mec="0.15",
                      label="4-position avg.", **marca)]
    ax[1].legend(handles=handles, loc="lower left", frameon=False,
                 markerscale=1, handlelength=0.8, borderpad=0.15,
                 labelspacing=0.2, handletextpad=0.3, fontsize=5.4)
    _save(fig, OUT_P)


def main():
    t1 = pd.read_csv(XAI / "results/stats/imagenet1k/table1.csv")
    d = t1[(t1.detector == "dom_proto")
           & (t1.ood.isin(["_mean_near", "_mean_far"]))]
    vals = {(r.model, r.ood.replace("_mean_", "")): (100 * r.auroc,
                                                     100 * r.fpr95)
            for r in d.itertuples()}
    RP = XAI / "results/protocol_placement/imagenet1k"
    bias = pd.read_csv(RP / "placement_bias.csv")
    aur = pd.read_csv(RP / "placement_auroc.csv")
    models = [m for m in MODELS if m in set(bias.model)]

    fig_recipe(vals)
    fig_placement(bias, aur, models)

    rec = "\n".join(
        f"    {MLAB[m]:<6} near AUROC {vals[(m,'near')][0]:5.1f}  "
        f"far {vals[(m,'far')][0]:5.1f}  |  FPR95 near "
        f"{vals[(m,'near')][1]:5.1f}  far {vals[(m,'far')][1]:5.1f}"
        for m in MODELS)
    b = bias[(bias.definition == DEF) & (~bias.pos.isin(["all", "std"]))]
    pl = "\n".join(
        f"    {MLAB[m]:<6} mean |bias| "
        f"{b[b.model == m].signed_bias.abs().mean():.3f}"
        for m in models)
    a2 = aur[aur.definition == DEF]
    gain = "\n".join(
        f"    {MLAB[m]:<6} mean gain of the 4-position mean "
        f"{100 * (a2[a2.model == m].auroc_avg4 - a2[a2.model == m].auroc_pos0).mean():+.2f} pt"
        for m in models)

    (OUTDIR / OUT_R.with_suffix(".txt").name).write_text(f"""\
{OUT_R.name} -- generated by fig5_code/fig5_placement.py

Source: external_xai_ood/results/stats/imagenet1k/table1.csv

WHAT IT SHOWS AND WHY
  Effect of the training recipe on the detector, in AUROC (a) and FPR95 (b),
  near-OOD against far-OOD. This is the central asymmetry of the paper: far
  stays flat across recipes and near slips.

  Replotted at 3.35 in with the fonts redefined for that width. The original
  was 5.5 in with 8 pt fonts; shrinking it would put the ticks at about
  4.9 pt.

Files:
  {OUT_R.name}   figure (column, 3.35 in)
  {OUT_R.with_suffix('.png').name}   400 dpi, for inspection
  {OUT_R.with_suffix('.txt').name}   this file

Numeric summary:
{rec}

How to read it:
  1. Flat far bars across recipes with falling near bars = regularization
     costs near, not far.
  2. FPR95 tells the same story at an operating threshold; read it together
     with (a), because high AUROC with high FPR95 means a bad tail.
""")

    (OUTDIR / OUT_P.with_suffix(".txt").name).write_text(f"""\
{OUT_P.name} -- generated by fig5_code/fig5_placement.py

Source: external_xai_ood/results/protocol_placement/imagenet1k/placement_*.csv

WHAT IT SHOWS AND WHY
  Effect of the query POSITION in the mosaic: (a) the per-position bias in the
  ID score and (b) what averaging the four positions recovers, set by set.
  This is the justification for the paper's default reading being the mean of
  the four.

Files:
  {OUT_P.name}   figure (column, 3.35 in)
  {OUT_P.with_suffix('.png').name}   400 dpi, for inspection
  {OUT_P.with_suffix('.txt').name}   this file

Numeric summary:
{pl}
{gain}

How to read it:
  1. In (a), a bar outside the grey band (+-0.01) means that position biases
     the ID score. The bias grows under LS and mixup.
  2. In (b), a hollow circle is the fixed corner and a filled one the mean of
     the four, following the legend in the lower-left of the panel itself,
     whose marker is the same size as the plot marker (both read MS_DOT), on
     purpose, so the correspondence is direct.
     A long segment means the position protocol matters on that set.
  3. A large bias in (a) without a gain in (b) would mean the bias is common
     to every class and cancels in AUROC.
""")
    for _o in (OUT_R, OUT_P):
        print(f"wrote {OUTDIR / _o.with_suffix('.txt').name}")


if __name__ == "__main__":
    main()
