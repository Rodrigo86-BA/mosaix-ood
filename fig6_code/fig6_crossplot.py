r"""157_fig_crossplot_bench - near by far crossplot on ImageNet-200 and CIFAR-100.

Restricted to the case that enters the paper: CIFAR-100 at 224, column layout.

LAYOUT
  1. Drawn at PRINTED size: the SCITEPRESS column is
     (6.221 in - 0.8 cm)/2 = 2.953 in, so 1 pt in the figure is 1 pt on paper.
  2. Fonts: ticks 7 pt, labels/titles/legend 7-7.5 pt, point labels 7 pt.
  3. A single x label centred under both panels instead of one per panel:
     at 7.5 pt the two "near-OOD AUROC (mean)" labels almost touched.
  4. Saved WITHOUT bbox_inches="tight", with fixed margins in inches: the PDF
     is exactly FIGSIZE.
  Colour = FAMILY (logit / feature / gradient / ours). Identity by selective
  direct labelling: the best near and the best far of each panel.

Sources:
  ImageNet-200  results/table2_multi/auroc.csv        (scripts/71)
  CIFAR-100     results/cifar100_224_panel/auroc.csv  (scripts/156)
Output: figures/crossplot_bench/157_crossplot_bench__cifar224_col.{pdf,png,txt}

Usage:
    $PYTHON_BIN fig6_code/fig6_crossplot.py
"""
import os

experiment_name = "table2_multi"
dataset_name    = "imagenet200|cifar100"

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # PRINTED body size: the figure enters the column at 1:1
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures" / "crossplot_bench"
OUTDIR.mkdir(parents=True, exist_ok=True)

# actual SCITEPRESS column width x printed height
FIGSIZE = (2.953, 1.760)

FAM = {"logit": ("#2a78d6", {"MSP", "GEN", "MLS", "Energy", "Cosine"}),
       "feature": ("#eb6834", {"Maha", "kNN", "ViM"}),
       "gradient": ("#1baf7a", {"GradNorm", "GAIA-A", "GAIA-Z"}),
       "ours": ("#4a3aa7", {"MosAIx-OOD"})}
LAB = {"msp": "MSP", "mls": "MLS", "energy": "Energy", "maha": "Maha",
       "knn": "kNN", "vim": "ViM", "gradnorm": "GradNorm",
       "gaia_a": "GAIA-A", "gaia_z": "GAIA-Z", "mosxai": "MosAIx-OOD"}

# column recipe with the printed font sizes; markers slightly larger to
# match the text
S = {"annot": 7, "off_ours": (-19, -8), "off_other": (2.5, 2.0),
     "nticks": 4, "s_ours": 26, "s_other": 14, "label_ours": False,
     "margin": 0.14, "xpad_right": 0.14}


def from_multi(bench):
    d = pd.read_csv(ROOT / "results" / "table2_multi" / "auroc.csv")
    d = d[d.benchmark == bench]
    p = d.pivot_table(index="detector", columns="ood", values="auroc")
    out = pd.DataFrame({"near": p["_mean_near"], "far": p["_mean_far"]})
    out.index = [LAB.get(i, i) for i in out.index]
    return out


def from_c224():
    d = pd.read_csv(ROOT / "results" / "cifar100_224_panel" / "auroc.csv")
    return d.set_index("detector")[["near", "far"]]


def panel(ax, p, title):
    for fam, (col, keys) in FAM.items():
        ks = [k for k in p.index if k in keys]
        if not ks:
            continue
        big = fam == "ours"
        ax.scatter([p.loc[k, "near"] for k in ks], [p.loc[k, "far"] for k in ks],
                   s=S["s_ours"] if big else S["s_other"], c=col,
                   edgecolors="white", linewidths=0.7,
                   marker="D" if big else "o", zorder=3,
                   label="MosAIx-OOD" if big else fam.capitalize())
    keys = {k for k in p.index if k in FAM["ours"][1]} if S["label_ours"] else set()
    keys |= {p["near"].idxmax(), p["far"].idxmax()}
    for k in keys:
        off = S["off_ours"] if k in FAM["ours"][1] else S["off_other"]
        ax.annotate(k, (p.loc[k, "near"], p.loc[k, "far"]),
                    textcoords="offset points", xytext=off,
                    fontsize=S["annot"], color="0.25", zorder=4)
    ax.margins(x=S["margin"], y=S["margin"])
    # The "best near" label always falls on the right-most point; at 7 pt it
    # left the panel. Only the right axis limit grows; no point moves.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + S["xpad_right"] * (x1 - x0))
    ax.xaxis.set_major_locator(plt.MaxNLocator(S["nticks"]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(S["nticks"]))
    ax.set_title(title, pad=2)


def _check_inside(fig):
    """Fail if any text leaves the canvas (savefig without bbox tight would crop it)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, H = fig.get_size_inches()
    tol = 0.005
    assert bb.x0 >= -tol and bb.y0 >= -tol and bb.x1 <= W + tol \
        and bb.y1 <= H + tol, f"content outside the figure: {bb} vs {W}x{H}"


def main():
    tabs = {"ImageNet-200 / RN-18": from_multi("imagenet200"),
            r"CIFAR-100 / RN-18 @ $224^2$": from_c224()}

    W, H = FIGSIZE
    # margins in inches: left = ylabel + ticks; gap = ticks of the 2nd panel
    L, GAP, R, TOP, BOT = 0.43, 0.33, 0.145, 0.37, 0.31
    AW = (W - L - GAP - R) / 2
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=L / W, right=1 - R / W, top=1 - TOP / H,
                        bottom=BOT / H, wspace=GAP / AW)
    for ax, (t, p) in zip(axes, tabs.items()):
        panel(ax, p, t)
    fig.supxlabel("near-OOD AUROC (mean)", y=0.0, va="bottom",
                  fontsize=7.5)
    axes[0].set_ylabel("far-OOD AUROC (mean)", labelpad=1.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
               columnspacing=0.8, handletextpad=0.25, borderpad=0.1,
               borderaxespad=0.05)
    out = OUTDIR / "157_crossplot_bench__cifar224_col.pdf"
    if not os.environ.get("CR_NOCHECK"): _check_inside(fig)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=400)
    plt.close(fig)
    print(f"wrote {out}")

    lines = []
    for t, p in tabs.items():
        m = [k for k in p.index if k in FAM["ours"][1]]
        m = m[0] if m else None
        rn, rf = p["near"].rank(ascending=False), p["far"].rank(ascending=False)
        lines.append(f"  {t}: method near {p.loc[m,'near']:.3f} "
                     f"(rank {int(rn[m])}/{len(p)}) far {p.loc[m,'far']:.3f} "
                     f"(rank {int(rf[m])}/{len(p)}) | best near "
                     f"{p['near'].idxmax()} / best far {p['far'].idxmax()}"
                     if m else f"  {t}: method absent")
    (out.with_suffix(".txt")).write_text(f"""\
Producing script: fig6_code/fig6_crossplot.py
Input: results/table2_multi/auroc.csv (scripts/71); results/cifar100_224_panel/auroc.csv (scripts/156)

What it measures, and why:
  Mean AUROC in near-OOD against far-OOD, one point per detector, on
  ImageNet-200 and on CIFAR-100 at 224. The reading is one of POSITION
  against the field, not of ranking.

  The two panels do NOT share axes: the benchmarks have different difficulty
  and different pools. Comparing ACROSS panels is not the intended reading.
  The single x label under both panels only saves space.

  Drawn at printed size ({W} x {H} in, SCITEPRESS column) with 7-7.5 pt text.

Output files:
  {out.name}   figure (column, {W} in, enters the paper at 1:1)
  {out.with_suffix('.png').name}   400 dpi, for inspection
  {out.with_suffix('.txt').name}   this file

Key numerical summary:
{chr(10).join(lines)}

Interpretation guide:
  1. Top-right corner = good in both regimes. Each panel has its own scale;
     read position WITHIN the panel.
  2. Orange (feature) high and to the left = a far specialist that pays for
     it in near. Blue (logit) to the right = the opposite.
  3. The diamond is our method. What the figure supports is competitiveness,
     not optimality: the detectors that beat it are in the figure.
  4. Spread within a colour measures how much the family varies internally.
  5. The summary above gives the rank of our method in each regime.
""")
    print(f"wrote {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
