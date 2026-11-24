"""Figure 2, the method diagram, with REAL images and maps.

experiment_name = 'fig_method'
dataset_name    = 'imagenet1k'

WHY
---
Reads the dump of fig2_code/fig2_method_data.py (which runs the whole path live, with gates
against the frozen run) and draws the diagram. NOTHING here is illustrative:
the training thumbnails are the training images actually considered, the three
references are the ones the run policy chose, the "embeddings" cards are real
layer4 channels, the CAM and the rectification are the measured 14x14 grids,
and the printed `s` is the one from the frozen dump.

The visual grammar (palette, stacked cards, rounded backbone box, Dark2
arrows, dashed grouping box) comes from the template in
`figures/squematic/CSKD_overview.svg`. The retrained classification head and
the self-distillation of that template do NOT appear: they do not exist in
this method.

Outputs (figures/method/):
  fig2_method.pdf   plus fig2_method.txt (companion)

Usage:
  $PYTHON_BIN fig2_code/fig2_method.py
  $PYTHON_BIN fig2_code/fig2_method.py --query ood
"""
import os

experiment_name = "fig_method"
dataset_name    = "imagenet1k"

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans", "font.size": 7,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "method"
OUT.mkdir(parents=True, exist_ok=True)

BLUE   = "#cfe2f3"
BLUE_BAR = "#1f4e79"
PURPLE = "#d9d2e9"
GREY   = "#cccccc"
YELLOW = "#ffe599"
GREEN  = "#1b9e77"
ORANGE = "#d95f02"
INK    = "#000000"

PAGE_W = 7.60
FIG_H  = 5.10
SC     = PAGE_W / 6.9
SKY    = "#2e75b6"


SNOW = ROOT / "figures" / "method" / "assets" / "snowflake_cskd.png"
_SNOW = None


def snowflake(fig, xc, yc, r=0.085):
    """The snowflake from the template (`CSKD_overview.svg`), extracted from the SVG
    itself and reused as a PNG with alpha, instead of the approximate drawing that
    was here before. Coordinates and radius in INCHES."""
    global _SNOW
    if _SNOW is None:
        _SNOW = plt.imread(SNOW)
    a = fig.add_axes([(xc - r) / PAGE_W, (yc - r) / FIG_H,
                      2 * r / PAGE_W, 2 * r / FIG_H])
    a.set_xticks([]); a.set_yticks([]); a.set_facecolor("none")
    for sp in a.spines.values():
        sp.set_visible(False)
    a.imshow(_SNOW)
    a.set_zorder(9)
    a.patch.set_alpha(0.0)
    return a


def ax_at(fig, x, y, w, h):
    """An axis in INCHES, with the origin at the bottom-left corner of the figure."""
    a = fig.add_axes([x / PAGE_W, y / FIG_H, w / PAGE_W, h / FIG_H])
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    return a


def txt(fig, x, y, s, size=6.4, weight="normal", color=INK, ha="center",
        va="center", style="normal", rot=0):
    return fig.text(x / PAGE_W, y / FIG_H, s, fontsize=size * SC,
                    fontweight=weight,
                    color=color, ha=ha, va=va, style=style,
                    rotation=rot, zorder=10)


def arrow(fig, x0, y0, x1, y1, color=INK, lw=1.5, ms=9, ls="-"):
    fig.add_artist(FancyArrowPatch(
        (x0 / PAGE_W, y0 / FIG_H), (x1 / PAGE_W, y1 / FIG_H),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms,
        lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=8))


def block(fig, x, y, w, h, label, fill=BLUE, size=7.0, pad=0.02, lw=1.2):
    """Caixa arredondada do template (backbone, resultado)."""
    fig.add_artist(FancyBboxPatch(
        (x / PAGE_W, y / FIG_H), w / PAGE_W, h / FIG_H,
        boxstyle=f"round,pad=0,rounding_size={pad}", transform=fig.transFigure,
        facecolor=fill, edgecolor=INK, lw=lw, zorder=4))
    if label:
        txt(fig, x + w / 2, y + h / 2, label, size=size, weight="bold")


def show(ax, img, cmap=None, vmin=None, vmax=None, ec=INK, lw=0.8, ls="-"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest" if cmap else "antialiased")
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(ec)
        s.set_linewidth(lw); s.set_linestyle(ls)
    return ax


def cards(fig, x, y, w, h, maps, cmap="Purples", dx=0.070, dy=0.058):
    """Stacked cards from the template. One difference from the template: the back
    sheets are NOT flat fills. Each one is a real layer4 channel, so that the
    "embeddings" box is not decorative."""
    axes = []
    for k in range(len(maps) - 1, -1, -1):
        a = ax_at(fig, x + k * dx, y + k * dy, w, h)
        a.set_zorder(3 + (len(maps) - k))
        M = np.asarray(maps[k], float)
        hi = float(np.percentile(M, 97)) or float(M.max()) or 1.0
        show(a, M, cmap=cmap, vmin=0.0, vmax=hi, lw=0.8)
        axes.append(a)
    return axes


def badge(fig, x, y, ok):
    """Correct or incorrect mark. A white outline instead of a disc: a disc drawn in
    figure fractions becomes an ellipse, since the figure is not square."""
    t = txt(fig, x, y, "\u2713" if ok else "\u2717", size=7.0, weight="bold",
            color=GREEN if ok else "#c0392b")
    t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def color_row(fig, xc, y, parts, size):
    """Draws [(text, colour), ...] side by side, the set centred on xc.

    It measures each piece with the renderer and only then positions it: mathtext
    has no predictable width, and guessing offsets misaligns the fraction."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ts = [txt(fig, 0, y, t_, size=size, color=c_, ha="left")
          for t_, c_ in parts]
    ws = [t.get_window_extent(renderer=r).width / fig.dpi for t in ts]
    x = xc - sum(ws) / 2
    for t, w in zip(ts, ws):
        t.set_position((x / PAGE_W, y / FIG_H))
        x += w
    return sum(ws)


def elbow(fig, pts, color=INK, lw=1.5, ms=9):
    """Elbow connector: the polyline first, then the head on the last segment."""
    X = [p[0] / PAGE_W for p in pts]
    Y = [p[1] / FIG_H for p in pts]
    fig.add_artist(plt.Line2D(X[:-1], Y[:-1], transform=fig.transFigure,
                              lw=lw, color=color, solid_capstyle="butt",
                              solid_joinstyle="miter", zorder=8))
    arrow(fig, *pts[-2], *pts[-1], color=color, lw=lw, ms=ms)


def main(args):
    z = np.load(ROOT / "results" / "method_fig" /
                f"method_data__{args.key}__{args.query}.npz", allow_pickle=True)
    cname = str(z["cname"])
    s, mu = float(z["s"]), float(z["mu"])
    eq, er = float(z["eq"]), float(z["er"])
    L, E, A = z["L"], z["E"], z["A"]
    ch, contrib = z["A_ch"], z["A_contrib"]

    fig = plt.figure(figsize=(PAGE_W, FIG_H))
    fig.patch.set_facecolor("white")

    Y0, YA, YB = 4.10, 2.36, 0.72
    QS, TH, MS = 0.66, 0.44, 1.20
    Y_ELB = 1.44

    bi, bp, bch = z["b_img"], z["b_pred"], z["b_chosen"]
    bname, bcls = str(z["b_name"]), int(z["b_cls"])
    nb = len(bi)

    TS = 0.64
    xt = 0.10
    ti = np.concatenate([bi[np.where(bch)[0][0]][None], z["t_img"]])
    for k in range(len(ti) - 1, -1, -1):
        a = ax_at(fig, xt + k * 0.046, Y0 - TS / 2 - k * 0.042, TS, TS)
        a.set_zorder(4 + (len(ti) - k))
        show(a, ti[k], ec=GREEN if k == 0 else INK, lw=1.6 if k == 0 else 0.9)
    txt(fig, xt + TS / 2 + 0.05, Y0 + TS / 2 + 0.22, "Training set",
        size=6.6, weight="bold")

    xb0, wb0, hb0 = xt + TS + 0.55, 0.80, 0.60
    arrow(fig, xt + TS + 0.20, Y0, xb0 - 0.04, Y0, color=GREEN)
    block(fig, xb0, Y0 - hb0 / 2, wb0, hb0, "Backbone\n(frozen)", size=6.4)
    snowflake(fig, xb0 + wb0 - 0.10, Y0 + hb0 / 2 - 0.09)

    bgw = nb * TH + (nb - 1) * 0.05
    ox0, ow = xb0 + wb0 + 0.55, bgw + 0.28
    oy0, oh = Y0 - 0.42, 0.86
    arrow(fig, xb0 + wb0 + 0.05, Y0, ox0 - 0.05, Y0, color=GREEN)
    fig.add_artist(Rectangle(
        (ox0 / PAGE_W, oy0 / FIG_H), ow / PAGE_W, oh / FIG_H,
        transform=fig.transFigure, facecolor="none", edgecolor="#5a5a5a",
        lw=0.9, zorder=2))
    txt(fig, ox0 + ow / 2, Y0 + 0.34,
        f"{bname.title()} class example", size=6.2, weight="bold")
    for k in range(nb):
        xa = ox0 + 0.13 + k * (TH + 0.05)
        ya = Y0 - 0.20
        a = ax_at(fig, xa, ya, TH, TH)
        a.set_zorder(5)
        keep = bool(bch[k])
        show(a, bi[k], ec=GREEN if keep else "#c0392b",
             lw=1.9 if keep else 1.4)
        badge(fig, xa + TH - 0.08, ya + TH - 0.08, int(bp[k]) == bcls)
    krej = [k for k in range(nb) if int(bp[k]) != bcls]
    if krej:
        xa = ox0 + 0.13 + krej[0] * (TH + 0.05)
        txt(fig, xa + TH / 2, Y0 - 0.31,
            f"classified as '{str(z['b_rej_name'][0])}'", size=5.2,
            color="#c0392b")
    txt(fig, ox0 + ow / 2, oy0 - 0.10,
        "keep the first 3 classified correctly, ranked by cosine",
        size=5.7)
    txt(fig, ox0 + ow / 2, oy0 - 0.20,
        "similarity to the class centroid in feature space",
        size=5.7)

    gw = 3 * TH + 2 * 0.05
    bx0, bw = ox0 + ow + 0.60, gw + 0.26
    by0, bh = Y0 - 0.32, 0.76
    arrow(fig, ox0 + ow + 0.06, Y0, bx0 - 0.05, Y0, color=GREEN)
    for k in (2, 1):
        fig.add_artist(Rectangle(
            ((bx0 + k * 0.052) / PAGE_W, (by0 + k * 0.050) / FIG_H),
            bw / PAGE_W, bh / FIG_H, transform=fig.transFigure,
            facecolor="#f0f0f0", edgecolor="#808080", lw=0.8, zorder=2))
    fig.add_artist(Rectangle(
        (bx0 / PAGE_W, by0 / FIG_H), bw / PAGE_W, bh / FIG_H,
        transform=fig.transFigure, facecolor="white", edgecolor=INK, lw=1.2,
        zorder=3))
    txt(fig, bx0 + bw / 2 + 0.05, by0 + bh + 0.30,
        "Reference bank", size=6.6, weight="bold")
    txt(fig, bx0 + bw / 2 + 0.05, by0 + bh + 0.18,
        "one row per class", size=5.8, style="italic")
    txt(fig, bx0 + bw / 2, by0 + bh - 0.10, f"row of '{bname}'", size=5.8)
    for k in range(3):
        xa = bx0 + 0.12 + k * (TH + 0.05)
        a = ax_at(fig, xa, Y0 - 0.26, TH, TH)
        a.set_zorder(5)
        show(a, bi[np.where(bch)[0][k]], ec=GREEN, lw=1.7)

    x = 0.10
    aq = ax_at(fig, x, YA - QS / 2, QS, QS)
    show(aq, z["q_img"], ec=ORANGE, lw=1.8)
    txt(fig, x + QS / 2, YA + QS / 2 + 0.11,
        "Test image" if args.query == "id" else "Test image (OOD)",
        size=6.6, weight="bold", color=ORANGE)
    txt(fig, x + QS / 2, YA - QS / 2 - 0.13, r"$H\times W\times C$",
        size=6.6)

    xb, wb, hb = x + QS + 0.48, 0.80, 0.60
    arrow(fig, x + QS + 0.05, YA, xb - 0.04, YA, color=ORANGE)
    block(fig, xb, YA - hb / 2, wb, hb, "Backbone\n(frozen)", size=6.4)
    snowflake(fig, xb + wb - 0.10, YA + hb / 2 - 0.09)

    xg, wg, hg = xb + wb + 0.48, 1.02, 0.54
    arrow(fig, xb + wb + 0.05, YA, xg - 0.05, YA)
    ag = ax_at(fig, xg, YA - hg / 2, wg, hg)
    v = np.asarray(z["bar_v"], float)
    v = (v - v.min()) / max(v.max() - v.min(), 1e-9) * 0.92 + 0.08
    cols = [BLUE_BAR] + [YELLOW] * (len(v) - 1)
    ag.bar(np.arange(len(v)), v, width=0.78, color=cols, edgecolor=INK,
           linewidth=0.45)
    ag.set_xlim(-0.7, len(v) - 0.3); ag.set_ylim(0, 1.12)
    for sp in ("top", "right"):
        ag.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ag.spines[sp].set_visible(True); ag.spines[sp].set_linewidth(0.8)
    txt(fig, xg + wg / 2, YA + hg / 2 + 0.12, "class logits", size=6.3,
        weight="bold")
    txt(fig, xg + 0.06, YA - hg / 2 - 0.12,
        rf"$\hat{{c}}$ = '{cname}'", size=6.2, weight="bold", color=BLUE_BAR,
        ha="left")

    xk, wk, hk = xg + wg + 0.52, gw + 0.26, 0.80
    ky0 = YA - 0.32
    arrow(fig, xg + wg + 0.06, YA, xk - 0.05, YA)
    for k in (2, 1):
        fig.add_artist(Rectangle(
            ((xk + k * 0.052) / PAGE_W, (ky0 + k * 0.050) / FIG_H),
            wk / PAGE_W, hk / FIG_H, transform=fig.transFigure,
            facecolor="#f0f0f0", edgecolor="#808080", lw=0.8, zorder=2))
    fig.add_artist(Rectangle(
        (xk / PAGE_W, ky0 / FIG_H), wk / PAGE_W, hk / FIG_H,
        transform=fig.transFigure, facecolor="white", edgecolor=INK, lw=1.2,
        zorder=3))
    txt(fig, xk + wk / 2 + 0.05, ky0 + hk + 0.20, "Reference bank", size=6.4,
        weight="bold")
    txt(fig, xk + wk / 2, ky0 + hk - 0.10,
        rf"row of $\hat{{c}}$ = '{cname}'", size=5.8)
    for k in range(3):
        a = ax_at(fig, xk + 0.12 + k * (TH + 0.05), YA - 0.26, TH, TH)
        a.set_zorder(5)
        show(a, z["r_img"][k], ec=GREEN, lw=1.7)
    txt(fig, xk + wk / 2, ky0 - 0.13, "references", size=6.3, weight="bold",
        color=GREEN)

    xm = xk + wk + 0.52
    arrow(fig, xk + wk + 0.06, YA, xm - 0.05, YA)
    am = ax_at(fig, xm, YA - MS / 2, MS, MS)
    show(am, z["m_img"], lw=1.0)
    am.add_patch(Rectangle((-.5, -.5), 224, 224, fill=False, ec=ORANGE,
                           lw=2.0, zorder=6))
    txt(fig, xm + MS * 0.25, YA + MS / 2 - 0.10, "query", size=5.8,
        weight="bold", color="white")
    for k, lab in enumerate(["ref 1", "ref 2", "ref 3"]):
        cx = xm + MS * (0.75 if k == 0 else (0.25 if k == 1 else 0.75))
        cy = YA + MS * (0.28 if k == 0 else -0.22)
        txt(fig, cx, cy, lab, size=5.4, color="white")
    txt(fig, xm + MS * 0.24, YA + MS / 2 + 0.11, r"2$\times$2 mosaic",
        size=6.6, weight="bold")
    txt(fig, xm + MS * 0.22, YA - MS / 2 - 0.13,
        r"$2H\times 2W\times C$", size=6.6)

    xb2, wb2, hb2 = 0.30, 0.86, 0.66
    elbow(fig, [(xm + MS / 2, YA - MS / 2 - 0.02), (xm + MS / 2, Y_ELB),
                (xb2 - 0.17, Y_ELB), (xb2 - 0.17, YB), (xb2 - 0.03, YB)])
    txt(fig, (xb2 + xm) / 2, Y_ELB + 0.09,
        "one forward of the assembled mosaic", size=6.0, style="italic")

    block(fig, xb2, YB - hb2 / 2, wb2, hb2, "Backbone\n(frozen)", size=6.4)
    snowflake(fig, xb2 + wb2 - 0.11, YB + hb2 / 2 - 0.10)
    txt(fig, xb2 + wb2 / 2, YB - hb2 / 2 - 0.11, "same weights", size=5.8,
        style="italic")

    xe, ES = xb2 + wb2 + 0.34, 0.68
    arrow(fig, xb2 + wb2 + 0.05, YB, xe - 0.05, YB)
    cards(fig, xe, YB - ES / 2 - 0.058, ES, ES, A[:3])
    txt(fig, xe + ES / 2 + 0.07, YB + ES / 2 + 0.16, "intermediate features",
        size=6.4, weight="bold")
    txt(fig, xe + ES / 2 + 0.07, YB - ES / 2 - 0.19,
        r"$A \in \mathbb{R}^{f_c \times 2f_h \times 2f_w}$", size=6.4)

    xc, CS = xe + ES + 0.86, 0.80
    arrow(fig, xe + ES + 0.26, YB, xc - 0.05, YB)
    txt(fig, (xe + ES + 0.26 + xc) / 2, YB + 0.15,
        r"$\langle w_{\hat{c}},\, A\rangle$", size=6.8)
    ac = ax_at(fig, xc, YB - CS / 2, CS, CS)
    show(ac, L, cmap="magma", lw=0.8)
    txt(fig, xc + CS / 2, YB + CS / 2 + 0.12,
        r"CAM $L_{\hat{c}}$ ($2f_h \times 2f_w$)", size=6.4, weight="bold")

    xz = xc + CS + 1.00
    arrow(fig, xc + CS + 0.05, YB, xz - 0.05, YB)
    txt(fig, (xc + CS + xz) / 2, YB + 0.15, r"$\mathrm{relu}(L_{\hat{c}}-\mu)$",
        size=6.6)
    az = ax_at(fig, xz, YB - CS / 2, CS, CS)
    show(az, E, cmap="magma", lw=0.8)
    az.add_patch(Rectangle((-.5, -.5), 7, 7, fill=False, ec=ORANGE, lw=1.7,
                           zorder=6))
    for x0, y0 in ((6.5, -.5), (-.5, 6.5), (6.5, 6.5)):
        az.add_patch(Rectangle((x0, y0), 7, 7, fill=False, ec=GREEN, lw=1.5,
                               zorder=6))
    txt(fig, xz + CS / 2, YB + CS / 2 + 0.12, r"rectified $E$", size=6.4,
        weight="bold")
    for x_, y_, lab, col, ha_, va_ in (
            (xz - 0.06, YB + CS / 2 + 0.05, r"$\bar E_q$", ORANGE,
             "right", "bottom"),
            (xz + CS + 0.06, YB - CS / 2 - 0.05, r"$\bar E_\mathrm{ref}$",
             GREEN, "left", "top")):
        t_ = txt(fig, x_, y_, lab, size=6.8, weight="bold", color=col,
                 ha=ha_, va=va_)
        t_.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    xs = xz + CS + 0.40
    ws = (xm + MS) - xs
    arrow(fig, xz + CS + 0.05, YB, xs - 0.05, YB)
    fig.add_artist(FancyBboxPatch(
        (xs / PAGE_W, (YB - 0.36) / FIG_H), ws / PAGE_W, 0.72 / FIG_H,
        boxstyle="round,pad=0,rounding_size=0.02", transform=fig.transFigure,
        facecolor="none", edgecolor=INK, lw=1.2, zorder=4))
    cx = xs + ws / 2 + 0.14
    txt(fig, cx - 0.42, YB, "$s\\;=$", size=8.4, ha="right")
    color_row(fig, cx, YB + 0.15, [(r"$\bar E_q$", ORANGE)], 8.4)
    wden = color_row(fig, cx, YB - 0.17,
                     [(r"$\bar E_q$", ORANGE), (r"$\,+\,$", INK),
                      (r"$\bar E_\mathrm{ref}$", GREEN)], 8.4)
    fig.add_artist(plt.Line2D([(cx - wden / 2) / PAGE_W,
                               (cx + wden / 2) / PAGE_W],
                              [(YB + 0.03) / FIG_H, (YB + 0.03) / FIG_H],
                              transform=fig.transFigure, lw=1.0, color=INK,
                              zorder=10))

    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    xlo, xhi = PAGE_W, 0.0
    for a in list(fig.axes) + list(fig.texts) + list(fig.artists):
        try:
            bb = a.get_window_extent(renderer=rr)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        xlo = min(xlo, bb.x0 / fig.dpi)
        xhi = max(xhi, bb.x1 / fig.dpi)
    PADX = 0.07
    x0b, x1b = xlo - PADX, xhi + PADX
    print(f"  conteudo: x {xlo:.3f} .. {xhi:.3f} in  ->  caixas "
          f"{x0b:.3f} .. {x1b:.3f}  (canvas {PAGE_W})")
    for y0_, y1_, col in ((3.38, 5.02, SKY), (0.06, 3.26, ORANGE)):
        fig.add_artist(Rectangle(
            (x0b / PAGE_W, y0_ / FIG_H), (x1b - x0b) / PAGE_W,
            (y1_ - y0_) / FIG_H, transform=fig.transFigure, facecolor="none",
            edgecolor=col, lw=1.3, linestyle=(0, (6, 4)), zorder=1))
    txt(fig, 0.10, by0 + bh + 0.30, "Run once for the whole train dataset",
        size=7.0, weight="bold", color=SKY, ha="left")
    txt(fig, 0.10, ky0 + hk + 0.20, "Run per query", size=7.0, weight="bold",
        color=ORANGE, ha="left")

    stem = f"fig2_method{'' if args.query == 'id' else '_ood'}"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(pdf, format="pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    plt.close(fig)

    (OUT / f"{stem}.txt").write_text(f"""\
{stem}.pdf  --  Figure 2 of the VISAPP paper (the method diagram)

1. PRODUCING SCRIPT
   fig2_code/fig2_method.py   (data: fig2_code/fig2_method_data.py)
   $PYTHON_BIN fig2_code/fig2_method.py --query {args.query}

2. WHAT THE FIGURE SHOWS AND WHY
   The complete readout path, from the test image to the scalar s, with REAL
   data at every point. No box is illustrative. This is the figure that
   answers "what exactly is computed", separate from Figure 1, which shows
   the effect. Drawing the backbone TWICE is literal: it runs once to predict
   the class of the query (which decides which references enter) and once
   over the assembled mosaic. The weights are the same and they are frozen.
   The method trains nothing.

3. OUTPUT FILES
   {stem}.pdf   vector, SCITEPRESS PAGE width (6.9 in)
   {stem}.png   220 dpi, for quick inspection
   {stem}.txt   this file

4. NUMBERS (backbone {args.key}, ResNet-50 / ImageNet-1K from scratch)
   predicted class    {int(z['pred'])} ('{cname}')
   query              {Path(str(z['qpath'])).name}   [{str(z['group'])}]
   references         {', '.join(str(x) for x in z['r_ids'])} (train ids)
   references classified correctly   {int((z['r_pred'] == int(z['pred'])).sum())}/3
   layer4 channels displayed         {ch.tolist()}
     contribution w_c[d]*GAP(A_d)    {np.round(contrib, 3).tolist()}
   mu (mosaic mean)                  {mu:.4f}
   E_q                               {eq:.4f}
   E_ref                             {er:.4f}
   s                                 {s:.4f}
   s WITHOUT the relu                {float(z['raw_ratio']):.6f}   (= 3/2 ALWAYS)

5. READING GUIDE
   1) The DASHED RULE is the central claim: above it nothing depends on the
      query, and the cost is paid once for the whole dataset; below it is
      what runs for each test image.
   2) The top band builds the bank. It deliberately uses a class DIFFERENT
      from the query: the construction is indifferent to the query, and using
      the same class would give the opposite impression. The class shown is
      one of the 30 where the correctness filter actually bites, so the
      candidate marked in red is a MEASURED rejection, not a drawing.
   3) Orange = the QUERY path; green = the REFERENCE path; black = the common
      trunk once the mosaic is assembled.
   4) The dashed orange arrow says "indexes", not "selects": the bank already
      exists, and the predicted class only picks the row. That is why two
      queries with the same prediction receive the SAME references, which is
      what makes the comparison in Figure 1 a matched pair.
   5) The purple cards are real layer4 channels of the mosaic, the ones of
      largest contribution w_c[d]*GAP(A_d) to the class logit.
   6) The CAM L is the whole 14x14 grid of the mosaic; the white lines mark
      the four quadrants, and the top-left one is the query.
   7) The rectification is where the whole method happens. WITHOUT it the
      ratio equals 3/2 for ANY image, a consequence of
      mu = (mL_q + 3 mL_ref)/4, an identity when the quadrants are equal. All
      the signal comes from the relu.
   8) The figure no longer prints mu, E_q, E_ref or the value of s: the
      diagram says what is computed, and the values for this query are in
      section 4 above. They still match the frozen dump (gate |Delta s| = 0
      in fig2_code/fig2_method_data.py) and Figure 1.
""")
    print(f"[ok] {pdf}")
    print(f"     s={s:.4f}  mu={mu:.4f}  Eq={eq:.4f}  Eref={er:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="ce", choices=["ce", "ls", "mixup"])
    ap.add_argument("--query", default="id", choices=["id", "ood"])
    ap.add_argument("--n-thumbs", type=int, default=4)
    main(ap.parse_args())
