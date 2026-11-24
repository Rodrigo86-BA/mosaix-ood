"""Figures of the four query positions.

Usage:
  $PYTHON_BIN supplementary/transformers/code/fig_pe_positions.py
  $PYTHON_BIN supplementary/transformers/code/fig_pe_positions.py --sub smoke --n 32
"""
import os

experiment_name = "fig_pe_positions"
dataset_name    = "imagenet1k"

import sys
import json
import argparse
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 6, "axes.titlesize": 6.2, "axes.labelsize": 6,
    "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "legend.fontsize": 5.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
})
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
XAI_OOD_ROOT = Path(os.environ.get("XAI_OOD_ROOT", ROOT / "external_xai_ood"))
for _d in (XAI_OOD_ROOT, REPO / "table_code", REPO / "shared_code", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_d))
m164 = importlib.import_module("transformer_quarter")
m168 = importlib.import_module("tab_field_mosaic")
m178 = importlib.import_module("pe_mosaic_positions")
from src.mosaic import build_mosaic

RES = ROOT / "results" / "pe_mosaic_positions"
OUT = ROOT / "figures" / "pe_positions"
EPS = 1e-12
NEAR, FAR = m178.NEAR, m178.FAR
ROWS = ["rn50_448", "vit_shrink", "vit_interp", "vit_periodic", "vit_smooth"]
LABEL = {"rn50_448": "ResNet-50\nmosaic 448",
         "rn50_224": "ResNet-50\ntiles 112, mosaic 224",
         "vit_shrink": "ViT, shrink\ntiles 112, mosaic 224",
         "vit_interp": "ViT, interpolate\ntiles 224, PE $14\\to28$",
         "vit_periodic": "ViT, extrapolate\nperiodic PE",
         "vit_smooth": "ViT, extrapolate\nsmooth PE"}
SHORT = {"rn50_448": "RN-50, 448", "rn50_224": "RN-50, tiles 112",
         "vit_shrink": "ViT shrink", "vit_interp": "ViT interpolate",
         "vit_periodic": "ViT extrap. periodic", "vit_smooth": "ViT extrap. smooth"}
COR = {"rn50_448": "black", "rn50_224": "0.55", "vit_shrink": "#e69f00",
       "vit_interp": "#0072b2", "vit_periodic": "#009e73", "vit_smooth": "#d55e00"}
QPOS = ["top-left", "top-right", "bottom-left", "bottom-right"]
GNOME = {"id": "ImageNet val (ID)", "ssb_hard": "SSB-hard (near-OOD)",
         "inaturalist": "iNaturalist (far-OOD)"}


def arquivos(sub, n):
    base = RES / sub if sub else RES
    ps = sorted(base.glob(f"178_per_sample_{n}samples_*.csv"))
    if not ps:
        raise FileNotFoundError(f"no 178_per_sample_{n}samples_* in {base}")
    ps = ps[-1]
    stamp = ps.stem.split("_")[-1]
    tag = f"{n}samples_{stamp}"
    return (ps, base / f"178_auroc_{tag}.csv",
            {c: base / f"178_maps_{c}_{tag}.npz" for c in m178.CONDS}, tag)


def escolhe(D, grupos, n_pick):
    d = D[D.cond == "rn50_448"]
    out = {}
    for g in grupos:
        dd = d[(d.group == g) & (d.idx < n_pick)].sort_values("idx")
        v = dd.s_q0.to_numpy()
        j = int(np.argsort(v)[len(v) // 2])
        r = dd.iloc[j]
        out[g] = dict(idx=int(r.idx), file=r.file, s=float(r.s_q0),
                      rank=f"{len(v) // 2 + 1}/{len(v)}",
                      smin=float(v.min()), smax=float(v.max()))
    return out


def denorm(x, tf):
    mean = torch.tensor(tf.mean).view(-1, 1, 1)
    std = torch.tensor(tf.std).view(-1, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(ax, img, L):
    px = img.shape[0]
    up = F.interpolate(torch.as_tensor(L, dtype=torch.float32)[None, None],
                       size=px, mode="bilinear", align_corners=False)[0, 0].numpy()
    lo, hi = np.percentile(up, [2, 98])
    nrm = np.clip((up - lo) / (hi - lo + EPS), 0, 1)
    ext = (-0.5, px - 0.5, px - 0.5, -0.5)
    ax.imshow(img, extent=ext)
    ax.imshow(nrm, cmap="magma", extent=ext, alpha=0.72 * nrm ** 2.0,
              vmin=0, vmax=1)


def figura_paineis(g, info, D, mapas, man, path_pdf, path_png):
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(6.9, 1.66 * len(ROWS)))
    numeros = []
    for i, cond in enumerate(ROWS):
        model, tile, pek = m178.CONDS[cond]
        tf = m168.W_OF[model].transforms()
        z = mapas[cond]
        row = np.where((z["group"] == g) & (z["idx"] == info["idx"]))[0][0]
        cls = int(z["pred"][row])
        rec = D[(D.cond == cond) & (D.group == g) & (D.idx == info["idx"])].iloc[0]
        x1 = m164.load_official([info["file"]], tf, pre=(g == "id"))
        refs = m164.load_official(man["prototypes"][model][str(cls)], tf, pre=False)
        q = m168.resize_tiles(x1, tile)[0]
        r = m168.resize_tiles(refs, tile)
        px = 2 * tile
        numeros.append(f"  {cond:<13} class {cls:<4} "
                       + "  ".join(f"s(qpos {p}) {rec[f's_q{p}']:.3f}"
                                   for p in range(4)))
        for p in range(4):
            ax = axes[i, p]
            img = denorm(build_mosaic(q, r, qpos=p), tf)
            overlay(ax, img, z["L"][row, p].astype(np.float32))
            rr, cc = divmod(p, 2)
            ax.add_patch(Rectangle((cc * tile - 0.5, rr * tile - 0.5), tile, tile,
                                   fill=False, edgecolor="#56b4e9", lw=1.0))
            if cond == "vit_smooth":
                ax.add_patch(Rectangle((-0.5, -0.5), 224, 224, fill=False,
                                       edgecolor="white", lw=0.7, ls="--"))
            ax.text(0.97, 0.04, f"$s={rec[f's_q{p}']:.3f}$",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.6, color="white",
                    bbox=dict(boxstyle="round,pad=0.16", fc="black", ec="none",
                              alpha=0.55))
            if i == 0:
                ax.set_title(f"query {QPOS[p]} (qpos {p})", fontsize=6.0, pad=3)
            if p == 0:
                ax.set_ylabel(LABEL[cond], fontsize=6.0, labelpad=3)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle(f"{GNOME.get(g, g)}: {Path(info['file']).name}",
                 fontsize=6.4, y=0.995)
    fig.subplots_adjust(wspace=0.04, hspace=0.08, top=0.965)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=170)
    plt.close(fig)
    return numeros


def tabela_posicoes(A, D, mapas):
    m = A[A.ood.str.startswith("_") & A.cond.isin(m178.CONDS)]
    au = m.pivot_table(index=["cond", "score"], columns="ood", values="auroc")
    reg = D.group.map(lambda g: "id" if g == "id" else
                      ("near" if g in NEAR else "far"))
    s_pos = D.assign(reg=reg).groupby(["cond", "reg"])[
        ["s_q0", "s_q1", "s_q2", "s_q3"]].mean()
    cor = D.assign(reg=reg).groupby(["cond", "reg"])[
        ["corr_q1", "corr_q2", "corr_q3"]].median()
    share = {}
    for cond in ROWS:
        z = mapas[cond]
        Q = z["Q"].astype(np.float64)
        sh = Q / (Q.sum(2, keepdims=True) + EPS)
        for reg_, mask in (("id", z["group"] == "id"),
                           ("far", np.isin(z["group"], FAR))):
            share[(cond, reg_)] = sh[mask].mean(0)
    return au, s_pos, cor, share


def figura_posicoes(au, s_pos, cor, share, path_pdf, path_png):
    fig = plt.figure(figsize=(6.9, 5.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.95, 0.95], hspace=0.55)

    g1 = gs[0].subgridspec(1, 3, wspace=0.32)
    xt = ["q0", "q1", "q2", "q3", "mean\n$s$", "mean\nmap"]
    scs = ["s_q0", "s_q1", "s_q2", "s_q3", "s_mean4", "s_map4"]
    conds = ["rn50_448", "rn50_224", "vit_shrink", "vit_interp", "vit_periodic",
             "vit_smooth"]
    for j, (col, tit) in enumerate((("_mean_near", "near-OOD AUROC"),
                                    ("_mean_far", "far-OOD AUROC"))):
        ax = fig.add_subplot(g1[0, j])
        for cond in conds:
            if cond not in au.index.get_level_values(0):
                continue
            v = [au.loc[(cond, s), col] for s in scs]
            ax.plot(range(4), v[:4], "-o", ms=2.4, lw=0.9, color=COR[cond],
                    label=SHORT[cond])
            ax.plot([4, 5], v[4:], "D", ms=2.6, color=COR[cond])
        ax.axvline(3.5, color="0.7", lw=0.5, ls=":")
        ax.set_xticks(range(6))
        ax.set_xticklabels(xt)
        ax.set_title(f"{tit}, by query position", fontsize=6.0, pad=3)
        if j == 0:
            ax.set_ylabel("AUROC")
    ax = fig.add_subplot(g1[0, 2])
    for cond in conds:
        if cond not in s_pos.index.get_level_values(0):
            continue
        ax.plot(range(4), s_pos.loc[(cond, "id")].to_numpy(), "-o", ms=2.4,
                lw=0.9, color=COR[cond], label=SHORT[cond])
        ax.plot(range(4), s_pos.loc[(cond, "far")].to_numpy(), "--", lw=0.7,
                color=COR[cond])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["q0", "q1", "q2", "q3"])
    ax.set_title("mean $s$: ID (solid), far-OOD (dashed)", fontsize=6.0, pad=3)
    ax.set_ylabel("mean $s$")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              handlelength=1.4)

    for linha, reg_ in ((1, "id"), (2, "far")):
        gl = gs[linha].subgridspec(1, len(ROWS) + 1, wspace=0.30,
                                   width_ratios=[1] * len(ROWS) + [1.25])
        for j, cond in enumerate(ROWS):
            ax = fig.add_subplot(gl[0, j])
            M = share[(cond, reg_)]
            ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
            for a in range(4):
                for b in range(4):
                    ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center",
                            fontsize=4.8, color="white" if M[a, b] > 0.5 else "black",
                            fontweight="bold" if a == b else "normal")
            ax.set_xticks(range(4))
            ax.set_xticklabels(["TL", "TR", "BL", "BR"], fontsize=5)
            ax.set_yticks(range(4))
            ax.set_yticklabels([f"q{p}" for p in range(4)], fontsize=5)
            if j == 0:
                ax.set_ylabel(f"{'ID' if reg_ == 'id' else 'far-OOD'}\n"
                              "query position", labelpad=1.5)
            ax.set_xlabel("tile of the mosaic", labelpad=1.0)
            ax.set_title(SHORT[cond], fontsize=5.8, pad=2, color=COR[cond])
        ax = fig.add_subplot(gl[0, len(ROWS)])
        w = 0.15
        for k, cond in enumerate(ROWS):
            if (cond, reg_) not in cor.index:
                continue
            v = cor.loc[(cond, reg_)].to_numpy()
            ax.bar(np.arange(3) + (k - 2) * w, v, width=w, color=COR[cond])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["q1", "q2", "q3"])
        ax.set_ylim(0, 1.02)
        ax.set_title("query map vs. its map at q0\n(median Pearson)",
                     fontsize=5.8, pad=2)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=170)
    plt.close(fig)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    ps, pa, pm, tag = arquivos(args.sub, args.n)
    D = pd.read_csv(ps)
    A = pd.read_csv(pa)
    mapas = {c: np.load(p) for c, p in pm.items() if p.exists()}
    man = json.loads(m178.MAN168.read_text())
    print(f"input: {ps.name}", flush=True)

    escolha = escolhe(D, args.groups, args.n_pick)
    sufixo = f"_{args.sub}" if args.sub else ""
    for g, info in escolha.items():
        pdf = OUT / f"179_fig_panels_{g}{sufixo}.pdf"
        png = OUT / f"179_fig_panels_{g}{sufixo}.png"
        nums = figura_paineis(g, info, D, mapas, man, pdf, png)
        (OUT / f"179_fig_panels_{g}{sufixo}.txt").write_text(
            "fig_pe_positions.py\n\n"
            "WHAT IT MEASURES\n"
            f"  One query of {GNOME.get(g, g)} in the 2x2 mosaic, in the four\n"
            "  positions (columns) and in five conditions (rows): RN-50 at 448\n"
            "  (control) and the ViT-B/16 shrunk, interpolated and extrapolated\n"
            "  (periodic and smooth). It shows how the explanation behaves when the\n"
            "  query changes place.\n\n"
            "FILES\n"
            f"  {pdf.name} / .png  the mosaic with the map on top. Blue frame = query\n"
            "    tile. White dashes (smooth only) = the 224 px whose positional\n"
            "    embedding is the trained one; outside it the positions are the\n"
            "    continuation of the sinusoids.\n"
            f"  input: results/pe_mosaic_positions/{ps.name}\n\n"
            "NUMBERS (s of each panel)\n" + "\n".join(nums) + "\n\n"
            "QUERY CHOICE\n"
            f"  median s under RN-50 448, qpos 0, among the first {args.n_pick}\n"
            f"  of the pool: s={info['s']:.3f}, rank {info['rank']} (min "
            f"{info['smin']:.3f}, max {info['smax']:.3f})\n  {info['file']}\n\n"
            "HOW TO READ\n"
            "  1. Along a row, the explanation should follow the query: the same\n"
            "     pattern inside the blue frame in the four columns.\n"
            "  2. RN-50: the convolution is almost equivariant to translation; the\n"
            "     query map changes little across columns (border effect only).\n"
            "  3. Periodic ViT: changing place permutes tokens with the same\n"
            "     embedding; the map moves with the tiles and s does not change.\n"
            "  4. Smooth ViT: in column 0 the query is on the trained range; in the\n"
            "     other columns it is on continued positions. If the map stays on\n"
            "     the dashes instead of following the query, the explanation\n"
            "     depends on the position and not on the content.\n"
            "  5. Interpolated and shrunk ViT: each tile falls on a different region\n"
            "     of the trained embedding; s changes with the position.\n\n"
            "CAVEATS\n"
            "  a. The color is normalized by percentiles 2 and 98 of each panel;\n"
            "     only s compares panels.\n"
            "  b. One query is not a trend; fig_positions has the whole set.\n"
            "  c. RN-50 and ViT have different prototypes (chosen in the feature\n"
            "     space of each model, as in tab_field_mosaic.py) and can predict\n"
            "     different classes for the same query.\n")
        print(f"[ok] {pdf}", flush=True)

    au, s_pos, cor, share = tabela_posicoes(A, D, mapas)
    pdf = OUT / f"179_fig_positions{sufixo}.pdf"
    png = OUT / f"179_fig_positions{sufixo}.png"
    figura_posicoes(au, s_pos, cor, share, pdf, png)
    tab_au = au[["_mean_near", "_mean_far"]].rename(
        columns={"_mean_near": "near", "_mean_far": "far"})
    diag = "\n".join(
        f"  {c:<13} ID: share of the energy on the query tile "
        + " ".join(f"{share[(c, 'id')][p, p]:.2f}" for p in range(4))
        + " | on the top-left tile "
        + " ".join(f"{share[(c, 'id')][p, 0]:.2f}" for p in range(4))
        for c in ROWS)
    (OUT / f"179_fig_positions{sufixo}.txt").write_text(
        "fig_pe_positions.py\n\n"
        "WHAT IT MEASURES\n"
        "  The effect of moving the query to another quadrant, on the whole set\n"
        "  (n per set in the name of the input file), for the RN-50 and for the\n"
        "  three adaptations of the ViT-B/16 to the mosaic.\n\n"
        "FILES\n"
        f"  {pdf.name} / .png\n"
        "    row 1: near and far AUROC with the query in each position (q0..q3)\n"
        "      and the two means of 4 positions (diamonds); mean s by position,\n"
        "      ID (solid) and far-OOD (dashed).\n"
        "    rows 2 and 3: for each condition, the mean share of the rectified\n"
        "      energy of the map on each tile (columns TL, TR, BL, BR) when the\n"
        "      query is in each position (rows q0..q3); ID and far-OOD. The\n"
        "      diagonal (bold) is the energy on the query tile. On the right, the\n"
        "      median correlation between the query sub-map at position p and at\n"
        "      position 0.\n"
        f"  input: results/pe_mosaic_positions/{ps.name}\n\n"
        "NUMBERS\n" + tab_au.to_string(float_format=lambda v: f"{v:.3f}") + "\n\n"
        + s_pos.to_string(float_format=lambda v: f"{v:.3f}") + "\n\n"
        + cor.to_string(float_format=lambda v: f"{v:.3f}") + "\n\n" + diag + "\n\n"
        "HOW TO READ\n"
        "  1. AUROC by position: a flat line means that detection does not\n"
        "     depend on where the query is. The periodic one is flat by design.\n"
        "  2. Mean s by position: the positional bias. What matters for the AUROC\n"
        "     is the distance between the solid (ID) and the dashed (far-OOD)\n"
        "     lines in each position.\n"
        "  3. Energy matrices: if the explanation follows the query, the diagonal\n"
        "     dominates. If one column dominates in every row, the map stays on\n"
        "     a fixed tile, whatever its content.\n"
        "  4. With uniform energy each tile would have 0.25.\n"
        "  5. Correlation of the query map: 1 = the explanation of the query does\n"
        "     not change when it changes place.\n\n"
        "CAVEATS\n"
        "  a. RN-50 with 112 tiles has a 7x7 map: the tile border falls inside a\n"
        "     cell. It only enters the AUROC and s curves, with the quadrant cut\n"
        "     of tab_field_mosaic.py.\n"
        "  b. In some OOD queries the query map is almost constant and the\n"
        "     correlation is ill-defined; hence the median.\n")
    print(f"[ok] {pdf}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-pick", type=int, default=16)
    ap.add_argument("--groups", nargs="+", default=["id", "ssb_hard", "inaturalist"])
    main(ap.parse_args())
