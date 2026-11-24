"""Periodic and smooth continuations of the ViT-B/16 positional embedding.

Usage:
  $PYTHON_BIN supplementary/transformers/code/pe_continuation.py
"""
import os

experiment_name = "pe_continuation"
dataset_name    = "imagenet1k"

import argparse
import time
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
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import interpolate_embeddings

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
XAI_OOD_ROOT = Path(os.environ.get("XAI_OOD_ROOT", ROOT / "external_xai_ood"))
OUT = ROOT / "results" / experiment_name
FIG = ROOT / "figures" / experiment_name
G = 14
GE = 28
PATCH = 16
EPS = 1e-12
COR = {"interp": "#0072b2", "periodic": "#009e73", "smooth": "#d55e00",
       "orig": "black"}
ROTULO = {"interp": "interpolated (bicubic)", "periodic": "periodic (DFT)",
          "smooth": "smooth sinusoidal fit"}


def pe_table():
    sd = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).state_dict()
    pe = sd["encoder.pos_embedding"].detach()[0]
    return sd, pe[0].clone(), pe[1:].reshape(G, G, -1).clone()


def freqs(fam, K):
    if fam == "sep":
        return ([(k, 0) for k in range(1, K + 1)]
                + [(0, l) for l in range(1, K + 1)])
    out = []
    for k in range(0, K + 1):
        for l in range(-K, K + 1):
            if k == 0 and l <= 0:
                continue
            out.append((k, l))
    return out


def design(r, c, fam, T, K):
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    cols = [np.ones_like(r)]
    for k, l in freqs(fam, K):
        th = 2 * np.pi * (k * r + l * c) / T
        cols += [np.cos(th), np.sin(th)]
    return np.stack(cols, 1)


def fit(r, c, Y, fam, T, K, lam_rel):
    Phi = design(r, c, fam, T, K)
    A = Phi.T @ Phi
    lam = lam_rel * np.trace(A) / A.shape[0]
    return np.linalg.solve(A + lam * np.eye(A.shape[0]), Phi.T @ Y)


def predict(r, c, C, fam, T, K):
    return design(r, c, fam, T, K) @ C


def cv(P, cands, B, lam_rel):
    rr, cc = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    rows = []
    for r0 in (0, G - B):
        for c0 in (0, G - B):
            inb = ((rr >= r0) & (rr < r0 + B) & (cc >= c0) & (cc < c0 + B))
            band = ~inb
            Yf, Yb = P[inb], P[band]
            m = Yf.mean(0)
            den = float(((Yb - m) ** 2).sum())
            dr = np.maximum(r0 - rr[band], rr[band] - (r0 + B - 1)).clip(0)
            dc = np.maximum(c0 - cc[band], cc[band] - (c0 + B - 1)).clip(0)
            dist = np.maximum(dr, dc)
            pred = {"constant": np.broadcast_to(m, Yb.shape),
                    "clamp": P[np.clip(rr[band], r0, r0 + B - 1),
                               np.clip(cc[band], c0, c0 + B - 1)],
                    "periodic": P[r0 + (rr[band] - r0) % B,
                                  c0 + (cc[band] - c0) % B]}
            for fam, T, K in cands:
                C = fit(rr[inb], cc[inb], Yf, fam, T, K, lam_rel)
                pred[f"smooth|{fam}|T{T}|K{K}"] = predict(rr[band], cc[band],
                                                          C, fam, T, K)
            for nome, Yh in pred.items():
                err = ((Yh - Yb) ** 2).sum(1)
                base = ((Yb - m) ** 2).sum(1)
                rows.append(dict(corner=f"r{r0}c{c0}", cand=nome,
                                 rel_err=float(err.sum() / den),
                                 **{f"rel_err_d{d}": float(err[dist == d].sum()
                                                           / base[dist == d].sum())
                                    for d in range(1, G - B + 1)}))
    return pd.DataFrame(rows)


def continua(P, sd, sel, lam_rel):
    Pn = P.double().numpy()
    D = Pn.shape[-1]
    rr, cc = np.meshgrid(np.arange(GE), np.arange(GE), indexing="ij")
    out = {}

    sd448 = interpolate_embeddings(GE * PATCH, PATCH, dict(sd))
    out["interp"] = sd448["encoder.pos_embedding"][0, 1:].reshape(GE, GE, D)

    out["periodic"] = P[rr % G, cc % G].clone()

    fam, T, K = sel
    r0, c0 = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    C = fit(r0.ravel(), c0.ravel(), Pn.reshape(-1, D), fam, T, K, lam_rel)
    E = predict(rr.ravel(), cc.ravel(), C, fam, T, K).reshape(GE, GE, D)
    fit_in = E[:G, :G].copy()
    sm = torch.from_numpy(E).float()
    sm[:G, :G] = P
    out["smooth"] = sm
    return out, fit_in


def diagnosticos(P, ext, fit_in):
    Pn = P.double().numpy()
    D = Pn.shape[-1]
    flat = Pn.reshape(-1, D)
    mu = flat.mean(0)
    r2 = 1 - ((fit_in - Pn) ** 2).sum() / ((Pn - mu) ** 2).sum()

    passo = np.r_[np.linalg.norm(np.diff(Pn, axis=0), axis=-1).ravel(),
                  np.linalg.norm(np.diff(Pn, axis=1), axis=-1).ravel()]
    norma_in = float(np.linalg.norm(flat, axis=1).mean())
    fn = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + EPS)

    linhas = [f"  smooth fit on the trained range: R^2 = {r2:.4f} "
              "(diagnostic only; the range uses the original table)",
              f"  typical step between neighboring trained positions: median "
              f"{np.median(passo):.3f}, p90 {np.percentile(passo, 90):.3f}",
              f"  mean norm of the embedding on the trained range: {norma_in:.3f}"]
    diag = {"r2_fit_in_range": float(r2), "step_median": float(np.median(passo)),
            "norm_in_range": norma_in}
    for nome, E in ext.items():
        En = E.double().numpy()
        fora = np.ones((GE, GE), bool)
        if nome != "interp":
            fora[:G, :G] = False
        v = En[fora]
        nv = float(np.linalg.norm(v, axis=1).mean())
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)
        cmax = (vn @ fn.T).max(1)
        s = (f"  {nome:<9} positions outside the range: {fora.sum():3d} | mean norm "
             f"{nv:.3f} ({nv / norma_in:.2f}x the range) | cosine to the closest "
             f"trained embedding: median {np.median(cmax):.3f}, "
             f"min {cmax.min():.3f}")
        if nome != "interp":
            salto = np.r_[np.linalg.norm(En[G, :G] - En[G - 1, :G], axis=-1),
                          np.linalg.norm(En[:G, G] - En[:G, G - 1], axis=-1)]
            s += (f"\n            jump at the 13->14 border: median "
                  f"{np.median(salto):.3f} "
                  f"({np.median(salto) / np.median(passo):.2f}x the typical step)")
            diag[f"{nome}_jump_ratio"] = float(np.median(salto) / np.median(passo))
        diag[f"{nome}_norm_ratio"] = nv / norma_in
        diag[f"{nome}_cosmax_median"] = float(np.median(cmax))
        linhas.append(s)
    return linhas, diag


def figura(P, ext, path_pdf, path_png, linha_perfil):
    Pn = P.double().numpy()
    D = Pn.shape[-1]
    flat = Pn.reshape(-1, D)
    mu = flat.mean(0)
    _, s, vt = np.linalg.svd(flat - mu, full_matrices=False)
    V = vt[:3]
    proj0 = (flat - mu) @ V.T
    lo, hi = proj0.min(0), proj0.max(0)

    def rgb(grid):
        g = grid.double().numpy() if torch.is_tensor(grid) else grid
        p = (g.reshape(-1, D) - mu) @ V.T
        return np.clip((p - lo) / (hi - lo + EPS), 0, 1).reshape(
            g.shape[0], g.shape[1], 3)

    fn = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + EPS)
    Es = ext["smooth"].double().numpy().reshape(-1, D)
    Es = Es / (np.linalg.norm(Es, axis=1, keepdims=True) + EPS)
    cmax = (Es @ fn.T).max(1).reshape(GE, GE)

    fig = plt.figure(figsize=(6.9, 3.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.78], hspace=0.42)
    gt = gs[0].subgridspec(1, 5, wspace=0.22, width_ratios=[1, 1, 1, 1, 1.18])
    paineis = [("trained $14\\times14$\n(PCA to RGB)", rgb(P), G, None),
               ("interpolated $28\\times28$", rgb(ext["interp"]), GE, None),
               ("periodic continuation", rgb(ext["periodic"]), GE, G),
               ("smooth continuation", rgb(ext["smooth"]), GE, G)]
    for j, (tit, img, n, caixa) in enumerate(paineis):
        ax = fig.add_subplot(gt[0, j])
        ax.imshow(img, interpolation="nearest")
        if caixa:
            ax.add_patch(Rectangle((-0.5, -0.5), caixa, caixa, fill=False,
                                   edgecolor="white", lw=1.0, ls="--"))
        ax.axhline(linha_perfil, color="white", lw=0.5, alpha=0.8)
        ax.set_title(tit, fontsize=6.0, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = fig.add_subplot(gt[0, 4])
    im = ax.imshow(cmax, cmap="viridis", vmin=min(0.0, cmax.min()), vmax=1,
                   interpolation="nearest")
    ax.add_patch(Rectangle((-0.5, -0.5), G, G, fill=False, edgecolor="white",
                           lw=1.0, ls="--"))
    ax.set_title("smooth: cosine to the closest\ntrained embedding", fontsize=6.0,
                 pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=4.8, width=0.4, length=1.4)
    cb.outline.set_linewidth(0.4)

    gb = gs[1].subgridspec(1, 3, wspace=0.28)
    xs = np.arange(GE)
    var = s ** 2 / (s ** 2).sum()
    for j in range(3):
        ax = fig.add_subplot(gb[0, j])
        for nome in ("interp", "periodic", "smooth"):
            g = ext[nome].double().numpy()[linha_perfil if nome != "interp"
                                           else 2 * linha_perfil]
            ax.plot(xs, (g - mu) @ V[j], lw=0.9, color=COR[nome],
                    label=ROTULO[nome])
        ax.plot(np.arange(G), (Pn[linha_perfil] - mu) @ V[j], "o", ms=2.2,
                color=COR["orig"], label="trained table", zorder=5)
        ax.axvline(G - 0.5, color="0.5", lw=0.6, ls="--")
        ax.set_title(f"PC{j + 1} ({var[j]:.0%} of variance), row "
                     f"{linha_perfil}", fontsize=6.0, pad=3)
        ax.set_xlabel("column position on the $28\\times28$ grid", labelpad=1.5)
        if j == 0:
            ax.set_ylabel("projection", labelpad=1.5)
            hs, ls = ax.get_legend_handles_labels()
    fig.legend(hs, ls, frameon=False, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), handlelength=1.6)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return var[:3]


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sd, cls_pe, P = pe_table()
    Pn = P.double().numpy()
    log = [f"177_pe_continuation.py | ViT-B/16 IMAGENET1K_V1 | table {tuple(P.shape)}"
           f" + CLS | validation block {args.block}x{args.block} | "
           f"relative ridge {args.lam_rel:g}"]

    cands = [(fam, T, K) for fam in ("sep", "full") for T in args.periods
             for K in range(1, args.kmax + 1)]
    df = cv(Pn, cands, args.block, args.lam_rel)
    df.to_csv(OUT / "177_cv.csv", index=False)
    media = df.groupby("cand").mean(numeric_only=True).sort_values("rel_err")
    suaves = media[media.index.str.startswith("smooth|")]
    melhor = suaves.index[0]
    _, fam, T, K = melhor.split("|")
    sel = (fam, int(T[1:]), int(K[1:]))
    log.append("\nCROSS-VALIDATION (error relative to the constant predictor, mean "
               "of 4 corners; d = distance to the block)")
    cols = ["rel_err"] + [c for c in media.columns if c.startswith("rel_err_d")]
    ref = media.loc[["constant", "clamp", "periodic"], cols]
    log.append(pd.concat([suaves[cols].head(8), ref]).to_string(
        float_format=lambda v: f"{v:.3f}"))
    log.append(f"\nCHOICE: {melhor}  -> family {sel[0]}, T={sel[1]}, K={sel[2]}")

    ext, fit_in = continua(P, sd, sel, args.lam_rel)

    for nome in ("periodic", "smooth"):
        if not torch.equal(ext[nome][:G, :G], P):
            raise AssertionError(f"{nome}: the trained range changed")
    per_ok = all(torch.equal(ext["periodic"][a * G:(a + 1) * G, b * G:(b + 1) * G], P)
                 for a in (0, 1) for b in (0, 1))
    if not per_ok:
        raise AssertionError("periodic: some tile does not repeat the table")
    log.append("\nGATES\n  range 0..13 x 0..13 equal to the original table, bit by bit: "
               "periodic ok, smooth ok\n  periodic: the four 14x14 tiles are "
               "exact copies of the table: ok")

    linhas_diag, diag = diagnosticos(P, ext, fit_in)
    log.append("\nDIAGNOSTICS\n" + "\n".join(linhas_diag))

    pt = OUT / "177_pe_extended.pt"
    torch.save({"cls": cls_pe, "grid224": P, **ext,
                "smooth_spec": {"family": sel[0], "T": sel[1], "K": sel[2],
                                "lam_rel": args.lam_rel, "block": args.block},
                "diag": diag}, pt)

    pdf = FIG / "177_fig_pe_continuation.pdf"
    png = FIG / "177_fig_pe_continuation.png"
    var = figura(P, ext, pdf, png, args.row)
    (FIG / "177_fig_pe_continuation.txt").write_text(
        "pe_continuation.py\n\n"
        "WHAT IT MEASURES\n"
        "  Three ways to give a positional embedding to the 28x28 grid of a 448\n"
        "  mosaic on the ViT-B/16, whose checkpoint only has the 14x14 grid.\n"
        "  Interpolated: the resampled table (torchvision, bicubic). Periodic:\n"
        "  the continuation of the DFT of the table, which repeats it. Smooth:\n"
        "  the continuation of a fit of low-frequency sinusoids, with positions\n"
        "  the model never saw.\n\n"
        "FILES\n"
        "  177_fig_pe_continuation.pdf/.png  top row: the embeddings in PCA-RGB\n"
        "    (PCA basis and color scale of the trained table, values outside\n"
        "    the scale clipped), and the cosine of each position of the smooth\n"
        "    continuation to the closest trained embedding. Bottom row: the\n"
        "    first three components along one row of the grid.\n"
        f"  {pt.relative_to(ROOT)}  the tensors used by pe_mosaic_positions.py.\n\n"
        "NUMBERS\n"
        f"  choice of the smooth continuation: family {sel[0]}, T={sel[1]}, K={sel[2]}\n"
        f"  PC1-3 explain {var[0]:.1%}, {var[1]:.1%}, {var[2]:.1%} of the variance\n"
        + "\n".join(linhas_diag) + "\n\n"
        "HOW TO READ\n"
        "  1. White dashed box = trained range (positions 0..13). Inside it the\n"
        "     periodic and smooth continuations are the original table.\n"
        "  2. Interpolated panel: the whole table stretched to 28x28. Each 224\n"
        "     tile falls on a quarter of the original range, at twice the\n"
        "     resolution.\n"
        "  3. Periodic panel: the table appears four times. The ViT cannot know\n"
        "     in which tile a patch is; only the content tells the tiles apart.\n"
        "  4. Smooth panel: outside the box, colors that do not appear inside it\n"
        "     (or clipped colors) mark positions new to the model.\n"
        "  5. Cosine to the closest trained embedding: 1 inside the range; the\n"
        "     lower it is outside, the farther from any position the model saw.\n"
        "  6. Profiles: black dots = trained table. The green line repeats the\n"
        "     dots after the dashed line; the orange one follows the fitted\n"
        "     sinusoids; the blue one is the table stretched to twice its size.\n\n"
        "CAVEATS\n"
        "  a. The smooth continuation depends on the basis of sinusoids. The\n"
        "     cross-validation picks the basis that best predicts up to 4\n"
        "     positions beyond a 10x10 block; the actual continuation reaches 14\n"
        "     positions beyond the range.\n"
        "  b. The trained range uses the original table, so there is a jump at\n"
        "     the 13->14 border of the size of the fit residual (see NUMBERS).\n")

    log.append(f"\n[ok] {pt}\n[ok] {pdf}\n[ok] {png}  [{time.time() - t0:.0f}s]")
    (OUT / "177_summary.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--lam-rel", type=float, default=1e-3)
    ap.add_argument("--periods", type=int, nargs="+",
                    default=[14, 20, 28, 42, 56])
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--row", type=int, default=6,
                    help="row of the trained grid used in the profiles")
    main(ap.parse_args())
