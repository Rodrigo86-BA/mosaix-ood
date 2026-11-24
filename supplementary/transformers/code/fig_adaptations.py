"""The two adaptations of the mosaic to a ViT, and example maps.

Usage:
  $PYTHON_BIN supplementary/transformers/code/fig_adaptations.py
  $PYTHON_BIN supplementary/transformers/code/fig_adaptations.py --n-pick 8 --n-cand 16
"""
import os

experiment_name = "fig_transformer_adapt"
dataset_name    = "imagenet1k"

import sys
import time
import argparse
import importlib
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
})
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import interpolate_embeddings

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
XAI_OOD_ROOT = Path(os.environ.get("XAI_OOD_ROOT", ROOT / "external_xai_ood"))
for _d in (XAI_OOD_ROOT, REPO / "table_code", REPO / "shared_code", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_d))
m07 = importlib.import_module("rood_qualify_protocol")
m164 = importlib.import_module("transformer_quarter")
m168 = importlib.import_module("tab_field_mosaic")
from src.mosaic import build_mosaic

OUT = ROOT / "figures" / "transformer_adapt"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
PATCH = 16
MODELS = ("rn50", "vit")
CONDS = ("single224", "single448", "mos448", "mos224")
COND_LABEL = {
    "single224": "image alone\n224",
    "single448": "image alone\n448",
    "mos448": "mosaic 448\ntile 224",
    "mos224": "mosaic 224\ntile 112",
}


def denorm(x, tf):
    mean = torch.tensor(tf.mean).view(-1, 1, 1)
    std = torch.tensor(tf.std).view(-1, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def resize(x, px):
    if x.shape[-1] == px:
        return x
    return F.interpolate(x, size=px, mode="bicubic",
                         align_corners=False, antialias=True)


def proto_for_class(model, cls, tf, tr, n_cand, bs, rng):
    files = tr[cls]
    take = np.sort(rng.choice(len(files), min(n_cand, len(files)),
                              replace=False))
    cand = [str(files[i]) for i in take]
    x = m164.load_official(cand, tf, pre=False)
    pred, z = model.pred_pool(x, bs)
    zn = z / (z.norm(dim=1, keepdim=True) + EPS)
    cen = zn.mean(0)
    cen = cen / (cen.norm() + EPS)
    ordem = torch.argsort(-(zn @ cen)).numpy()
    ok = [i for i in ordem if pred[i] == cls] or list(ordem)
    return [cand[i] for i in ok[:3]]


def quadrantes(L):
    mu = L.flatten(1).mean(1, keepdim=True)[:, :, None]
    E = torch.relu(L - mu)
    h = E.shape[-1] // 2
    eq = E[:, :h, :h].flatten(1).mean(1)
    er = ((E[:, :h, h:].flatten(1).mean(1)
           + E[:, h:, :h].flatten(1).mean(1)
           + E[:, h:, h:].flatten(1).mean(1)) / 3)
    return (eq / (eq + er + EPS)).numpy(), E


def mapa_e_s(nome, dev, x1, proto_files, tf, cond, bs):
    if cond.startswith("single"):
        px = int(cond.replace("single", ""))
        xin = resize(x1, px)
        w = m168.wrap(nome, dev, px)
        cls, _ = w.pred_pool(xin, bs)
        L = w.cam(xin, torch.tensor(cls), bs)
        del w
        torch.cuda.empty_cache()
        return xin[0], L[0], None, int(cls[0])

    tile = 224 if cond == "mos448" else 112
    w224 = m168.wrap(nome, dev, 224)
    cls, _ = w224.pred_pool(x1, bs)
    del w224
    torch.cuda.empty_cache()

    refs = m164.load_official(proto_files, tf, pre=False)
    q = resize(x1, tile)[0]
    r = resize(refs, tile)
    mos = build_mosaic(q, r, qpos=0).unsqueeze(0)

    w = m168.wrap(nome, dev, tile * 2)
    L = w.cam(mos, torch.tensor(cls), 1)
    del w
    torch.cuda.empty_cache()
    s, _ = quadrantes(L)
    return mos[0], L[0], float(s[0]), int(cls[0])


def escolhe_queries(dev, tf_rn, tr, pools, args, log):
    rng = np.random.default_rng(m07.SEED)
    escolha = {}
    for grupo in ("id", args.ood):
        files = pools[grupo][:args.n_pick]
        x = m164.load_official(files, tf_rn, pre=(grupo == "id"))
        w224 = m168.wrap("rn50", dev, 224)
        pred, _ = w224.pred_pool(x, args.bs)
        del w224
        torch.cuda.empty_cache()

        protos = {}
        for c in sorted(set(int(p) for p in pred)):
            w = m168.wrap("rn50", dev, 224)
            protos[c] = proto_for_class(w, c, tf_rn, tr, args.n_cand,
                                        args.bs, rng)
            del w
            torch.cuda.empty_cache()

        w448 = m168.wrap("rn50", dev, 448)
        vals = []
        for i, c in enumerate(pred):
            refs = m164.load_official(protos[int(c)], tf_rn, pre=False)
            mos = build_mosaic(x[i], refs, qpos=0).unsqueeze(0)
            L = w448.cam(mos, torch.tensor([int(c)]), 1)
            vals.append(quadrantes(L)[0][0])
        del w448
        torch.cuda.empty_cache()

        vals = np.array(vals)
        idx = int(np.argsort(vals)[len(vals) // 2])
        escolha[grupo] = dict(file=files[idx], s_rn50_448=float(vals[idx]),
                              rank=f"{len(vals)//2+1}/{len(vals)}",
                              s_min=float(vals.min()), s_max=float(vals.max()))
        log(f"  {grupo:<12} median s={vals[idx]:.3f} "
            f"(min {vals.min():.3f}, max {vals.max():.3f})  "
            f"{Path(files[idx]).name}")
    return escolha


def grid_patches(ax, px, lw=0.25, cor="white", alpha=0.55):
    n = px // PATCH
    for k in range(1, n):
        ax.axhline(k * PATCH - 0.5, color=cor, lw=lw, alpha=alpha)
        ax.axvline(k * PATCH - 0.5, color=cor, lw=lw, alpha=alpha)


def moldura_query(ax, px, tile, cor="#d55e00", lw=1.1):
    ax.add_patch(Rectangle((-0.5, -0.5), tile, tile, fill=False,
                           edgecolor=cor, lw=lw))


def pos_embed_grids():
    w = ViT_B_16_Weights.IMAGENET1K_V1
    sd224 = vit_b_16(weights=w).state_dict()
    sd448 = interpolate_embeddings(448, PATCH, dict(sd224))
    k = "encoder.pos_embedding"
    p224 = sd224[k][0, 1:].detach().numpy()
    p448 = sd448[k][0, 1:].detach().numpy()
    return p224.reshape(14, 14, -1), p448.reshape(28, 28, -1)


def pca_rgb(grid):
    h, w, d = grid.shape
    flat = grid.reshape(-1, d)
    flat = flat - flat.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    proj = flat @ vt[:3].T
    lo, hi = proj.min(0, keepdims=True), proj.max(0, keepdims=True)
    return ((proj - lo) / (hi - lo + EPS)).reshape(h, w, 3), s


def cos_para(grid, frac=(0.5, 0.5)):
    h, w, d = grid.shape
    flat = grid.reshape(-1, d)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + EPS)
    r, c = int(h * frac[0]), int(w * frac[1])
    return (flat @ flat[r * w + c]).reshape(h, w), (r, c)


def figura_esquema(disp, path, log):
    fig = plt.figure(figsize=(6.9, 4.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.30)

    gtop = gs[0].subgridspec(1, 3, wspace=0.16)
    painel = [
        ("Original input\n$224$, query is the whole field",
         disp["single224"], 224, 224, "$14\\times14=196$ tokens on the query"),
        ("Adaptation 1, resample the mosaic\n"
         "$112$ tiles, mosaic back to $224$",
         disp["mos224"], 224, 112, "$7\\times7=49$ tokens on the query"),
        ("Adaptation 2, interpolate the model\n"
         "$224$ tiles, mosaic at $448$",
         disp["mos448"], 448, 224, "$14\\times14=196$ tokens, grid $28\\times28$"),
    ]
    for j, (titulo, img, px, tile, rodape) in enumerate(painel):
        ax = fig.add_subplot(gtop[0, j])
        ax.imshow(img, extent=(-0.5, px - 0.5, px - 0.5, -0.5))
        grid_patches(ax, px)
        moldura_query(ax, px, tile)
        ax.set_title(titulo, fontsize=6.0, pad=3)
        ax.set_xlabel(rodape, fontsize=5.8, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    p224, p448 = pos_embed_grids()
    rgb224, sv = pca_rgb(p224)
    rgb448, _ = pca_rgb(p448)
    cos224, rc224 = cos_para(p224)
    cos448, rc448 = cos_para(p448)
    log(f"  pos-emb: 14x14x{p224.shape[-1]} -> 28x28x{p448.shape[-1]}; "
        f"the first 3 components explain "
        f"{(sv[:3]**2).sum()/(sv**2).sum():.1%} of the variance")

    gbot = gs[1].subgridspec(1, 5, wspace=0.42,
                             width_ratios=[1, 1, 1, 1, 1.25])
    for j, (img, titulo, cmap) in enumerate([
            (rgb224, "pos. embedding $14\\times14$\nPCA to RGB", None),
            (rgb448, "interpolated $28\\times28$\nPCA to RGB", None),
            (cos224, "cosine to the centre\nposition, $14\\times14$", "viridis"),
            (cos448, "cosine to the centre\nposition, $28\\times28$", "viridis")]):
        ax = fig.add_subplot(gbot[0, j])
        im = ax.imshow(img, cmap=cmap, interpolation="nearest")
        if cmap:
            ax.plot(*(rc224[::-1] if j == 2 else rc448[::-1]), marker="+",
                    ms=4, mew=0.8, color="#d55e00")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=4.8, width=0.4, length=1.4)
            cb.outline.set_linewidth(0.4)
        ax.set_title(titulo, fontsize=5.9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax = fig.add_subplot(gbot[0, 4])
    perfil224 = cos224[rc224[0]]
    perfil448 = cos448[rc448[0]]
    ax.plot(np.linspace(0, 1, 14), perfil224, "-o", ms=2.2, lw=0.9,
            color="#0072b2", label="$14\\times14$")
    ax.plot(np.linspace(0, 1, 28), perfil448, "-", lw=0.9,
            color="#d55e00", label="$28\\times28$ interp.")
    ax.set_title("row through the centre", fontsize=5.9, pad=3)
    ax.set_xlabel("relative position", labelpad=1.5)
    ax.set_ylabel("cosine", labelpad=1.5)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.legend(frameon=False, loc="lower center", handlelength=1.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figura_mapas(res, disp_all, path):
    linhas = [(m, g) for m in MODELS for g in res]
    fig, axes = plt.subplots(len(linhas), len(CONDS),
                             figsize=(6.9, 1.62 * len(linhas)))
    for i, (modelo, grupo) in enumerate(linhas):
        for j, cond in enumerate(CONDS):
            ax = axes[i, j]
            cel = res[grupo][modelo][cond]
            img = disp_all[grupo][modelo][cond]
            px = img.shape[0]
            L = cel["mapa"]
            up = F.interpolate(torch.tensor(L)[None, None].float(),
                               size=px, mode="bilinear",
                               align_corners=False)[0, 0].numpy()
            lo, hi = np.percentile(up, [2, 98])
            norm = np.clip((up - lo) / (hi - lo + EPS), 0, 1)
            ext = (-0.5, px - 0.5, px - 0.5, -0.5)
            ax.imshow(img, extent=ext)
            ax.imshow(norm, cmap="magma", extent=ext,
                      alpha=0.72 * norm ** 2.0, vmin=0, vmax=1)
            if modelo == "vit":
                grid_patches(ax, px, lw=0.16, alpha=0.28)
            tile = {"single224": 224, "single448": 448,
                    "mos448": 224, "mos224": 112}[cond]
            moldura_query(ax, px, tile, lw=0.9)
            if cel["s"] is not None:
                ax.text(0.97, 0.04, f"$s={cel['s']:.3f}$",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=5.6, color="white",
                        bbox=dict(boxstyle="round,pad=0.16", fc="black",
                                  ec="none", alpha=0.55))
            if i == 0:
                ax.set_title(COND_LABEL[cond], fontsize=6.0, pad=3)
            if j == 0:
                rot = "ResNet-50" if modelo == "rn50" else "ViT-B/16"
                ax.set_ylabel(f"{rot}\n{grupo}", fontsize=6.0, labelpad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.subplots_adjust(wspace=0.04, hspace=0.10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(args):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    linhas_log = []

    def log(s):
        print(s, flush=True)
        linhas_log.append(s)

    t0 = time.time()
    tf = {m: m168.W_OF[m].transforms() for m in MODELS}
    pools = m07.query_pools(max(args.n_pick, 8))
    tr, _ = m07.imagefolder_files(m07.P["imagenet1k"]["train"])
    log(f"device {dev} | seed {m07.SEED} | ood {args.ood}")

    log("query choice, median s under RN-50 tile 224:")
    escolha = escolhe_queries(dev, tf["rn50"], tr, pools, args, log)

    rng = np.random.default_rng(m07.SEED)
    res, disp_all, rows = {}, {}, []
    for grupo, info in escolha.items():
        res[grupo], disp_all[grupo] = {}, {}
        for modelo in MODELS:
            x1 = m164.load_official([info["file"]], tf[modelo],
                                    pre=(grupo == "id"))
            w = m168.wrap(modelo, dev, 224)
            cls, _ = w.pred_pool(x1, 1)
            protos = proto_for_class(w, int(cls[0]), tf[modelo], tr,
                                     args.n_cand, args.bs, rng)
            del w
            torch.cuda.empty_cache()

            res[grupo][modelo], disp_all[grupo][modelo] = {}, {}
            for cond in CONDS:
                xin, L, s, c = mapa_e_s(modelo, dev, x1, protos, tf[modelo],
                                        cond, args.bs)
                res[grupo][modelo][cond] = dict(mapa=L.numpy(), s=s, cls=c)
                disp_all[grupo][modelo][cond] = denorm(xin, tf[modelo])
                rows.append(dict(group=grupo, model=modelo, cond=cond,
                                 pred_class=c, s=s,
                                 map_side=L.shape[-1],
                                 file=info["file"]))
                log(f"  {grupo:<12} {modelo:<5} {cond:<10} "
                    f"class {c:<4} map {tuple(L.shape)} "
                    f"s={'--' if s is None else f'{s:.3f}'}")

    disp_esq = {c: disp_all["id"]["vit"][c] for c in
                ("single224", "mos224", "mos448")}
    pa = OUT / "173_fig_transformer_adapt_esquema.pdf"
    pb = OUT / "173_fig_transformer_adapt_mapas.pdf"
    figura_esquema(disp_esq, pa, log)
    figura_mapas(res, disp_all, pb)

    df = pd.DataFrame(rows)
    pc = OUT / "173_medidas.csv"
    df.to_csv(pc, index=False)

    tabela = df[df.s.notna()].pivot_table(
        index=["group", "model"], columns="cond", values="s")
    base = {(g, m): res[g][m]["single224"]["cls"] for g in res for m in MODELS}
    flips = [r for r in rows
             if r["pred_class"] != base[(r["group"], r["model"])]]
    (OUT / "173_fig_transformer_adapt_esquema.txt").write_text(
        "fig_adaptations.py\n\n"
        "WHAT IT MEASURES\n"
        "  The two adaptations a ViT needs to read a 2x2 mosaic, and what each\n"
        "  one costs. The RN-50 is the control: equivariant to the input size\n"
        "  and without positional embedding, it reads the three sizes\n"
        "  unchanged. The white grid is the 16 px patch grid; the orange frame\n"
        "  is the query tile.\n\n"
        "FILES\n"
        "  173_fig_transformer_adapt_esquema.pdf  diagram of the two\n"
        "    adaptations with the patch grid, plus the positional embeddings of\n"
        "    the ViT-B/16 in PCA-RGB and in cosine similarity, at 14x14 and\n"
        "    interpolated to 28x28.\n"
        "  173_fig_transformer_adapt_mapas.pdf  heatmaps, 2 queries x 2 models\n"
        "    x 4 conditions. CAM at layer4 for the RN-50, Grad-CAM at the\n"
        "    penultimate encoder block for the ViT.\n"
        "  173_medidas.csv  one row per cell, with s and the predicted class.\n\n"
        "NUMBERS\n" + tabela.to_string() + "\n\n"
        "HOW TO READ\n"
        "  1. Patch grid in the middle column of the diagram: the mosaic is\n"
        "     back to 224 and the grid is still 14x14, so the query keeps 49\n"
        "     of 196 tokens. The loss is of token resolution, not of pixels.\n"
        "  2. Grid in the right column: the query keeps 196 tokens, but the\n"
        "     grid becomes 28x28 and the positional embedding has to be\n"
        "     interpolated.\n"
        "  3. PCA-RGB of the embeddings: the smooth spatial structure survives\n"
        "     the interpolation; the 28x28 map is the smoothed 14x14 one.\n"
        "  4. Cosine to the central position: the interpolation widens the\n"
        "     peak. The profile on the right shows both curves on the same\n"
        "     relative scale.\n"
        "  5. Heatmaps: compare row by row. Where s drops between mos448 and\n"
        "     mos224 on the ViT and not on the RN-50, the cost is of the\n"
        "     resampling, not of the construction.\n"
        "  6. s only exists in the two mosaic columns. Without a mosaic there\n"
        "     is no reference quadrant and the ratio is not defined.\n\n"
        "CAVEATS\n"
        "  a. Each map is normalized by percentiles 2 and 98 inside its own\n"
        "     panel. The color compares positions inside a panel, never across\n"
        "     panels; s is what compares panels.\n"
        "  a1. Two queries are not a trend. The s of one example can rise\n"
        "     between mos448 and mos224 while the AUROC of the set drops. The\n"
        "     AUROC values use n=1000 per set; these maps show the mechanism\n"
        "     and do not estimate the effect.\n"
        "  a2. s exactly 0.000 is not a failure. It means that no pixel of the\n"
        "     query quadrant exceeded the mean of the mosaic, so the rectified\n"
        "     energy of the query is zero: the extreme far-OOD case.\n"
        "  b. The reference tiles differ between the RN-50 and the ViT. The\n"
        "     exemplars are chosen in the feature space of each model, as in\n"
        "     tab_field_mosaic.py, so the same class gives different banks.\n"
        "  c. The predicted class can change between 224 and 448. The cells\n"
        "     where it changed are listed below; the map always follows the\n"
        "     class predicted in its own condition.\n"
        + ("".join(f"     {r['group']} {r['model']} {r['cond']}: class "
                   f"{r['pred_class']}\n" for r in flips) or
           "     none\n") + "\n"
        "QUERY CHOICE\n"
        "  Fixed rule: the query of MEDIAN s under RN-50 with tile 224, among\n"
        f"  the first {args.n_pick} of each group. It is not the best-separating one.\n"
        + "".join(f"  {g:<12} s={i['s_rn50_448']:.3f} rank {i['rank']} "
                  f"(min {i['s_min']:.3f}, max {i['s_max']:.3f})\n"
                  f"    {i['file']}\n" for g, i in escolha.items()) + "\n"
        "LOG\n" + "\n".join(linhas_log) + "\n")

    log(f"\n[ok] {pa}")
    log(f"[ok] {pb}")
    log(f"[ok] {pc}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", default="inaturalist",
                    choices=["ssb_hard", "ninco", "inaturalist",
                             "openimage_o", "textures"])
    ap.add_argument("--n-pick", type=int, default=16)
    ap.add_argument("--n-cand", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    main(ap.parse_args())
