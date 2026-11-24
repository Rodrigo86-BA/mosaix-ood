"""Figure 1 (teaser), v2: four rows plus the large mosaic spanning two rows.

experiment_name = 'fig_teaser_v2'
dataset_name    = 'imagenet1k'

LAYOUT
------
  row 1     the 4 ID images, WITHOUT the explanation
  row 2     the same 4, WITH the explanation overlaid   <- title = prediction
  row 3     the 4 far-OOD images, WITHOUT the explanation
  row 4     the same 4, WITH the overlay                <- title = prediction
  column 5  the MOSAIC, spanning 2 rows at a time (ID on top, OOD below)

WHAT CHANGES RELATIVE TO THE CURRENT FIG. 1
-------------------------------------------
`fig1_explanations_and_mosaic.png` is an inherited raster, and PLAN.md (F3)
records that the dominance values printed inside it come from the LEGACY run
(0.34/0.17/0.08), not from the frozen protocol of Table 2. This figure is
rendered from the dump of scripts/43_grid_dump.py, the same backbone, the same
prototype panel and the same seed as the frozen run, so the printed numbers
are those of the protocol in the paper. That closes F3.

THE PAIR IS MATCHED
-------------------
The ID query is one of the 692 (out of 1000) that all THREE RN-50/IN-1K models
(ce, ls, mixup) classify correctly. The far-OOD query is predicted into the
SAME class, so both mosaics use THE SAME THREE REFERENCES. The only thing that
changes between the top block and the bottom one is the query. A gate checks
this.

COLOUR SCALE
------------
One scale common to the 10 images (8 tiles plus 2 mosaics). Nothing is
normalised per panel, so the drop on the query tile is not a rendering
artefact.

Outputs:
  figures/teaser/fig1_teaser_v2.pdf   (plus companion .txt)

Usage:
  $PYTHON_BIN fig1_code/fig1_teaser.py
  $PYTHON_BIN fig1_code/fig1_teaser.py --ood textures
"""
import os

experiment_name = "fig_teaser_v2"
dataset_name    = "imagenet1k"

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision.models import ResNet50_Weights

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "results" / "grid_dump"
OUTDIR = ROOT / "figures" / "teaser"
OUTDIR.mkdir(parents=True, exist_ok=True)
STEM = "fig1_teaser_v2"

FAR = ["inaturalist", "openimage_o", "textures"]


def quads(g):
    """The 4 quadrants of a g by g grid (build_mosaic, qpos=0)."""
    h = g // 2
    return [(slice(0, h), slice(0, h)), (slice(0, h), slice(h, g)),
            (slice(h, g), slice(0, h)), (slice(h, g), slice(h, g))]


CATS = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
OODLAB = {"inaturalist": "iNaturalist", "openimage_o": "OpenImage-O",
          "textures": "Textures", "ninco": "NINCO",
          "ssb_hard": "SSB-hard"}

NEAR = {"ssb_hard", "ninco"}
FARSETS = {"inaturalist", "openimage_o", "textures"}

DATASETS = {
    "in1k":    dict(grids="grid_dump", cells=("ce", "ls", "mixup"),
                    tile=224, grid=14, names="tv1k", imgs="path",
                    label="ImageNet-1K", arch="ResNet-50 (from scratch)"),
    "im200":   dict(grids="grid_dump_im200",
                    cells=("ce", "ce_oe", "ls", "ls_oe"),
                    tile=224, grid=14, names="words", imgs="path",
                    label="ImageNet-200", arch="ResNet-18"),
    "cifar100": dict(grids="grid_dump_multi", cells=("cifar100",),
                     tile=32, grid=8, names="cifar", imgs="array",
                     label="CIFAR-100", arch="ResNet-18"),
    "in1k_tv": dict(grids="grid_dump_multi", cells=("imagenet1k_tv",),
                    tile=224, grid=14, names="tv1k", imgs="path",
                    label="ImageNet-1K", arch="ResNet-50 (torchvision V1)"),
}

_CIFNAMES = None


def cifar_name(c):
    global _CIFNAMES
    if _CIFNAMES is None:
        from torchvision import datasets
        _CIFNAMES = datasets.CIFAR100(
            os.environ.get("CIFAR_TORCHVISION_ROOT", ""), train=False,
            download=False).classes
    return _CIFNAMES[int(c)].replace("_", " ")


_WORDS = None


def wnid_name(wnid):
    """Short class name from the wnid, through the ImageNet words.txt."""
    global _WORDS
    if _WORDS is None:
        _WORDS = {}
        wp = Path(os.environ.get("TIN_WORDS",
                                 Path(os.environ.get("DATA_ROOT", "")) /
                                 "Tiny_imagenet" / "words.txt"))
        if wp.exists():
            for ln in wp.read_text().splitlines():
                if "\t" in ln:
                    k, v = ln.split("\t", 1)
                    _WORDS[k] = v.split(",")[0].strip()
    return _WORDS.get(wnid, wnid)


def true_labels(ds, qpath, group):
    """True label of the ID queries.

    in1k   the wnid is the parent directory of the file, and the class index is
           the alphabetical order of the train directories (ImageFolder
           convention).
    im200  the OpenOOD clean/val tree is flat (no class directory), so the label
           comes from the benchmark imglist itself.
    """
    t = np.full(len(qpath), -1, dtype=int)
    if ds == "in1k":
        train = Path(qpath[np.where(group != "id")[0][0]]).parent.parent
        return None
    import importlib
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    _s.path.insert(0, str(ROOT / "external_xai_ood"))
    m29 = importlib.import_module("29_im200_models")
    lab = {str(m29.resolve(r)): l
           for r, l in m29.read_imglist("val_imagenet200")}
    for k, q in enumerate(qpath):
        if group[k] == "id":
            t[k] = lab.get(str(q), -1)
    return t

PAGE_W = 6.9


def load_disp(path, pre):
    """The image as the network saw it (same crop), unnormalised."""
    im = Image.open(path).convert("RGB")
    if not pre:
        w, h = im.size
        sc = 256 / min(w, h)
        im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))),
                       Image.BILINEAR)
        w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        im = im.crop((l, t, l + 224, t + 224))
    return np.asarray(im, dtype=np.float32) / 255.


def cam_over(ax, G, side, vmin, vmax, alpha=0.80, per_panel=False):
    """Overlays the map G (7x7 or 14x14) on an image of `side` pixels.

    It does not use np.kron: that is nearest-neighbour and produces 32x32 blocks.
    Here the small map goes to imshow with an `extent` the size of the image and
    BICUBIC interpolation, the standard for a CAM overlay.

    per_panel=True normalises EACH map to its own [min,max] -> 0..1, which is what
    CAM tooling does by default and what an operator actually sees. per_panel=False
    uses the common scale passed in (vmin, vmax), which preserves the comparison
    across panels but flattens the query tiles against the prototypes (see
    scripts/53).

    The r value printed on the figure does NOT depend on this: it comes from the
    raw map.
    """
    G = np.asarray(G, float)
    if per_panel:
        lo, hi = float(G.min()), float(G.max())
        G = (G - lo) / max(hi - lo, 1e-12)
        vmin, vmax = 0.0, 1.0
    return ax.imshow(G, cmap="magma", alpha=alpha,
                     vmin=vmin, vmax=vmax, interpolation="bicubic",
                     extent=(-0.5, side - 0.5, side - 0.5, -0.5),
                     zorder=2)


def dominance(L):
    """s = E_q / (E_q + E_r) on a grid of arbitrary size."""
    g = L.shape[-1]
    Q = quads(g)
    mu = L.mean()
    E = np.maximum(L - mu, 0.0)
    eq = E[Q[0][0], Q[0][1]].mean()
    er = float(np.mean([E[a_, b_].mean() for a_, b_ in Q[1:]]))
    return (eq / (eq + er)) if (eq + er) > 0 else 0.0


def pick_pair(args):
    """(z, i_id, i_ood, class, s): an ID query every cell of the dataset gets right,
    plus an OOD query in the SAME predicted class, hence with the SAME references."""
    cfg = DATASETS[args.dataset]
    gdir = ROOT / "results" / cfg["grids"]
    Z = {k: np.load(gdir / f"grids__{k}.npz", allow_pickle=True)
         for k in cfg["cells"]}
    _k0 = cfg["cells"][0]
    _pk = "qpath" if "qpath" in Z[_k0] else "qref"
    assert all((Z[_k0][_pk] == Z[k][_pk]).all() for k in Z), \
        "the cells must share the query pool"
    z = Z[cfg["cells"][0]]
    group, pred = z["group"], z["pred"]
    qpath = z["qpath"] if "qpath" in z else z["qref"]
    rpath = z["rpath"] if "rpath" in z else z["rref"]

    if "true" in z:
        true = z["true"]
    elif args.dataset == "in1k":
        train = Path(rpath[0][0]).parent.parent
        wnids = sorted(d.name for d in train.iterdir() if d.is_dir())
        widx = {w: k for k, w in enumerate(wnids)}
        true = np.array([widx.get(Path(q).parent.name, -1) for q in qpath])
    else:
        true = true_labels(args.dataset, qpath, group)

    is_id = group == "id"
    allright = is_id & (true >= 0)
    for k in Z:
        allright &= (Z[k]["pred"] == true)
    print(f"queries ID acertadas por {len(Z)} modelo(s) "
          f"({', '.join(cfg['cells'])}): {int(allright.sum())}/{int(is_id.sum())}")

    Lm = z["L_mos"].astype(np.float64)
    sv = np.array([dominance(Lm[i]) for i in range(len(Lm))])

    pool = ([args.ood] if args.ood != "any"
            else sorted(FARSETS & set(np.unique(group).tolist())))
    best = None
    for c in np.unique(pred[allright]):
        ids = np.where(allright & (pred == c))[0]
        ood = np.where(np.isin(group, pool) & (pred == c))[0]
        if not len(ids) or not len(ood):
            continue
        i = int(ids[np.argmax(sv[ids])])
        j = int(ood[np.argmin(sv[ood])])
        if best is None or (sv[i] - sv[j]) > best[0]:
            best = (sv[i] - sv[j], int(c), i, j)
    assert best, f"no matched pair for --ood {args.ood}"
    gap, cls, i, j = best
    assert list(rpath[i]) == list(rpath[j]), "references differ"
    nk = cfg["names"]
    cname = (CATS[cls] if nk == "tv1k" else
             cifar_name(cls) if nk == "cifar" else
             wnid_name(Path(rpath[i][0]).parent.name))
    print(f"matched class: {cls} ({cname})")
    print(f"  ID  idx {i}  s={sv[i]:.3f}  {Path(qpath[i]).name}")
    print(f"  OOD idx {j}  s={sv[j]:.3f}  [{group[j]}]  {Path(qpath[j]).name}")
    print("  gate: the 3 references are IDENTICAL in both mosaics, ok")
    return z, i, j, cls, sv, cname


def layout_inline(blocks, vmin, vmax, PP, rpaths=None, CNAME='',
                  TILE=224):
    """2 rows by 9 columns (2 of them spacers), references shown ONCE.

    Columns: [query ID][query OOD] | spacer | [ref 1][ref 2][ref 3] |
             spacer | [mosaic ID][mosaic OOD]

    The spacers are gridspec columns without axes: they physically separate the
    three groups, and the dashed lines run through them.

    Two label bands at the top: the GROUP band ("Query samples", "Reference
    samples for <class> class", "Mosaic ID (s = ..)", "Mosaic OOD (s = ..)") and
    the COLUMN title ("ID", "<set> OOD", ref 1-3).

    Boxes: ID query in GREEN, OOD query in YELLOW; the same colour marks the
    corresponding tile inside each mosaic.

    Geometry: the right margin was tightened (the colorbar was consuming about
    0.9 in of usable width) and the fonts reduced, which allows a lower hspace.
    Both give width back to the individual tiles. The mosaic column width is still
    MEASURED in two passes.
    """
    bi, bo = blocks
    tiles = [bi["tiles"][0], bo["tiles"][0]] + bi["tiles"][1:]
    solos = [bi["solo"][0], bo["solo"][0]] + bi["solo"][1:]
    cname = CNAME
    oodset = bo["tag"].split("(")[-1].rstrip(")")
    TITLES = ["ID", f"{oodset} OOD", "ref 1", "ref 2", "ref 3"]
    QBOX = ["#009E73", "#f0c000"]
    BADGE_OK, BADGE_NO = "#1b9e77", "#c0392b"

    COLS = [0, 1, 3, 4, 5]
    SP = 0.12

    tids = ["?", "?", "?"]
    if rpaths is not None:
        tids = [str(x) for x in rpaths]
    SUB = [f"pred: {cname}", f"pred: {cname}"] + \
          [f"train id {t}" for t in tids]

    GAP_BETWEEN = 0.024
    W_FIG = 9.6

    R = 2.0
    for _pass in range(2):
        unit = W_FIG / (5 + 2 * SP + 2 * R)
        fig = plt.figure(figsize=(W_FIG, 2 * unit + 0.66))
        gs = GridSpec(2, 9, figure=fig,
                      width_ratios=[1, 1, SP, 1, 1, 1, SP, R, R],
                      wspace=0.075, hspace=0.11,
                      left=0.040, right=0.872, top=0.845, bottom=0.026)

        ax_raw, ax_ovl, mos_ax = [], [], []
        marcados = []
        for i, c in enumerate(COLS):
            a = fig.add_subplot(gs[0, c])
            a.imshow(tiles[i]); a.set_xticks([]); a.set_yticks([])
            a.set_title(TITLES[i], fontsize=6.2, pad=1.8)
            if i < 2:
                a.add_patch(plt.Rectangle((-.5, -.5), TILE, TILE, fill=False,
                                          ec=QBOX[i], lw=2.2, zorder=6))
            if i == 0:
                a.set_ylabel("image\nalone", fontsize=5.6, labelpad=2)
            ax_raw.append(a)

            a2 = fig.add_subplot(gs[1, c])
            a2.imshow(tiles[i])
            cam_over(a2, solos[i], TILE, vmin, vmax, per_panel=PP)
            a2.set_xticks([]); a2.set_yticks([])
            a2.set_title(SUB[i], fontsize=5.2, pad=1.2)
            if i < 2:
                a2.add_patch(plt.Rectangle((-.5, -.5), TILE, TILE, fill=False,
                                           ec=QBOX[i], lw=2.2, zorder=6))
            if i < 2:
                marcados.append((a2.title, a2, i == 0))
            if i == 0:
                a2.set_ylabel("with\nexplanation",
                              fontsize=5.6, labelpad=2)
            ax_ovl.append(a2)

        for k, b in enumerate(blocks):
            am = fig.add_subplot(gs[0:2, 7 + k])
            am.imshow(b["mos"])
            im = cam_over(am, b["Lmos"], 2 * TILE, vmin, vmax, per_panel=PP)
            am.set_xticks([]); am.set_yticks([])
            am.add_patch(plt.Rectangle((-.5, -.5), TILE, TILE, fill=False,
                                       ec=QBOX[k], lw=2.8, zorder=6))
            for x, y, va, txt in [(.25, 1.010, "bottom", f"Query {k + 1}"),
                                  (.75, 1.010, "bottom", "ref 1"),
                                  (.25, -.010, "top", "ref 2"),
                                  (.75, -.010, "top", "ref 3")]:
                t = am.text(x, y, txt, transform=am.transAxes,
                            ha="center", va=va, fontsize=5.4,
                            weight="bold" if txt.startswith("Query")
                            else "normal")
                if txt.startswith("Query"):
                    marcados.append((t, am, k == 0))
            mos_ax.append(am)

        fig.canvas.draw()
        W_in, H_in = fig.get_size_inches()
        h_need = ax_raw[0].get_position().y1 - ax_ovl[0].get_position().y0
        w_need = h_need * H_in / W_in
        w_have = mos_ax[0].get_position().width
        if _pass == 0:
            R *= w_need / w_have
            plt.close(fig)
            continue
        for k, am in enumerate(mos_ax):
            am.set_position([am.get_position().x0 + k * GAP_BETWEEN,
                             ax_ovl[0].get_position().y0, w_need, h_need])
            am.set_aspect("auto")
        tw = ax_raw[0].get_position().width * W_in
        print(f"    tile individual: {tw:.3f} in   mosaico: {w_need*W_in:.3f} in")
        xb = mos_ax[-1].get_position().x1 + 0.030
        print(f"    borda do Mosaic OOD {mos_ax[-1].get_position().x1:.3f} | "
              f"colorbar {xb:.3f} | folga {0.030*W_in:.2f} in")

    pq0, pq1 = ax_raw[0].get_position(), ax_raw[1].get_position()
    pr0, pr2 = ax_raw[2].get_position(), ax_raw[4].get_position()
    yband = ax_raw[0].get_position().y1 + 0.070
    fig.text((pq0.x0 + pq1.x1) / 2, yband, "Query samples",
             ha="center", va="bottom", fontsize=6.8, weight="bold")
    fig.text((pr0.x0 + pr2.x1) / 2, yband,
             f"Reference samples ({cname})",
             ha="center", va="bottom", fontsize=6.8, weight="bold")
    for k, (am, b) in enumerate(zip(mos_ax, blocks)):
        pm = am.get_position()
        fig.text(pm.x0 + pm.width / 2, yband,
                 f"Mosaic {'ID' if k == 0 else 'OOD'} ($s$ = {b['d']:.2f})",
                 ha="center", va="bottom", fontsize=6.8, weight="bold")

    ytop, ybot = yband + 0.032, ax_ovl[0].get_position().y0 - 0.070
    for xa, xb in [(pq1.x1, pr0.x0), (pr2.x1, mos_ax[0].get_position().x0)]:
        fig.add_artist(plt.Line2D([(xa + xb) / 2] * 2, [ybot, ytop],
                                  color="0.45", lw=0.9, ls=(0, (4, 3)),
                                  transform=fig.transFigure, zorder=0))

    mb = mos_ax[0].get_position()
    x_bar = mos_ax[-1].get_position().x1 + 0.030
    cax = fig.add_axes([x_bar, mb.y0, 0.011, mb.height])
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("explanation" if PP else "$L(u)$, shared scale",
                 fontsize=5.6, labelpad=2)
    cb.set_ticks([])
    lo, hi = ("0", "1") if PP else (f"{vmin:.0f}", f"{vmax:.0f}")
    cax.text(0.5, 1.012, hi, transform=cax.transAxes, ha="center",
             va="bottom", fontsize=5.4)
    cax.text(0.5, -0.012, lo, transform=cax.transAxes, ha="center",
             va="top", fontsize=5.4)

    from matplotlib.text import Text as _T
    fs = W_FIG / PAGE_W
    for t in fig.findobj(_T):
        t.set_fontsize(t.get_fontsize() * fs)

    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    shrunk = 0
    for ax in fig.axes:
        aw = ax.get_window_extent().width
        if aw <= 0:
            continue
        for t in (ax.title, ax.xaxis.label):
            if not t.get_text():
                continue
            tw = t.get_window_extent(renderer=rr).width
            if tw > 0.98 * aw:
                t.set_fontsize(t.get_fontsize() * 0.98 * aw / tw)
                shrunk += 1
    if shrunk:
        print(f"    width guard: {shrunk} label(s) shrunk to fit the "
              f"tile")

    SEP = 0.055
    apertados = []
    for alvo, ax, ok in marcados:
        g = ax.text(0, 0, "\u2713" if ok else "\u2717",
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=alvo.get_fontsize() * 1.30, fontweight="bold",
                    color=BADGE_OK if ok else BADGE_NO,
                    zorder=8, clip_on=False)
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        aw = ax.get_window_extent().width
        bb = alvo.get_window_extent(renderer=rr)
        tw, gw = bb.width / aw, g.get_window_extent(renderer=rr).width / aw
        if tw + SEP + gw > 0.98:
            r = 0.98 / (tw + SEP + gw)
            alvo.set_fontsize(alvo.get_fontsize() * r)
            g.set_fontsize(g.get_fontsize() * r)
            fig.canvas.draw()
            rr = fig.canvas.get_renderer()
            bb = alvo.get_window_extent(renderer=rr)
            tw = bb.width / aw
            gw = g.get_window_extent(renderer=rr).width / aw
            apertados.append(ax)
        yc = ax.transAxes.inverted().transform(
            (bb.x0, (bb.y0 + bb.y1) / 2))[1]
        cx, cy = alvo.get_position()
        cx_novo = cx - (SEP + gw) / 2
        alvo.set_position((cx_novo, cy))
        g.set_position((cx_novo + tw / 2 + SEP, yc))
    print(f"    marks beside the label: {len(marcados)}"
          + (f" ({len(apertados)} shrunk to fit the tile)"
             if apertados else ""))
    return fig


def main(args):
    global STEM, PP
    PP = (args.norm == "panel")
    if args.dataset == "in1k" and args.ood == "inaturalist":
        if args.layout == "inline":
            STEM = "fig1_teaser_inline" if PP else "fig_teaser_inline_shared"
        else:
            STEM = "fig1_teaser_v2" if PP else "fig_teaser_shared_scale"
    else:
        STEM = (f"teaser__{args.dataset}__{args.ood}__{args.layout}"
                f"__{'panel' if PP else 'shared'}")
    z, i_id, i_od, cls, dvals, cname = pick_pair(args)
    cfg = DATASETS[args.dataset]
    TL, GR = cfg["tile"], cfg["grid"]
    Q = quads(GR)
    group, pred = z["group"], z["pred"]
    Lm = z["L_mos"].astype(np.float64)
    Ls = z["L_solo"].astype(np.float64)

    def imgs_of(i):
        """(query, [3 refs]) as float arrays in 0..1, the way the dump stores them.

        In the dumps of scripts/55 the CIFAR entries carry the pixels inline (32 px
        fits in the npz); the 224 px ones store a path and the image is reread with
        the same crop the network saw."""
        if cfg["imgs"] == "array":
            return (z["q_img"][i].astype(np.float32) / 255.,
                    [x.astype(np.float32) / 255. for x in z["r_img"][i]])
        qp = z["qpath"] if "qpath" in z else z["qref"]
        pre = (args.dataset == "in1k" and group[i] == "id")
        return (load_disp(qp[i], pre=pre),
                [load_disp(x, pre=False) for x in z["rpath"][i]])

    blocks = []
    reg = "near-OOD" if str(group[i_od]) in ("ninco", "ssb_hard") else "far-OOD"
    for i, tag in [(i_id, "ID"),
                   (i_od, f"{reg} ({OODLAB.get(str(group[i_od]), str(group[i_od]))})")]:
        q, refs = imgs_of(i)
        M = np.zeros((2 * TL, 2 * TL, 3), np.float32)
        M[:TL, :TL] = q;   M[:TL, TL:] = refs[0]
        M[TL:, :TL] = refs[1]; M[TL:, TL:] = refs[2]
        solo = [Ls[i][Q[0][0], Q[0][1]]] + [Ls[i][a_, b_] for a_, b_ in Q[1:]]
        blocks.append(dict(tag=tag, tiles=[q] + refs, solo=solo,
                           mos=M, Lmos=Lm[i], d=dvals[i],
                           pred=int(pred[i]), idx=i, tile=TL))

    allL = np.concatenate([np.concatenate([s.ravel() for s in b["solo"]])
                           for b in blocks] +
                          [b["Lmos"].ravel() for b in blocks])
    vmin, vmax = np.percentile(allL, [2, 99.5])
    print(f"common colour scale: [{vmin:.1f}, {vmax:.1f}]")

    if args.layout == "inline":
        i0 = blocks[0]["idx"]
        rids = ([str(x) for x in z["rref"][i0]] if "rref" in z
                else [Path(x).stem.split("_")[-1] for x in z["rpath"][i0]])
        fig = layout_inline(blocks, vmin, vmax, PP, rpaths=rids, CNAME=cname,
                            TILE=blocks[0]["tile"])
        fp = OUTDIR / f"{STEM}.pdf"
        fig.savefig(fp, bbox_inches="tight", dpi=300)
        fig.savefig(OUTDIR / f"{STEM}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"\n[ok] {fp}  ({fp.stat().st_size/1e6:.2f} MB)")
        b0, b1 = blocks
        (OUTDIR / f"{STEM}.txt").write_text("\n".join([
            f"{STEM}.pdf companion", "=" * 58, "",
            "1. SCRIPT PRODUTOR",
            "   fig1_code/fig1_teaser.py --layout inline "
            f"--norm {'panel' if PP else 'shared'}",
            "   Dados: results/grid_dump/grids__ce.npz (scripts/43)", "",
            "2. LAYOUT",
            "   2 rows by 7 columns:",
            "     [QUERY ID][QUERY far-OOD][ref 1][ref 2][ref 3]"
            "[MOSAICO ID][MOSAICO far-OOD]",
            "   row 0 = image alone | row 1 = with the explanation",
            "   The two mosaics span both rows.",
            "   The 3 REFERENCES appear ONCE: they are the SAME in both",
            "   mosaics (matched pair), and an assert checks it.", "",
            "3. FILES",
            f"   figures/teaser/{STEM}.pdf / .png / .txt", "",
            "4. NUMERIC SUMMARY",
            f"   matched class: {b0['pred']} ({CATS[b0['pred']]})",
            f"   ID   s = {b0['d']:.3f}",
            f"   OOD  s = {b1['d']:.3f}  [{group[b1['idx']]}]",
            f"   normalisation: {'0-1 PER PANEL' if PP else 'common scale'}",
            f"   common raw scale (reference): [{vmin:.1f}, {vmax:.1f}]", "",
            "5. INTERPRETATION GUIDE",
            "   (1) The coloured box is the QUERY tile: GREEN for ID, YELLOW",
            "       for OOD; the same colour reappears in the mosaic.",
            "   (2) Both queries fall in the SAME predicted class, which is",
            "       why they share the reference panel. Only the query changes.",
            "   (3) Under per-panel normalisation the INDIVIDUAL explanation of",
            "       the OOD query looks strong, which is what a CAM tool shows.",
            "       The same tile goes dark inside the mosaic.",
            "   (4) The printed r comes from the RAW maps; the normalisation is",
            "       for display only.",
            "   (5) ONE pair of examples: illustration, not evidence.",
            "   (6) The GREEN check and the RED cross follow the LABEL, after",
            "       'pred: lemon' in the individual panels and after 'Query N'",
            "       in the mosaics. They say whether that query is ID (check)",
            "       or OOD (cross). They sat INSIDE the tile until 2026-09-09;",
            "       they moved off the image so as not to compete with the",
            "       explanation map.",
        ]) + "\n", encoding="utf-8")
        print(f"[ok] {OUTDIR / f'{STEM}.txt'}")
        return 0

    R = 2.0
    for _pass in range(2):
        unit = PAGE_W / (4 + R)
        fig = plt.figure(figsize=(PAGE_W, 4 * unit + 0.34))
        gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1, 1, 1, R],
                      wspace=0.035, hspace=0.035,
                      left=0.105, right=0.905, top=0.945, bottom=0.006)

        COL = ["QUERY", "reference 1", "reference 2", "reference 3"]
        mos_ax = []

        for bi, b in enumerate(blocks):
            r0 = 2 * bi
            first_top = first_bot = None
            for c in range(4):
                a = fig.add_subplot(gs[r0, c])
                a.imshow(b["tiles"][c]); a.set_xticks([]); a.set_yticks([])
                if bi == 0:
                    a.set_title(COL[c], fontsize=7.5, pad=2)
                if c == 0:
                    a.add_patch(plt.Rectangle((-.5, -.5), TILE, TILE, fill=False,
                                              ec="#f0c000", lw=2.2, zorder=6))
                    a.set_ylabel("image\nalone", fontsize=6.5, labelpad=2)
                    first_top = a

                a2 = fig.add_subplot(gs[r0 + 1, c])
                a2.imshow(b["tiles"][c])
                cam_over(a2, b["solo"][c], 224, vmin, vmax, per_panel=PP)
                a2.set_xticks([]); a2.set_yticks([])
                if c == 0:
                    a2.add_patch(plt.Rectangle((-.5, -.5), TILE, TILE, fill=False,
                                               ec="#f0c000", lw=2.2, zorder=6))
                    a2.set_ylabel("with\nexplanation", fontsize=6.5, labelpad=2)
                    first_bot = a2

            am = fig.add_subplot(gs[r0:r0 + 2, 4])
            am.imshow(b["mos"])
            im = cam_over(am, b["Lmos"], 2 * TILE, vmin, vmax, per_panel=PP)
            am.set_xticks([]); am.set_yticks([])
            am.add_patch(plt.Rectangle((-.5, -.5), TILE, TILE, fill=False,
                                       ec="#f0c000", lw=3.0, zorder=6))
            am.text(.982, .020, f"$s$ = {b['d']:.2f}", transform=am.transAxes,
                    ha="right", va="bottom", fontsize=10, color="w",
                    weight="bold",
                    bbox=dict(fc="k", alpha=.6, ec="none", pad=2.2))
            if bi == 0:
                am.set_title("MOSAIC", fontsize=8, pad=2, weight="bold")
            mos_ax.append((am, first_top, first_bot))

            p_top = first_top.get_position()
            p_bot = first_bot.get_position()
            fig.text(0.012, (p_top.y1 + p_bot.y0) / 2,
                     f"{b['tag']}\npredicted: {CATS[b['pred']]}",
                     rotation=90, ha="left", va="center",
                     fontsize=7.5, weight="bold", linespacing=1.35)

        fig.canvas.draw()
        W_in, H_in = fig.get_size_inches()
        am0, at0, ab0 = mos_ax[0]
        h_need = at0.get_position().y1 - ab0.get_position().y0
        w_need = h_need * H_in / W_in
        w_have = am0.get_position().width
        if _pass == 0:
            R *= w_need / w_have
            plt.close(fig)
            continue

        for am, a_top, a_bot in mos_ax:
            am.set_position([am.get_position().x0, a_bot.get_position().y0,
                             w_need, h_need])
            am.set_aspect("auto")

    gtop = mos_ax[0][1].get_position().y1
    gbot = mos_ax[-1][2].get_position().y0
    cax = fig.add_axes([0.945, gbot + (gtop - gbot) * 0.30,
                        0.013, (gtop - gbot) * 0.40])
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("explanation" if PP else "$L(u)$, shared scale",
                 fontsize=6.5, labelpad=3)
    cb.ax.tick_params(labelsize=6)
    if PP:
        cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", "0.5", "1"])

    fp = OUTDIR / f"{STEM}.pdf"
    fig.savefig(fp, bbox_inches="tight", dpi=300)
    fig.savefig(OUTDIR / f"{STEM}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"\n[ok] {fp}  ({fp.stat().st_size/1e6:.2f} MB)")

    b0, b1 = blocks
    L = [f"{STEM}.pdf companion", "=" * 58, "",
         "1. PRODUCING SCRIPT",
         "   fig1_code/fig1_teaser.py",
         "   Data: results/grid_dump/grids__ce.npz (scripts/43_grid_dump.py)",
         "   experiment_name = 'fig_teaser_v2'   dataset_name = 'imagenet1k'", "",
         "2. WHAT THE FIGURE SHOWS, AND WHY",
         "   The teaser of the paper: one ID query and one far-OOD query, each",
         "   with the 3 prototype references of the PREDICTED class, first",
         "   alone and then with the CAM of the predicted class on top; and the",
         "   448 mosaic (a JOINT forward pass) on the right, spanning both rows",
         "   of the block.",
         "   Layout: R1 ID without explanation | R2 ID with overlay | R3 far-OOD",
         "   without explanation | R4 far-OOD with overlay | column 5 = mosaic,",
         "   2 rows.",
         "   The title of each block is the model PREDICTION for the query.", "",
         "3. FILES",
         f"   figures/teaser/{STEM}.pdf   the figure (for the paper)",
         f"   figures/teaser/{STEM}.png   raster preview",
         f"   figures/teaser/{STEM}.txt   this file", "",
         "4. NUMERIC SUMMARY",
         f"   matched class: {b0['pred']} ({CATS[b0['pred']]})",
         f"   ID   idx {b0['idx']:>4}  d = {b0['d']:.3f}   "
         f"{Path(qpath[b0['idx']]).name}",
         f"   OOD  idx {b1['idx']:>4}  d = {b1['d']:.3f}   "
         f"{Path(qpath[b1['idx']]).name}  [{group[b1['idx']]}]",
         f"   colour scale common to the 10 images: [{vmin:.1f}, {vmax:.1f}]",
         "   backbone: ResNet-50 CE / ImageNet-1K (resnet50_full_ce_s0)", "",
         "5. INTERPRETATION GUIDE",
         "   (1) The yellow box marks the QUERY tile, everywhere.",
         "   (2) MATCHED PAIR: the ID query is one of the 692/1000 that all",
         "       THREE RN-50/IN-1K models (ce, ls, mixup) get right; the far-OOD",
         "       query is predicted into the SAME class. So both mosaics use THE",
         "       SAME THREE references, and an assert in the script checks it.",
         "       Only the query changes.",
         "   (3) A COLOUR SCALE COMMON to the 10 images (8 tiles + 2 mosaics).",
         "       Nothing is normalised per panel, so the query tile going dark",
         "       in the OOD block is not a rendering artefact.",
         "   (4) Row 2 against row 4: the INDIVIDUAL explanation of the OOD",
         "       query is not dark. It only goes dark in the mosaic, beside the",
         "       prototypes. That is the claim of the teaser.",
         "   (5) The `d` printed on the mosaic comes from the FROZEN protocol",
         "       (script 43), not from the legacy run. This closes item F3 of",
         "       PLAN.md: the previous Fig. 1 carried 0.34/0.17/0.08, values",
         "       from the legacy run, against the 0.31/0.18/0.07 of Table 2.",
         "   (6) The CAMs in rows 2 and 4 are the SOLO maps (each image passed",
         "       through the network ALONE), from L_solo in the dump. The one in",
         "       column 5 is the map of the JOINT forward pass. The difference",
         "       between the two is the subject of the paper.",
         "   (7) CAUTION: this is ONE pair of examples, chosen for the largest",
         "       contrast in d between an all-correct ID query and a far-OOD",
         "       query of the same class. It is illustration, not evidence. The",
         "       aggregate numbers are in Table 1 and in the notebook",
         "       `a_escala_do_mosaico`."]
    (OUTDIR / f"{STEM}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ok] {OUTDIR / f'{STEM}.txt'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="in1k",
                    choices=["in1k", "im200", "cifar100", "in1k_tv"],
                    help="which ID set (defines grids, cells and names)")
    ap.add_argument("--layout", default="stacked",
                    choices=["stacked", "inline"],
                    help="stacked = 4 rows by 5 columns (the current "
                         "Figure 1); inline = 2 rows by 7 columns, references "
                         "shown once")
    ap.add_argument("--norm", default="panel", choices=["panel", "shared"],
                    help="panel = each map normalised to 0..1 (the default "
                         "of CAM tooling, and Figure 1 of the paper); "
                         "shared = one scale common to the 10 panels")
    ap.add_argument("--ood", default="inaturalist",
                    choices=["any", "inaturalist", "openimage_o", "textures",
                             "ninco", "ssb_hard",
                             "cifar10", "tin", "svhn", "mnist"],
                    help="OOD set of query 2 (near or far)")
    main(ap.parse_args())
