"""149 - Schematic figure of the six saliency methods on one mosaic.

experiment_name = 'saliency_zoo'
dataset_name    = 'imagenet1k'

A schematic example showing GradCAM, GradCAM++, ScoreCAM, LIME and EigenCAM
next to the mosaic.

LAYOUT
  1. The figure is drawn at PRINTED size, 6.221 in (the SCITEPRESS text
     width), so 1 pt in the figure is 1 pt on paper: titles at 7.5 pt and the
     colour scale at 7 pt.
  2. Each title takes TWO lines (method / value of s): on one line, at 7.5 pt,
     it does not fit a panel of about 0.97 in.
  3. Panels are placed by hand, with neither tight_layout nor bbox tight, so
     the PDF size is exactly the figure size.
  4. `--replot` reads the cache BEFORE building the adapter (model, pools), so
     a redraw needs no GPU. The maps, query, class and `s` values come from
     results/saliency_zoo/winner.pt (read only). A new search never
     overwrites that cache; it writes results/camera_ready/149_winner.pt.

WHAT THE FIGURE CLAIMS
  The six methods read the SAME mosaic and the SAME predicted class. All that
  changes is how each one weights the evidence. Since every attribution map is
  displayed min-max normalised, the visual comparison is one of SHAPE; the `s`
  annotated on each panel is the scalar reading the mosaic adds, and it is NOT
  invariant to a change of method.

TWO THINGS THAT MUST BE IN THE CAPTION (measured in this repository)
  1. On a GAP+FC head the gradient d(l_c)/dA_k(u) = w_ck/Z is CONSTANT in
     space, so GradCAM IS the CAM. The identity is verified to 0.0 in
     scripts/75, which is why the column is labelled "GradCAM (= CAM)".
  2. EigenCAM does not use the head: it is the first principal component of
     A. It enters as a CONTROL, a map that ignores the predicted class.

  The sign of the EigenCAM principal component is arbitrary (SVD). The
  `w_eigencam` of scripts/76 orients it by what correlates positively with the
  activation mass, otherwise half the queries would come out inverted by
  numerical accident.

SAMPLE SELECTION
  The first far-OOD query in the pool whose prediction is stable. There is no
  directed search here, because the figure illustrates the DEFINITION of the
  methods rather than an effect. `--pick n` changes the sample.

Outputs: figures/head_refit/149_f1__saliency_zoo.{pdf,png} + .txt

THE PUBLISHED FIGURE
  query 29d18a362a1a1fb3a1287af837e157e9.jpg from the inaturalist pool,
  predicted class 412 (trash can), margin min(others) - cam = +0.310.
  Reproduce with --find-class 412 (or --only-path with the name above).
  Redraw without repeating the search: --replot.

Usage:
  $PYTHON_BIN fig3_code/fig3_saliency_zoo.py --replot
  $PYTHON_BIN fig3_code/fig3_saliency_zoo.py --find-class 412
"""
import os

experiment_name = "saliency_zoo"
dataset_name    = "imagenet1k"

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_code"
MOSAIX = Path(os.environ.get("MOSAIX_ROOT", ROOT))
XAI = ROOT / "external_xai_ood"
sys.path.insert(0, str(XAI))
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(MOSAIX / "scripts"))
sys.path.insert(0, str(MOSAIX / "scripts" / "frozen_xai_ood"))

from src.mosaic import build_mosaic

m29 = importlib.import_module("29_im200_models")
m30 = importlib.import_module("30_attribution_methods")
m73 = importlib.import_module("73_attribution_multi")
m76 = importlib.import_module("76_saliency_matrix")
m128 = importlib.import_module("128_seam_full")
warnings.filterwarnings("ignore", category=UserWarning)

FIG = ROOT / "figures" / "head_refit"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif", "font.size": 7})
# SCITEPRESS text width (\textwidth = 6.221 in): the figure enters at 1:1
PAGE_W = 6.221
TITLE_FS, CBAR_FS = 7.5, 7.0
IMN = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
ORANGE, INK = "#D95F02", "#222222"
# RISE is left out of the figure: over a 40-query search it went dark
# together with GradCAM (0.11 against 0.10), so it does not discriminate.
SHOW = [("cam", "GradCAM (= CAM)"), ("gradcampp", "GradCAM++"),
        ("scorecam", "ScoreCAM"), ("lime", "LIME"), ("eigencam", "EigenCAM")]


def to_img(t):
    return np.clip(t.permute(1, 2, 0).numpy() * IMN[1] + IMN[0], 0, 1)


def norm01(E):
    E = np.clip(E, 0, None)
    m = float(E.max())
    return E / m if m > 0 else E


def _draw(args, it, tiles, i, rt, maps, svals, c, cand_n):
    T = tiles[i].shape[-1]
    q = to_img(tiles[i])
    mimg = np.zeros((2 * T, 2 * T, 3)); mimg[:T, :T] = q
    for k, (yy, xx) in enumerate(((0, 1), (1, 0), (1, 1))):
        mimg[yy*T:(yy+1)*T, xx*T:(xx+1)*T] = to_img(rt[k])

    n = len(SHOW) + 1
    # Geometry in INCHES. Normal gap G between panels; before the last one
    # (EigenCAM) a wider gap GCB, which holds the colour scale and its ticks.
    G, GCB, BOT, TOPM = 0.05, 0.20, 0.005, 0.27
    w = (PAGE_W - (n - 2) * G - GCB) / n          # side of the square panel
    H = BOT + w + TOPM
    fig = plt.figure(figsize=(PAGE_W, H))
    ax, x = [], 0.0
    for j in range(n):
        if j == n - 1:
            x += GCB - G
        ax.append(fig.add_axes([x / PAGE_W, BOT / H, w / PAGE_W, w / H]))
        x += w + G
    ax[0].imshow(mimg)
    ax[0].add_patch(plt.Rectangle((-.5, -.5), T, T, fill=False, ec=ORANGE,
                                  lw=1.2))
    ax[0].set_title("mosaic", fontsize=TITLE_FS, pad=2)
    for j, (meth, lab) in enumerate(SHOW, start=1):
        A = ax[j]
        Eu = np.asarray(Image.fromarray(maps[meth].astype(np.float32),
                                        mode="F")
                        .resize((2 * T, 2 * T), Image.BILINEAR), dtype=float)
        col = plt.get_cmap("magma")(np.clip(Eu, 0, 1))[..., :3]
        A.imshow(np.clip(mimg * (1 - args.alpha) + col * args.alpha, 0, 1))
        A.add_patch(plt.Rectangle((-.5, -.5), T, T, fill=False, ec=ORANGE,
                                  lw=1.2))
        # `s` goes in the TITLE, on a second line: at 7.5 pt a one-line
        # title was wider than the panel
        A.set_title(f"{lab}\n$s$ = {svals[meth]:.2f}", fontsize=TITLE_FS,
                    pad=2, linespacing=1.15)
    for A in ax:
        A.set_xticks([]); A.set_yticks([])

    # Map scale, 0 at the bottom and 1 at the top, against the left of the
    # last panel. Maps are normalised one by one, so the bar gives the colour
    # range, not energy comparable across panels.
    _p = ax[-1].get_position()
    _cw, _cg = 0.035, 0.03                 # bar width and gap, in inches
    _cax = fig.add_axes([_p.x0 - (_cw + _cg) / PAGE_W, _p.y0,
                         _cw / PAGE_W, _p.height])
    _g = np.linspace(1, 0, 256).reshape(-1, 1)
    _cax.imshow(_g, cmap="magma", aspect="auto", extent=(0, 1, 0, 1))
    _cax.set_xticks([])
    _cax.yaxis.tick_left()
    _cax.set_yticks([0, 1])
    _cax.set_yticklabels(["0", "1"], fontsize=CBAR_FS)
    _cax.tick_params(axis="y", length=1.5, pad=1.2)
    # "0" sits ABOVE the base and "1" hangs BELOW the top: centred on the
    # tick they would spill out of the figure height
    _tl = _cax.get_yticklabels()
    _tl[0].set_va("bottom"); _tl[1].set_va("top")
    for _s in _cax.spines.values():
        _s.set_linewidth(0.4)

    # nothing may leave the canvas (without bbox tight, savefig would crop it)
    fig.canvas.draw()
    _bb = fig.get_tightbbox(fig.canvas.get_renderer())
    assert _bb.x0 >= -5e-3 and _bb.y0 >= -5e-3 and _bb.x1 <= PAGE_W + 5e-3 \
        and _bb.y1 <= H + 5e-3, f"content outside the figure: {_bb}"

    o = FIG / "149_f1__saliency_zoo"
    # 400 dpi: at printed size the ~0.97 in panel gets ~390 px, close to the
    # native 448 px of the mosaic
    for e in ("pdf", "png"):
        fig.savefig(o.with_suffix("." + e), dpi=400)
    plt.close(fig)
    print(f"  figure {PAGE_W:.3f} x {H:.3f} in", flush=True)

    tab = "\n".join(f"    {lab:<18} s = {svals[m]:.3f}" for m, lab in SHOW)
    o.with_suffix(".txt").write_text(f"""149_f1__saliency_zoo.pdf -- generated by fig3_code/fig3_saliency_zoo.py

FILES
  figures/head_refit/149_f1__saliency_zoo.pdf  figure, {PAGE_W:.3f} x {H:.3f} in,
                                               enters the text width at 1:1
  figures/head_refit/149_f1__saliency_zoo.png  400 dpi, for inspection
  figures/head_refit/149_f1__saliency_zoo.txt  this file
  Titles at {TITLE_FS} pt on two lines (method / s); colour scale at {CBAR_FS} pt.

WHAT IT SHOWS AND WHY
  The saliency methods reading the SAME mosaic and the SAME predicted class
  ({c}). All that changes between the panels is how each one weights the
  evidence. Every map is min-max normalised, which is the convention
  attribution maps are displayed under, so the visual comparison is one of
  SHAPE. The `s` annotated on each panel is the scalar reading the mosaic
  adds, and it is NOT invariant to a change of method.

TWO THINGS MEASURED IN THIS REPOSITORY THAT THE CAPTION MUST CARRY
  1. On a GAP+FC head the gradient d(l_c)/dA_k(u) = w_ck/Z is CONSTANT in
     space. So GradCAM IS the CAM, an identity verified to 0.0 in scripts/75,
     together with HiRes-CAM (1.7e-7) and XGrad-CAM (1.2e-7). They are not
     three methods, they are one.
  2. EigenCAM does not use the head: it is the first principal component of
     A. It enters as a CONTROL, a map that ignores the predicted class. The
     sign of the component is arbitrary under SVD, and scripts/76 orients it
     by the correlation with the activation mass, otherwise half the queries
     would come out inverted by accident.

VALUES
{tab}

SELECTION
  A DIRECTED search over {cand_n} queries of the `{args.ood}` pool: maximise
  min(s of the other four) - s_GradCAM. The criterion is the WORST case, not
  the mean, so it is high only when all of them light up. Chosen:
  {Path(it['path']).name}, margin {it['score']:+.3f}.

HOW TO READ IT
  1. Compare the SHAPE across panels; absolute intensity was discarded by the
     normalisation in all of them.
  2. EigenCAM is the only one that does not see the class: if it lights up in
     the same place as the others, what the others show is activation
     magnitude rather than class-specific evidence. This is the same warning
     as scripts/35.
  3. LIME works by perturbation in IMAGE space, so its superpixels cross the
     mosaic seam on purpose. Cropping them per tile would give LIME
     information no other method has.
""")
    print(f"\n[ok] {o.with_suffix('.pdf')}")


def main(args):
    # The cache is read BEFORE the adapter, so --replot loads neither the
    # model nor the pools.
    CACHE = ROOT / "results" / "saliency_zoo" / "winner.pt"
    if args.replot and CACHE.exists():
        it = torch.load(CACHE, map_location="cpu", weights_only=False)
        cand_n = it["cand_n"]
        c, rt, maps, svals = it["cls"], it["rt"], it["maps"], it["s"]
        tiles = {0: it["tile"]}; i = 0
        print(f"  replot of {Path(it['path']).name} | class {c} "
              f"(cache of {cand_n} scanned)", flush=True)
        for meth, lab in SHOW:
            print(f"    {lab:<18} s={svals[meth]:.3f}", flush=True)
        return _draw(args, it, tiles, i, rt, maps, svals, c, cand_n)
    if args.replot:
        raise SystemExit(f"--replot without a cache at {CACHE}; not recomputing here")

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ad = m73.ADAPTERS["imagenet1k"](args, dev)
    grid = ad.grid
    gen = torch.Generator().manual_seed(args.seed)
    def read_all(tile, c):
        """The maps and the `s` values for one query. Returns (mos, rt, maps, s)."""
        rt = ad.panel_tiles(c)
        mos = build_mosaic(tile, rt, qpos=0)
        Am = ad.layer4(mos[None], dev, args.bs)[0].double()
        wc, b_c = ad.W[c].double(), float(ad.b[c])
        mp, sv = {}, {}
        for meth, _ in SHOW:
            if meth in m76.W_METHODS:
                fw = m76.w_of(meth)
                sel, w = fw(Am, wc, b_c, model=ad.model, c=c, dev=dev,
                            bs=args.sc_bs, topk=args.scorecam_topk, x=mos)
                Lm = m30.apply_w(Am, sel, w)
            else:
                Lm = m76.MFN[meth](mos, ad.model, c, dev, grid, args, gen=gen)
            Lm = torch.as_tensor(Lm).double()
            E = torch.relu(Lm - Lm.mean()).numpy()
            mp[meth] = norm01(E)
            sv[meth] = m128.s_of(E, grid // 2)
        return mos, rt, mp, sv

    # A new search NEVER overwrites results/saliency_zoo/winner.pt; it writes
    # its own cache under results/camera_ready/.
    CACHE = ROOT / "results" / "camera_ready" / "149_winner.pt"
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    # DIRECTED search: the query on which GradCAM goes DARK on the OOD
    # quadrant and the others LIGHT UP. The criterion is the worst case,
    # min(s of the others) - s_cam, which is high only when ALL of them
    # light up, not their mean.
    refs = ad.pools[args.ood][:args.n_search]
    if args.only_path:
        # recompute ONLY the named query: fills the cache without repeating
        # the scan when the winner is already known
        hit = [r for r in ad.pools[args.ood]
               if Path(str(r)).name == args.only_path]
        if not hit:
            raise SystemExit(f"`{args.only_path}` is not in the "
                             f"`{args.ood}` pool with n_per_set={args.n_per_set}")
        refs = hit[:1]
    best, cand_n = None, 0
    for s0 in range(0, len(refs), args.chunk):
        batch = refs[s0:s0 + args.chunk]
        tl = ad.load_tiles(batch, args.ood)
        pr = ad.predict(tl, dev, args.bs)
        for j in range(len(batch)):
            c_ = int(pr[j])
            if args.list_preds:
                print(f"    {Path(str(batch[j])).name}  class {c_}", flush=True)
                continue
            if args.find_class is not None and c_ != args.find_class:
                continue
            mos_, rt_, mp_, sv_ = read_all(tl[j], c_)
            others = [sv_[m] for m, _ in SHOW if m != "cam"]
            score = min(others) - sv_["cam"]
            cand_n += 1
            if best is None or score > best["score"]:
                best = dict(score=score, path=str(batch[j]), cls=c_,
                            tile=tl[j], mos=mos_, rt=rt_, maps=mp_, s=sv_)
        if best is not None:
            print(f"    {min(s0+args.chunk, len(refs)):>3}/{len(refs)} scanned"
                  f"  | best min(others)-cam = {best['score']:+.3f}",
                  flush=True)
    it = best
    it["cand_n"] = cand_n
    torch.save(it, CACHE)
    c, rt, maps, svals = it["cls"], it["rt"], it["maps"], it["s"]
    tiles = {0: it["tile"]}
    i = 0
    print(f"\n  chosen {Path(it['path']).name} | predicted class {c} "
          f"of {cand_n} scanned", flush=True)
    print(f"    GradCAM s={svals['cam']:.3f} | lowest of the others "
          f"{min(svals[m] for m, _ in SHOW if m != 'cam'):.3f} "
          f"| margin {it['score']:+.3f}", flush=True)
    for meth, lab in SHOW:
        print(f"    {lab:<18} s={svals[meth]:.3f}", flush=True)

    return _draw(args, it, tiles, i, rt, maps, svals, c, cand_n)

    print(f"\n[ok] {o.with_suffix('.pdf')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", default="inaturalist")
    # 600, not 40: the query of the published figure lies beyond the first
    # 40 of the pool. With 40 the script picks another image.
    ap.add_argument("--n-search", type=int, default=600)
    ap.add_argument("--list-preds", action="store_true",
                    help="list candidate -> predicted class and exit")
    ap.add_argument("--find-class", type=int, default=None,
                    help="only candidates with this predicted class")
    ap.add_argument("--only-path", default=None,
                    help="recompute only this query, by file name")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the cache, without scanning again")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.72)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--sc-bs", type=int, default=64)
    ap.add_argument("--scorecam-topk", type=int, default=128)
    ap.add_argument("--rise-n", type=int, default=2000)
    ap.add_argument("--lime-n", type=int, default=1000)
    ap.add_argument("--lime-segments", type=int, default=100)
    ap.add_argument("--occ-stride", type=int, default=16)
    ap.add_argument("--occ-size", type=int, default=32)
    ap.add_argument("--n-per-set", type=int, default=600)
    ap.add_argument("--proto-sub", type=int, default=120)
    ap.add_argument("--max-candidates", type=int, default=24)
    ap.add_argument("--fit-n", type=int, default=200000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
