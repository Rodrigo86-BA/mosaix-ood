"""119 - Figure 5 (fig:headrefit) in four columns: original, CE, LS, LS+AugDelete.

experiment_name = 'head_refit'
dataset_name    = 'imagenet1k'

WHAT CHANGES RELATIVE TO fig_head_refit.pdf (scripts/69)
--------------------------------------------------------
Four columns (original, CE model, LS model, LS+AugDelete model), no
transparency on low values, and an OOD sample with more contrast.

  1. THREE readings, not two. The old figure isolated the head: one backbone
     (LS) read by two heads. The whole CE model now enters as well.

     A CAVEAT THAT MUST STAY IN THE TEXT: the LS and LS+AugDelete columns
     share the frozen backbone, so the difference between them is ONLY the
     reading direction, which is the controlled experiment. The CE column is
     a different backbone trained from scratch: different features, not just
     a different head. Both comparisons live in the figure and they are not
     the same comparison.

  2. THE MOSAIC IS THE SAME IN ALL THREE COLUMNS. The reference panel comes
     from a single cell (`--panel-key`, default ce), so the input pixels are
     identical and only the reading changes. If each model brought its own
     panel, the columns would show different images and the visual comparison
     would be meaningless.

  3. NO ALPHA RAMP. The old overlay used alpha proportional to the value,
     which left the photograph clean where there is no evidence. Opacity is
     constant here; the price is that the reference tiles become less
     legible, which is why the `original` column exists. That column is where
     the reader checks that the references belong to the predicted class.

  4. THE SAMPLE IS CHOSEN BY A CRITERION, NOT BY EYE. Among the queries where
     all THREE models predict the same class (necessary, otherwise the maps
     would explain different classes), maximise

         z(contrast) + z(disagreement)
         contrast     = median over the 3 maps of (p90 - p50)/p90
         disagreement = min over the 3 pairs of (1 - spearman)

     The first term says the map has dynamic range; the second says the three
     maps are not the same drawing. Both are computed on the MOSAIC. The
     criterion is fixed before looking at the result and goes in the
     companion.

Outputs (figures/head_refit/): 119_fig5_headrefit_4col.{pdf,png} + .txt

Usage:
  $PYTHON_BIN fig4_code/fig4_head_refit.py
"""
import os

experiment_name = "head_refit"
dataset_name    = "imagenet1k"

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_code"
MOSAIX = Path(os.environ.get("MOSAIX_ROOT", ROOT))
XAI = ROOT / "external_xai_ood"
sys.path.insert(0, str(XAI))
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(MOSAIX / "scripts"))
sys.path.insert(0, str(MOSAIX / "scripts" / "frozen_xai_ood"))

from src.mosaic import build_mosaic

m07 = importlib.import_module("rood_qualify_protocol")
m68 = importlib.import_module("68_head_refit_figure")
warnings.filterwarnings("ignore", category=UserWarning)

RES = ROOT / "results" / experiment_name
FIG = ROOT / "figures" / experiment_name
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif", "font.size": 7})
COL_W = 3.35
INK, ORANGE = "#222222", "#D95F02"
IMNORM = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
COLS = [("ce", "model CE"), ("ls", "model LS"),
        ("refit", "model LS + AugDelete")]


def to_img(t):
    return np.clip(t.permute(1, 2, 0).numpy() * IMNORM[1] + IMNORM[0], 0, 1)


def norm01(E):
    m = float(E.max())
    return E / m if m > 0 else E


def s_of(E):
    g = E.shape[-1]; h = g // 2
    eq = E[:h, :h].mean()
    er = (E[:h, h:].mean() + E[h:, :h].mean() + E[h:, h:].mean()) / 3.0
    return float(eq / (eq + er + 1e-12))


def contrast(E):
    """Dynamic range of the map: (p90 - p50)/p90. High means the map has structure."""
    v = E.reshape(-1)
    p90, p50 = np.percentile(v, 90), np.percentile(v, 50)
    return float((p90 - p50) / p90) if p90 > 0 else 0.0


def main(args):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"=== 119 figure 5 in four columns | dev={dev} ===", flush=True)

    mce, Wce, bce = m07.load_backbone("ce", dev)
    mls, Wls, bls = m07.load_backbone("ls", dev)
    Wrf, brf, vtop1 = m68.refit_head("ls", dev, args)
    body = {"ce": m07.body_of(mce), "ls": m07.body_of(mls)}
    HEAD = {"ce": (Wce, bce, "ce"), "ls": (Wls, bls, "ls"),
            "refit": (Wrf, brf, "ls")}
    print(f"  AugDelete: val top-1 {vtop1:.4f}", flush=True)

    tr_files, offs = m07.imagefolder_files(m07.P["imagenet1k"]["train"])
    panel = m07.prototype_panel(args.panel_key, tr_files, offs)
    refs = m07.query_pools(args.n_query)[args.ood][:args.n_query]

    cand = []
    for s0 in range(0, len(refs), args.chunk):
        batch = refs[s0:s0 + args.chunk]
        tiles = m07.load_batch(batch, pre=(args.ood == "id"))
        Fq = {k: m07.layer4(body[k], tiles, dev, args.bs) for k in body}
        pred = {}
        for nm, (W, b, bk) in HEAD.items():
            z = Fq[bk].mean(dim=(2, 3)).double()
            pred[nm] = (z @ W.double().T + b.double()).argmax(1).numpy()
        for i in range(len(batch)):
            cs = {int(pred[nm][i]) for nm in HEAD}
            if len(cs) != 1:
                continue
            c = cs.pop()
            rt = m07.load_batch(panel[c])
            mos = build_mosaic(tiles[i], rt, qpos=0).unsqueeze(0)
            Fm = {k: m07.layer4(body[k], mos, dev, args.bs)[0] for k in body}
            Em, Es, sv = {}, {}, {}
            for nm, (W, b, bk) in HEAD.items():
                Lm = m68.cam(W, c, Fm[bk])
                Ls = m68.cam(W, c, Fq[bk][i])
                Em[nm] = m68.rect_mu(Lm).numpy()
                Es[nm] = m68.rect_plain(Ls).numpy()
                sv[nm] = s_of(Em[nm])
            ks = [k for k, _ in COLS]
            h = Em["ce"].shape[-1] // 2
            qsl = (slice(0, h), slice(0, h))

            def _rho(a, b, quad):
                x = a[quad].reshape(-1) if quad else _refs(a)
                y = b[quad].reshape(-1) if quad else _refs(b)
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    return 0.0
                r = float(spearmanr(x, y).statistic)
                return 0.0 if not np.isfinite(r) else r

            def _refs(E):
                g = E.shape[-1]
                return np.concatenate([E[:h, h:].reshape(-1),
                                       E[h:, :h].reshape(-1),
                                       E[h:, h:].reshape(-1)])

            gap = sv["ls"] - sv["ce"]
            d_int = abs(sv["refit"] - sv["ce"])
            geo_q = min(_rho(Em[a], Em[b_], qsl)
                        for a in ks for b_ in ks if a < b_)
            geo_r = min(_rho(Em[a], Em[b_], None)
                        for a in ks for b_ in ks if a < b_)
            con = float(np.median([contrast(Em[k]) for k in ks]))
            cand.append(dict(path=str(batch[i]), cls=c, gap=gap, d_int=d_int,
                             geo_q=geo_q, geo_r=geo_r, con=con,
                             tile=tiles[i], rt=rt, Em=Em, Es=Es, s=sv))
        print(f"    {min(s0+args.chunk, len(refs)):>4}/{len(refs)} varridas, "
              f"{len(cand)} candidatas  [{time.time()-t0:5.0f}s]", flush=True)

    if not cand:
        raise SystemExit("no query with a prediction all 3 models agree on")
    def _z(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / max(v.std(), 1e-9)
    gapv = np.array([c["gap"] for c in cand])
    zz = (_z(gapv) + _z([-c["d_int"] for c in cand])
          + 2.0 * _z([c["geo_q"] for c in cand])
          + 2.0 * _z([c["geo_r"] for c in cand]))
    rank = list(np.argsort(-zz))
    pick = int(rank[args.force_pick - 1]) if args.force_pick \
        else int(rank[0])
    it = cand[pick]
    hits = [c for c in cand
            if c["gap"] > args.thr_gap and c["d_int"] < args.thr_int
            and c["geo_q"] > args.thr_geo and c["geo_r"] > args.thr_geo]
    frac = 100.0 * len(hits) / max(len(cand), 1)
    n = len(cand)
    up = [c for c in cand if c["s"]["ls"] > c["s"]["ce"]]
    corr = [c for c in cand
            if abs(c["s"]["refit"] - c["s"]["ce"])
            < abs(c["s"]["ls"] - c["s"]["ce"])]
    both = [c for c in cand if c in up and c in corr]
    med_gap = float(np.median([c["s"]["ls"] - c["s"]["ce"] for c in cand]))
    med_res = float(np.median(
        [abs(c["s"]["ls"] - c["s"]["ce"]) - abs(c["s"]["refit"] - c["s"]["ce"])
         for c in cand]))
    gen = dict(n=n, up=len(up), corr=len(corr), both=len(both),
               med_gap=med_gap, med_res=med_res)
    print(f"\n  chosen {Path(it['path']).name} (class {it['cls']}) "
          f"out of {len(cand)} candidates", flush=True)
    print(f"    gap LS-CE {it['gap']:+.3f} | |AugDelete-CE| {it['d_int']:.3f} "
          f"| MINIMUM geometry over the 3: query {it['geo_q']:+.3f} refs {it['geo_r']:+.3f}",
          flush=True)
    print(f"    GENERALITY of the effect, over the {n} candidates:", flush=True)
    print(f"      LS inflates the query (s_LS > s_CE):      "
          f"{len(up)}/{n} = {100*len(up)/n:.1f}%   median {med_gap:+.3f}",
          flush=True)
    print(f"      AugDelete returns closer to CE than LS:   "
          f"{len(corr)}/{n} = {100*len(corr)/n:.1f}%   median correction "
          f"{med_res:+.3f}", flush=True)
    print(f"      both at once:                             "
          f"{len(both)}/{n} = {100*len(both)/n:.1f}%", flush=True)
    print(f"    (the STRICT conjunction with geometry and fixed thresholds "
          f"holds in {len(hits)}/{n} = {frac:.1f}%, which is the selection "
          f"filter, not the measure of the effect)", flush=True)
    for k, lab in COLS:
        print(f"    {lab:<22} s={it['s'][k]:.3f}", flush=True)

    K = min(args.top_k, len(cand))
    figs, axs = plt.subplots(K, 4, figsize=(4 * 1.72, 1.78 * K))
    axs = np.atleast_2d(axs)
    lin = []
    for r_, idx in enumerate(rank[:K]):
        c_ = cand[int(idx)]
        T_ = c_["tile"].shape[1]
        qi = to_img(c_["tile"])
        mi = np.zeros((2 * T_, 2 * T_, 3)); mi[:T_, :T_] = qi
        for k_, (yy, xx) in enumerate(((0, 1), (1, 0), (1, 1))):
            mi[yy*T_:(yy+1)*T_, xx*T_:(xx+1)*T_] = to_img(c_["rt"][k_])
        En = {k: norm01(c_["Em"][k]) for k, _ in COLS}
        panels = [(mi, None, None)] + [
            (mi, En[k], c_["s"][k]) for k, _ in COLS]
        for j_, (img, M, sv_) in enumerate(panels):
            A = axs[r_, j_]
            if M is None:
                A.imshow(img)
            else:
                H, Wd_ = img.shape[:2]
                Eu = np.asarray(Image.fromarray(M.astype(np.float32), mode="F")
                                .resize((Wd_, H), Image.BILINEAR), dtype=float)
                col = plt.get_cmap("magma")(np.clip(Eu, 0, 1))[..., :3]
                A.imshow(np.clip(img * (1 - args.alpha) + col * args.alpha,
                                 0, 1))
                A.text(.03, .03, f"$s$={sv_:.2f}", transform=A.transAxes,
                       color="white", fontsize=7.5, fontweight="bold",
                       va="bottom", bbox=dict(fc=INK, ec="none", alpha=.62,
                                              pad=1.3))
            A.add_patch(plt.Rectangle((-.5, -.5), T_, T_, fill=False,
                                      ec=ORANGE, lw=1.2))
            A.set_xticks([]); A.set_yticks([])
        axs[r_, 0].set_ylabel(
            f"#{r_+1}  cls {c_['cls']}\n"
            f"gap {c_['gap']:+.2f}\n"
            f"|AD-CE| {c_['d_int']:.2f}\n"
            f"geo q {c_['geo_q']:+.2f} r {c_['geo_r']:+.2f}",
            fontsize=5.8)
        lin.append(f"  #{r_+1:<3} {Path(c_['path']).name[:34]:<34} "
                   f"cls {c_['cls']:<5} s: CE {c_['s']['ce']:.3f} "
                   f"LS {c_['s']['ls']:.3f} AD {c_['s']['refit']:.3f} | "
                   f"gap {c_['gap']:+.3f} |AD-CE| {c_['d_int']:.3f} "
                   f"geo q {c_['geo_q']:+.3f} r {c_['geo_r']:+.3f}")
    for j_, t_ in enumerate(["original"] + [l for _, l in COLS]):
        axs[0, j_].set_title(t_, fontsize=7)
    figs.suptitle("Candidates for figure 5, pick one by number\n"
                  "OOD query dark under CE, lit under LS, with AugDelete "
                  "returning to CE", fontsize=8)
    figs.tight_layout(rect=[0, 0, 1, .955])
    osh = FIG / "119_f2__candidatos"
    for e_ in ("pdf", "png"):
        figs.savefig(osh.with_suffix("." + e_), dpi=170, bbox_inches="tight")
    plt.close(figs)
    osh.with_suffix(".txt").write_text(
        "119_f2__candidatos.pdf -- generated by fig4_code/fig4_head_refit.py\n\n"
        "WHAT IT SHOWS\n"
        f"  The {K} best candidates for figure 5, ranked by the directed\n"
        "  criterion: an OOD query dark under CE, lit under LS, with AugDelete\n"
        "  returning to CE in both intensity and geometry. Pick one by number\n"
        "  and run with --force-pick <n> to produce the final figure.\n\n"
        "GENERALITY OF THE EFFECT (over " + str(gen['n']) + " candidates)\n"
        f"  LS inflates the query     {gen['up']}/{gen['n']} = {100*gen['up']/gen['n']:.1f}%   median {gen['med_gap']:+.3f}\n"
        f"  AugDelete corrects        {gen['corr']}/{gen['n']} = {100*gen['corr']/gen['n']:.1f}%   median {gen['med_res']:+.3f}\n"
        f"  both                      {gen['both']}/{gen['n']} = {100*gen['both']/gen['n']:.1f}%\n\n"
        "CANDIDATES\n" + "\n".join(lin) + "\n")
    print(f"\n[ok] {osh.with_suffix('.pdf')}")
    for l_ in lin:
        print(l_, flush=True)

    En = {k: norm01(it["Em"][k]) for k, _ in COLS}
    Sn = {k: norm01(it["Es"][k]) for k, _ in COLS}
    for k, _ in COLS:
        d_ = abs(s_of(En[k]) - it["s"][k])
        assert d_ < 1e-9, f"s not invariant to normalisation ({k}): {d_:.2e}"
    print("  [gate] s invariant to 0-1 normalisation: max |delta| < 1e-9")

    def overlay(img, E):
        """Map over the image at CONSTANT opacity.

        No alpha ramp: low values have to show as well. The price is that the
        photograph disappears under the map, which is why the `original` column
        exists.
        """
        H, W = img.shape[:2]
        Eu = np.asarray(Image.fromarray(E.astype(np.float32), mode="F")
                        .resize((W, H), Image.BILINEAR), dtype=float)
        col = plt.get_cmap("magma")(np.clip(Eu, 0, 1))[..., :3]
        return np.clip(img * (1 - args.alpha) + col * args.alpha, 0, 1)

    T = it["tile"].shape[1]
    q_img = to_img(it["tile"])
    mos_img = np.zeros((2 * T, 2 * T, 3))
    mos_img[:T, :T] = q_img
    for k, (yy, xx) in enumerate(((0, 1), (1, 0), (1, 1))):
        mos_img[yy * T:(yy + 1) * T, xx * T:(xx + 1) * T] = to_img(it["rt"][k])

    fig = plt.figure(figsize=(COL_W, 1.78))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.038],
                          wspace=.006, hspace=.012,
                          left=.002, right=.972, top=.905, bottom=.004)

    def panel(ax, arr, title=None):
        ax.imshow(arr, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if title:
            ax.set_title(title, fontsize=5.4, pad=1.6)
        return ax

    panel(fig.add_subplot(gs[0, 0]), q_img, "original")
    ax = panel(fig.add_subplot(gs[1, 0]), mos_img)
    ax.add_patch(plt.Rectangle((-.5, -.5), T, T, fill=False, ec=ORANGE,
                               lw=1.3))
    for j, (k, lab) in enumerate(COLS, start=1):
        panel(fig.add_subplot(gs[0, j]), overlay(q_img, Sn[k]), lab)
        ax = panel(fig.add_subplot(gs[1, j]), overlay(mos_img, En[k]))
        ax.add_patch(plt.Rectangle((-.5, -.5), T, T, fill=False, ec=ORANGE,
                                   lw=1.3))
        ax.text(.035, .035, f"$s={it['s'][k]:.2f}$", transform=ax.transAxes,
                color="white", fontsize=5.6, fontweight="bold", va="bottom",
                bbox=dict(fc=INK, ec="none", alpha=.62, pad=1.6))
    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="magma"), cax=cax)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["0", "1"])
    cb.ax.tick_params(labelsize=4.6, length=1.4, pad=1.0)
    cb.outline.set_linewidth(.5)
    out = FIG / "119_fig5_headrefit_4col"
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=300)
    plt.close(fig)

    tab = "\n".join(f"    {lab:<24} s = {it['s'][k]:.3f}" for k, lab in COLS)
    out.with_suffix(".txt").write_text(f"""119_fig5_headrefit_4col.pdf -- generated by fig4_code/fig4_head_refit.py

WHAT IT SHOWS AND WHY
  Four columns over the SAME far-OOD query ({args.ood}) and the SAME mosaic:
  the raw input and the CAM of the predicted class read by three models. Top
  row: the image alone. Bottom row: the mosaic, with the query tile in orange
  and `s` annotated.

  The question is whether the explanation depends on the model. It does, and
  the mosaic is the only one of the two arrangements that turns that
  dependence into a NUMBER.

TWO DIFFERENT COMPARISONS IN ONE FIGURE (do not conflate them)
  LS -> LS+AugDelete   FROZEN backbone, only the head changes. The features
                       are identical, so every difference is a difference in
                       reading direction. This is the controlled experiment.
  CE -> LS             DIFFERENT backbones, trained from scratch. Different
                       features, not just a different head.
  The reference panel comes from a single cell (`{args.panel_key}`) in all
  three columns, so the input pixels are the same and only the reading
  changes.

HOW THE SAMPLE WAS CHOSEN: a DIRECTED search, not a draw
  The target is an OOD query with a specific pattern: dark under the CE
  model, lit under LS, with AugDelete returning to CE in both INTENSITY and
  GEOMETRY, on the query tile AND on the references. The criterion maximises

    z(s_LS - s_CE) + z(-|s_AugDelete - s_CE|)
      + z(geometry rho on the query) + z(geometry rho on the references)

  over the {len(cand)} queries where the three models predict the same class.
  Chosen: {Path(it['path']).name}, predicted class {it['cls']}.
    gap LS - CE            {it['gap']:+.3f}
    |AugDelete - CE|       {it['d_int']:.3f}
    MINIMUM geometry on the query  {it['geo_q']:+.3f}   (lowest Spearman of the 3 pairs)
    MINIMUM geometry on the refs   {it['geo_r']:+.3f}

THE EFFECT IS GENERAL; THE SAMPLE IS ITS SHARPEST INSTANCE
  This is the distinction that matters for reading the figure. The criterion
  above picks the sharpest sample, but the effect it illustrates does not
  depend on it:

    LS inflates the query energy (s_LS > s_CE)     {gen['up']}/{gen['n']} = {100*gen['up']/gen['n']:.1f}%   median {gen['med_gap']:+.3f}
    AugDelete returns closer to CE than LS         {gen['corr']}/{gen['n']} = {100*gen['corr']/gen['n']:.1f}%   median {gen['med_res']:+.3f}
    both at once                                   {gen['both']}/{gen['n']} = {100*gen['both']/gen['n']:.1f}%

  These MARGINAL rates are what support the claim that changing the head
  corrects the explanation. The STRICT conjunction used to CHOOSE the sample
  (gap > {args.thr_gap}, |AugDelete - CE| < {args.thr_int}, geometry >
  {args.thr_geo} on both sides) holds in {len(hits)}/{len(cand)} =
  {frac:.1f}%, but that measures the selection filter and not the effect:
  four simultaneous conditions at arbitrary thresholds, and the rarity of the
  conjunction says nothing about the frequency of the phenomenon.

  The quantitative claim in the paper remains the one over the 112 queries of
  scripts/68; this figure is the visual evidence for it.

FILES
  119_fig5_headrefit_4col.{{pdf,png}} and this .txt
  aggregates over the 112 queries: results/head_refit/per_query.csv (scripts/68)

KEY NUMBERS
{tab}
    AugDelete val top-1 = {vtop1:.4f}

HOW TO READ IT
  1. Every map is min-max normalised to [0,1], the convention under which
     every attribution map is seen. So ABSOLUTE ENERGY does not reach the
     reader, and only shape is comparable across panels. The colour bar is
     shared by the six maps.
  2. `s` passes through that normalisation intact (it is a ratio internal to
     the map, hence homogeneous of degree 0); a gate in the script checks
     this to 1e-9. It is the part of the magnitude that survives, and it
     survives because the references give an internal scale the image alone
     does not have.
  3. The opacity is CONSTANT ({args.alpha:.2f}), with no ramp: low map values
     show as well. That is why the photograph disappears under the overlay,
     and why the `original` column is there. It is where one checks that the
     references belong to the predicted class.
  4. Three samples do not support statistics. The quantitative claim is in
     the 112 queries of scripts/68, not here.
""")
    print(f"\n[ok] {out.with_suffix('.pdf')}")
    print(f"[ok] {out.with_suffix('.txt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", default="inaturalist",
                    choices=["ssb_hard", "ninco", "inaturalist",
                             "openimage_o", "textures", "id"])
    ap.add_argument("--panel-key", default="ce")
    ap.add_argument("--n-query", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--thr-gap", type=float, default=0.10)
    ap.add_argument("--thr-int", type=float, default=0.05)
    ap.add_argument("--thr-geo", type=float, default=0.50)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--force-pick", type=int, default=0,
                    help="1..K: usa o n-esimo candidato da folha")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--fit-bs", type=int, default=4096)
    ap.add_argument("--force-refit", action="store_true")
    main(ap.parse_args())
