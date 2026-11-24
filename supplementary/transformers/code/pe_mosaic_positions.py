"""The mosaic under every adaptation, with the query in the four positions.

Usage:
  $PYTHON_BIN supplementary/transformers/code/pe_mosaic_positions.py
  $PYTHON_BIN supplementary/transformers/code/pe_mosaic_positions.py --n 32 --out-sub smoke
"""
import os

experiment_name = "pe_mosaic_positions"
dataset_name    = "imagenet1k"

import sys
import json
import time
import argparse
import importlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import vit_b_16

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
XAI_OOD_ROOT = Path(os.environ.get("XAI_OOD_ROOT", ROOT / "external_xai_ood"))
for _d in (XAI_OOD_ROOT, REPO / "table_code", REPO / "shared_code", Path(__file__).resolve().parent):
    sys.path.insert(0, str(_d))
m07 = importlib.import_module("rood_qualify_protocol")
m164 = importlib.import_module("transformer_quarter")
m168 = importlib.import_module("tab_field_mosaic")
from src.mosaic import build_mosaic

STAMP = time.strftime("%Y%m%d-%H%M")
NEAR = ["ssb_hard", "ninco"]
FAR = ["inaturalist", "openimage_o", "textures"]
GROUPS = ["id"] + NEAR + FAR
EPS = 1e-12
R168 = ROOT / "results" / "mosaic_sizes"
MAN168 = Path(os.environ.get("MAN168", R168 / "168_manifest_20260913-1027.json"))
AUC168 = Path(os.environ.get("AUC168", R168 / "168_auroc_1000samples_20260913-1027.csv"))
AUC167 = Path(os.environ.get("AUC167", ROOT / "results" / "field_and_posemb"
                             / "167_auroc_field_im1k_1000samples_20260913-1007.csv"))
PE177 = ROOT / "results" / "pe_continuation" / "177_pe_extended.pt"

CONDS = {
    "rn50_448":     ("rn50", 224, None),
    "rn50_224":     ("rn50", 112, None),
    "vit_shrink":   ("vit", 112, None),
    "vit_interp":   ("vit", 224, "interp"),
    "vit_periodic": ("vit", 224, "periodic"),
    "vit_smooth":   ("vit", 224, "smooth"),
}
REF168 = {"rn50_448": ("rn50", 224), "rn50_224": ("rn50", 112),
          "vit_interp": ("vit", 224), "vit_shrink": ("vit", 112)}
SCORES = ["s_q0", "s_q1", "s_q2", "s_q3", "s_mean4", "s_map4"]


class ViTPE(m168.ViT):

    def __init__(self, dev, grid=None, cls=None):
        self.dev = dev
        base = vit_b_16(weights=m168.W_OF["vit"])
        if grid is None:
            self.size, self.m = 224, base.to(dev).eval()
            return
        n = grid.shape[0]
        self.size = n * 16
        sd = base.state_dict()
        sd["encoder.pos_embedding"] = torch.cat(
            [cls[None], grid.reshape(n * n, -1)])[None].contiguous()
        m = vit_b_16(image_size=self.size)
        m.load_state_dict(sd)
        self.m = m.to(dev).eval()


def tf_key(tf):
    return (tuple(tf.resize_size), tuple(tf.crop_size), str(tf.interpolation),
            tuple(tf.mean), tuple(tf.std))


def carrega(files, tf, pre, workers):
    from PIL import Image
    import torchvision.transforms as T
    rsz, crop = int(tf.resize_size[0]), int(tf.crop_size[0])
    steps = ([] if pre else [T.Resize(rsz, interpolation=tf.interpolation),
                             T.CenterCrop(crop)])
    comp = T.Compose(steps + [T.ToTensor(), T.Normalize(tf.mean, tf.std)])

    def um(p):
        with Image.open(p) as im:
            return comp(im.convert("RGB"))

    with ThreadPoolExecutor(workers) as ex:
        return torch.stack(list(ex.map(um, files)))


def encolhe(x, px, chunk=256):
    if x.shape[-1] == px:
        return x
    return torch.cat([m168.resize_tiles(x[i:i + chunk], px)
                      for i in range(0, len(x), chunk)])


def monta(q, refs, qpos):
    B, C, h, w = q.shape
    mos = q.new_zeros(B, C, 2 * h, 2 * w)
    ri = 0
    for k in range(4):
        r, c = divmod(k, 2)
        src = q if k == qpos else refs[:, ri]
        mos[:, :, r * h:(r + 1) * h, c * w:(c + 1) * w] = src
        if k != qpos:
            ri += 1
    return mos


def quadrantes(L):
    mu = L.flatten(1).mean(1)[:, None, None]
    E = torch.relu(L - mu)
    h = L.shape[-1] // 2
    return torch.stack([E[:, :h, :h].flatten(1).mean(1),
                        E[:, :h, h:].flatten(1).mean(1),
                        E[:, h:, :h].flatten(1).mean(1),
                        E[:, h:, h:].flatten(1).mean(1)], 1)


def s_de_Q(Q):
    idx = np.arange(4)
    eq = Q[:, idx, idx]
    er = (Q.sum(2) - eq) / 3
    return eq / (eq + er + EPS)


def tile_of(k, p):
    return -1 if k == p else k - (1 if k > p else 0)


def s_map4(Q):
    acc = np.zeros((Q.shape[0], 4))
    for p in range(4):
        for k in range(4):
            acc[:, tile_of(k, p) + 1] += Q[:, p, k]
    acc /= 4
    q, r = acc[:, 0], acc[:, 1:].mean(1)
    return np.divide(q, q + r, out=np.zeros_like(q), where=(q + r) > 0)


def canon(L, p):
    h = L.shape[-1] // 2
    sl = [L[:, :h, :h], L[:, :h, h:], L[:, h:, :h], L[:, h:, h:]]
    return torch.stack([sl[p]] + [sl[k] for k in range(4) if k != p], 1)


def pearson(a, b):
    a = a.flatten(1).double()
    b = b.flatten(1).double()
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + EPS)).numpy()


def roda_cond(reader, Xt, Rt, pred, args):
    out = {}
    for g in GROUPS:
        Ls = []
        x = Xt[g]
        for i in range(0, len(x), args.bs_q):
            q = x[i:i + args.bs_q].to(reader.dev)
            cls = pred[g][i:i + args.bs_q]
            R = torch.stack([Rt[int(c)] for c in cls]).to(reader.dev)
            mos = torch.cat([monta(q, R, p) for p in range(4)])
            cl4 = torch.as_tensor(np.tile(cls, 4))
            L = reader.cam(mos, cl4, args.bs_cam).float()
            B = len(q)
            Ls.append(L.reshape(4, B, *L.shape[1:]).transpose(0, 1))
            del mos
        out[g] = torch.cat(Ls)
    return out


@torch.no_grad()
def mls(model, x, dev, bs, size):
    out = []
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(dev, non_blocking=True)
        if size != b.shape[-1]:
            b = F.interpolate(b, size=size, mode="bicubic",
                              align_corners=False, antialias=True)
        out.append(model(b).max(1).values.float().cpu())
    return torch.cat(out).numpy()


def auroc_ood(sid, sood):
    y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
    return 1 - roc_auc_score(y, np.r_[sid, sood])


def linhas_auroc(nome, score, por_grupo):
    rows, per = [], {}
    for g in NEAR + FAR:
        per[g] = auroc_ood(por_grupo["id"], por_grupo[g])
        rows.append(dict(cond=nome, score=score, ood=g, auroc=per[g]))
    rows += [dict(cond=nome, score=score, ood="_mean_near",
                  auroc=float(np.mean([per[g] for g in NEAR]))),
             dict(cond=nome, score=score, ood="_mean_far",
                  auroc=float(np.mean([per[g] for g in FAR])))]
    return rows


def main(args):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    out = ROOT / "results" / experiment_name / (args.out_sub or "")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.n}samples_{STAMP}"
    linhas = []

    def log(s):
        print(s, flush=True)
        linhas.append(s)

    log(f"=== 178 | dev={dev} | AMP=off | torch.compile=off | matmul_tf32="
        f"{torch.backends.cuda.matmul.allow_tf32} | cudnn_tf32="
        f"{torch.backends.cudnn.allow_tf32} | n={args.n}/set | "
        f"bs_q={args.bs_q} bs_cam={args.bs_cam} ===")
    man = json.loads(MAN168.read_text())
    pools = {g: man["pools"][g][:args.n] for g in GROUPS}
    completo = args.n == man["n_per_set"]
    if completo:
        ref = m07.query_pools(args.n)
        if not all(ref[g] == man["pools"][g] for g in GROUPS):
            raise AssertionError("pools of the 168 manifest != m07.query_pools")
        log("[gate 0] pools of the 168 manifest = m07.query_pools(1000): ok")
    else:
        log(f"[gate 0] smoke: first {args.n} queries of each 168 pool "
            "(the gates against 167/168 only apply with n=1000)")
    pe = torch.load(PE177)
    log(f"177 embedding: smooth continuation {pe['smooth_spec']}")

    qa, ra = torch.randn(3, 8, 8), torch.randn(3, 3, 8, 8)
    for p in range(4):
        if not torch.equal(monta(qa[None], ra[None], p)[0],
                           build_mosaic(qa, ra, qpos=p)):
            raise AssertionError(f"monta != build_mosaic at qpos={p}")
    log("[gate 1] monta() = build_mosaic in the 4 positions: ok")

    conds = [c for c in CONDS if c in args.conds]
    recs, auc_rows, mls_sc, maps = [], [], {}, {}
    cache_img = {}
    t_all = time.time()
    for model in ("rn50", "vit"):
        mc = [c for c in conds if CONDS[c][0] == model]
        if not mc:
            continue
        t0 = time.time()
        tf = m168.W_OF[model].transforms()
        k = tf_key(tf)
        if k not in cache_img:
            cache_img = {k: {g: carrega(pools[g], tf, g == "id", args.workers)
                             for g in GROUPS}}
        X = cache_img[k]
        for g in GROUPS:
            if not torch.equal(m164.load_official(pools[g][:2], tf,
                                                  pre=(g == "id")), X[g][:2]):
                raise AssertionError(f"parallel loader != load_official ({g})")
        log(f"\n--- {model} | images loaded [{time.time() - t0:.0f}s] | "
            "[gate 2] parallel loader = m164.load_official: ok")

        w224 = m168.RN50(dev) if model == "rn50" else ViTPE(dev)
        pred = {g: w224.pred_pool(X[g], args.bs)[0] for g in GROUPS}
        mls_sc[f"mls_{model}_224"] = {g: mls(w224.m, X[g], dev, args.bs, 224)
                                      for g in GROUPS}
        if model == "rn50":
            mls_sc["mls_rn50_448"] = {g: mls(w224.m, X[g], dev, args.bs, 448)
                                      for g in GROUPS}
        else:
            with torch.no_grad():
                tok = w224.m.conv_proj(X["id"][:64].to(dev)).flatten(2)
                nt = tok.norm(dim=1).mean().item()
            npe = {kk: pe[kk].reshape(-1, pe[kk].shape[-1]).norm(dim=1).mean().item()
                   for kk in ("grid224", "interp", "periodic", "smooth")}
            log(f"  mean norm of the patch token (before the embedding): {nt:.2f} | "
                "mean norm of the positional embedding: "
                + ", ".join(f"{kk} {v:.2f}" for kk, v in npe.items()))
        del w224
        torch.cuda.empty_cache()

        classes = sorted({int(c) for g in GROUPS for c in pred[g]})
        files = [f for c in classes for f in man["prototypes"][model][str(c)]]
        Rall = carrega(files, tf, False, args.workers)
        R224 = {c: Rall[3 * j:3 * j + 3] for j, c in enumerate(classes)}
        log(f"  {len(classes)} predicted classes, {len(files)} references "
            f"[{time.time() - t0:.0f}s]")

        for cond in mc:
            t1 = time.time()
            _, tile, pek = CONDS[cond]
            if model == "rn50":
                reader = m168.RN50(dev)
            elif pek is None:
                reader = ViTPE(dev)
            else:
                reader = ViTPE(dev, grid=pe[pek], cls=pe["cls"])
                mls_sc[f"mls_vit_448_{pek}"] = {
                    g: mls(reader.m, X[g], dev, args.bs, 448) for g in GROUPS}
            Xt = {g: encolhe(X[g], tile) for g in GROUPS}
            Rt = {c: encolhe(R224[c], tile) for c in classes}
            if tile != 224:
                um = torch.cat([m168.resize_tiles(X["id"][i:i + 1], tile)
                                for i in range(4)])
                dif = (um - Xt["id"][:4]).abs().max().item()
                if dif > 1e-5:
                    raise AssertionError(f"batched resize differs from 168: {dif}")
            Lg = roda_cond(reader, Xt, Rt, pred, args)
            del reader, Xt, Rt
            torch.cuda.empty_cache()

            S = next(iter(Lg.values())).shape[-1]
            Ls, Qs, grp, idx, prd = [], [], [], [], []
            for g in GROUPS:
                L = Lg[g]
                N = L.shape[0]
                Q = quadrantes(L.reshape(-1, S, S)).reshape(N, 4, 4).numpy()
                s = s_de_Q(Q)
                sm = s_map4(Q)
                if S % 2 == 0:
                    c0 = canon(L[:, 0], 0)
                    cq = np.stack([pearson(canon(L[:, p], p)[:, 0], c0[:, 0])
                                   for p in (1, 2, 3)], 1)
                    ca = np.stack([pearson(canon(L[:, p], p), c0)
                                   for p in (1, 2, 3)], 1)
                else:
                    cq = ca = np.full((N, 3), np.nan)
                for i in range(N):
                    recs.append(dict(cond=cond, group=g, idx=i, file=pools[g][i],
                                     pred=int(pred[g][i]),
                                     **{f"s_q{p}": float(s[i, p]) for p in range(4)},
                                     s_mean4=float(s[i].mean()), s_map4=float(sm[i]),
                                     **{f"corr_q{p}": float(cq[i, p - 1])
                                        for p in (1, 2, 3)},
                                     **{f"corr_all{p}": float(ca[i, p - 1])
                                        for p in (1, 2, 3)}))
                Ls.append(L.half().numpy())
                Qs.append(Q.astype(np.float32))
                grp += [g] * N
                idx += list(range(N))
                prd += [int(c) for c in pred[g]]
            np.savez_compressed(out / f"178_maps_{cond}_{tag}.npz",
                                L=np.concatenate(Ls), Q=np.concatenate(Qs),
                                group=np.array(grp), idx=np.array(idx),
                                pred=np.array(prd))
            d = pd.DataFrame([r for r in recs if r["cond"] == cond])
            for sc in SCORES:
                auc_rows += linhas_auroc(cond, sc, {
                    g: d.loc[d.group == g, sc].to_numpy() for g in GROUPS})
            a = pd.DataFrame(auc_rows)
            a = a[(a.cond == cond) & a.ood.str.startswith("_")]
            p0 = a.pivot(index="score", columns="ood", values="auroc")
            log(f"  {cond:<13} tile {tile} -> {2 * tile}  [{time.time() - t1:.0f}s]  "
                + "  ".join(f"{sc} {p0.loc[sc, '_mean_near']:.3f}/"
                            f"{p0.loc[sc, '_mean_far']:.3f}"
                            for sc in ("s_q0", "s_mean4")))

    for nome, sc in mls_sc.items():
        auc_rows += linhas_auroc(nome, "mls", sc)
    A = pd.DataFrame(auc_rows)
    A.to_csv(out / f"178_auroc_{tag}.csv", index=False)
    D = pd.DataFrame(recs)
    D.to_csv(out / f"178_per_sample_{tag}.csv", index=False)

    log("\nGATES")
    if completo:
        r168 = pd.read_csv(AUC168)
        for cond, (mm, tt) in REF168.items():
            if cond not in conds:
                continue
            a = A[(A.cond == cond) & (A.score == "s_q0")].set_index("ood").auroc
            b = r168[(r168.model == mm) & (r168.tile == tt)].set_index("ood").auroc
            dm = float((a - b).abs().max())
            log(f"  [gate 3] {cond:<11} qpos=0 against 168 ({mm}, tile {tt}): "
                f"|dAUROC| max {dm:.4f}" + ("  ok" if dm < args.tol else "  FAILED"))
            if dm >= args.tol:
                raise AssertionError(f"{cond} does not reproduce 168: {dm:.4f}")
        r167 = pd.read_csv(AUC167)
        for nome, (mm, cc) in {"mls_rn50_224": ("rn50", "orig"),
                               "mls_vit_224": ("vit", "orig"),
                               "mls_vit_448_interp": ("vit", "up448")}.items():
            if nome not in mls_sc:
                continue
            a = A[A.cond == nome].set_index("ood").auroc
            b = r167[(r167.model == mm) & (r167.cond == cc)].set_index("ood").auroc
            dm = float((a - b).abs().max())
            log(f"  [gate 4] {nome:<19} against 167 ({mm} {cc}): |dAUROC| max "
                f"{dm:.4f}" + ("  ok" if dm < args.tol else "  FAILED"))
            if dm >= args.tol:
                raise AssertionError(f"{nome} does not reproduce 167: {dm:.4f}")
    if "vit_periodic" in conds:
        d = D[D.cond == "vit_periodic"]
        ds = float(np.abs(d[["s_q1", "s_q2", "s_q3"]].to_numpy()
                          - d[["s_q0"]].to_numpy()).max())
        cmin = float(d[["corr_all1", "corr_all2", "corr_all3"]].min().min())
        log(f"  [gate 5] vit_periodic: max |s(qpos) - s(0)| = {ds:.2e}; lowest "
            f"correlation of the realigned mosaic = {cmin:.6f}"
            + ("  ok" if ds < 1e-3 else "  FAILED"))
        if ds >= 1e-3:
            raise AssertionError("periodic is not invariant to the position")

    m = A[A.ood.str.startswith("_")].pivot_table(index=["cond", "score"],
                                                 columns="ood", values="auroc")
    m.columns = ["far", "near"]
    log("\nAUROC (mean near / far)\n" + m[["near", "far"]].to_string(
        float_format=lambda v: f"{v:.3f}"))

    reg = D.group.map(lambda g: "id" if g == "id" else
                      ("near" if g in NEAR else "far"))
    vies = D.assign(reg=reg).groupby(["cond", "reg"])[
        ["s_q0", "s_q1", "s_q2", "s_q3"]].mean()
    log("\nMEAN s BY QUERY POSITION (id / near / far)\n"
        + vies.to_string(float_format=lambda v: f"{v:.3f}"))

    cor = D.assign(reg=reg).groupby(["cond", "reg"])[
        ["corr_q1", "corr_q2", "corr_q3", "corr_all1", "corr_all2",
         "corr_all3"]].median()
    log("\nCORRELATION OF THE EXPLANATION WITH THE ONE AT POSITION 0 (median; q = query "
        "sub-map, all = realigned mosaic)\n"
        + cor.to_string(float_format=lambda v: f"{v:.3f}"))

    log(f"\n[ok] {out / f'178_auroc_{tag}.csv'}  [{time.time() - t_all:.0f}s]")
    (out / f"178_manifest_{tag}.json").write_text(json.dumps(
        {"script": "pe_mosaic_positions.py", "n_per_set": args.n,
         "manifest_168": str(MAN168), "pe_177": str(PE177),
         "smooth_spec": pe["smooth_spec"], "conds": conds,
         "qpos_order": "row-major: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right",
         "tf32_matmul": args.tf32, "bs_q": args.bs_q, "bs_cam": args.bs_cam},
        indent=1))
    (out / f"178_summary_{tag}.txt").write_text("\n".join(linhas) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--bs-q", type=int, default=32,
                    help="queries per batch (each one yields 4 mosaics)")
    ap.add_argument("--bs-cam", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tf32", action="store_true",
                    help="TF32 in matmuls (168 ran without it)")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="AUROC tolerance against 167/168")
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--out-sub", default="")
    main(ap.parse_args())
