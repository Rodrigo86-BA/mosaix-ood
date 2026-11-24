"""ViM with a principal-subspace residual on the torchvision ResNet-18.

experiment_name = 'vim_principal_rn18'
dataset_name    = 'imagenet1k'

WHY
---
In the full detector table the ViM row of the RN-18 was identical to Energy.
It is not a table error: the `fit_vim` of src/detector_maps.py uses the null
space of the head, R = Vh[K:], with K = 1000 classes. The RN-18 feature has
512 dimensions, so W (1000x512) has no null space, R comes out empty, the
residual is zero and ViM collapses to Energy. Here ViM is recomputed in the
form of the original method (Wang et al., 2022), which measures the residual
outside the PRINCIPAL subspace of the training features and is defined for any
dimension. The output CSV feeds the RN-18 ViM row of the full detector table
(table_code/make_tables.py).

WHAT IS THE SAME AS scripts/64 (so the row is comparable to the others)
-----------------------------------------------------------------------
  * pools: those of the full run (n_per_set = 10^9, ID on the raw tree with
    Resize(256)+CenterCrop(224), OOD through query_pools of scripts/62);
  * model and transform: `build_tv` and `load_tile` of scripts/64 itself, and
    the same numerical flags (TF32 off for matmul, on for convolutions);
  * training features: the same cache and the SAME 200,000-sample subset of
    `build_params` (SEED 0) that Maha/kNN/ViM of the table used;
  * metric: `src.metrics.auroc`/`fpr95`, ID positive, high score = ID.

THE ORIGINAL ViM
----------------
  u  = -pinv(W) b                          origin where the logits are linear
  S  = (Z-u)^T (Z-u) / N                   covariance without re-centring
  P  = the D eigenvectors of largest eigenvalue;  NS = the complement (C-D columns)
  alpha = mean max-logit / mean ||(z-u) NS|| on the training set
  ViM(z) = logsumexp(W z + b) - alpha ||(z-u) NS||
D follows the rule of the official code, fixed BEFORE looking at OOD: C >= 2048
-> 1000, C >= 768 -> 512, else C/2. For the RN-18, D = 256. D = 128 and 384 are
computed only as sensitivity and choose nothing.

GATES (stop the script if they fail)
------------------------------------
1. count and order per group equal to those of scores__resnet18.npz of the
   full run;
2. predictions >= 99.99% equal to those of the npz, and per-sample Energy
   within TF32 noise (median < 1e-3, max < 5e-2). Convolutions run in TF32 and
   the algorithm depends on the batch size, so ~1e-3 is expected; a wrong pool
   or transform flips predictions (the double crop flips 16%);
2b. on the full split, the AUROC of the recomputed Energy matches auroc.csv
   within 1e-4 per set, checked BEFORE any output is written;
3. the head applied to the validation cache reproduces the top-1 of the
   checkpoint (0.69758, tolerance 0.02): the training cache is of the same model;
4. the null-space ViM of the repository has R with zero columns on the RN-18
   (diagnostic).

Environment:
  MOSAIX_ROOT   working repository (scripts/64, results/depth_sweep/)
  TV_FEAT_ROOT  optional; cached torchvision features, overrides the path
                hard-coded in scripts/64 (expects resnet18_v1/ inside)

Outputs (results/vim_principal_rn18/ of $MOSAIX_ROOT, or .../smoke/ with --limit):
  176_auroc_rn18_<ts>.csv    arch, group, detector, D, auroc, fpr95
  176_scores_rn18.npz        per-sample scores (vim_D*, energy, group)
  176_vim_principal_rn18.txt companion

Usage:
  smoke: $PYTHON_BIN camera_ready_code/vim_principal_rn18.py --limit 64
  full:  $PYTHON_BIN camera_ready_code/vim_principal_rn18.py
"""
import os

experiment_name = "vim_principal_rn18"
dataset_name    = "imagenet1k"

import argparse
import hashlib
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
sys.path.insert(0, str(ROOT / "scripts"))
m64 = importlib.import_module("64_depth_sweep")      # brings sys.path, fp32 flags
if os.environ.get("TV_FEAT_ROOT"):
    m64.TV_FEAT = Path(os.environ["TV_FEAT_ROOT"])
m62 = m64.m62
from src.config import paths                          # noqa: E402
from src.metrics import auroc, fpr95                  # noqa: E402
from src.detector_maps import fit_vim                 # noqa: E402

ARCH = "resnet18"
OUT = ROOT / "results" / experiment_name
NPZ_REF = ROOT / "results" / "depth_sweep" / f"scores__{ARCH}.npz"
CSV_REF = ROOT / "results" / "depth_sweep" / "auroc.csv"
FDIR = m64.TV_FEAT / f"{ARCH}_v1"
NEAR, FAR = m64.NEAR, m64.FAR
TOP1_CKPT = m64.TOP1_TV[ARCH]
POOLS_SHA_RUN = "5a085218acf3"        # printed by the full run (summary.txt)
TAB3 = ["MSP", "GEN", "MLS", "Energy", "Cosine", "Maha", "kNN", "GradNorm",
        "GAIA-A", "GAIA-Z", "mosXai-OOD-4x+tau"]


def vim_dim(c):
    """Rule of the official ViM code for the dimension of the principal subspace."""
    return 1000 if c >= 2048 else (512 if c >= 768 else c // 2)


def build_pools(n_per_set=10 ** 9):
    """Replicates main() of scripts/64: query_pools of scripts/62 and ID on the whole raw tree."""
    class A:
        pass
    A.n_per_set = n_per_set
    pools = m62.query_pools("imagenet1k", A)
    rv = Path(paths()["imagenet200"]["raw_val"])
    f = [x for d in sorted(y for y in rv.iterdir() if y.is_dir())
         for x in sorted(d.iterdir())]
    rng = np.random.default_rng(m62.SEED)
    k = min(n_per_set, len(f))
    pools["id"] = [f[i] for i in np.sort(rng.choice(len(f), k, replace=False))]
    sha = hashlib.sha1("|".join(str(x) for v in pools.values()
                                for x in v).encode()).hexdigest()[:12]
    return pools, sha


class Refs(torch.utils.data.Dataset):
    def __init__(self, refs, load_tile):
        self.refs, self.load_tile = refs, load_tile

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        return self.load_tile(self.refs[i], False)


def train_subset(fit_n):
    """The same subset as m62.build_params (SEED, fit_n, sorted)."""
    emb = np.load(FDIR / m64.FEAT_NAMES[0], mmap_mode="r")
    rng = np.random.default_rng(m62.SEED)
    idx = np.sort(rng.choice(len(emb), min(fit_n, len(emb)), replace=False))
    return torch.from_numpy(np.array(emb[idx], dtype=np.float32))


def fit_vim_principal(Z, W, b, D):
    Z64, W64, b64 = Z.double(), W.double(), b.double()
    u = -torch.linalg.pinv(W64) @ b64
    X = Z64 - u
    cov = X.T @ X / X.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    NS = evecs[:, order[D:]]
    alpha = ((Z64 @ W64.T + b64).max(1).values.mean()
             / (X @ NS).norm(dim=1).mean()).item()
    return u, NS, alpha


def vim_score(z, energy, u, NS, alpha):
    return energy - alpha * ((z.double() - u) @ NS).norm(dim=1)


def means(tab, det):
    t = tab[tab.detector == det].set_index("group").auroc
    return float(t[NEAR].mean()), float(t[FAR].mean())


def main(args):
    t0 = time.time()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = OUT / "smoke" if args.limit else OUT
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== 176 | {ARCH} | dev={dev} | AMP=off | torch.compile=off | "
          f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32} | "
          f"cudnn_tf32={torch.backends.cudnn.allow_tf32} | bs={args.bs} | "
          f"{'SMOKE limit=' + str(args.limit) if args.limit else 'full split'}"
          " ===", flush=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    npz = np.load(NPZ_REF)
    ref = {k: npz[k] for k in ("group", "energy", "pred")}
    pools, sha = build_pools()
    log(f"pools_sha={sha} (full run: {POOLS_SHA_RUN}"
        f"{', equal' if sha == POOLS_SHA_RUN else ', DIFFERENT'})")
    if args.limit:
        sel = np.concatenate([np.flatnonzero(ref["group"] == g)[:args.limit]
                              for g in pools])
        pools = {g: v[:args.limit] for g, v in pools.items()}
        ref = {k: v[sel] for k, v in ref.items()}
    counts = {g: len(v) for g, v in pools.items()}
    rg, rc = np.unique(ref["group"], return_counts=True)
    ref_counts = dict(zip(rg.tolist(), rc.tolist()))
    log(f"counts: {counts}")
    if counts != ref_counts:
        raise SystemExit(f"GATE 1 FAILED: counts {counts} != npz {ref_counts}")
    order = np.concatenate([np.array([g] * n) for g, n in counts.items()])
    if not np.array_equal(order, ref["group"]):
        raise SystemExit("GATE 1 FAILED: group order differs from the npz")
    log("[gate 1] group counts and order equal to the npz: ok")

    model, W, b, load_tile = m64.build_tv(ARCH, dev, "V1")
    body = torch.nn.Sequential(*list(model.children())[:-2]).eval()
    C = W.shape[1]
    D = vim_dim(C)
    log(f"feature C={C}, classes K={W.shape[0]}, D by the official rule = {D}")

    Zs, E, P = [], [], []
    n_all = sum(counts.values())
    done = 0
    for g, refs in pools.items():
        dl = torch.utils.data.DataLoader(Refs(refs, load_tile), batch_size=args.bs,
                                         num_workers=args.workers, pin_memory=True)
        with torch.no_grad():
            for x in dl:
                fmap = body(x.to(dev, non_blocking=True))
                z = torch.flatten(model.avgpool(fmap), 1)
                lg = model.fc(z)
                Zs.append(z.cpu())
                E.append(torch.logsumexp(lg, 1).cpu())
                P.append(lg.argmax(1).cpu())
                done += len(x)
        log(f"  {g:<12} {len(refs):>6} images  ({done}/{n_all}, "
            f"{time.time()-t0:.0f}s)")
    Z = torch.cat(Zs)
    ene = torch.cat(E).numpy()
    pred = torch.cat(P).numpy()

    ad = np.abs(ene.astype(np.float32) - ref["energy"])
    same_pred = float((pred == ref["pred"]).mean())
    log(f"[gate 2] |Energy - npz|: mediana {np.median(ad):.1e}, p99 "
        f"{np.quantile(ad, 0.99):.1e}, max {ad.max():.1e}; equal predictions = "
        f"{same_pred:.6f}")
    if same_pred < 0.9999 or np.median(ad) > 1e-3 or ad.max() > 5e-2:
        raise SystemExit("GATE 2 FAILED: the forward pass does not reproduce the full run")

    zv = torch.from_numpy(np.load(FDIR / "imagenet1k_val_emb.npy").astype(np.float32))
    yv = np.load(FDIR / "imagenet1k_val_lbl.npy")
    t1c = float(((zv @ W.T + b).argmax(1).numpy() == yv).mean())
    log(f"[gate 3] top-1 of the head on the validation cache = {t1c:.5f} "
        f"(checkpoint {TOP1_CKPT:.5f})")
    if abs(t1c - TOP1_CKPT) > 0.02:
        raise SystemExit("GATE 3 FAILED: the cache is not of the same model")

    Zt = train_subset(args.fit_n)
    ns = fit_vim(Zt, W, b)
    log(f"[gate 4] null-space ViM of the repository: R {tuple(ns.R.shape)} "
        f"(zero columns = null residual = ViM identical to Energy)")

    energy64 = torch.logsumexp(Z.double() @ W.double().T + b.double(), dim=1)
    rows, scores = [], {"energy": ene.astype(np.float32), "group": ref["group"]}
    idm = ref["group"] == "id"
    for d in sorted({D, 128, 384}):
        u, NS, alpha = fit_vim_principal(Zt, W, b, d)
        s = vim_score(Z, energy64, u, NS, alpha).numpy()
        scores[f"vim_D{d}"] = s.astype(np.float32)
        tag = "ViM" if d == D else f"ViM-D{d}"
        for g in NEAR + FAR:
            m = ref["group"] == g
            rows.append(dict(arch=ARCH, group=g, detector=tag, D=d,
                             auroc=auroc(s[idm], s[m]), fpr95=fpr95(s[idm], s[m])))
        log(f"  ViM principal D={d:<4} alpha={alpha:.4f}")
    for g in NEAR + FAR:
        m = ref["group"] == g
        rows.append(dict(arch=ARCH, group=g, detector="Energy (recalc)", D=0,
                         auroc=auroc(ene[idm], ene[m]),
                         fpr95=fpr95(ene[idm], ene[m])))
    df = pd.DataFrame(rows)

    old = pd.read_csv(CSV_REF)
    old = old[old.arch == ARCH]
    e_old = old[old.detector == "Energy"].set_index("group").auroc
    e_new = df[df.detector == "Energy (recalc)"].set_index("group").auroc
    dA = float((e_old - e_new).abs().max())
    log(f"[gate 2b] AUROC of the recomputed Energy vs auroc.csv: |delta| max = {dA:.2e}"
        + ("  (smoke: not comparable)" if args.limit else ""))
    if not args.limit and dA > 1e-4:
        raise SystemExit("GATE 2b FAILED: the Energy AUROC does not reproduce auroc.csv")

    ts = time.strftime("%Y%m%d-%H%M")
    fcsv = out / f"176_auroc_rn18_{ts}.csv"
    df.to_csv(fcsv, index=False)
    np.savez_compressed(out / "176_scores_rn18.npz", **scores)

    log("")
    log(f"{'RN-18':<20}" + "".join(f"{g[:8]:>9}" for g in NEAR + FAR)
        + f"{'near':>8}{'far':>8}")
    for det in ["ViM", "ViM-D128", "ViM-D384"]:
        t = df[df.detector == det].set_index("group").auroc
        nm, fm = means(df, det)
        log(f"{det + ' (principal)':<20}"
            + "".join(f"{t[g]:>9.3f}" for g in NEAR + FAR) + f"{nm:>8.3f}{fm:>8.3f}")
    t = old[old.detector == "ViM"].set_index("group").auroc
    nm, fm = means(old, "ViM")
    log(f"{'ViM (null, old)':<20}" + "".join(f"{t[g]:>9.3f}" for g in NEAR + FAR)
        + f"{nm:>8.3f}{fm:>8.3f}")

    log("")
    log("RN-18 ranking with the new ViM (means; other rows from auroc.csv):")
    mean_rows = [(k, *means(old, k)) for k in TAB3 if k in set(old.detector)]
    mean_rows.append(("ViM (principal)", *means(df, "ViM")))
    for j, reg in ((1, "near"), (2, "far")):
        top = sorted(mean_rows, key=lambda r: -r[j])[:5]
        log(f"  {reg}: " + "  ".join(
            f"{r[0].replace('mosXai-OOD-4x+tau', 'MosAIx')} {r[j]:.3f}" for r in top))

    (out / "176_vim_principal_rn18.txt").write_text(
        "camera_ready_code/vim_principal_rn18.py\n\n"
        "WHAT IT MEASURES AND WHY\n"
        "  The RN-18 ViM in the full detector table was identical to Energy: the\n"
        "  null-space fit of the repository has no residual when the feature (512)\n"
        "  is smaller than the number of classes (1000). Here ViM is recomputed in\n"
        "  its original form, with the residual outside the principal subspace of\n"
        f"  the training features, D = {D} by the rule of the official code, fixed\n"
        "  before looking at OOD.\n\n"
        "FILES\n"
        f"  {fcsv.relative_to(ROOT)}  AUROC/FPR95 per set\n"
        f"  {(out / '176_scores_rn18.npz').relative_to(ROOT)}  per-sample scores\n\n"
        "NUMBERS AND GATES\n" + "\n".join("  " + s for s in lines) + "\n\n"
        "HOW TO READ IT\n"
        "  1. Gates 1, 2 and 2b ensure that the queries, the transform and the\n"
        "     forward pass are those of the full run: the only thing that changes\n"
        "     in the row is ViM.\n"
        "  2. The 'ViM (principal)' row replaces the RN-18 ViM row of the table.\n"
        "  3. D=128 and D=384 are sensitivity. If they stay close to D=256, the\n"
        "     number does not depend on the choice of D; neither is used to choose.\n"
        "  4. The ranking shows whether MosAIx-OOD stays ahead on the RN-18 with a\n"
        "     valid ViM. The other rows come from auroc.csv of the full run.\n")
    log(f"\n[ok] {fcsv}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-n", type=int, default=200000)
    ap.add_argument("--bs", type=int, default=32,
                    help="32 = batch of the full run. With 256 the TF32 algorithm of "
                         "the convolutions changes and flips 1 in 384 borderline predictions")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: only the first N of each group")
    main(ap.parse_args())
