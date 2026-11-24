"""169 - Why the kNN policy loses to the prototype policy.

experiment_name = 'knn_vs_prototype'
dataset_name    = 'imagenet1k'

The ablation table shows the query-adaptive kNN exemplar policy losing to the
prototype policy in far-OOD (0.861 against 0.904, RN-50 CE). This script
measures WHY, by separating the two quantities the ratio s hides.

s = Eq / (Eq + Er). The table reports only s. Here Eq and Er are recorded
separately, for both policies, on the same query, plus three selection
diagnostics computed from features alone:

  cos(q, centroid)    how typical of the class the query is
  cos(ref, centroid)  how typical the chosen references are
  cos(ref, q)         how similar to the query the references are

The hypothesis: kNN picks the nearest neighbours OF THE QUERY, so for an OOD
query it picks atypical members of the class, which carry less evidence. Er
falls along with Eq and the ratio cancels the fall. The prototype policy does
not do this because it never looks at the query.

The policies are identical to those in the frozen ablation
(src/exemplars.py):
  prototype  top-3 cosine to the class centroid   (independent of the query)
  knn        top-3 cosine to the query            (depends on the query)

Output: results/knn_vs_prototype/
  169_per_sample_<ts>.csv    one row per (query, policy)
  169_resumo_<ts>.csv        AUROC and means per set and policy
  169_knn_vs_prototype.text   companion

Usage:
  $PYTHON_BIN note_knn_code/knn_vs_prototype.py
  $PYTHON_BIN note_knn_code/knn_vs_prototype.py --n 1000
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

experiment_name = "knn_vs_prototype"
dataset_name = "imagenet1k"

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_code"
XAI = ROOT / "external_xai_ood"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(XAI))

m07 = importlib.import_module("rood_qualify_protocol")
from src.backbones import load_train_feats
from src.exemplars import prototype_indices, query_adaptive_indices
from src.metrics import auroc

EPS = 1e-8
SEED = m07.SEED
P = m07.P
OUT = ROOT / "results" / experiment_name


def l2n(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + EPS)


def idx_to_path(idx, per, offs):
    c = int(np.searchsorted(offs, idx, side="right") - 1)
    return str(per[c][int(idx - offs[c])])


def build_mosaic(q, refs):
    """Query no canto superior esquerdo, 2x2, tiles nativos."""
    top = torch.cat([q, refs[0]], dim=2)
    base = torch.cat([refs[1], refs[2]], dim=2)
    return torch.cat([top, base], dim=1)


class Backbone:
    def __init__(self, dev):
        self.m, W, _ = m07.load_backbone("ce", dev)
        self.W = W
        self.body = m07.body_of(self.m)
        self.dev = dev

    @torch.no_grad()
    def gap_e_pred(self, x, bs):
        gaps, preds = [], []
        for s in range(0, len(x), bs):
            A = self.body(x[s:s + bs].to(self.dev))
            z = A.mean((2, 3)).cpu()
            gaps.append(z)
            preds.append((z @ self.W.T).argmax(1))
        return torch.cat(gaps).numpy(), torch.cat(preds).numpy()

    @torch.no_grad()
    def energies(self, mos, cls, bs):
        """Mean Eq and Er per tile, with mu removed (relu(L - mean))."""
        eq, er = [], []
        for s in range(0, len(mos), bs):
            A = self.body(mos[s:s + bs].to(self.dev))
            w = self.W[cls[s:s + bs]].to(self.dev)
            L = torch.einsum("bc,bchw->bhw", w, A)
            mu = L.flatten(1).mean(1)[:, None, None]
            E = torch.relu(L - mu)
            h = E.shape[-1] // 2
            eq.append(E[:, :h, :h].flatten(1).mean(1).cpu())
            er.append((((E[:, :h, h:].flatten(1).mean(1)
                         + E[:, h:, :h].flatten(1).mean(1)
                         + E[:, h:, h:].flatten(1).mean(1)) / 3)).cpu())
        return torch.cat(eq).numpy(), torch.cat(er).numpy()


def main(args):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    log(f"device {dev} | seed {SEED} | n {args.n} per set")

    bb = Backbone(dev)
    gtr, ltr = load_train_feats("ce")
    per, offs = m07.imagefolder_files(P["imagenet1k"]["train"])
    pools = m07.query_pools(args.n)
    log(f"train features {gtr.shape} | pools {[len(v) for v in pools.values()]}")

    centroids, protos = {}, {}
    img_cache = {}
    records = []
    t0 = time.time()

    for group, files in pools.items():
        pre = group == "id"
        for s in range(0, len(files), args.bs):
            batch = files[s:s + args.bs]
            x = m07.load_batch(batch, pre=pre)
            z, pred = bb.gap_e_pred(x, args.bs)
            zn = l2n(z)

            for i, (img_path, c) in enumerate(zip(batch, pred)):
                c = int(c)
                if c not in centroids:
                    ic = np.where(ltr == c)[0]
                    Zc = l2n(np.asarray(gtr[ic], np.float32))
                    cen = Zc.mean(0)
                    centroids[c] = cen / (np.linalg.norm(cen) + EPS)
                    protos[c] = prototype_indices(gtr, ltr, c, 3)
                cen = centroids[c]
                knn_idx = query_adaptive_indices(gtr, ltr, c, z[i], 3)

                for pol, ridx in (("prototype", protos[c]), ("knn", knn_idx)):
                    ref_paths = [idx_to_path(j, per, offs) for j in ridx]
                    key = (pol, c) if pol == "prototype" else None
                    if key is not None and key in img_cache:
                        R = img_cache[key]
                    else:
                        R = m07.load_batch(ref_paths, pre=False)
                        if key is not None:
                            if len(img_cache) > 64:
                                img_cache.clear()
                            img_cache[key] = R
                    Zr = l2n(np.asarray(gtr[ridx], np.float32))
                    records.append(dict(
                        group=group, file=img_path, pred=c, policy=pol,
                        mosaico=build_mosaic(x[i], R),
                        cos_q_cen=float(zn[i] @ cen),
                        cos_ref_cen=float((Zr @ cen).mean()),
                        cos_ref_q=float((Zr @ zn[i]).mean()),
                    ))
            if s % (args.bs * 10) == 0:
                log(f"  {group} {s + len(batch)}/{len(files)}"
                    f" [{time.time() - t0:.0f}s]")

    log(f"built {len(records)} mosaics [{time.time() - t0:.0f}s]")

    for s in range(0, len(records), args.bs_mos):
        block = records[s:s + args.bs_mos]
        mos = torch.stack([r.pop("mosaico") for r in block])
        cls = torch.tensor([r["pred"] for r in block])
        eq, er = bb.energies(mos, cls, args.bs_mos)
        for r, a, b in zip(block, eq, er):
            r["e_q"], r["e_ref"] = float(a), float(b)
            r["s"] = float(a / (a + b + EPS))
        if s % (args.bs_mos * 20) == 0:
            log(f"  energies {s + len(block)}/{len(records)}"
                f" [{time.time() - t0:.0f}s]")

    df = pd.DataFrame(records)
    fcsv = OUT / f"169_per_sample_{ts}.csv"
    df.to_csv(fcsv, index=False)
    log(f"wrote {fcsv.relative_to(ROOT)}")

    NEAR = [k for k, v in P["ood_sets"].items() if v["split"] == "near"]
    FAR = [k for k, v in P["ood_sets"].items() if v["split"] == "far"]
    summary = []
    for pol in ("prototype", "knn"):
        d = df[df.policy == pol]
        sid = d[d.group == "id"].s.values
        for g in NEAR + FAR:
            so = d[d.group == g].s.values
            summary.append(dict(
                policy=pol, ood_set=g,
                split="near" if g in NEAR else "far",
                auroc=auroc(sid, so),
                s_id=sid.mean(), s_ood=so.mean(),
                eq_id=d[d.group == "id"].e_q.mean(),
                eq_ood=d[d.group == g].e_q.mean(),
                er_id=d[d.group == "id"].e_ref.mean(),
                er_ood=d[d.group == g].e_ref.mean(),
                cos_ref_cen_id=d[d.group == "id"].cos_ref_cen.mean(),
                cos_ref_cen_ood=d[d.group == g].cos_ref_cen.mean(),
                cos_ref_q_id=d[d.group == "id"].cos_ref_q.mean(),
                cos_ref_q_ood=d[d.group == g].cos_ref_q.mean(),
            ))
    res = pd.DataFrame(summary)
    fres = OUT / f"169_resumo_{ts}.csv"
    res.to_csv(fres, index=False)
    log(f"wrote {fres.relative_to(ROOT)}")

    log("")
    log("AUROC per policy (mean over the sets)")
    for pol in ("prototype", "knn"):
        d = res[res.policy == pol]
        log(f"  {pol:10s} near {d[d.split == 'near'].auroc.mean():.3f}"
            f"  far {d[d.split == 'far'].auroc.mean():.3f}")

    log("")
    log("correlation Eq against Er per policy and group")
    for pol in ("prototype", "knn"):
        d = df[df.policy == pol]
        for g in ["id"] + FAR:
            dd = d[d.group == g]
            r = np.corrcoef(dd.e_q, dd.e_ref)[0, 1]
            log(f"  {pol:10s} {g:<12} r = {r:+.3f}")

    comp = OUT / "169_knn_vs_prototype.txt"
    comp.write_text("\n".join([
        "Producing script: note_knn_code/knn_vs_prototype.py", "",
        "What it measures, and why:",
        "  The kNN policy loses to the prototype policy in far-OOD in the",
        "  ablation table. The table reports only the ratio s = Eq/(Eq+Er),",
        "  which hides whether the loss comes from Eq (the query) or from Er",
        "  (the reference). Both are recorded separately here, on the SAME",
        "  query, for both policies, with three selection diagnostics",
        "  computed from features alone.", "",
        "Output files:",
        f"  169_per_sample_{ts}.csv  one row per (query, policy):",
        "     group, file, pred, policy, e_q, e_ref, s,",
        "     cos_q_cen, cos_ref_cen, cos_ref_q",
        f"  169_resumo_{ts}.csv      AUROC and means per set and policy",
        "  169_knn_vs_prototype.txt this companion", "",
        "Numeric summary:",
        *[f"  {l}" for l in log_lines[-14:]], "",
        "Interpretation guide:",
        "  1. If er_ood < er_id under kNN and not under prototype, the",
        "     adaptive reference falls along with the query: the bar moves",
        "     with what it is supposed to measure.",
        "  2. If corr(Eq, Er) is higher under kNN, the ratio cancels part of",
        "     the signal, which is the direct mechanism of the AUROC loss.",
        "  3. cos_ref_cen_ood lower than cos_ref_cen_id under kNN shows the",
        "     selection picking atypical members of the class for OOD queries.",
        "  4. A much higher cos_ref_q under kNN confirms the policy is doing",
        "     what it promises; the loss is not a bug.",
    ]) + "\n")
    log(f"wrote {comp.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--bs-mos", type=int, default=16)
    main(ap.parse_args())
