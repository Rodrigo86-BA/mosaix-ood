"""REAL data for Figure 2, the method diagram.

experiment_name = 'method_fig'
dataset_name    = 'imagenet1k'

WHY THIS SCRIPT EXISTS
----------------------
Figure 2 is the method diagram, and a method diagram lies very easily: an
"embeddings" box with invented coloured rectangles, a hand-drawn heatmap, a
score value picked because it looks good. This script runs the WHOLE path
live, on the same backbone and the same protocol as the run that produces the
numbers in the paper (scripts/43 -> results/grid_dump/grids__ce.npz), and
dumps to disk every image, every map and every number the figure shows.
Nothing in the drawing is illustrative.

THE PATH, in the order the figure walks it
------------------------------------------
  0. reference bank    RESOLVED BEFORE ANY QUERY, for all 1000 classes at
                       once, by scripts/58. Policy: sort the training
                       exemplars of the class by cosine against the centroid
                       (`src.exemplars.prototype_indices`) and keep the first
                       3 the backbone CLASSIFIES CORRECTLY, by a live forward
                       pass. The panel without that filter (the policy of the
                       frozen run) is dumped too, under `base_*`, so the
                       figure can show what the filter rejected. `--no-bank`
                       returns to the old behaviour.
  1. indexing          the predicted class only INDEXES the bank: no
                       reference image depends on the query.
  2. query             one validation image, predicted c. `pre=True`: the
                       clean/val tree is already cropped to 224 (gate 10/11).
  3. 2x2 mosaic        query in the top-left tile (qpos=0) plus the 3
                       references -> 448x448.
  4. ONE forward       layer4 of the mosaic -> A in R^{2048x14x14}.
  5. embeddings        the K channels of largest contribution
                       |w_c[d]| * GAP(A_d). This is what the "embeddings" box
                       of the template displays, with real activation maps.
  6. class CAM         L = <w_c, A> on the 14x14 grid.
  7. rectification     mu = mean(L), E = relu(L - mu).
  8. the score         s = E_q / (E_q + E_ref).

GATES
-----
  * `s` recomputed here matches the frozen dump (tol 1e-5), which proves that
    the forward pass of this session reproduces the run in the paper.
  * the 3 references are IDENTICAL to those in the dump.
  * the unrectified ratio is 3/2 by construction (mu = (mLq + 3 mLr)/4). This
    is checked and printed, because it is what justifies the relu in the
    figure.

Writes nothing inside external_xai_ood/: it only reads.

Output: results/method_fig/method_data__<key>__<query>.npz

Usage:
  full: $PYTHON_BIN fig2_code/fig2_method_data.py
        $PYTHON_BIN fig2_code/fig2_method_data.py --query ood
  cell: --key ce | ls | mixup
"""
import os

experiment_name = "method_fig"
dataset_name    = "imagenet1k"

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_code"
XAI = ROOT / "external_xai_ood"
sys.path.insert(0, str(XAI))
sys.path.insert(0, str(SHARED))

from src.mosaic import build_mosaic
from torchvision.models import ResNet50_Weights

m07 = importlib.import_module("rood_qualify_protocol")

RES = ROOT / "results" / experiment_name
RES.mkdir(parents=True, exist_ok=True)

CATS = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
QUAD = [(slice(0, 7), slice(0, 7)), (slice(0, 7), slice(7, 14)),
        (slice(7, 14), slice(0, 7)), (slice(7, 14), slice(7, 14))]


def score(L):
    """s = E_q / (E_q + E_ref), the same arithmetic as fig1_code/fig1_teaser.py."""
    mu = float(L.mean())
    E = np.maximum(L - mu, 0.0)
    eq = float(E[QUAD[0][0], QUAD[0][1]].mean())
    er = float(np.mean([E[a, b].mean() for a, b in QUAD[1:]]))
    return mu, E, eq, er, (eq / (eq + er) if eq + er > 0 else 0.0)


def disp(path, pre):
    """The image as the network saw it (same crop), in uint8 for display."""
    im = Image.open(path).convert("RGB")
    if not pre:
        w, h = im.size
        sc = 256 / min(w, h)
        im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))),
                       Image.BILINEAR)
        w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        im = im.crop((l, t, l + 224, t + 224))
    return np.asarray(im, dtype=np.uint8)


def pick_query(z, args, widx):
    """Index of the query: an ID one predicted correctly with high s, or an OOD one
    with low s.

    It chooses among the classes that have both ID and OOD queries, so that the
    figure can later gain an OOD variant with THE SAME references.
    """
    group, pred, qpath = z["group"], z["pred"], z["qpath"]
    L = z["L_mos"].astype(np.float64)
    sv = np.array([score(L[i])[-1] for i in range(len(L))])
    true = np.array([widx.get(Path(q).parent.name, -1) for q in qpath])
    ok = (group == "id") & (true >= 0) & (pred == true)
    ood = np.isin(group, ["inaturalist", "openimage_o", "textures"])

    if args.cls is not None:
        c = int(args.cls)
    else:
        best = None
        for c_ in np.unique(pred[ok]):
            a = np.where(ok & (pred == c_))[0]
            b = np.where(ood & (pred == c_))[0]
            if not len(a) or not len(b):
                continue
            g = sv[a].max() - sv[b].min()
            if best is None or g > best[0]:
                best = (g, int(c_))
        c = best[1]
    pool = np.where((ok if args.query == "id" else ood) & (pred == c))[0]
    assert len(pool), f"no {args.query} query in class {c}"
    i = int(pool[np.argmax(sv[pool])] if args.query == "id"
            else pool[np.argmin(sv[pool])])
    return i, c, sv


def main(args):
    t0 = time.time()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = np.load(ROOT / "results" / "grid_dump" / f"grids__{args.key}.npz",
                allow_pickle=True)

    tr_files, offs = m07.imagefolder_files(m07.P["imagenet1k"]["train"])
    wnids = sorted(d.name for d in Path(m07.P["imagenet1k"]["train"]).iterdir()
                   if d.is_dir())
    widx = {w: k for k, w in enumerate(wnids)}

    i, c, sv = pick_query(z, args, widx)
    qpath = str(z["qpath"][i])
    rpath = [str(x) for x in z["rpath"][i]]
    cname = CATS[c]
    print(f"\n=== 56 method figure data | {args.key} | query {args.query} ===")
    print(f"  predicted class {c} ({cname})   idx {i}   s_dump {sv[i]:.4f}")
    print(f"  query {Path(qpath).name}  [{z['group'][i]}]")

    model, W, b = m07.load_backbone(args.key, dev)
    body = m07.body_of(model)
    wc = W[c].double()

    panel = m07.prototype_panel(args.key, tr_files, offs)
    assert [str(x) for x in panel[c]] == rpath, "references differ from the dump"
    print("  gate: rpath from the dump == prototype panel (frozen policy)")

    bank = None
    if args.bank:
        bank = np.load(ROOT / "results" / "ref_bank" / f"bank__{args.key}.npz",
                       allow_pickle=True)
        rpath = [str(x) for x in bank["paths"][c]]
        print(f"  filtered bank: class {c} "
              f"{'CHANGED' if bool(bank['changed'][c]) else 'unchanged'} by the "
              f"filtro de acerto; ranks {bank['ranks'][c].tolist()}")

    qt = m07.load_batch([qpath], pre=(str(z["group"][i]) == "id"))
    rt = m07.load_batch(rpath)
    mos = build_mosaic(qt[0], rt, qpos=0).unsqueeze(0)

    A = m07.layer4(body, mos, dev, args.bs)[0].double()
    L = torch.einsum("d,dhw->hw", wc, A).float().numpy().astype(np.float64)
    mu, E, eq, er, s = score(L)

    dl = float(np.abs(L - z["L_mos"][i].astype(np.float64)).max())
    ds = abs(s - sv[i])
    same_panel = rpath == [str(x) for x in z["rpath"][i]]
    print(f"  gate: |dL|max {dl:.2e}   |ds| {ds:.2e}"
          f"   (painel {'igual' if same_panel else 'TROCADO'} ao do dump)")
    if same_panel:
        assert ds < 1e-5, f"s diverge do dump: {s:.6f} vs {sv[i]:.6f}"

    mLq = float(L[QUAD[0][0], QUAD[0][1]].mean())
    mLr = float(np.mean([L[a, b].mean() for a, b in QUAD[1:]]))
    raw = (mLq - mu) / ((mLq - mu) + (mLr - mu))
    print(f"  gate: mu {mu:.4f} vs (mLq+3mLr)/4 {(mLq + 3 * mLr) / 4:.4f}"
          f"   |  s SEM relu {raw:.6f}  (3/2 por construcao)")
    assert abs(mu - (mLq + 3 * mLr) / 4) < 1e-6 and abs(raw - 1.5) < 1e-6, \
        "a identidade mu = (mLq+3mLr)/4 quebrou"

    gap = A.mean(dim=(1, 2))
    contrib = (wc * gap).numpy()
    ch = np.argsort(-contrib)[:args.k_ch]
    Ak = A[torch.from_numpy(ch.copy())].float().numpy()
    print(f"  embeddings: canais {ch.tolist()}  "
          f"contrib {contrib[ch].round(3).tolist()}  "
          f"(sum over 2048 = {contrib.sum():.2f} = logit-b)")

    r_pred = m07.predict(model, rt, dev, args.bs)
    print(f"  refs predicted {r_pred.tolist()} (class {c}) -> "
          f"{int((r_pred == c).sum())}/3 corretas")

    if bank is not None:
        depth = max(int(bank["ranks"][c].max()) + 1, args.n_cand)
        ridx = bank["rank_idx"][c][:depth]
        cand = [str(tr_files[c][int(g - offs[c])]) for g in ridx]
        c_pred = np.array([int(bank["cand_pred"][c][k]) for k in range(depth)],
                          dtype=np.int32)
        chosen = np.zeros(depth, bool)
        chosen[bank["ranks"][c]] = True
    else:
        rng = np.random.default_rng(0)
        others = [str(p) for p in tr_files[c] if str(p) not in rpath]
        pick = rng.choice(len(others), min(args.n_cand - 3, len(others)),
                          replace=False)
        cand = rpath + [others[k] for k in np.sort(pick)]
        ct = m07.load_batch(cand)
        c_pred = m07.predict(model, ct, dev, args.bs).astype(np.int32)
        chosen = np.array([p in rpath for p in cand])
    print(f"  pool de treino: {int((c_pred == c).sum())}/{len(cand)} corretas, "
          f"{int(chosen.sum())} escolhidas, ranks "
          f"{np.where(chosen)[0].tolist()}")

    bc = args.build_class
    assert bank is not None, "--build-class exige o banco (scripts/58)"
    bdepth = max(int(bank["ranks"][bc].max()) + 1, args.n_build)
    bidx = bank["rank_idx"][bc][:bdepth]
    bpaths = [str(tr_files[bc][int(g - offs[bc])]) for g in bidx]
    b_pred = np.array([int(bank["cand_pred"][bc][k]) for k in range(bdepth)],
                      dtype=np.int32)
    b_chosen = np.zeros(bdepth, bool)
    b_chosen[bank["ranks"][bc]] = True
    b_img = np.stack([disp(x, pre=False) for x in bpaths])
    rej = [int(b_pred[k]) for k in range(bdepth) if b_pred[k] != bc]

    with torch.no_grad():
        z_all = model(qt.to(dev)).cpu()[0].double()
    topk = torch.topk(z_all, args.n_bars)
    bar_v = topk.values.numpy()
    bar_c = topk.indices.numpy().astype(np.int32)
    assert int(bar_c[0]) == c, "the top logit must be the predicted class"
    print(f"  logits top-{args.n_bars}: "
          f"{[f'{CATS[int(k)]} {v:.1f}' for k, v in zip(bar_c[:4], bar_v[:4])]}")

    rng = np.random.default_rng(4)
    pool_t = [k for k in range(1000) if not (400 <= k <= 900)]
    tc = [pool_t[int(x)] for x in
          rng.choice(len(pool_t), args.n_train, replace=False)]
    t_img = np.stack([disp(str(tr_files[k][0]), pre=False) for k in tc])
    print(f"  construction class: {bc} ({CATS[bc]}), {bdepth} candidates, "
          f"mantidos ranks {bank['ranks'][bc].tolist()}, "
          f"rejected one(s) predicted as {[CATS[r] for r in rej]}")

    q_img = disp(qpath, pre=(str(z["group"][i]) == "id"))
    r_img = np.stack([disp(p, pre=False) for p in rpath])
    m_img = np.zeros((448, 448, 3), np.uint8)
    m_img[:224, :224] = q_img
    m_img[:224, 224:] = r_img[0]
    m_img[224:, :224] = r_img[1]
    m_img[224:, 224:] = r_img[2]

    out = RES / f"method_data__{args.key}__{args.query}.npz"
    np.savez_compressed(
        out,
        q_img=q_img, r_img=r_img, m_img=m_img,
        c_img=np.stack([disp(p, pre=False) for p in cand]),
        bar_v=bar_v.astype(np.float32), bar_c=bar_c,
        bar_name=np.array([CATS[int(k)] for k in bar_c]),
        t_img=t_img,
        b_img=b_img, b_pred=b_pred, b_chosen=b_chosen,
        b_cls=np.int32(bc), b_name=CATS[bc],
        b_rej_name=np.array([CATS[r] for r in rej]),
        bank_changed=np.bool_(bool(bank["changed"][c]))
        if bank is not None else np.bool_(False),
        c_pred=c_pred.astype(np.int32), c_chosen=chosen,
        r_pred=r_pred.astype(np.int32),
        A=Ak, A_ch=ch.astype(np.int32), A_contrib=contrib[ch],
        L=L.astype(np.float32), E=E.astype(np.float32),
        mu=np.float32(mu), eq=np.float32(eq), er=np.float32(er),
        s=np.float32(s), raw_ratio=np.float32(raw),
        mLq=np.float32(mLq), mLr=np.float32(mLr),
        pred=np.int32(c), cname=cname, group=str(z["group"][i]),
        qpath=qpath, rpath=np.array(rpath), cand=np.array(cand),
        r_ids=np.array([Path(p).stem.split("_")[-1] for p in rpath]))
    print(f"\n  E_q {eq:.4f}   E_ref {er:.4f}   s {s:.4f}")
    print(f"[ok] {out}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - t0:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="ce", choices=["ce", "ls", "mixup"])
    ap.add_argument("--query", default="id", choices=["id", "ood"])
    ap.add_argument("--cls", type=int, default=None,
                    help="force the predicted class (default: the teaser pair)")
    ap.add_argument("--k-ch", type=int, default=4,
                    help="how many layer4 channels to dump as embeddings")
    ap.add_argument("--bank", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="use the correctness-filtered bank of scripts/58")
    ap.add_argument("--build-class", type=int, default=817,
                    help="the class that ILLUSTRATES the bank construction; "
                         "the default is one of the 30 where the correctness "
                         "filter bites")
    ap.add_argument("--n-bars", type=int, default=10,
                    help="how many logits to draw in the prediction bar")
    ap.add_argument("--n-train", type=int, default=6,
                    help="images in the generic training stack of band 1")
    ap.add_argument("--n-build", type=int, default=4,
                    help="ranked candidates to draw in the construction")
    ap.add_argument("--n-cand", type=int, default=8,
                    help="training candidates for the selection box")
    ap.add_argument("--bs", type=int, default=32)
    main(ap.parse_args())
