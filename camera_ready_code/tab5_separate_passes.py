"""Table 5: joint mosaic forward pass versus separate forward passes.

experiment_name = 'camera_ready_tables'
dataset_name    = 'imagenet1k | imagenet200'

WHAT IT COMPARES
----------------
Query and references explained in separate forward passes, with the same ratio
computed, against the joint mosaic pass. Nothing is re-run here; the inputs are
per-query CSVs written by the working repository:
  * RN-50 CE, ImageNet-1K: scripts/23 -> results/why_mosaic/per_query.csv
      d_mos          the paper score (joint pass; = Table 1, GradCAM)
      d_solo_pooled  the 4 images in SEPARATE passes; the four 7x7 maps are
                     assembled into a 14x14 mosaic, centred by the mean of the
                     4x49 cells, relu, same ratio
      mls_plain      MLS on the query alone (reference)
  * RN-18, ImageNet-200, cells ce and ls (the models of the paper; CSV and
    text only, the LaTeX table shows ImageNet-1K only):
      scripts/29 -> results/im200_models/per_query__{ce,ls}.csv
      d_mos, d_solo (= d_solo_pooled), mls_plain
  The ce_oe and ls_oe cells of scripts/29 are NOT in the paper and stay out of
  the table; they are reported in the companion so their sign is not hidden.

CI: paired bootstrap (B=2000, seed 0), resampling ID and each OOD set
independently; the same resample serves both variants, so the CI is that of
the DIFFERENCE joint - separate. near mean = SSB-hard, NINCO; far =
iNaturalist, OpenImage-O, Textures.

GATE: d_mos of the RN-50 must give 0.721/0.899 (Table 1, GradCAM).

Outputs:
  results/camera_ready_tables/cr_213_separate.csv
  results/camera_ready_tables/cr_213_separate.txt   (companion)
  results/camera_ready_tables/tab_separate.tex       (the paper table body)

Usage:
  $PYTHON_BIN camera_ready_code/tab5_separate_passes.py
"""
import os

experiment_name = "camera_ready_tables"
dataset_name    = "imagenet1k|imagenet200"

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
OUT = ROOT / "results" / experiment_name
OUT.mkdir(parents=True, exist_ok=True)
NEAR, FAR = ["ssb_hard", "ninco"], ["inaturalist", "openimage_o", "textures"]
B, SEED = 2000, 0

CELLS = [  # (label, csv, separate-pass column)
    ("IN-1K / RN-50 CE", ROOT / "results/why_mosaic/per_query.csv", "d_solo_pooled"),
    ("IN-200 / RN-18 CE", ROOT / "results/im200_models/per_query__ce.csv", "d_solo"),
    ("IN-200 / RN-18 LS", ROOT / "results/im200_models/per_query__ls.csv", "d_solo"),
]
EXTRA = [
    ("IN-200 / RN-18 CE+OE", ROOT / "results/im200_models/per_query__ce_oe.csv", "d_solo"),
    ("IN-200 / RN-18 LS+OE", ROOT / "results/im200_models/per_query__ls_oe.csv", "d_solo"),
]


def auroc(pos, neg):
    """AUROC with ID positive (Mann-Whitney, ties = 1/2)."""
    r = rankdata(np.r_[pos, neg])
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cell_stats(path, sep):
    df = pd.read_csv(path)
    var = {"mls": "mls_plain", "sep": sep, "joint": "d_mos"}
    idv = {k: df[df.group == "id"][c].to_numpy() for k, c in var.items()}
    oodv = {g: {k: df[df.group == g][c].to_numpy() for k, c in var.items()}
            for g in NEAR + FAR}

    def means(id_idx, ood_idx):
        res = {}
        for k in var:
            a = {g: auroc(idv[k][id_idx], oodv[g][k][ood_idx[g]]) for g in NEAR + FAR}
            res[k] = (np.mean([a[g] for g in NEAR]), np.mean([a[g] for g in FAR]))
        return res

    full = means(np.arange(len(idv["mls"])),
                 {g: np.arange(len(oodv[g]["mls"])) for g in NEAR + FAR})
    rng = np.random.default_rng(SEED)
    dn, df_ = [], []
    for _ in range(B):
        ii = rng.integers(0, len(idv["mls"]), len(idv["mls"]))
        oo = {g: rng.integers(0, len(oodv[g]["mls"]), len(oodv[g]["mls"]))
              for g in NEAR + FAR}
        m = means(ii, oo)
        dn.append(m["joint"][0] - m["sep"][0])
        df_.append(m["joint"][1] - m["sep"][1])
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return full, ci(dn), ci(df_)


def main():
    rows = []
    for lab, path, sep in CELLS + EXTRA:
        full, cin, cif = cell_stats(path, sep)
        rows.append({"cell": lab, "in_paper": (lab, path, sep) in CELLS,
                     "mls_near": full["mls"][0], "mls_far": full["mls"][1],
                     "sep_near": full["sep"][0], "sep_far": full["sep"][1],
                     "joint_near": full["joint"][0], "joint_far": full["joint"][1],
                     "d_near": full["joint"][0] - full["sep"][0],
                     "d_far": full["joint"][1] - full["sep"][1],
                     "d_near_lo": cin[0], "d_near_hi": cin[1],
                     "d_far_lo": cif[0], "d_far_hi": cif[1]})
        print(f"  {lab}: ok")
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cr_213_separate.csv", index=False)

    g = t.iloc[0]
    gate = max(abs(g.joint_near - 0.7212925), abs(g.joint_far - 0.898855))
    status = "PASSED" if gate < 1e-6 else "FAILED"

    # ---- LaTeX table: ImageNet-1K only; the ImageNet-200 cells stay in the
    # CSV/companion and are quoted in the text ----
    P = t.iloc[[0]].reset_index(drop=True)
    f3 = lambda x: f"{x:.3f}".lstrip("0") if x >= 0 else f"$-${abs(x):.3f}".replace("0.", ".")
    pp = lambda x: f"{100 * x:+.1f}".replace("-", "$-$")
    ci = lambda lo, hi: (f"\\scriptsize[{100*lo:+.1f}, {100*hi:+.1f}]"
                         .replace("-", "$-$"))
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{c}}}" for c in P.cell)
    cm = "".join(f"\\cmidrule(lr){{{2+2*k}-{3+2*k}}}" for k in range(len(P)))
    nf = " & ".join(["near & far"] * len(P))

    def row(name, a, b):
        return name + " & " + " & ".join(f"{f3(P.loc[k, a])} & {f3(P.loc[k, b])}"
                                         for k in range(len(P))) + r" \\"
    dl = (r"$\Delta$ joint $-$ separate & " +
          " & ".join(f"{pp(P.loc[k, 'd_near'])} & {pp(P.loc[k, 'd_far'])}"
                     for k in range(len(P))) + r" \\")
    dc = (r"\quad 95\% CI & " +
          " & ".join(f"{ci(P.loc[k, 'd_near_lo'], P.loc[k, 'd_near_hi'])} & "
                     f"{ci(P.loc[k, 'd_far_lo'], P.loc[k, 'd_far_hi'])}"
                     for k in range(len(P))) + r" \\")
    tex = "\n".join([
        "% generated by camera_ready_code/tab5_separate_passes.py -- do not edit by hand",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Joint versus separate forward passes, mean AUROC on ImageNet-1K. "
        r"$\Delta$: joint minus separate, in AUROC points, with 95\% CI.}",
        r"\label{tab:separate}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "cc" * len(P) + "@{}}",
        r"\toprule",
        " & " + head + r" \\",
        cm,
        " & " + nf + r" \\",
        r"\midrule",
        row("MLS (single image)", "mls_near", "mls_far"),
        row("Separate passes", "sep_near", "sep_far"),
        row("Joint mosaic (\\method)", "joint_near", "joint_far"),
        r"\midrule",
        dl, dc,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    (OUT / "tab_separate.tex").write_text(tex + "\n")

    show = t.copy()
    for c in show.columns[2:]:
        show[c] = show[c].astype(float).round(4)
    txt = [
        "(1) Script: camera_ready_code/tab5_separate_passes.py",
        "(2) What it measures: mean near/far AUROC of the mosaic score (joint pass)",
        "    against the SAME ratio computed with query and references in separate",
        "    passes, and the difference with a paired bootstrap CI (B=2000).",
        "(3) Files: cr_213_separate.csv; tab_separate.tex (paper table).",
        f"(4) GATE d_mos RN-50 vs Table 1 (0.7213/0.8989): max|d| = {gate:.1e} -> {status}",
        show.to_string(index=False),
        "(5) Guide:",
        "  1. d_near/d_far > 0 with the CI above zero: the joint pass adds",
        "     separation beyond the same ratio without interaction.",
        "  2. CI crossing zero: the interaction is not distinguishable from zero in that cell.",
        "  3. in_paper=False: OE cells of scripts/29, not in the paper; their sign",
        "     has to be stated if the table is used as general evidence.",
    ]
    (OUT / "cr_213_separate.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
