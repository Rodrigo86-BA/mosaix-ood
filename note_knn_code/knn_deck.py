"""171 - Builds the .pptx of the two kNN slides.

experiment_name = 'ppt_knn'
dataset_name    = 'imagenet1k'

Reads the figures from note_knn_code/knn_slides.py and the numbers from note_knn_code/knn_vs_prototype.py and builds
a two-slide 16:9 deck with speaker notes.

Input:  figures/slides_knn/170_s1_exemplo.png
        figures/slides_knn/170_s2_medida.png
        results/knn_vs_prototype/169_per_sample_*.csv
Output: slides/knn_vs_prototype.pptx

Usage:
  $PYTHON_BIN note_knn_code/knn_deck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

experiment_name = "ppt_knn"
dataset_name = "imagenet1k"

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "slides_knn"
RES = ROOT / "results" / "knn_vs_prototype"
OUT = ROOT / "slides" / "knn_vs_prototype.pptx"

BLUE = RGBColor(0x00, 0x72, 0xB2)
ORANGE = RGBColor(0xD5, 0x5E, 0x00)
GREY = RGBColor(0x44, 0x44, 0x44)


def numbers():
    df = pd.read_csv(sorted(RES.glob("169_per_sample_*.csv"))[-1])
    near = {"ssb_hard", "ninco"}
    far = {"inaturalist", "openimage_o", "textures"}
    df["reg"] = df.group.map(lambda g: "ID" if g == "id"
                             else ("near" if g in near else "far"))
    d = {}
    for p in ("prototype", "knn"):
        s = df[df.policy == p]
        d[p] = dict(
            eq_id=s[s.reg == "ID"].e_q.mean(), eq_far=s[s.reg == "far"].e_q.mean(),
            er_id=s[s.reg == "ID"].e_ref.mean(), er_far=s[s.reg == "far"].e_ref.mean(),
            s_id=s[s.reg == "ID"].s.mean(), s_far=s[s.reg == "far"].s.mean(),
            cc_id=s[s.reg == "ID"].cos_ref_cen.mean(),
            cc_far=s[s.reg == "far"].cos_ref_cen.mean(),
        )
    return d


def title_box(slide, texto, sub):
    cx = slide.shapes.add_textbox(Inches(0.45), Inches(0.28),
                                  Inches(12.4), Inches(1.0))
    tf = cx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = texto
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = sub
    r2.font.size = Pt(14)
    r2.font.color.rgb = GREY


def bullets(slide, left, top, larg, height, items, size=14):
    cx = slide.shapes.add_textbox(left, top, larg, height)
    tf = cx.text_frame
    tf.word_wrap = True
    for i, (text, color, bold) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return cx


def main():
    if not (FIG / "170_s1_exemplo.png").exists():
        sys.exit("run note_knn_code/knn_slides.py first")
    n = numbers()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    s1 = prs.slides.add_slide(blank)
    title_box(s1, "The kNN policy picks references that resemble the query",
           "ResNet-50 trained from scratch on ImageNet-1K, cross-entropy. "
           "Grad-CAM of the predicted class, computed over the whole "
           "mosaic in a single forward pass.")
    s1.shapes.add_picture(str(FIG / "170_s1_exemplo.png"), Inches(0.30),
                          Inches(1.32), height=Inches(6.0))
    bullets(s1, Inches(8.35), Inches(1.70), Inches(4.65), Inches(5.4), [
        ("Top row, in-distribution query.", GREY, True),
        ("Both policies give s = 0.371. The choice of references does not "
         "matter when the query belongs to the class.", GREY, False),
        ("Bottom row, far-OOD query.", GREY, True),
        ("A paisley texture predicted as bib.", GREY, False),
        ("prototype: the references are real bibs, the query tile stays "
         "dark, s = 0.005.", BLUE, False),
        ("kNN: the references are three heavily patterned bibs, chosen "
         "because they resemble the paisley. The query tile lights up, "
         "s = 0.698.", ORANGE, False),
        ("Same query, same predicted class, same network. Only the "
         "reference policy changed.", RGBColor(0x11, 0x11, 0x11), True),
    ])
    s1.notes_slide.notes_text_frame.text = (
        'Figure from note_knn_code/knn_slides.py, data from note_knn_code/knn_vs_prototype.py. The far-OOD query is the one with the largest s(knn) - s(prototype) difference in the Textures set. The ID query is the one with the smallest difference, as a control.')

    s2 = prs.slides.add_slide(blank)
    title_box(s2, "The reference stops being a fixed bar",
           "1,000 queries per set, seed 0, OpenOOD v1.5 splits. "
           "s = Eq / (Eq + Eref): the ablation table reports only the ratio, "
           "so both terms are measured here on the same query.")
    s2.shapes.add_picture(str(FIG / "170_s2_medida.png"), Inches(0.45),
                          Inches(1.55), width=Inches(12.45))

    p, k = n["prototype"], n["knn"]
    lines = [
        (f"prototype   Eq falls {100 * (1 - p['eq_far'] / p['eq_id']):.0f}% "
         f"from ID to far-OOD, Eref moves "
         f"{100 * (p['er_far'] / p['er_id'] - 1):+.0f}%.  "
         f"s: {p['s_id']:.3f} to {p['s_far']:.3f}", BLUE, False),
        (f"kNN         Eq falls {100 * (1 - k['eq_far'] / k['eq_id']):.0f}%, "
         f"Eref falls {100 * (1 - k['er_far'] / k['er_id']):.0f}% with it. "
         f"The ratio cancels half the drop.  "
         f"s: {k['s_id']:.3f} to {k['s_far']:.3f}", ORANGE, False),
        ("Why the bar moves: for an OOD query the nearest same-class "
         "neighbours are the atypical members of the class. Cosine to the "
         f"class centroid falls {k['cc_id']:.2f} to {k['cc_far']:.2f} under "
         f"kNN, against {p['cc_id']:.2f} to {p['cc_far']:.2f} under "
         "prototype.", GREY, False),
        ("A reference that correlates with the query does not measure the "
         "query: corr(Eq, Eref) is +0.48 under kNN and -0.01 under "
         "prototype. Far-OOD AUROC 0.898 to 0.841.",
         RGBColor(0x11, 0x11, 0x11), True),
    ]
    bullets(s2, Inches(0.45), Inches(5.05), Inches(12.45), Inches(2.2),
            lines, size=13)
    s2.notes_slide.notes_text_frame.text = (
        'Numbers from results/knn_vs_prototype/169_per_sample_*.csv. The ablation table in the paper gives prototype far 0.904 and knn far 0.861; here 0.898 and 0.841. The difference is n=1000 ID queries instead of 10000, and one query position instead of four.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    for i, s in enumerate(prs.slides, 1):
        print(f"  slide {i}: {len(s.shapes)} shapes")


if __name__ == "__main__":
    main()
