"""Response Figure R1: schematic of the two adaptations of the mosaic to a ViT.

experiment_name = 'camera_ready_vit_schematic'
dataset_name    = 'imagenet1k'

Illustrates the paragraphs "Adapting the mosaic" and "Adapting the model" of
Section 6. Illustration only, no measurement. Four real images of the same
class (goldfinch, n01531178): query = the first image of the class in the raw
val/ tree, references = the first 3 of the raw train/ tree (alphabetical order,
no selection by content). Resize(256)+CenterCrop(224) on all of them.
  (a) Adapting the mosaic: each tile goes to 112 -> mosaic 224, a 14x14 grid of
      16 px patches, ViT unchanged.
  (b) Adapting the model: tiles at 224 -> mosaic 448, a 28x28 grid, position
      embeddings interpolated 14x14 -> 28x28.
Scale: both mosaics are drawn at the same size; the patch grid (14x14 against
28x28) shows the difference in resolution.

Environment:
  IN1K_RAW_TRAIN  raw ImageNet-1K training tree (wnid folders)
                  default ${DATA_ROOT}/imagenet1k/Imagenet/train
  IM200_RAW_VAL   raw ImageNet-1K validation tree (wnid folders)
                  default ${DATA_ROOT}/imagenet1k/Imagenet/val/ILSVRC2012_img_val

Outputs (figures/vit_schematic/ of $MOSAIX_ROOT):
  cr_229_vit_mosaic_schematic.pdf/.png, cr_229_vit_mosaic_schematic.txt (companion)

Usage:
  $PYTHON_BIN camera_ready_code/figR1_vit_schematic.py
"""
import os

experiment_name = "camera_ready_vit_schematic"
dataset_name    = "imagenet1k"

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MOSAIX_ROOT", REPO))
OUT = ROOT / "figures" / "vit_schematic"
OUT.mkdir(parents=True, exist_ok=True)
_DATA = Path(os.environ.get("DATA_ROOT", "."))
TRAIN_RAW = Path(os.environ.get("IN1K_RAW_TRAIN", _DATA / "imagenet1k/Imagenet/train"))
VAL_RAW = Path(os.environ.get("IM200_RAW_VAL", _DATA / "imagenet1k/Imagenet/val/ILSVRC2012_img_val"))
WNID = "n01531178"
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7, "font.family": "DejaVu Sans"})
ORANGE, BLUE, GREY = "#E69F00", "#0072B2", "#555555"


def crop224(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)))


def val_of_class():
    lab = VAL_RAW / WNID
    return sorted(lab.iterdir())[0]


q = crop224(val_of_class())
refs = [crop224(p) for p in sorted((TRAIN_RAW / WNID).iterdir())[:3]]
tiles = [q] + refs


def mosaic(side):
    t = [np.asarray(Image.fromarray(x).resize((side, side), Image.BILINEAR)) for x in tiles]
    return np.vstack([np.hstack(t[:2]), np.hstack(t[2:])])


def draw(ax, img, patch, title, black, blue):
    """Both mosaics are drawn at the same size, with the labels beside them; the patch grid shows the scale."""
    n = img.shape[0]
    ax.imshow(img, extent=(0, 1, 1, 0), interpolation="lanczos")
    for k in range(0, n + 1, patch):
        ax.plot([0, 1], [k / n, k / n], color="white", lw=0.25, alpha=0.7)
        ax.plot([k / n, k / n], [0, 1], color="white", lw=0.25, alpha=0.7)
    ax.add_patch(Rectangle((0, 0), 0.5, 0.5, fill=False, ec=ORANGE, lw=1.4))
    ax.set_xlim(0, 1.86); ax.set_ylim(1.0, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=7.5, loc="left", pad=3)
    y = 0.02
    for line in black:
        ax.text(1.05, y, line, ha="left", va="top", fontsize=6.2); y += 0.14
    y += 0.06
    for line in blue:
        ax.text(1.05, y, line, ha="left", va="top", fontsize=6.2, color=BLUE); y += 0.14


fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.08), gridspec_kw=dict(wspace=0.04))
draw(axs[0], mosaic(112), 16, "(a) Adapting the mosaic",
     ["224$\\times$224 input", "14$\\times$14 patches"], ["each image", "224$\\to$112", "same PE"])
draw(axs[1], mosaic(224), 16, "(b) Adapting the model",
     ["448$\\times$448 input", "28$\\times$28 patches"], ["each image 224", "PE 14$\\times$14", "$\\to$28$\\times$28"])
fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02)
base = OUT / "cr_229_vit_mosaic_schematic"
fig.savefig(base.with_suffix(".pdf"))
fig.savefig(base.with_suffix(".png"), dpi=300)

txt = f"""(1) Script: camera_ready_code/figR1_vit_schematic.py
(2) What it shows: schematic (no measurement) of the two ways of taking the 2x2 mosaic to ViT-B/16.
    (a) each tile resampled from 224 to 112, a 224 mosaic, the input the ViT already expects
    (14x14 patches of 16 px); (b) tiles at 224, a 448 mosaic (28x28 patches) and interpolated
    position embeddings. Both mosaics have the same size in the drawing. Orange box = query.
(3) Files: {base.name}.pdf/.png (column width, 3.35 in), this .txt.
(4) Images: class {WNID} (goldfinch); query = {val_of_class().name} (val);
    references = the first 3 of train/{WNID} in alphabetical order. Resize(256)+CenterCrop(224).
(5) Guide:
  1. In (a) the query takes the same fraction of the input (1/4) but each object has half the side:
     each patch covers 4x more area of the original image.
  2. In (b) each patch covers the same area as in the isolated image, but the ViT sees a grid of
     positions it did not see in training.
  3. The numbers of each adaptation are in the tab_field table (Transformers and Image Size).
"""
base.with_suffix(".txt").write_text(txt)
print(txt)
