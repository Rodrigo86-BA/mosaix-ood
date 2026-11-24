# Reproducing the figures and tables

Every entry gives the producing script, the command, the environment variables
from `.env` it consumes, and the intermediate artifacts it reads. Run

```
$PYTHON_BIN configs/render_paths.py --check
```

to see which of these are satisfied on the current machine. Paths below are
relative to `$MOSAIX_ROOT` unless prefixed with `$XAI_OOD_ROOT`.

Two of the six figures (5 and 6) read frozen CSVs only: no checkpoint, no
dataset, no GPU. The other four load a backbone.

---

## Fig. 1 - teaser  `figures/teaser/fig1_teaser_inline.pdf`

```
$PYTHON_BIN fig1_code/fig1_teaser.py --bench in1k --layout inline --norm panel
```

| | |
|---|---|
| env | `CIFAR_TORCHVISION_ROOT`, `TIN_WORDS`, both only for class names |
| reads | `results/grid_dump/grids__{ce,ls,mixup}.npz` |
| backbone | none loaded here |

This script loads no checkpoint. It renders from the frozen grid dumps of
`scripts/43_grid_dump.py`, which were produced with the three from-scratch
RN-50 backbones at seed 0. `CE_CKPT`, `LS_CKPT` and `MIXUP_CKPT` are needed
only to regenerate those dumps.

## Fig. 2 - method  `figures/method/fig2_method.pdf`

```
$PYTHON_BIN fig2_code/fig2_method_data.py --key ce --query <id>
$PYTHON_BIN fig2_code/fig2_method.py
```

| | |
|---|---|
| env | `CE_CKPT`, `CE_FEATDIR`, `IN1K_TRAIN` |
| reads | `results/method_fig/`, `results/ref_bank/bank__ce.npz`, `figures/method/assets/snowflake_cskd.png` |
| backbone | RN-50 CE |

Script 56 runs the backbone live over `IN1K_TRAIN` to select the prototypes and
dump the panels; script 57 only draws them.

## Fig. 3 - saliency zoo  `figures/head_refit/149_f1__saliency_zoo.pdf`

```
$PYTHON_BIN fig3_code/fig3_saliency_zoo.py
```

| | |
|---|---|
| env | `CE_CKPT`, `IN1K_TRAIN`, `OOD_INATURALIST`, `OOD_OPENIMAGE_O`, `OOD_TEXTURES` |
| reads | `results/saliency_zoo/winner.pt` |
| backbone | RN-50 CE |

`winner.pt` caches the query the figure shows (the *trash can*). Delete it to
re-run the search, which needs the OOD sets above.

## Fig. 4 - head refit  `figures/head_refit/119_fig5_headrefit_4col.pdf`

```
$PYTHON_BIN fig4_code/fig4_head_refit.py
```

| | |
|---|---|
| env | `CE_CKPT`, `LS_CKPT`, `IN1K_TRAIN`, and `LS_FEATDIR` if the refit head is not cached |
| reads | both backbones live, plus `results/head_refit/head_ce_refit__ls.pt` |
| backbone | RN-50 CE and RN-50 LS |

CE and LS are two backbones trained from scratch, not one backbone refit; only
the head differs in the refit arm, so every difference is a difference in
gradient direction. The third column needs the refit head, which is read from
the cached artifact above; without it the script refits from the LS training
features.

## Fig. 5 - placement ablation  `figures/ablations_col/163_ablations_placement_col.pdf`

```
$PYTHON_BIN fig5_code/fig5_placement.py
```

| | |
|---|---|
| env | none |
| reads | `$XAI_OOD_ROOT/results/stats/imagenet1k/table1.csv`, `$XAI_OOD_ROOT/results/protocol_placement/imagenet1k/placement_*.csv` |

## Fig. 6 - cross-benchmark  `figures/crossplot_bench/157_crossplot_bench__cifar224_col.pdf`

```
$PYTHON_BIN fig6_code/fig6_crossplot.py --cifar 224
```

| | |
|---|---|
| env | none |
| reads | `results/table2_multi/auroc.csv`, `results/cifar100_224_panel/auroc.csv` |

The upstream runs that produced those CSVs used `IM200_*` (ImageNet-200 / RN-18)
and `CIFAR_*` (CIFAR-100 / RN-18); the figure itself does not.

---

## Tables

`tables/*.tex` is **generated** by `table_code/make_tables.py` from the
frozen CSVs. Running it reproduces the tables the paper inputs.

### tab:field - image size and positional embeddings

```
$PYTHON_BIN table_code/tab_field_posemb.py
$PYTHON_BIN table_code/tab_field_mosaic.py
```

| | |
|---|---|
| env | `IN1K_VAL`, `OOD_SSB_HARD`, `OOD_NINCO`, `OOD_INATURALIST`, `OOD_OPENIMAGE_O`, `OOD_TEXTURES`, `TORCH_HOME` |
| writes | `results/mosaic_sizes/` |
| backbone | **public torchvision** `ResNet50_Weights.IMAGENET1K_V1` and `ViT_B_16_Weights.IMAGENET1K_V1`, downloaded on demand |

This is the one table measured on public weights rather than the from-scratch
backbones. All ten rows are paired: same seed (0), same 1,000-query pools per
set. Script 168 writes a manifest listing every query file and every prototype
**by path**, so the pools can be reconstructed without the original index
arrays. Script 167 reproduces the earlier `shared_code/transformer_quarter.py` measurement to 0.00 pp
on all four ViT cells, which is the check that the pools really are the same.

Two implementation notes that are easy to get wrong:

1. The ViT Grad-CAM readout is taken on the **penultimate** encoder block. After
   the last block only the CLS token reaches the head, so the gradients on the
   spatial tokens are exactly zero.
2. Reading a ViT at 448 requires `interpolate_embeddings(448, 16, state_dict)`
   from `torchvision.models.vision_transformer`. The rows that use it are marked
   with a dagger in the table.
