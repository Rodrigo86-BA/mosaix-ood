# Code of the final-version tables and figures

The scripts here produce Tables 2, 5 and 8, Figures 7 and 8, and the ResNet-18
ViM row of the full detector table.

Most of them import modules that live in the working repository:
`64_depth_sweep`, `62_mosaic_rood`, `68_head_refit_figure`,
`07_rood_qualify_protocol`, `cr_200_loc_common` and `cr_201_loc_gate`. These
modules are not copied here. Each script puts `$MOSAIX_ROOT/scripts` on
`sys.path` and imports them from there, the same way `fig3_code/` does.
`MOSAIX_ROOT` also sets where every script reads `results/` and writes
`results/` and `figures/`. If it is unset, the scripts fall back to the root of
this repository. That works only for the scripts that read CSVs and import
nothing (Tables 2, 5 and 8).

Some of the imported working-repository modules hard-code machine-local paths.
`64_depth_sweep.TV_FEAT` is one, the cached torchvision features. `cr_200_loc_common`
hard-codes `VAL_RAW` and `IMS_VAL`. Where a script below depends on one of these,
it says so.

The working-repository modules read the dataset and checkpoint paths from
`$XAI_OOD_ROOT/configs/paths.yaml`. The variables listed under
**Environment** below are the `.env.example` entries that file has to agree
with.

All commands are run from the root of this repository.

## Table 5, joint versus separate forward passes: `tab5_separate_passes.py`

**Produces** `results/camera_ready_tables/tab_separate.tex`, the body of Table 5,
plus `cr_213_separate.csv` and the companion `cr_213_separate.txt`

```
$PYTHON_BIN camera_ready_code/tab5_separate_passes.py
```

**Environment** MOSAIX_ROOT

**Reads** `results/why_mosaic/per_query.csv` (ImageNet-1K, RN-50 CE),
`results/im200_models/per_query__{ce,ls,ce_oe,ls_oe}.csv` (ImageNet-200, RN-18)

It compares the score of the joint mosaic forward pass with the same ratio
computed from separate forward passes of the query and the references. The CI
of the difference comes from a paired bootstrap with B=2000 and seed 0. A gate
checks that the joint-pass AUROC of the RN-50 matches Table 1 (0.721/0.899).
The LaTeX table shows ImageNet-1K only. The ImageNet-200 cells are written to
the CSV and the companion. The script needs no GPU.

## Table 2, multi-instance coverage: `tab2_multiinstance.py`

**Produces** `results/camera_ready_tables/tab_multiinst.tex`, the body of Table 2,
plus the companion `cr_214_multiinst.txt`

```
$PYTHON_BIN camera_ready_code/tab2_multiinstance.py
```

**Environment** MOSAIX_ROOT

**Reads** `results/camera_ready_localization/cr_207_scope_summary.csv`,
`results/camera_ready_localization/cr_207_link.csv`

The ID side covers the 363 ImageNet-S300 validation images that are correctly
classified and contain two or more objects of the class. It reports the
fraction of objects lit, where an object counts as lit when the map maximum
inside it reaches half of the global maximum. The OOD side reports the near and
far AUROC of the mosaic from Table 1. The arrows mark a 95% bootstrap CI of the
paired difference against GradCAM. The two input CSVs come from the
working-repository localization runs `cr_205` and `cr_207`, which are not
shipped here.

## Response Figure R2, failure cases: `figR2_failure_mosaics.py`

**Produces** `results/camera_ready_localization/cr_215_failure_mosaics.pdf` (Figure R2 of the response to the reviewers; the paper describes the two cases in the Limitations section),
plus `.png`, `.csv` (the two chosen cases) and the companion `.txt`

```
$PYTHON_BIN camera_ready_code/figR2_failure_mosaics.py
```

**Environment** MOSAIX_ROOT, CE_CKPT, IN1K_VAL, OOD_SSB_HARD, OOD_NINCO,
OOD_INATURALIST, OOD_OPENIMAGE_O, OOD_TEXTURES (through the ImageNet-1K
adapter of scripts/73 and `07_rood_qualify_protocol`). It needs a CUDA GPU.

**Imports** `cr_200_loc_common` and `cr_201_loc_gate` from `$MOSAIX_ROOT/scripts`

**Reads** the frozen per-query GradCAM CSV of scripts/76,
`results/saliency_matrix/76_per_query_cam_methods_im1k_1000samples_20260909-0647.csv`
(named by `cr_201_loc_gate.FROZEN_CAM`)

A fixed rule picks the two cases. Case (a) is the misclassified ID query with
the lowest GradCAM `s`. Case (b) is the SSB-hard or NINCO query with the
highest `s`. The script recomputes the mosaic and the map live, and a gate
checks that the recomputed `s` and the prediction match the frozen CSV.

## Table 8, AugDelete

Table 8 compares the original head of two ResNet-50 models (torchvision V2 and
our mixup) with a refit head; the LS model is measured the same way and quoted in the text. The refit head is a linear layer trained
with plain cross-entropy on the frozen training features, with SGD lr 0.1,
momentum 0.9, weight decay 1e-4, 50 epochs, a cosine schedule, batch 512 and
seed 42. Three scripts produce the table. Run them in this order.

### 1. Refit the head: `tab8_refit_head_sgd.py`

**Produces** `results/camera_ready_augdelete/head_ce_refitsgd__<key>.pt` and
`cr_222_mls__<key>.csv`, the MLS AUROC of each head on the cached OOD features

```
$PYTHON_BIN camera_ready_code/tab8_refit_head_sgd.py --key v2
$PYTHON_BIN camera_ready_code/tab8_refit_head_sgd.py --key ls
$PYTHON_BIN camera_ready_code/tab8_refit_head_sgd.py --key mixup
```

**Environment** MOSAIX_ROOT, LS_FEATDIR and MIXUP_FEATDIR (through
`configs/paths.yaml` of the frozen repository), LS_CKPT and MIXUP_CKPT. For
`--key v2` it also needs V2_FEATDIR and V2_REF_CE_HEAD. It needs a CUDA GPU.

**Imports** `07_rood_qualify_protocol` from `$MOSAIX_ROOT/scripts`

**Reads** `imagenet1k_{train,val}_{emb,lbl}.npy` and `ood/<set>/*_emb.npy`
in each feature directory. For V2 it also reads `trained_fc_ce_retrained.pth`
in V2_FEATDIR and the reference head in V2_REF_CE_HEAD. For LS and mixup it
reads the superseded AdamW heads `head_ce_refit__<key>.pt` and
`head_ce_refit20__<key>.pt` in `results/camera_ready_augdelete/`, which are
compared in the CSV only. The first comes from `tab8_augdelete_eval.py --run heads`.
The second comes from the working-repository script `cr_220`. The script stops
if either file is missing.

### 2. Evaluate the mosaic and MLS: `tab8_augdelete_eval.py`

**Produces** `results/camera_ready_augdelete/auroc__<run>_fullsample.csv` and
`scores4__<run>_fullsample.npz`

```
$PYTHON_BIN camera_ready_code/tab8_augdelete_eval.py --run gate
$PYTHON_BIN camera_ready_code/tab8_augdelete_eval.py --run v2_refitsgd
$PYTHON_BIN camera_ready_code/tab8_augdelete_eval.py --run ls_ckpt
$PYTHON_BIN camera_ready_code/tab8_augdelete_eval.py --run ls_refitsgd
$PYTHON_BIN camera_ready_code/tab8_augdelete_eval.py --run mixup_ckpt
$PYTHON_BIN camera_ready_code/tab8_augdelete_eval.py --run mixup_refitsgd
```

**Environment** MOSAIX_ROOT, LS_CKPT, MIXUP_CKPT, LS_FEATDIR, MIXUP_FEATDIR,
IN1K_TRAIN, IN1K_VAL, IM200_RAW_VAL (the raw validation tree for the V2 runs),
and OOD_*. TV_FEAT_ROOT is optional. It overrides the torchvision feature
directory hard-coded in scripts/64, which the `gate` and `v2_refitsgd` runs
use. The script needs a CUDA GPU.

**Imports** `64_depth_sweep` (which imports `62_mosaic_rood`),
`68_head_refit_figure` and `07_rood_qualify_protocol` from `$MOSAIX_ROOT/scripts`

**Reads** `head_ce_refitsgd__<key>.pt` from step 1. The `gate` run also reads
`results/depth_sweep/64_auroc_tvV2_grad_rn50_1000samples_20260909-0702.csv`.

This script runs `run_arch` of scripts/64 unchanged and swaps from outside only
the backbone, the head and the feature directory. The `gate` run checks that
the wrapper reproduces the earlier V2 measurement to 1e-4. The `ckpt` runs
check the top-1 against the checkpoint.

The script also accepts `--run heads`, `ls_refit`, `mixup_refit`, `ls_refit20`
and `mixup_refit20`. These select AdamW heads: `refit` trains 3 epochs and
`refit20` trains 20. The SGD `refitsgd` head replaced both, and the paper uses
only `refitsgd`. The AdamW options remain so the earlier result files can be
regenerated. `--n 1000` runs a 1000-per-set subsample instead of the full sets.

### 3. Build the table: `tab8_augdelete_table.py`

**Produces** `results/camera_ready_augdelete/tab_augdelete.tex`, the body of
Table 8, plus `cr_218_summary.csv` and the companion `cr_218_summary.txt`

```
$PYTHON_BIN camera_ready_code/tab8_augdelete_table.py
```

**Environment** MOSAIX_ROOT

**Reads** `results/depth_sweep/64_auroc_tvV2_rn50_fullsample_20260909-0036.csv`
(V2 original head, scripts/64) and the six `auroc__*_fullsample.csv` files from
step 2. It checks that every run covers each OOD set in full. If a file is
missing, it stops and writes nothing.

## Response Figure R1, ViT mosaic schematic: `figR1_vit_schematic.py`

**Produces** `figures/vit_schematic/cr_229_vit_mosaic_schematic.pdf` (Figure R1 of the response to the reviewers),
plus `.png` and the companion `.txt`

```
$PYTHON_BIN camera_ready_code/figR1_vit_schematic.py
```

**Environment** MOSAIX_ROOT (output location), IN1K_RAW_TRAIN, IM200_RAW_VAL.
If either of the last two is unset, the script falls back to
`${DATA_ROOT}/imagenet1k/Imagenet/train` and
`${DATA_ROOT}/imagenet1k/Imagenet/val/ILSVRC2012_img_val`.

**Reads** four raw images of class n01531178 (goldfinch). The query is the first
validation image of the class, and the references are the first three training
images in alphabetical order. All four get Resize(256) and CenterCrop(224).

The figure is an illustration and measures nothing. Panel (a) shrinks each tile
to 112 to form a 224 mosaic. Panel (b) keeps the tiles at 224, forms a 448
mosaic and interpolates the position embeddings.

## ResNet-18 ViM row of the full detector table: `vim_principal_rn18.py`

**Produces** `results/vim_principal_rn18/176_auroc_rn18_<timestamp>.csv`, which
`table_code/make_tables.py` reads, plus `176_scores_rn18.npz` and the companion
`176_vim_principal_rn18.txt`

```
$PYTHON_BIN camera_ready_code/vim_principal_rn18.py --limit 64
$PYTHON_BIN camera_ready_code/vim_principal_rn18.py
```

The first command is a smoke run and writes to `results/vim_principal_rn18/smoke/`.

**Environment** MOSAIX_ROOT, IM200_RAW_VAL and OOD_* (through
`configs/paths.yaml` of the frozen repository). TV_FEAT_ROOT is optional. It
overrides the torchvision feature directory hard-coded in scripts/64 and must
contain `resnet18_v1/`. The script falls back to the CPU, but a CUDA GPU is
recommended.

**Imports** `64_depth_sweep` (and through it `62_mosaic_rood`) from
`$MOSAIX_ROOT/scripts`, plus `src.config`, `src.metrics` and `src.detector_maps`
from the frozen repository.

**Reads** `results/depth_sweep/scores__resnet18.npz` and
`results/depth_sweep/auroc.csv` of the full scripts/64 run, which the gates
compare against, and `resnet18_v1/imagenet1k_{train,val}_*.npy`.

The null-space ViM of `src/detector_maps.py` reduces to Energy on the RN-18.
The RN-18 feature has 512 dimensions, fewer than the 1000 classes, so the head
has no null space. This script recomputes ViM in its original form, with the
residual taken outside the principal subspace of the training features. D is
256, set by the rule of the official code before looking at OOD. D = 128 and
D = 384 are reported as sensitivity only. Gates check that the pools, the
transform and the forward pass reproduce the full run before any output is
written.
