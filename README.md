# MosAIx, VISAPP 2027

This repository holds the scripts that produce the figures and tables of the
paper, the figures and tables themselves, and the configuration needed to run
those scripts on another machine. The paper source is not included.

> [!WARNING]
> **The in-distribution images are read from a pre-cropped tree.**
>
> `IN1K_VAL` points at `imagenet1k_pre/clean/val`, an ImageNet-1K validation
> tree that was already resized and center-cropped to 224x224 on disk. Every
> measurement in the paper reads it that way. The OOD sets and the training
> tree are stored raw and take `Resize(256)` and `CenterCrop(224)` at load
> time.
>
> `shared_code/rood_qualify_protocol.py` exposes this as
> `load_batch(paths, pre=False)`. `pre=True` selects `TFM_PRE`, which only
> converts and normalizes. `pre=False` selects `TFM_RAW`, which crops first.
> The call sites pass `pre=True` for in-distribution queries and `pre=False`
> everywhere else.
>
> Cropping the pre-cropped tree a second time changes 16 percent of the
> predictions, and nothing in the code stops it. The top-1 gate in
> `head_gate` compares the classifier head against the cached features, so it
> never sees the live image path.
>
> If you point `IN1K_VAL` at a raw validation tree instead, invert the flag:
> pass `pre=False` for the in-distribution queries as well, or the images
> reach the backbone at their original size.

## Layout

One folder per figure, one for the tables, one for what they share.

```
fig1_code/      the teaser
fig2_code/      the method diagram
fig3_code/      the saliency zoo
fig4_code/      the head refit
fig5_code/      the placement ablation
fig6_code/      the cross-benchmark plot
table_code/     every table, plus the tab:field measurement
shared_code/    the query pools, loaders and geometry the others import
note_knn_code/  the kNN versus prototype note, not a paper figure
camera_ready_code/  the tables and figures added in the camera-ready version

figures/        the figures the paper includes, as PDF
tables/         the generated .tex table bodies
configs/        the .env reader and the paths.yaml template
notes/          analyses that did not fit in the paper
supplementary/  supplementary material (transformer backbones)
```

The supplementary material on [transformer backbones](supplementary/transformers/README.md)
holds the figures and tables of every transformer experiment.

Each `*_code` folder has a README naming what it produces, the command, and
the environment variables it needs.

The camera-ready version adds two tables (joint versus separate forward passes;
objects lit and AUROC on multi-instance images), two figures (the ViT mosaic
schematic and the failure cases), and replaces the AugDelete table with three
ResNet-50 models (torchvision V2, label smoothing and mixup). Their code is in
[camera_ready_code](camera_ready_code/README.md).

Two of the figure folders need `MOSAIX_ROOT` set, because they import
attribution modules that live in the working repository rather than here.

## Downloading the checkpoints

The scripts here load three ResNet-50 checkpoints trained from scratch on
ImageNet-1K, one per loss, all at seed 0. They are shared at

https://drive.google.com/drive/folders/1KFOULKRAiJyvdVRWMaBMvPscxtG2RHHs?usp=sharing

The runs with seeds 1 and 2 are not shared. The folder holds 301 MB: the three
`last.pt` files, about 98 MB each, their training logs, and the refit head that
Figure 4 reads. It has one subfolder per checkpoint. With `<download>` as the
directory you saved it to, the three lines in `.env` are

```
CE_CKPT=<download>/resnet50_full_ce_s0/last.pt
LS_CKPT=<download>/resnet50_full_ls_standard_a0.1_s0/last.pt
MIXUP_CKPT=<download>/resnet50_mixup1.0_s0/last.pt
```

Copy `head_refit/head_ce_refit__ls.pt` to `$MOSAIX_ROOT/results/head_refit/`.
Inside the folder, `sha256sum -c SHA256SUMS` checks the download, and
`MANIFEST.txt` lists what each file is.

Google Drive warns that the three `last.pt` files are too large for it to scan
for viruses. `Download anyway` fetches them.

The curated GAP features behind `CE_FEATDIR`, `LS_FEATDIR` and `MIXUP_FEATDIR`
come to 46 GB and are not shared. They are derived from these checkpoints.

## Setting the environment variables

The environment variables are only needed to re-run the scripts. Building the
paper does not use them.

### Step 1, copy the template

```
cp .env.example .env
```

`.env.example` lists every variable with a comment saying which figure needs
it. `.env` is in `.gitignore`, so your local paths stay out of the repository.

### Step 2, fill in the paths

Open `.env` and set each variable to a path on your machine. The variables fall
into five groups.

**Repository roots.** `MOSAIX_ROOT` is the working repository, the one that
holds `scripts/`, `results/` and `figures/`. `XAI_OOD_ROOT` is the frozen
source repository, which holds `src/` and the frozen `results/`. In the working
repository the frozen one is reachable as the symlink `external_xai_ood`.

**Datasets.** `DATA_ROOT` is the directory that contains every dataset tree.
The other dataset variables are written relative to it, so setting `DATA_ROOT`
correctly usually sets them all. `IN1K_TRAIN` and `IN1K_VAL` are the ImageNet-1K
trees. `OOD_SSB_HARD` and `OOD_NINCO` are the near-OOD sets. `OOD_INATURALIST`,
`OOD_OPENIMAGE_O` and `OOD_TEXTURES` are the far-OOD sets.

**Checkpoints.** `CE_CKPT`, `LS_CKPT` and `MIXUP_CKPT` point at the three
ResNet-50 checkpoints trained from scratch on ImageNet-1K, one per loss. Each
is a plain state dict named `last.pt`, and the download link is under
[Downloading the checkpoints](#downloading-the-checkpoints). `CE_FEATDIR`,
`LS_FEATDIR` and `MIXUP_FEATDIR` point at the curated GAP features for the
same three backbones.

**Smaller benchmarks.** `IM200_*` and `CIFAR_*` are needed only to regenerate
the CSVs behind Figure 6. Figure 6 itself reads those CSVs and does not touch
the datasets.

**Tools.** `PYTHON_BIN` is the interpreter the scripts run under. `TORCH_HOME`
is where torchvision caches the public weights it downloads. Set `TORCH_HOME`
to a disk with a few gigabytes free.

You can write one variable in terms of another. The reader expands `${NAME}`
against variables defined above it in the file:

```
DATA_ROOT=/data
OOD_TEXTURES=${DATA_ROOT}/dtd
```

### Step 3, check what you set

```
$PYTHON_BIN configs/render_paths.py --check
```

This prints every variable with `OK` or `--` depending on whether the path
exists, then prints each figure with the variables and files it needs. A figure
marked `OK` can be produced on this machine. A figure marked `--` lists what is
missing under `MISSING env` or `MISSING artifacts`.

### Step 4, render the configuration file

```
$PYTHON_BIN configs/render_paths.py
```

This writes `configs/paths.yaml` from `configs/paths.yaml.template`, replacing
each `${NAME}` with the value from `.env`. It then runs the same check as
step 3.

### Step 5, point the frozen repository at it

The scripts do not read `.env`. They resolve their own root as the parent of
`scripts/`, and the ones that need machine-local paths read
`external_xai_ood/configs/paths.yaml` through `src/config.py`. Copy the
rendered file there:

```
cp configs/paths.yaml $XAI_OOD_ROOT/configs/paths.yaml
```

`shared_code/rood_qualify_protocol.py` is the exception. It already reads
`CE_CKPT`, `CE_FEATDIR` and `CE_TOP1` from the environment and overrides the
yaml with them. The variable names in `.env` follow that convention.

## What the scripts need beyond this repository

Reproducing a figure needs three things this repository does not contain:

1. The frozen source repository at `XAI_OOD_ROOT`, for `src/` and for the
   frozen CSVs that Figure 5 reads.
2. The working repository at `MOSAIX_ROOT` with its `results/` tree. That tree
   holds the grid dumps, the reference banks and the CSVs behind Figures 1, 2,
   3 and 6.
3. `scripts/frozen_xai_ood/` in the working repository, which several of these
   scripts import by number.

`configs/render_paths.py --check` reports each of these as a missing artifact
instead of letting a script fail part way through.

`FIGURES.md` gives the command, the variables and the artifacts for each of the
six figures.

## Notes

`notes/knn_vs_prototype.md` explains why the query-adaptive kNN reference
policy loses to the fixed prototypes in the design ablation. It renders on
GitHub, with both figures inline. The same material as a deck is
`notes/knn_vs_prototype.pptx`, which GitHub offers for download rather than
rendering.

## Data and models

The datasets follow the OpenOOD v1.5 protocol. The in-distribution set is the
ImageNet-1K validation split. The near-OOD sets are SSB-hard and NINCO. The
far-OOD sets are iNaturalist, OpenImage-O and Textures. The two smaller
benchmarks are ImageNet-200 and CIFAR-100, both with a ResNet-18.

There are two families of backbones, and the tables state which one they use.

The first family is trained from scratch: ResNet-50 on ImageNet-1K, one recipe
and three losses, with seeds 0, 1 and 2. The losses are cross-entropy, label
smoothing at 0.1, and mixup with Beta(1,1). Validation top-1 is 75.51 +/- 0.13,
76.19 +/- 0.19 and 74.89 +/- 0.43. The three recipes are within a point and a
half of each other, so differences downstream come from what the representation
encodes rather than from classification quality. The paths are `CE_CKPT`,
`LS_CKPT` and `MIXUP_CKPT`.

The second family is the public torchvision `IMAGENET1K_V1` weights, used by
`tab:field` and by scripts 164, 167 and 168. These download on demand into
`TORCH_HOME`.
