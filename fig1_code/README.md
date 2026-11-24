# Figure 1, the teaser

**Produces** `figures/teaser/fig1_teaser_inline.pdf`

```
$PYTHON_BIN fig1_code/fig1_teaser.py --bench in1k --layout inline --norm panel
```

**Environment** CIFAR_TORCHVISION_ROOT and TIN_WORDS, both only for class names. This script loads no checkpoint: it reads the frozen grid dumps. CE_CKPT, LS_CKPT and MIXUP_CKPT are needed only if you regenerate those dumps, which is done by scripts/43 in the working repository.

**Reads** results/grid_dump/grids__{ce,ls,mixup}.npz

One ID query and one far-OOD query predicted into the same class, so both mosaics use the same three references. A gate in the script checks that. The colour scale is common to the ten images, so the query tile going dark in the OOD block is not a rendering artefact.
