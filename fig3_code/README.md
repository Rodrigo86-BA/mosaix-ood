# Figure 3, the saliency zoo

**Produces** `figures/head_refit/149_f1__saliency_zoo.pdf`

```
$PYTHON_BIN fig3_code/fig3_saliency_zoo.py --replot
```

**Environment** none for `--replot`. A new search (`--find-class 412`, or no
flag) needs CE_CKPT, IN1K_TRAIN, OOD_INATURALIST, OOD_OPENIMAGE_O,
OOD_TEXTURES. The backbone is resolved by the ImageNet-1K adapter of scripts/73
in the working repository, which loads the CE cell.

**Reads** results/saliency_zoo/winner.pt

Saliency methods reading the same mosaic and the same predicted class. Needs
MOSAIX_ROOT set, because it imports attribution modules from the working
repository (`29_im200_models`, `30_attribution_methods`,
`73_attribution_multi`, `76_saliency_matrix`, `128_seam_full`).

The figure is drawn at its printed size, 6.221 in wide (the SCITEPRESS text
width), with 7.5 pt titles on two lines (method, then `s`) and a 7 pt colour
scale, and saved without `bbox_inches="tight"`, so it enters the paper at 1:1.
`--replot` reads the cache before building the model, so the redraw needs no
GPU. A new search never overwrites `results/saliency_zoo/winner.pt`; it writes
`results/camera_ready/149_winner.pt` instead.
