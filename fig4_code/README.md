# Figure 4, the head refit

**Produces** `figures/head_refit/119_fig5_headrefit_4col.pdf`

```
$PYTHON_BIN fig4_code/fig4_head_refit.py
```

**Environment** CE_CKPT, LS_CKPT, IN1K_TRAIN, and LS_FEATDIR if the refit head is not cached.

**Reads** both backbones live, plus the refit head cached at `$MOSAIX_ROOT/results/head_refit/head_ce_refit__ls.pt`. That file is in the checkpoint download, under `head_refit/`. Without it the script refits the head from the LS training features, which takes longer and needs LS_FEATDIR.

The LS and LS+AugDelete columns share a frozen backbone, so the difference between them is only the reading direction. The CE column is a different backbone. The two comparisons are not the same. Needs MOSAIX_ROOT.
