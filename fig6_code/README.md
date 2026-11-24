# Figure 6, the cross-benchmark plot

**Produces** `figures/crossplot_bench/157_crossplot_bench__cifar224_col.pdf`

```
$PYTHON_BIN fig6_code/fig6_crossplot.py
```

**Environment** none. Optional: `CR_NOCHECK=1` skips the assertion that no
text falls outside the canvas.

**Reads** results/table2_multi/auroc.csv, results/cifar100_224_panel/auroc.csv

Frozen CSVs only. The upstream runs that produced them needed IM200_* and CIFAR_*.

The figure is drawn at its printed size (2.953 x 1.760 in, one SCITEPRESS
column) with 7 to 7.5 pt text, and saved without `bbox_inches="tight"`, so the
PDF enters the paper at 1:1. It always draws the ImageNet-200 panel and the
CIFAR-100 at 224 panel; there are no command-line options.
