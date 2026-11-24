# The kNN versus prototype note

**Produces** `figures/slides_knn/, notes/knn_vs_prototype.pptx`

```
$PYTHON_BIN note_knn_code/knn_vs_prototype.py --n 1000
$PYTHON_BIN note_knn_code/knn_slides.py --ood textures
$PYTHON_BIN note_knn_code/knn_deck.py
```

**Environment** CE_CKPT, CE_FEATDIR, IN1K_TRAIN, IN1K_VAL, the five OOD set variables

**Reads** results/knn_vs_prototype/

Not a paper figure. It explains why the query-adaptive kNN reference policy loses to the fixed prototypes in the design ablation. The written version is notes/knn_vs_prototype.md.
