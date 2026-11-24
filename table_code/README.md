# Every table in the paper

**Produces** `tables/*.tex`

```
$PYTHON_BIN table_code/make_tables.py
```

**Environment** none for make_tables.py; see below for the tab:field measurement

**Reads** $XAI_OOD_ROOT/results/

`make_tables.py` writes all the table bodies from the frozen CSVs. One table has its own measurement code: `tab:field` comes from `tab_field_posemb.py` (max-logit under three conditions, no mosaic) and `tab_field_mosaic.py` (the mosaic at two sizes). Those two need IN1K_VAL, the five OOD set variables and TORCH_HOME, and they use the public torchvision weights rather than the from-scratch backbones. Run them before make_tables.py if you want to regenerate that table.
