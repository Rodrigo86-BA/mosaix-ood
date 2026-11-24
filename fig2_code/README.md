# Figure 2, the method diagram

**Produces** `figures/method/fig2_method.pdf`

```
$PYTHON_BIN fig2_code/fig2_method_data.py --key ce --query id
$PYTHON_BIN fig2_code/fig2_method.py
```

**Environment** CE_CKPT, CE_FEATDIR, IN1K_TRAIN. `--key` selects the backbone and defaults to `ce`; `ls` and `mixup` read LS_CKPT and MIXUP_CKPT through configs/paths.yaml.

**Reads** results/method_fig/, results/ref_bank/bank__ce.npz

`fig2_method_data.py` runs the whole readout path live and dumps every image, map and number the diagram shows. `fig2_method.py` only draws them. Nothing in the figure is illustrative.
