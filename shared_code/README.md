# Shared by several of the folders above

**Produces** `nothing on its own`

```
not run directly
```

**Environment** CE_CKPT, CE_FEATDIR, CE_TOP1 are read as overrides

**Reads** $XAI_OOD_ROOT/configs/paths.yaml

> [!WARNING]
> `load_batch(paths, pre=False)` decides the transform. `pre=True` skips the
> crop and is the correct flag for `IN1K_VAL`, which is stored already cropped
> to 224x224. `pre=False` applies `Resize(256)` and `CenterCrop(224)` and is
> the correct flag for the OOD sets and the training tree. Cropping the
> pre-cropped tree twice changes 16 percent of the predictions, and no
> assertion in this file catches it.

`rood_qualify_protocol.py` holds the query pools, the image loading, the backbone loader and the mosaic geometry, and is imported by the figure and table code. `transformer_quarter.py` holds the checkpoint-specific transforms and the quarter-field construction, and is imported by the tab:field measurement.
