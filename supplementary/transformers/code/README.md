# Code of the transformer experiments

| script | section of the page | reads | writes |
|---|---|---|---|
| `fig_adaptations.py` | 3 and 8, the two adaptations and the example maps | the query pools | `figures/transformer_adapt/` |
| `pe_continuation.py` | 4 | torchvision weights only | `results/pe_continuation/`, `figures/pe_continuation/` |
| `pe_mosaic_positions.py` | 5 to 8 | the output of `pe_continuation.py`, `tab_field_mosaic.py` and `tab_field_posemb.py` | `results/pe_mosaic_positions/` |
| `fig_pe_positions.py` | 7 and 8 | the output of `pe_mosaic_positions.py` | `figures/pe_positions/` |

Sections 2 and 3 also use `shared_code/transformer_quarter.py`,
`table_code/tab_field_posemb.py` and `table_code/tab_field_mosaic.py`.

Every path is a variable:

| variable | meaning | default |
|---|---|---|
| `MOSAIX_ROOT` | directory that holds `results/` and `figures/` | the root of this repository |
| `XAI_OOD_ROOT` | source repository with `src/` | `$MOSAIX_ROOT/external_xai_ood` |
| `MAN168`, `AUC168` | manifest and AUROC file written by `tab_field_mosaic.py` | the names of the original run |
| `AUC167` | AUROC file written by `tab_field_posemb.py` | the name of the original run |

`shared_code/rood_qualify_protocol.py` reads `external_xai_ood/` at the root of
this repository, so link it there once:

```
ln -s $XAI_OOD_ROOT external_xai_ood
```

Run `pe_continuation.py`, then `pe_mosaic_positions.py`, then the two figures.
