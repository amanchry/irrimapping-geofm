# Comparison of the experiments

!!! info "Status: planned (Morocco first, Utah later)"
    This page will hold all runs of E01–E04 side by side once they are finished. Each experiment page reports only
    its own runs; E05 (Prithvi frozen / LoRA / fine-tuned / from scratch) is compared within itself on its
    [own page](E05_prithvi_strategies.md).

## What will be compared

All runs of a region are scored on the **same out-of-fold pixels**: Morocco 2023/24 and 2024/25 pooled, 473,675
labelled pixels of 8,775 surveyed parcels, two classes (irrigated, rainfed).

| Run | Experiment | Model |
| --- | --- | --- |
| E01 random forest | [E01](E01_s2_pixel_baseline.md) | pixel time series, random forest (**reference**) |
| E01 gradient boosting | [E01](E01_s2_pixel_baseline.md) | pixel time series, histogram gradient boosting |
| E02 TempCNN | [E02](E02_temporal_nn.md) | pixel time series, temporal CNN |
| E03 U-TAE | [E03](E03_utae.md) | whole tiles, trained from scratch (**reference** for E04) |
| E04 Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA | [E04](E04_frozen_geofm.md) | whole tiles, frozen foundation model + L-TAE + UPerNet |

## Planned tables and figures

| Content | How |
| --- | --- |
| **Scores** | macro-F1 (irrigated, rainfed; headline), rainfed F1, precision and recall, with 95% block-bootstrap CI |
| **Paired tests against the E01 random forest** | does a model beat the pixel baseline? Difference on the same pixels, 95% CI, Holm-corrected p |
| **Paired tests against E03 U-TAE** | does pretraining add anything beyond the spatial context of the tile? |
| **Figures** | scores of all runs with CIs; confusion matrices of the random forest, U-TAE and the best foundation model |

A difference counts when its 95% CI excludes 0 and the Holm-corrected p < 0.05 ([evaluation protocol](../phase2_models/evaluation.md)).
Tables and tests: `pixelwise_run/05_compare_experiments.py` (`experiments/comparison/<region>/`); figures:
`experiments/report_figures.py` (`docs/experiments/figures/`).

## Results

_To be filled when E01–E04 are finished._
