# Experiments

The experiments classify the **10 m pixels of the surveyed parcels** as irrigated or rainfed, comparing models and
ways of adapting foundation models; the best model will later produce the annual, continuous maps (where cropland is
is decided in the mapping phase, [D19](../decisions.md#d19-two-class-experiments-cropland-extent-in-the-mapping-phase)).

!!! success "Status: Morocco done, all experiments E01–E05 (22 November 2026); Utah next"
    Redesigned on 20 November 2026: two classes, no cropland mask, earlier 3-class results removed. Code ready and
    tested; the plan is on the [experiment plan](plan.md) page. Pixel samples: Morocco 2023/24 and 2024/25
    (473,675 pixels of 8,775 parcels).

## Experiment list

| ID | Experiment | Input | Model | Status |
| --- | --- | --- | --- | --- |
| [E01](E01_s2_pixel_baseline.md) | Sentinel-2 pixel baseline | pixel time series, 12 months × 12 bands | random forest, gradient boosting | Morocco done: macro-F1 0.894 (RF), 0.893 (GB) |
| [E02](E02_temporal_nn.md) | Temporal neural network | pixel time series | TempCNN | Morocco done: macro-F1 0.912 |
| [E03](E03_utae.md) | U-TAE from scratch | 128 px tiles | U-TAE | Morocco done: macro-F1 0.900 |
| [E04](E04_frozen_geofm.md) | Frozen GeoFMs + decoder (RQ1) | 128 px tiles | Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA + L-TAE + UPerNet | Morocco done: best TerraMind 0.927, then Clay 0.923, AgriFM 0.921, Prithvi 0.917 |
| [Comparison](comparison.md) | all runs of E01–E04 side by side, paired tests (RQ1) | | | |
| [E05](E05_prithvi_strategies.md) | Prithvi frozen, LoRA, fully fine-tuned or from scratch, with 10–100% of the training parcels (RQ2) | 128 px tiles | Prithvi-EO-2.0 + L-TAE + UPerNet (as E04) | Morocco done: frozen, LoRA, fine-tuned 0.90–0.92 at every share; from scratch 0.80–0.83 |

Common set-up, metrics and run order: [experiment plan](plan.md). Data: [pixel samples](../phase1_data/pixel_samples.md).

## Code

| File | Content |
| --- | --- |
| [`pixelwise_run/pixel_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/pixel_common.py) | pixel samples, Sentinel-2 features, PASTIS rotation, metrics, block bootstrap, outputs (all experiments) |
| [`pixelwise_run/nn_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/nn_common.py) | L-TAE, TempCNN, pixel training loop (E02) |
| [`pixelwise_run/tile_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/tile_common.py) | tile data, foundation-model segmenter (frozen, LoRA, fine-tuned or from scratch), UPerNet head, U-TAE, label subsets, tile training loop (E03–E05) |
| [`pixelwise_run/encoders.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/encoders.py) | the six foundation models: Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA (E04) |
| [`pixelwise_run/05_compare_experiments.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/05_compare_experiments.py) | one table of all finished runs of E01–E04 and paired tests (block bootstrap, Holm) against the reference runs (E01 random forest, E03 U-TAE); E05 has its own tests (`experiments/E05_prithvi_strategies/report.py`) |

Each experiment folder `experiments/E0<N>_<name>/` holds its `run.py` (settings at the top) and, after a run,
`results/<region>/<run>/`. Region, encoder and quick-test mode can be set per Slurm job with
`EXPERIMENT_REGION`, `EXPERIMENT_ENCODER` and `EXPERIMENT_QUICK_TEST`.
