# Experiments

The experiments classify **every 10 m pixel** as irrigated or rainfed, so the trained model can produce an annual,
continuous map of the whole area, not only of the surveyed parcels.

!!! info "Status: scripts ready, runs pending"
    The plan is on the [experiment plan](plan.md) page. Pixel samples are ready for Morocco 2023/24 and 2024/25 and
    Utah 2023/24. Every script was checked on a small subset; the experiments run one at a time on the cluster,
    starting with E01, and only `results/` is copied back.


## Experiment list

| ID | Experiment | Input | Model | Status |
| --- | --- | --- | --- | --- |
| [E01](E01_s2_pixel_baseline.md) | Sentinel-2 pixel baseline | pixel time series, 12 months × 12 bands | random forest, gradient boosting | ✅ done: cropland macro-F1 0.876 Morocco, 0.729 Utah (RF) |
| [E02](E02_temporal_nn.md) | Temporal neural networks | pixel time series | TempCNN, L-TAE | ✅ done: L-TAE 0.890 Morocco, 0.741 Utah (+0.01, not significant) |
| [E03](E03_frozen_geofm.md) | Frozen GeoFMs + decoder (RQ1) | 128 px tiles | Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA + UPerNet | ✅ ready |
| [E04](E04_utae.md) | U-TAE from scratch | 128 px tiles | U-TAE | ✅ ready |

Fine-tuning with LoRA (RQ2) is deferred and will be added after E01–E04.

Common set-up, metrics and run order: [experiment plan](plan.md). Data: [pixel samples](../phase1_data/pixel_samples.md),
[cropland mask](../phase1_data/cropland_mask.md).

## Code

| File | Content |
| --- | --- |
| [`pixelwise_run/pixel_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/pixel_common.py) | pixel samples, Sentinel-2 features, PASTIS rotation, metrics, block bootstrap, outputs (all experiments) |
| [`pixelwise_run/nn_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/nn_common.py) | L-TAE, TempCNN, pixel training loop (E02) |
| [`pixelwise_run/tile_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/tile_common.py) | tile data, foundation-model segmenter, UPerNet head, U-TAE, LoRA, tile training loop (E03, E04) |
| [`pixelwise_run/encoders.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/encoders.py) | the six foundation models: Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA (E03) |
| [`pixelwise_run/05_compare_experiments.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/05_compare_experiments.py) | one table of all finished runs and paired tests (block bootstrap, Holm) against a reference run |

Each experiment folder `experiments/E0<N>_<name>/` holds its `run.py` (settings at the top) and, after a run,
`results/<region>/<run>/`. Region, encoder and quick-test mode can be set per Slurm job with
`EXPERIMENT_REGION`, `EXPERIMENT_ENCODER` and `EXPERIMENT_QUICK_TEST`.
