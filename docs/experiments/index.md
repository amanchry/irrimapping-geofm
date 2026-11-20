# Experiments

The experiments classify **every 10 m pixel** as irrigated or rainfed, so the trained model can produce an annual,
continuous map of the whole area, not only of the surveyed parcels.

!!! success "Status: E01–E04 done in both regions"
    Results are on the experiment pages; the plan is on the [experiment plan](plan.md) page. E03 covers Prithvi,
    Clay, TerraMind and AgriFM; Presto and TESSERA are still to be run. Pixel samples: Morocco 2023/24 and 2024/25,
    Utah 2023/24.

## Experiment list

| ID | Experiment | Input | Model | Status |
| --- | --- | --- | --- | --- |
| [E01](E01_s2_pixel_baseline.md) | Sentinel-2 pixel baseline | pixel time series, 12 months × 12 bands | random forest, gradient boosting | ✅ done: cropland macro-F1 0.876 Morocco, 0.729 Utah (RF) |
| [E02](E02_temporal_nn.md) | Temporal neural networks | pixel time series | TempCNN, L-TAE | ✅ done: L-TAE 0.890 Morocco, 0.741 Utah (+0.01, not significant) |
| [E03](E03_frozen_geofm.md) | Frozen GeoFMs + decoder (RQ1) | 128 px tiles | Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA + UPerNet | ✅ done (4 of 6 encoders): best 0.900 Morocco (AgriFM, not significant), 0.797 Utah (TerraMind, +0.068 significant) |
| [E04](E04_utae.md) | U-TAE from scratch | 128 px tiles | U-TAE | ✅ done: 0.897 Morocco (not significant), 0.766 Utah (+0.037 significant) |
| [E05](E05_prithvi_strategies.md) | Prithvi frozen, fully fine-tuned or from scratch, with 10–100% of the training parcels (RQ2) | 128 px tiles | Prithvi-EO-2.0 + L-TAE + UPerNet (as E03) | ⏳ code ready, Utah not run yet |

Fine-tuning with LoRA (low-rank adapters) is not part of E05; `add_lora` in `tile_common.py` is ready for a later run.

Common set-up, metrics and run order: [experiment plan](plan.md). Data: [pixel samples](../phase1_data/pixel_samples.md),
[cropland mask](../phase1_data/cropland_mask.md).

## Code

| File | Content |
| --- | --- |
| [`pixelwise_run/pixel_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/pixel_common.py) | pixel samples, Sentinel-2 features, PASTIS rotation, metrics, block bootstrap, outputs (all experiments) |
| [`pixelwise_run/nn_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/nn_common.py) | L-TAE, TempCNN, pixel training loop (E02) |
| [`pixelwise_run/tile_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/tile_common.py) | tile data, foundation-model segmenter (frozen, LoRA, fine-tuned or from scratch), UPerNet head, U-TAE, label subsets, tile training loop (E03–E05) |
| [`pixelwise_run/encoders.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/encoders.py) | the six foundation models: Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA (E03) |
| [`pixelwise_run/05_compare_experiments.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/05_compare_experiments.py) | one table of all finished runs of E01–E04 and paired tests (block bootstrap, Holm) against the reference runs (E01 random forest, E04 U-TAE); E05 has its own tests (`experiments/E05_prithvi_strategies/report.py`) |

Each experiment folder `experiments/E0<N>_<name>/` holds its `run.py` (settings at the top) and, after a run,
`results/<region>/<run>/`. Region, encoder and quick-test mode can be set per Slurm job with
`EXPERIMENT_REGION`, `EXPERIMENT_ENCODER` and `EXPERIMENT_QUICK_TEST`.
