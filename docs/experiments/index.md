# Experiments · pixel-wise classification

The experiments classify **every 10 m pixel** as irrigated or rainfed, so the trained model can produce an annual,
continuous map of the whole area, not only of the surveyed parcels.

!!! info "Status: planned, data being prepared"
    The plan is on the [experiment plan](plan.md) page. Pixel samples: Morocco 2023/24 ready; Morocco 2024/25 and
    Utah on the cluster. Experiments run one at a time on the cluster, starting with P01.

The earlier parcel-wise experiments (E01–E05, one label per parcel) are in the
[Archive](../archive/index.md): [results](../archive/parcel_wise_classification/results.md) ·
[experiments](../archive/parcel_wise_classification/index.md).

## Experiment list

| ID | Experiment | Input | Model | Status |
| --- | --- | --- | --- | --- |
| P01 | Sentinel-2 pixel baseline | pixel time series, 12 months × 12 bands | random forest, gradient boosting | 📝 next |
| P02 | Temporal neural networks | pixel time series | TempCNN, L-TAE | 📝 planned |
| P03 | AlphaEarth pixel embeddings | 64-d annual embedding | random forest, MLP | 📝 planned |
| P04 | Frozen GeoFMs + decoder (RQ1) | 128 px tiles | Prithvi, Clay, TerraMind, AgriFM + UPerNet | 📝 planned |
| P05 | U-TAE from scratch | 128 px tiles | U-TAE | 📝 planned |
| P06 | Fine-tuning (RQ2) | 128 px tiles | best P04 encoder + LoRA | 📝 planned |

Common set-up, metrics and run order: [experiment plan](plan.md). Data: [pixel samples](../phase1_data/pixel_samples.md),
[cropland mask](../phase1_data/cropland_mask.md).
