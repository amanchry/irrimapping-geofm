# Phase 1 · Data preparation

Phase 1 provides the data on which all later phases rely: labelled reference parcels, a spatial partition for
validation, and a common store of Sentinel-1/2 imagery from which every model input is derived. The design ensures
that differences between models are attributable to the models rather than to their input data.

## Components

| Component | Content | Status | Page |
| --- | --- | --- | --- |
| **1a · Morocco training data** | 8,781 labelled cropland parcels (7,585 irrigated, 1,196 rainfed); 5 spatial folds of 20 km blocks; 2,987 tiles of 128 × 128 px with 12 monthly composites × 19 bands; parcel features, pixel sets and patch-aligned chips | ✅ complete | [Morocco training data](morocco.md) |
| **1b · Utah training data** (second study area; temporal test) | 12,015 WRLU crop fields in 550 sampled cells: WRLU 2024 training year (5,000 irrigated, 1,013 rainfed) and WRLU 2021 test year (4,987 irrigated, 1,015 rainfed); 5 spatial folds of 20 km blocks; tile plan of 5,436 tiles | 🔄 labels, folds and tile plan done; download and views to run | [Utah training data](utah.md) |
| **1c · Multi-year imagery, Morocco** | tile store for 2021–2025 for the annual maps of RQ3 | 📝 prepared in [Phase 5](../phase5_maps/index.md) | – |

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1 | `ml_dataset_prep/01_prepare_labels_Morocco.py` or `01_prepare_labels_Utah.py` | cleaned, labelled cropland parcels (one script per region, same output layout) |
| 2 | `ml_dataset_prep/02_make_spatial_folds.py` | fold (0–4) per parcel |
| 3 | `ml_dataset_prep/03_build_tiles.py` | tile store from Google Earth Engine |
| 4 | `ml_dataset_prep/04_make_views.py` | model inputs (views A–C) |
| check | `models_run/01_inspect_dataset.py` | dataset inspection figures |

Scripts 02–04 and the inspection serve both regions: set `REGION = "morocco"` or `"utah"` at the top. The settings
that differ per region (projection, agricultural year, block size, map boundaries) are in `ml_dataset_prep/regions.py`.
All outputs of a region are in `outputs/<region>/` (`dataset/`, and for Morocco also `study/` and `analysis/`).

The scripts, design choices, figures and output formats are documented in [Morocco training data](morocco.md) and
[Utah training data](utah.md). The data
providers are listed in [Data sources](../data_source/index.md); departures from the proposal are recorded as
decisions [D1, D2, D4 and D9](../decisions.md).

## Reuse in later phases

The scripts are region- and period-independent: the Utah dataset (1b) is produced with the same scripts 02–04, and
the multi-year imagery (1c) by changing the agricultural years.
