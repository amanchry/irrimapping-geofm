# Phase 1 · Data preparation

Phase 1 provides the data on which all later phases rely: labelled reference parcels, a spatial partition for
validation, and a common store of Sentinel-1/2 imagery from which every model input is derived. The design ensures
that differences between models are attributable to the models rather than to their input data.

## Components

| Component | Content | Status | Page |
| --- | --- | --- | --- |
| **1a · ML-ready dataset, Morocco** | 8,781 labelled cropland parcels (7,585 irrigated, 1,196 rainfed); 5 spatial folds of 20 km blocks; 2,987 tiles of 128 × 128 px with 12 monthly composites × 19 bands; parcel features, pixel sets and patch-aligned chips | ✅ complete | [ML-ready dataset](dataset.md) |
| **1b · Transfer-region dataset** | reference labels and the same tile store for the region of RQ4 (candidates: eastern Washington, Utah, Tunisia) | 📝 prepared at the start of [Phase 3](../phase3_transfer/index.md) | – |
| **1c · Multi-year imagery, Morocco** | tile store for 2021–2025 for the annual maps of RQ3 | 📝 prepared in [Phase 5](../phase5_maps/index.md) | – |

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 1 | `ml_dataset_prep/01_prepare_labels.py` | cleaned, labelled cropland parcels |
| 2 | `ml_dataset_prep/02_make_spatial_folds.py` | fold (0–4) per parcel |
| 3 | `ml_dataset_prep/03_build_tiles.py` | tile store from Google Earth Engine |
| 4 | `ml_dataset_prep/04_make_views.py` | model inputs (views A–C) |
| check | `models_run/01_inspect_dataset.py` | dataset inspection figures |

The scripts, design choices, figures and output formats are documented in [ML-ready dataset](dataset.md). The data
providers are listed in [Data sources](../data_source/index.md); departures from the proposal are recorded as
decisions [D1, D2, D4 and D9](../decisions.md).

## Reuse in later phases

The scripts are region- and period-independent: the transfer-region dataset (1b) and the multi-year imagery (1c) are
produced with the same pipeline by changing the input labels, the study area and the agricultural years at the top
of each script.
