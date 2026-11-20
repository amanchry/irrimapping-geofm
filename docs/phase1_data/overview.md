# Phase 1 · Data preparation

Phase 1 provides the data on which all later phases rely: the cropland mask that defines the mapped area, labelled
reference parcels, a spatial partition for validation, a common store of monthly imagery from which every model
input is derived, the labelled pixels of the pixel-wise models and the wall-to-wall Sentinel-2 tiles for the maps.
Every model uses Sentinel-2 only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)). The design ensures
that differences between models are attributable to the models rather than to their input data.

## Components

| Component | Content | Status | Page |
| --- | --- | --- | --- |
| **1a · Morocco training data** | 8,781 labelled cropland parcels (7,585 irrigated, 1,196 rainfed); 5 spatial folds of 20 km blocks; 2,987 tiles of 128 × 128 px with 12 monthly composites × 19 bands (2023/24 and 2024/25) | ✅ complete | [Morocco training data](morocco.md) |
| **1b · Utah training data** (second study area, [D13](../decisions.md#d13-utah-as-a-second-independent-study-area)) | 12,015 WRLU crop fields in 550 sampled cells: WRLU 2024 (5,000 irrigated, 1,013 rainfed) and WRLU 2021 for the temporal test (4,987 irrigated, 1,015 rainfed); 5 spatial folds of 20 km blocks; tile plan of 5,436 tiles | ✅ 2023/24 complete (labels, folds, tiles, pixel samples); 2020/21 labels, folds and label rasters prepared for the temporal test | [Utah training data](utah.md) |
| **1d · Cropland extent and mask** | union of 4 cropland products per region and agricultural year ([D17](../decisions.md#d17-cropland-mask-from-four-products-and-a-not-cropland-class)); Morocco 103,068 km², Utah 9,979 km² of cropland (2023/24) | ✅ Morocco 2023/24 and 2024/25, Utah 2023/24 and 2020/21 | [Cropland extent and mask](cropland_mask.md) |
| **1e · Pixel samples** (input of every experiment) | labelled pixels: irrigated, rainfed, not cropland, with folds; Morocco 2023/24 (402,756) and 2024/25 (119,369); Utah 2023/24 (656,081, of which 654,382 in a fold) | ✅ Morocco 2023/24 and 2024/25, Utah 2023/24; Utah 2020/21 to be built for the temporal test | [Pixel samples](pixel_samples.md) |
| **1f · Wall-to-wall Sentinel-2 tiles** (STAC) | 12 monthly Sentinel-2 composites for every 128 px tile with ≥ 1% cropland, from the Earth Search STAC catalogue (`s2_download/`); Utah 18,472 and Morocco 112,536 tiles (2023/24) | 🔄 Utah 2023/24: 17,278 of 18,472 tiles, the rest to be rerun after a bug fix; Morocco on hold | [Cropland extent and mask §5](cropland_mask.md#5-extent-of-the-wall-to-wall-tiles) |
| **1c · Multi-year imagery, Morocco** | tiles for 2021–2025 for the annual maps | 📝 prepared in [Phase 5](../phase5_maps/index.md) | – |

!!! note "Archived: parcel-wise set-up"
    The parcel-wise set-up derived three further views from the tile store (parcel feature tables, pixel sets and
    patch-aligned 64 px chips, `parcelwise_run/04_make_views.py`). These views are no longer used; see the
    [Archive](../archive/index.md).

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| 0 | `cropland_extent/make_cropland_mask.py` | cropland mask per region and agricultural year |
| 01 | `pixelwise_run/01_prepare_labels_Morocco.py` or `01_prepare_labels_Utah.py` | cleaned, labelled cropland parcels (one script per region, same output layout), with their share inside the mask |
| 02 | `pixelwise_run/02_make_spatial_folds.py` | fold (0–4) per parcel |
| 03 | `pixelwise_run/03_build_tiles.py` | tile store from Google Earth Engine |
| 04 | `pixelwise_run/04_make_pixel_samples.py` | labelled pixels for the pixel-wise models |
| 06 | `pixelwise_run/06_build_cropland_tiles.py` | wall-to-wall Sentinel-2 tiles over the cropland mask from Earth Engine (earlier route for the mapping stage) |
| S01 | `s2_download/01_s2_grid_overview.py` | download blocks of 10.24 km over the cropland tiles, the Sentinel-2 tiles over them, overview figures |
| S02 | `s2_download/02_download_s2_stac.py` (cluster wrapper `s2_dl.sbatch`) | monthly Sentinel-2 composites (SCL cloud mask, monthly median) of every tile with ≥ 1% cropland, from the Earth Search STAC catalogue (`sentinel-2-c1-l2a`) |

The archived parcel-wise pipeline (`parcelwise_run/`) uses the same steps 01–03, then `04_make_views.py` (parcel
features and chips) and the parcel model scripts 05–09 ([Archive](../archive/index.md)).

Steps 02–06 serve both regions: set `REGION = "morocco"` or `"utah"` at the top (the STAC download reads the
region from the environment variable `EXPERIMENT_REGION`). The settings
that differ per region (projection, agricultural year, block size, map boundaries) are in `pixelwise_run/regions.py`.
All outputs of a region are in `outputs/<region>/` (`dataset/`, `cropland_extent/`, `s2_download/`, `study/` and,
for Morocco, `analysis/`). The STAC download is documented in `s2_download/README.md`; its tiles
follow the grid and `tile_id` of the cropland mask and serve training, experiments and maps of the STAC-based
approach.

The scripts, design choices, figures and output formats are documented in [Morocco training data](morocco.md) and
[Utah training data](utah.md). The data
providers are listed in [Data sources](../data_source/index.md); departures from the proposal are recorded as
decisions [D1, D2, D4, D9, D13, D14, D16 and D17](../decisions.md).

## Reuse in later phases

The scripts are region- and period-independent: the Utah dataset (1b) is produced with the same scripts 02–04, and
the multi-year imagery (1c) by changing the agricultural years.
