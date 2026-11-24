# Phase 1 · Data preparation

Phase 1 provides the data on which all later phases rely: labelled reference parcels, a spatial partition for
validation, a common store of monthly imagery from which every model input is derived, the labelled pixels of the
pixel-wise models and, for the mapping phase, the annual Dynamic World cropland and the
wall-to-wall imagery. All imagery is downloaded once per region and agricultural year on one grid of 10.24 km blocks;
the training tiles are cut from these blocks and the maps are predicted from them
([D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)). The labels, the pixel samples and the experiments do not use a cropland mask; the Dynamic World cropland serves the
download and the maps only.
Every model uses Sentinel-2 only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)). The design ensures
that differences between models are attributable to the models rather than to their input data.

## Components

| Component | Content | Status | Page |
| --- | --- | --- | --- |
| **1a · Morocco training data** | 8,781 labelled cropland parcels (7,585 irrigated, 1,196 rainfed); 5 spatial folds of 20 km blocks; 2,987 tiles of 128 × 128 px with 12 monthly composites × 19 bands (2023/24 and 2024/25) | ✅ complete | [Morocco training data](morocco.md) |
| **1b · Utah training data** (second study area, [D13](../decisions.md#d13-utah-as-a-second-independent-study-area)) | 12,015 WRLU crop fields in 550 sampled cells: WRLU 2024 (5,000 irrigated, 1,013 rainfed) and WRLU 2021 for the temporal test (4,987 irrigated, 1,015 rainfed); 5 spatial folds of 20 km blocks; tile plan of 5,436 tiles | ✅ 2023/24 labels, folds and tiles complete; pixel samples to be rebuilt with the two-class script; 2020/21 labels, folds and label rasters prepared for the temporal test | [Utah training data](utah.md) |
| **1d · Cropland extent** | annual Dynamic World cropland per region and agricultural year (`data_download/02a_dynamic_world.py`: "crops" the most frequent class in at least 2 of the 12 months), the only cropland layer: it selects the downloaded blocks and the pixels classified on the maps; the experiments do not use it. The earlier union of 4 cropland products ([D17](../decisions.md#d17-cropland-mask-from-four-products-and-a-not-cropland-class)) is archived ([Union cropland mask](../archive/cropland_union_mask.md)) | 🔄 Dynamic World cropland; union mask archived | [Cropland extent](cropland_extent.md) |
| **1e · Pixel samples** (input of every experiment) | labelled pixels of the surveyed parcels, two classes (irrigated, rainfed), with folds (`04_pixel_samples/`); Morocco 2023/24 (367,882) and 2024/25 (105,793): 473,675 pixels of 8,775 parcels in 2,015 tiles | ✅ Morocco 2023/24 and 2024/25; ⏳ Utah 2023/24 to be rebuilt with the two-class script; Utah 2020/21 to be built for the temporal test | [Pixel samples](pixel_samples.md) |
| **1f · Wall-to-wall imagery** (10.24 km blocks) | binary Dynamic World cropland of the whole mapping area (one file per year); 12 monthly composites of Sentinel-2 and Sentinel-1 for every block with ≥ 164 cropland pixels (1% of a 128 px tile) and every block holding training tiles (`data_download/`, Earth Engine; [D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)). Replaces the retired STAC download of Sentinel-2 ([D18](../decisions.md#d18-wall-to-wall-composites-from-stac-for-the-maps)) | 🔄 block download tested on one Morocco block (2023/24); region-wide downloads to follow | [Cropland extent §2](cropland_extent.md#2-blocks-and-downloads) |
| **1c · Multi-year imagery, Morocco** | imagery for 2021–2025 for the annual maps (the block download of 1f for further agricultural years) | 📝 prepared in [Phase 5](../phase5_maps/index.md) | – |

!!! note "Archived: parcel-wise set-up"
    The parcel-wise set-up derived three further views from the tile store (parcel feature tables, pixel sets and
    patch-aligned 64 px chips, `parcelwise_run/04_make_views.py`). These views are no longer used; see the
    [Archive](../archive/index.md).

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| T01 | `data_training_prep/01_prepare_labels_Morocco.py` or `01_prepare_labels_Utah.py` | cleaned, labelled cropland parcels (one script per region, same output layout) |
| T02 | `data_training_prep/02_make_spatial_folds.py` | fold (0–4) per parcel |
| T03 | `data_training_prep/03_plan_tiles.py` | tile plan, label rasters and tile index of the 128 px training tiles (no download) |
| D01 | `data_download/01_plan_blocks.py` | every 10.24 km block of the region's mapping area (no cropland mask), planned once per region, the same blocks for every year, without training data (the training tiles per block are counted from the tile plan when 02b/02c load the grid) (`outputs/<region>/data_download/grid/blocks.parquet`, `.geojson`; map of the block extents `blocks.png`); Morocco: 4,199 blocks (414,906 km² mapping area), of which 169 (2023/24) and 65 (2024/25) hold training tiles |
| D02a | `data_download/02a_dynamic_world.py` | binary cropland of the whole region, one 10 m GeoTIFF per agricultural year downloaded with `geemap`: CROPLAND = 1 where "crops" is the most frequent Dynamic World class in at least 2 of the 12 months of the agricultural year, else 0 (uint8, 255 = no image); every block and training tile is a window of it (`outputs/<region>/data_download/dynamic_world/cropland_<season>.tif`) |
| D02b–c | `data_download/02b_s2.py`, `02c_s1.py` | monthly Sentinel-2 composites (Cloud Score+) and monthly Sentinel-1 (ascending, descending), only for the blocks with cropland (≥ 164 Dynamic World cropland pixels, 1% of a 128 px tile) or with training tiles (`outputs/<region>/data_download/<source>/<season>/<bx>_<by>.tif`); later sources (ERA5) follow the same rule |
| D03 | `data_download/03_build_vrt.py` | one VRT mosaic per source and agricultural year |
| T04 | `data_training_prep/04_cut_tiles.py` | the 128 px training tiles (`_img.tif`, 228 bands; `_dw.tif`, binary Dynamic World cropland) cut from the downloaded blocks |
| T05 | `data_training_prep/05_tile_quicklooks.py` | quick-look figures of selected tiles |
| 04 | `pixelwise_run/04_make_pixel_samples.py` | labelled pixels (irrigated, rainfed) of the surveyed parcels for the pixel-wise models (`04_pixel_samples/<season>/`) |
| D04a–c | `data_download/04a_ref_esa_worldcover.py`, `04b_ref_lgrip30.py`, `04c_ref_dynamic_world_annual.py` | reference maps: ESA WorldCover 2021, LGRIP30, annual Dynamic World layer per calendar year: all 9 classes in one band (cropland = crops), one 10 m GeoTIFF per region via `geemap` (`outputs/<region>/data_download/reference/`) |

The run order is D01 and D02a (with the cropland maps of each year; both independent of the labels), T01–T03, D02b, D02c, D03, T04–T05, then 04: D02b and D02c read the tile plan when they load the grid, so the blocks of the training tiles are
always downloaded, and the training tiles are cut from the downloaded blocks. Tiles cut from a downloaded
block were checked to be identical (values, band names, position) to the training tiles downloaded earlier tile by
tile, so the existing training tiles are kept. Only the training area gets 128 px tile files; for the maps, each
block is classified on its own (cropland pixels only: tile models read 128 px windows from the block, pixel models
read each pixel's series) and the block maps are merged at the end
([D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)). The earlier routes to wall-to-wall imagery (Sentinel-2 from the Earth Search STAC
catalogue, `s2_download/`, and the Earth Engine tile route `pixelwise_run/06_build_cropland_tiles.py`) are retired; the
union cropland mask of four products is archived (`data_download/archive/00_union_cropland_mask.py`).

The archived parcel-wise pipeline (`parcelwise_run/`) uses the labels, folds and tiles of `data_training_prep/`, then
`04_make_views.py` (parcel features and chips) and the parcel model scripts 05–09
([Archive](../archive/index.md)).

All scripts serve both regions: set `REGION = "morocco"` or `"utah"` at the top. The settings that differ per region
(projection, agricultural year, block size of the folds, map boundaries) are in `data_download/regions.py`, the only
copy; the block grid and the download loop are in `data_download/download_common.py`. All outputs of a region are
in `outputs/<region>/` (`dataset/`, `download/`, `cropland_extent/`, `study/` and, for Morocco, `analysis/`).

The scripts, design choices, figures and output formats are documented in [Morocco training data](morocco.md) and
[Utah training data](utah.md). The data
providers are listed in [Data sources](../data_source/index.md); departures from the proposal are recorded as
decisions [D1, D2, D4, D9, D13, D14, D16, D17 and D20](../decisions.md).

## Reuse in later phases

The scripts are region- and period-independent: the Utah dataset (1b) is produced with the same scripts (only the label script is region-specific), and
the multi-year imagery (1c) by changing the agricultural years.
