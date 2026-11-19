# ml_dataset_prep/

Part 1 · dataset: labels, spatial folds, tile store and model inputs (scripts 01–04), for two regions.
The cropland mask that scripts 01, 03c and 04c read is made first in [`cropland_extent/`](../cropland_extent/README.md)
(outputs in `outputs/<region>/cropland_extent/`).

| Script | Region |
| --- | --- |
| `01_prepare_labels_Morocco.py`, `01_prepare_labels_Utah.py` | one label script per region (same output layout) |
| `02_make_spatial_folds.py`, `03_build_tiles.py`, `04_make_views.py` | both: set `REGION = "morocco"` or `"utah"` at the top |
| `03b_tile_quicklooks.py` | both: figures of tiles from the tile store for a visual check |
| `03c_build_cropland_tiles.py` | both: wall-to-wall Sentinel-2 tiles over the cropland mask (same grid as 03), for the maps |
| `04c_make_pixel_samples.py` | both: labelled pixels (irrigated / rainfed / not cropland) with folds, for pixel-wise models |
| `regions.py` | settings that differ per region (projection, agricultural year, block size, map boundaries, mapping area) |

Outputs: `outputs/<region>/dataset/` (`01_labels`, `02_folds`, `03_tiles_128px`, `03c_cropland_tiles_128px`, `04c_pixel_samples`).

Each script has its settings at the top and a description of what it does and writes.

**Documentation:** [https://amanchry.github.io/irrimapping-geofm/phase1_data/morocco/](https://amanchry.github.io/irrimapping-geofm/phase1_data/morocco/) (source: [`docs/phase1_data/morocco.md`](../docs/phase1_data/morocco.md))
