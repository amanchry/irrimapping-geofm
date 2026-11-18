# ml_dataset_prep/

Part 1 · dataset: cropland mask, labels, spatial folds, tile store and model inputs (scripts 00–04), for two regions.

| Script | Region |
| --- | --- |
| `00_make_cropland_mask.py` | both: cropland mask (union of 4 cropland products, Earth Engine) for the wall-to-wall maps; run first |
| `01_prepare_labels_Morocco.py`, `01_prepare_labels_Utah.py` | one label script per region (same output layout) |
| `02_make_spatial_folds.py`, `03_build_tiles.py`, `04_make_views.py` | both: set `REGION = "morocco"` or `"utah"` at the top |
| `03b_tile_quicklooks.py` | both: figures of tiles from the tile store for a visual check |
| `regions.py` | settings that differ per region (projection, agricultural year, block size, map boundaries, mapping area) |

Outputs: `outputs/<region>/dataset/` (`00_cropland_mask`, `01_labels`, `02_folds`, `03_tiles_128px`).

Each script has its settings at the top and a description of what it does and writes.

**Documentation:** [https://amanchry.github.io/irrimapping-geofm/phase1_data/morocco/](https://amanchry.github.io/irrimapping-geofm/phase1_data/morocco/) (source: [`docs/phase1_data/morocco.md`](../docs/phase1_data/morocco.md))
