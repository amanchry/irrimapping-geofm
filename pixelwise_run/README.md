# pixelwise_run/

The pixel-wise pipeline (current): every 10 m pixel of the surveyed parcels is classified as **irrigated or
rainfed** (2 classes). The experiments compare models on these labelled pixels only; no cropland mask and no
not-cropland class are used (where cropland is, is a question of the mapping phase: `../cropland_extent/`). Scripts
are numbered in run order; run them from the project folder, settings at the top of each script (`REGION`,
seasons, ...).

| Step | Script | What it does | Output |
| --- | --- | --- | --- |
| 01 | `01_prepare_labels_Morocco.py`, `01_prepare_labels_Utah.py` | labelled parcels of the region (all cropland: irrigated or rainfed) | `outputs/<region>/dataset/01_labels/` |
| 02 | `02_make_spatial_folds.py` | 20 km blocks, 5 spatial folds | `…/dataset/02_folds/` |
| 03 | `03_build_tiles.py` | 128 px tiles around the parcels: 12 monthly Sentinel-2 (+ Sentinel-1) composites and label rasters (Earth Engine) | `…/dataset/03_tiles_128px/agyear/` |
| 03b | `03b_tile_quicklooks.py` | optional: one figure per tile | `…/03_tiles_128px/agyear/quicklooks/` |
| 04 | `04_make_pixel_samples.py` | labelled pixels (irrigated, rainfed) of the parcels with their 12-month series, folds and positions | `…/dataset/04_pixel_samples/<season>/` |
| – | `../experiments/E01…E05/run.py` | the experiments, one at a time | `experiments/<ID>/results/<region>/` |
| 05 | `05_compare_experiments.py` | one table of all finished runs, paired tests (block bootstrap, Holm) | `experiments/comparison/<region>/` |
| 06 | `06_build_cropland_tiles.py` | mapping phase only (not used by the experiments): wall-to-wall tiles over a cropland mask, Earth Engine route; the STAC route is `../s2_download/` | `…/dataset/03c_cropland_tiles_128px/<season>/` |

The earlier 3-class samples (with not-cropland pixels from the cropland mask) are kept unchanged in
`04c_pixel_samples/`; the 2-class samples of step 04 go to `04_pixel_samples/`.

Helper modules (imported, not run; no number because Python cannot import a name that starts with a digit):

| File | Content |
| --- | --- |
| `regions.py` | region settings (projection, agricultural year, boundaries, output folders) |
| `pixel_common.py` | pixel samples, Sentinel-2 features, PASTIS rotation, metrics, block bootstrap, outputs |
| `nn_common.py` | L-TAE, TempCNN, pixel training loop (E02) |
| `tile_common.py` | tile data, foundation-model segmenter + UPerNet head, U-TAE, LoRA, tile training loop (E03–E05) |
| `encoders.py` | the six foundation models: Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA (weights in `checkpoints/<model>/`) |

Steps 01–03 and `regions.py` are also in `../parcelwise_run/` (same scripts, kept with the
archived pipeline).

**Documentation:** https://amanchry.github.io/irrimapping-geofm/experiments/
