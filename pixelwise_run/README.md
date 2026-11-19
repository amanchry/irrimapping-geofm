# pixelwise_run/

The pixel-wise pipeline (current): every 10 m pixel is classified as irrigated, rainfed or not cropland, for the
annual continuous maps. Scripts are numbered in run order; run them from the project folder, settings at the top of
each script (`REGION`, seasons, ...).

| Step | Script | What it does | Output |
| --- | --- | --- | --- |
| 0 | `../cropland_extent/make_cropland_mask.py` | cropland mask per region and year (union of 4 products, Earth Engine); needed by steps 01, 04, 06 | `outputs/<region>/cropland_extent/cropland_mask/` |
| 01 | `01_prepare_labels_Morocco.py`, `01_prepare_labels_Utah.py` | labelled parcels of the region, mask check | `outputs/<region>/dataset/01_labels/` |
| 02 | `02_make_spatial_folds.py` | 20 km blocks, 5 spatial folds | `…/dataset/02_folds/` |
| 03 | `03_build_tiles.py` | 128 px tiles around the parcels: 12 monthly Sentinel-2 (+ Sentinel-1) composites and label rasters (Earth Engine) | `…/dataset/03_tiles_128px/agyear/` |
| 03b | `03b_tile_quicklooks.py` | optional: one figure per tile | `…/03_tiles_128px/agyear/quicklooks/` |
| 04 | `04_make_pixel_samples.py` | labelled pixels (irrigated, rainfed, not cropland) with their 12-month series, folds and positions | `…/dataset/04c_pixel_samples/<season>/` |
| – | `../experiments/E01…E05/run.py` | the experiments, one at a time | `experiments/<ID>/results/<region>/` |
| 05 | `05_compare_experiments.py` | one table of all finished runs, paired tests (block bootstrap, Holm) | `experiments/comparison/<region>/` |
| 06 | `06_build_cropland_tiles.py` | later, for the maps: wall-to-wall tiles over the cropland mask | `…/dataset/03c_cropland_tiles_128px/<season>/` |

The output folder names are those of the earlier script numbers (`04c_pixel_samples`, `03c_cropland_tiles_128px`)
so that existing data on the Mac and the cluster stays valid.

Helper modules (imported, not run; no number because Python cannot import a name that starts with a digit):

| File | Content |
| --- | --- |
| `regions.py` | region settings (projection, agricultural year, boundaries, output folders) |
| `cropland_mask.py` | mask statistics per parcel (step 01) |
| `pixel_common.py` | pixel samples, Sentinel-2 features, PASTIS rotation, metrics, block bootstrap, outputs |
| `nn_common.py` | L-TAE, TempCNN, pixel training loop (E02) |
| `tile_common.py` | tile data, foundation-model segmenter + UPerNet head, U-TAE, LoRA, tile training loop (E03–E05) |
| `encoders.py` | the six foundation models: Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA (weights in `checkpoints/<model>/`) |

Steps 01–03 and `regions.py` / `cropland_mask.py` are also in `../parcelwise_run/` (same scripts, kept with the
archived pipeline).

**Documentation:** https://amanchry.github.io/irrimapping-geofm/experiments/
