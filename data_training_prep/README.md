# data_training_prep/ — labels, spatial folds and the 128 px training tiles

Turns the field labels into training data, shared by the pixel-wise pipeline (`pixelwise_run/`) and the archived
parcel-wise pipeline (`parcelwise_run/`). The images are not downloaded here: they come from the 10.24 km blocks of
`data_download/`, and only the tiles around the labelled parcels are cut out as files.

| Step | Script | What it does | Output (`outputs/<region>/data_training_prep/…`) |
| --- | --- | --- | --- |
| 01 | `01_prepare_labels_Morocco.py`, `01_prepare_labels_Utah.py` | labelled parcels of the region (all cropland: irrigated or rainfed) | `01_labels/` |
| 02 | `02_make_spatial_folds.py` | 20 km blocks, 5 spatial folds | `02_folds/` |
| 03 | `03_plan_tiles.py` | 128 px tile plan around the parcels (own tiles + context), label rasters, tile index; no download | `03_tiles_128px/agyear/` |
| 04 | `04_cut_tiles.py` | cuts the planned tiles out of the downloaded blocks: `<tile_id>_img.tif` (Sentinel-2 + Sentinel-1, 228 bands, the file every model reads), `<tile_id>_dw.tif` (Dynamic World, annual + monthly); QA; refreshes the index | `03_tiles_128px/agyear/tiles/` |
| 05 | `05_tile_quicklooks.py` | optional: one figure per tile (visual check) | `03_tiles_128px/agyear/quicklooks/` |

`tile_store.py` holds the paths of the tile store; the grid and `regions.py` are imported from `data_download/`.

```bash
python data_training_prep/01_prepare_labels_Morocco.py   # or 01_prepare_labels_Utah.py
python data_training_prep/02_make_spatial_folds.py
python data_training_prep/03_plan_tiles.py
#   then data_download/01_plan_blocks.py, 02a_dynamic_world.py, 02b_s2.py, 02c_s1.py
python data_training_prep/04_cut_tiles.py
python pixelwise_run/04_make_pixel_samples.py       # pixel samples of the experiments
```

**Same tiles as before.** `04_cut_tiles.py` writes the 228-band layout of the tiles downloaded earlier per tile. A test
on a downloaded Morocco block (2026-11-24) gave tiles identical in values, band names and position to the existing
training tiles; tiles that already exist are kept (`OVERWRITE = False`), so the earlier training tiles need no new
download. `03_plan_tiles.py` reproduces `parcels.csv` and the label rasters exactly.
