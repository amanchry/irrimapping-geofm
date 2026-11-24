# data_download/ — every data source, once per region and agricultural year, on one block grid

All data are downloaded **once** per region and agricultural year on one grid of **10.24 km blocks** (1024 × 1024 px
at 10 m) in the region's projection, **block by block**. The blocks cover the whole mapping area of the region, without
a mask. Dynamic World comes first, as one binary cropland image per block; the other sources are downloaded only for
blocks with cropland (or with training tiles). The same blocks serve training (the 128 px training
tiles are cut out of them by `data_training_prep/04_cut_tiles.py`), testing and the maps (each block is classified on its
own, on its cropland pixels, and the block maps are merged at the end). Each source has its own script and folder, so a
new source (e.g. ERA5 for Presto) is one more `02x` script on the same blocks. Downloads use Google Earth Engine with the
service-account key in `secrets/` (never commit it).

| Step | Script | What it does | Output (`outputs/<region>/data_download/…`) |
| --- | --- | --- | --- |
| 01 | `01_plan_blocks.py` | the block grid, **once per region** (the same blocks every year): every block of the mapping area (no mask), with the number of training tiles per block and year (warning if a training tile lies outside the area); map of the block extents over the area | `grid/blocks.parquet, .geojson, .png` |
| 02a | `02a_dynamic_world.py` | binary cropland image for **every** block: `CROPLAND` = 1 where the most frequent Dynamic World class of the calendar year in which the agricultural year ends is crops, else 0 (uint8, 255 = no image); per year a map of the cropland share (320 m cells) and a map of the blocks with and without cropland; `DOWNLOAD = False` only redraws the maps | `dynamic_world/<season>/<bx>_<by>.tif`, `dynamic_world/cropland_map_<season>.png`, `dynamic_world/cropland_blocks_<season>.png` |
| 02b | `02b_s2.py` | Sentinel-2 L2A monthly composites (12 bands + clear-observation count; Cloud Score+), only blocks with cropland (≥ 164 cropland pixels, 1% of a 128 px tile) or training tiles | `s2/<season>/<bx>_<by>.tif` |
| 02c | `02c_s1.py` | Sentinel-1 GRD monthly composites, ascending and descending (VV, VH, scene count), same blocks | `s1/<season>/<bx>_<by>.tif` |
| 03 | `03_build_vrt.py` | one virtual mosaic per source and year (no copy of the data) | `<source>/<season>/mosaic.vrt` |
| 04a | `04a_ref_esa_worldcover.py` | reference map: ESA WorldCover 10 m of the region + PNG (Planetary Computer) | `outputs/<region>/data_download/reference/esa_worldcover/` |
| 04b | `04b_ref_lgrip30.py` | reference map: LGRIP30 irrigated / rainfed cropland 30 m + PNG (NASA Earthdata token) | `outputs/<region>/data_download/reference/lgrip30/` |
| 04c | `04c_ref_dynamic_world_annual.py` | reference map: annual Dynamic World of the whole region as one 10 m GeoTIFF (`geemap.download_ee_image`, needs `geedim` < 2): one band `label` (all 9 classes; cropland = crops, as in 02a; about 11 GB uncompressed for Morocco, much less on disk), optional `crop_freq` for a 50% threshold; PNG of the land cover with legend and PNG of the cropland | `outputs/<region>/data_download/reference/dynamic_world/` |
| 00 | `00_union_cropland_mask.py` | union of 4 cropland products (ESA WorldCover, Esri LULC, GCEP30, GLAD); **kept for later, not used** by the download | `outputs/<region>/data_download/reference/cropland_mask/<season>/` |

Shared modules: `regions.py` (projection, agricultural year, mapping area and boundaries per region; the only copy,
used by every folder) and `download_common.py` (block grid, file layout, the cropland check of a block, Earth Engine login, the
per-block download; `download_blocks(..., require_cropland=True)` keeps only blocks with cropland or training tiles).

```bash
python data_download/01_plan_blocks.py             # after data_training_prep/03_plan_tiles.py
python data_download/02a_dynamic_world.py          # binary cropland image of every block
python data_download/02b_s2.py                     # blocks with cropland; resumable; N_TEST_BLOCKS = 1 for a test
python data_download/02c_s1.py
python data_download/03_build_vrt.py
```

Every script has `REGION` (and `SEASONS`) in its settings block. Run the whole-region downloads on the cluster.

**Grid.** Block (bx, by) covers x = bx · 10,240 m … + 10,240 m and y = by · 10,240 m … + 10,240 m in the region's
projection; it holds the 128 px tiles ix = 8 bx … 8 bx + 7, iy = 8 by … 8 by + 7 of the training data, so every
training tile and every map window is an exact cut-out of a block.

**Request size.** Earth Engine limits one request to about 50 MB and counts about 3 bytes per int16 pixel, so a block
is fetched in groups of 9 bands (Sentinel-2: 18 requests per block and year).

**Size and time** (measured on one Morocco block, 2023/24, one worker): Sentinel-2 159 MB and 164 s, Sentinel-1
82 MB and 92 s, Dynamic World (binary) 3–25 KB and about 6 s. Morocco has 4,199 blocks over its mapping area (414,906 km²), Utah 2,213 (219,809 km²);
Dynamic World is fetched for all of them, Sentinel-2 and Sentinel-1 only for those with cropland.

**Weather and other coarse sources.** ERA5 and similar data (about 9–11 km per pixel) use the same blocks with a coarse
pixel (`download_blocks(..., size_px=…)`, e.g. 11 × 11 values per block) and are resampled when a model reads them.
