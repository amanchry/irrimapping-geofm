# data_download/ — every data source, once per region and agricultural year, on one block grid

All data are downloaded **once** per region and agricultural year on one grid of **10.24 km blocks** (1024 × 1024 px
at 10 m) in the region's projection, **block by block**. The blocks cover the whole mapping area of the region, without
a mask. Dynamic World comes first, as binary cropland for every block (with a VRT mosaic of the region per year); the other sources are downloaded only for
blocks with cropland (or with training tiles). The same blocks serve training (the 128 px training
tiles are cut out of them by `data_training_prep/04_cut_tiles.py`), testing and the maps (each block is classified on its
own, on its cropland pixels, and the block maps are merged at the end). Each source has its own script and folder, so a
new source (e.g. ERA5 for Presto) is one more `02x` script on the same blocks. Sentinel-2 and Sentinel-1 come from
open STAC catalogues (no account, no quota; `stac_common.py`); Google Earth Engine (service-account key in `secrets/`,
never commit it) is used only for Dynamic World (02a, 04c).

| Step | Script | What it does | Output (`outputs/<region>/data_download/…`) |
| --- | --- | --- | --- |
| 01 | `01_plan_blocks.py` | the block grid, **once per region** (the same blocks every year): every block of the mapping area (no mask, no training data; the training tiles per block are counted from the current tile plan when 02b/02c load the grid, with a warning for a training tile outside it); map of the block extents over the area | `grid/blocks.parquet, .geojson, .png` |
| 02a | `02a_dynamic_world.py` | binary cropland of **every planned block** (no mask), one Earth Engine request (`ee.data.computePixels`) and one 10 m GeoTIFF per block and agricultural year: `CROPLAND` = 1 where "crops" is the most frequent Dynamic World class in at least 2 of the 12 months of the agricultural year, else 0 (uint8, 255 = outside the mapping area or no image); resumable (finished blocks kept, training blocks first, stops after 20 failed blocks in a row); a VRT mosaic per year at the end; per year a map of the cropland share (320 m cells) and a map of the blocks with and without cropland; `DOWNLOAD = False` only rebuilds the mosaic and redraws the maps | `dynamic_world/<season>/<bx>_<by>.tif`, `dynamic_world/<season>/mosaic.vrt`, `dynamic_world/cropland_map_<season>.png`, `dynamic_world/cropland_blocks_<season>.png` |
| 02b | `02b_s2.py` | Sentinel-2 L2A monthly composites (12 bands + clear-observation count) from **Earth Search** (STAC, `sentinel-2-c1-l2a`): offset of Collection 1 removed (reflectance × 10,000), SCL cloud mask, monthly median; only blocks with cropland (≥ 164 cropland pixels, 1% of a 128 px tile) or training tiles | `s2/<season>/<bx>_<by>.tif` |
| 02c | `02c_s1.py` | Sentinel-1 monthly composites, ascending and descending (VV, VH, scene count) from **Planetary Computer** (STAC, `sentinel-1-rtc`: gamma0, radiometrically terrain corrected), mean in linear power, dB × 100; same blocks | `s1/<season>/<bx>_<by>.tif` |
| 03 | `03_build_vrt.py` | one virtual mosaic per source and year (no copy of the data) | `<source>/<season>/mosaic.vrt` |
| 04a | `04a_ref_esa_worldcover.py` | reference map: ESA WorldCover 10 m of the region + PNG (Planetary Computer) | `outputs/<region>/data_download/reference/esa_worldcover/` |
| 04b | `04b_ref_lgrip30.py` | reference map: LGRIP30 irrigated / rainfed cropland 30 m + PNG (NASA Earthdata token) | `outputs/<region>/data_download/reference/lgrip30/` |
| 04c | `04c_ref_dynamic_world_annual.py` | reference map: annual Dynamic World of the whole region as one 10 m GeoTIFF (`geemap.download_ee_image`, needs `geedim` < 2): one band `label` (all 9 classes; cropland = crops, as in 02a; about 11 GB uncompressed for Morocco, much less on disk), optional `crop_freq` for a 50% threshold; PNG of the land cover with legend and PNG of the cropland | `outputs/<region>/data_download/reference/dynamic_world/` |

`archive/` holds the archived union cropland mask of four products (`00_union_cropland_mask.py`); it is not used, the
annual Dynamic World cropland of `02a` is the only cropland layer.

Shared modules: `stac_common.py` (STAC search and reading of scenes onto a block), `regions.py` (projection, agricultural year, mapping area and boundaries per region; the only copy,
used by every folder) and `download_common.py` (block grid, file layout, the cropland check of a block, Earth Engine login for Dynamic World, the
per-block download; `download_blocks(..., require_cropland=True)` keeps only blocks with cropland or training tiles).

```bash
python data_download/01_plan_blocks.py             # first; independent of data_training_prep/
python data_download/02a_dynamic_world.py          # binary cropland of every block, resumable; VRT mosaic at the end
python data_download/02b_s2.py                     # blocks with cropland; resumable; N_TEST_BLOCKS = 1 for a test
python data_download/02c_s1.py
python data_download/03_build_vrt.py
```

Every script has `REGION` (and `SEASONS`) in its settings block. Run the whole-region downloads on the cluster.

**Grid.** Block (bx, by) covers x = bx · 10,240 m … + 10,240 m and y = by · 10,240 m … + 10,240 m in the region's
projection; it holds the 128 px tiles ix = 8 bx … 8 bx + 7, iy = 8 by … 8 by + 7 of the training data, so every
training tile and every map window is an exact cut-out of a block.

**Reading from STAC.** Only the window of the block is read from each scene's cloud-optimised GeoTIFF, resampled
onto the block grid (nearest neighbour); a failed read is retried, then the scenes are read one by one, and if more
than 2% of a block's scenes cannot be read the block is not written (a rerun fetches it). The band names, order,
scaling and NODATA are those of the earlier Earth Engine download, so the training tiles and the models read the
blocks unchanged.

**Check against Earth Engine** (Morocco tile 2023-24_384_2795, 2023/24, all 12 months): Sentinel-2 agrees with the
earlier Earth Engine composite (correlation 0.98–1.00 per band and month, median difference 0 in most months; about
10% fewer clear observations, as SCL is stricter than Cloud Score+). Sentinel-1 RTC is gamma0, about 1.3 dB (VV) and
3 dB (VH) above Earth Engine's sigma0, and Planetary Computer lacks about 10% of the acquisitions (10 of about 95 for
this tile); only models trained on these data read Sentinel-1 (the foundation models use Sentinel-2, D14).

**Size and time** (one Morocco block, 2023/24, on a laptop): Sentinel-2 from STAC 958 s (239 scenes); with Earth
Engine it was 164 s (Sentinel-2) and 92 s (Sentinel-1). Dynamic World: 13 s per block (two blocks in parallel, Earth Engine
project in restricted mode). Morocco has 4,199 blocks over its mapping area (414,906 km²), Utah 2,213 (219,809 km²);
Dynamic World is fetched for all of them, Sentinel-2 and Sentinel-1 only for those with cropland.

**Weather and other coarse sources.** ERA5 and similar data (about 9–11 km per pixel) use the same blocks with a coarse
pixel (`process_blocks(..., size_px=…)`, e.g. 11 × 11 values per block) and are resampled when a model reads them.
