# s2_download — Sentinel-2 monthly composites from STAC (no Earth Engine)

Wall-to-wall Sentinel-2 imagery for the continuous maps: 12 monthly composites per 128 px tile over the whole
cropland mask, read from a public STAC catalogue (Earth Search, `sentinel-2-c1-l2a`, no account). The tiles have the
same grid, `tile_id` and file format as the Earth Engine tiles (`pixelwise_run/03_build_tiles.py`,
`06_build_cropland_tiles.py`), so the experiment and mapping code read them unchanged.

| Step | Script | What it does |
| --- | --- | --- |
| 01 | `01_s2_grid_overview.py` | Cropland blocks (10.24 km = 8 × 8 tiles of 128 px), the Sentinel-2 tiles over them, overview figures |
| 02 | `02_download_s2_stac.py` | Per block and agricultural year: all scenes → SCL cloud mask → monthly median + clear count → 128 px tiles. `TILE_SET = "parity"` (parcel-store tiles of 8 blocks spread over the region) or `"cropland"` (all cropland tiles) |
| 03 | `03_parity_check.py` | STAC vs Earth Engine on the same tiles: reflectance bias / RMSE per band and month, cloud-mask agreement |

Figures: `figures/` (`<region>_s2_blocks_tiles.png`, explained in `figures/figures_explained.txt`).

## Units

| Unit | Size | Role | Morocco | Utah |
| --- | --- | --- | --- | --- |
| Sentinel-2 tile (MGRS) | 109.8 km | how the images are published | 66 | 41 |
| Block | 10.24 km, 1024 px | one download job | 3,143 | 1,307 |
| Model tile | 1.28 km, 128 px | stored, modelled, predicted | 112,536 | 18,472 |

(2023/24 mask, tiles with ≥ 1% cropland.)

## Compositing (matches the Earth Engine tiles except the cloud mask)

| Item | STAC (this folder) | Earth Engine tiles |
| --- | --- | --- |
| Scenes | Earth Search `sentinel-2-c1-l2a`, `eo:cloud_cover` < 60 | `COPERNICUS/S2_SR_HARMONIZED`, `CLOUDY_PIXEL_PERCENTAGE` < 60 |
| Offset | −1000 for processing baseline ≥ 04.00 (values < 0 → 0) | harmonised collection |
| Cloud mask | SCL not in {0, 1, 2, 3, 8, 9, 10} | Cloud Score+ `cs_cdf` ≥ 0.60 |
| Resampling to 10 m | nearest | nearest (default) |
| Composite | monthly median of clear observations; `S2_NCLEAR` = count | same |
| Overlapping Sentinel-2 tiles | every scene counts | every scene counts |

Unreadable scenes are retried, then logged per block (`blocks_done.jsonl`); a block with more than 2% unreadable
scenes is skipped and logged in `blocks_failed.jsonl` (no tiles written, so a rerun fetches it again).

Run order: 01 → 02 with `TILE_SET = "parity"` → 03; if the sources agree (and the model scores hold on both inputs),
02 with `TILE_SET = "cropland"` on the cluster (job array: `S2_PART`, `S2_N_PARTS`). Outputs:
`outputs/<region>/s2_download/{grid,parity,cropland}/<season>/`.
