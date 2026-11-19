# s2_download — Sentinel-2 monthly composites from STAC

Wall-to-wall Sentinel-2 imagery: 12 monthly composites per 128 px tile over the whole cropland mask, read from a
public STAC catalogue (Earth Search, `sentinel-2-c1-l2a`, no account). This is the imagery of the STAC-based
approach: training tiles, experiments and maps all use these tiles. The tiles follow the grid and `tile_id` of the
cropland mask (`cropland_extent/make_cropland_mask.py`).

| Step | Script | What it does |
| --- | --- | --- |
| 01 | `01_s2_grid_overview.py` | Cropland blocks (10.24 km = 8 × 8 tiles of 128 px), the Sentinel-2 tiles over them, overview figures |
| 02 | `02_download_s2_stac.py` | Per block and agricultural year: all scenes → SCL cloud mask → monthly median + clear count → 128 px tiles, for every tile with ≥ 1% cropland |

Figures: `figures/` (`<region>_s2_blocks_tiles.png`, explained in `figures/figures_explained.txt`).

## Units

| Unit | Size | Role | Morocco | Utah |
| --- | --- | --- | --- | --- |
| Sentinel-2 tile (MGRS) | 109.8 km | how the images are published | 66 | 41 |
| Block | 10.24 km, 1024 px | one download job | 3,143 | 1,307 |
| Model tile | 1.28 km, 128 px | stored, modelled, predicted | 112,536 | 18,472 |

(2023/24 mask, tiles with ≥ 1% cropland. Morocco 2024/25: 113,319 tiles in 3,220 blocks.)

## Compositing

| Item | Setting |
| --- | --- |
| Scenes | Earth Search `sentinel-2-c1-l2a`, `eo:cloud_cover` < 60 |
| Offset | −1000 for processing baseline ≥ 04.00 (values < 0 → 0) |
| Cloud mask | SCL not in {0, 1, 2, 3, 8, 9, 10} |
| Resampling to 10 m | nearest |
| Composite | monthly median of clear observations; `S2_NCLEAR` = count |
| Overlapping Sentinel-2 tiles | every scene counts |

Unreadable scenes are retried, then logged per block (`blocks_done.jsonl`); a block with more than 2% unreadable
scenes is skipped and logged in `blocks_failed.jsonl` (no tiles written, so a rerun fetches it again).

## Run

Years: Morocco 2023/24 and 2024/25, Utah 2023/24 (`SEASONS` in the script; `S2_SEASONS` overrides it per job).
On the cluster as a job array, one share of the blocks per task (compute nodes reach the STAC hosts through the
proxy); the script is resumable, finished tiles are skipped:

```bash
EXPERIMENT_REGION=utah S2_PART=<k> S2_N_PARTS=<n> python s2_download/02_download_s2_stac.py
```

Output: `outputs/<region>/s2_download/cropland/<season>/` (`tiles/`, `tiles_qa.jsonl`, `blocks_done.jsonl`,
`blocks_failed.jsonl`, `tiles_info.json`); grid overview: `outputs/<region>/s2_download/grid/<season>/`.
