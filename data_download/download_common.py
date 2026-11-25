"""
download_common.py  —  block grid, file layout, Earth Engine login and the per-block processing loop (not run alone)
==================================================================================================================

Every data source is downloaded ONCE per region and agricultural year on one grid of 10.24 km blocks
(1024 x 1024 px at 10 m), in the region's projection (regions.py). The grid is the 128 px tile grid of the training
data, grouped 8 x 8: block (bx, by) holds the tiles ix = 8 bx ... 8 bx + 7, iy = 8 by ... 8 by + 7, so every training
tile and every map window is an exact cut-out of a block.

    outputs/<region>/data_download/
        grid/blocks.parquet, .geojson               the block grid over the whole mapping area, the same for every
                                                    year (01_plan_blocks.py); the training tiles per block are counted
                                                    when the blocks are loaded (load_blocks)
        dynamic_world/<season>/<bx>_<by>.tif        binary Dynamic World cropland of every block (02a_dynamic_world.py)
        dynamic_world/<season>/mosaic.vrt           VRT mosaic of these blocks (whole region)
        <source>/<season>/<bx>_<by>.tif             one file per block, 12 months x bands of the source
        <source>/<season>/<bx>_<by>_scenes.json     scene IDs per month (provenance), where available

A new source (e.g. ERA5) is one more script 02x_<source>.py: it defines compute(ag_season, bx, by) -> one block and
calls process_blocks(), which selects the blocks, skips finished ones, runs blocks in parallel and writes the files.
Sentinel-2 and Sentinel-1 come from open STAC catalogues (stac_common.py); Earth Engine (service-account key in
secrets/, KEY_FILE; never commit it) is used only for Dynamic World (02a, 04c).
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from regions import dataset_dir, settings

ROOT = Path(__file__).resolve().parents[1]                          # project folder
KEY_FILE = ROOT / "secrets" / "amanchaudhary.web-service-account.json"   # Earth Engine key (never commit)

GSD_M = 10                     # pixel size (m)
TILE_PX = 128                  # model / training tile (1.28 km)
TILES_PER_BLOCK = 8            # a block is 8 x 8 tiles
BLOCK_PX = TILE_PX * TILES_PER_BLOCK                # 1024 px = 10.24 km
TILE_M, BLOCK_M = TILE_PX * GSD_M, BLOCK_PX * GSD_M
N_MONTHS = 12
NODATA = -32768                # int16 no-data of the image files
MIN_CROP_PIXELS = 164          # a block "has cropland" from this many cropland pixels (1% of a 128 px tile, 1.6 ha)
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]


# ------------------------------------------------------------ paths and grid
def download_dir(region):
    """outputs/<region>/data_download/"""
    return ROOT / "outputs" / region / "data_download"


def season_tag(ag_season):
    return ag_season.replace("/", "-")                              # "2023/24" -> "2023-24"


def block_name(bx, by):
    return f"{bx}_{by}"


def block_file(region, source, ag_season, bx, by):
    return download_dir(region) / source / season_tag(ag_season) / f"{block_name(bx, by)}.tif"


def grid_file(region):
    """The block grid of a region (01_plan_blocks.py): the same blocks for every agricultural year."""
    return download_dir(region) / "grid" / "blocks.parquet"


def training_plan_file(region):
    """The training-tile plan of data_training_prep/03_plan_tiles.py (ix, iy and ag_season of every tile)."""
    return dataset_dir(region) / "03_tiles_128px" / "agyear" / "tiles.parquet"


DW_NODATA = 255                # no-data of the Dynamic World cropland blocks (uint8)
DW_SOURCE = "dynamic_world"    # folder of the Dynamic World cropland blocks (02a_dynamic_world.py)


def dw_block_file(region, ag_season, bx, by):
    """Binary Dynamic World cropland of one block and agricultural year (02a_dynamic_world.py)."""
    return block_file(region, DW_SOURCE, ag_season, bx, by)


def dw_mosaic_file(region, ag_season):
    """VRT mosaic over all Dynamic World cropland blocks of one agricultural year (written by 02a at the end)."""
    return download_dir(region) / DW_SOURCE / season_tag(ag_season) / "mosaic.vrt"


def block_crop_pixels(region, ag_season, blocks):
    """Cropland pixels per block (list, same order as blocks); None for a block whose Dynamic World file
    (02a_dynamic_world.py) is not there yet."""
    out = []
    for bx, by in zip(blocks["bx"], blocks["by"]):
        path = dw_block_file(region, ag_season, bx, by)
        if not path.exists():
            out.append(None)
            continue
        with rasterio.open(path) as src:
            out.append(int((src.read(1) == 1).sum()))
    return out


def crop_pixels(region, ag_season, bx, by):
    """Cropland pixels of one block (Dynamic World, 02a); None if its block file is not there."""
    return block_crop_pixels(region, ag_season, pd.DataFrame({"bx": [bx], "by": [by]}))[0]


def block_transform(bx, by):
    """Top-left corner of block (bx, by): x = bx * 10240 m, top y = (by + 1) * 10240 m."""
    return Affine(GSD_M, 0, bx * BLOCK_M, 0, -GSD_M, (by + 1) * BLOCK_M)


def block_of_tile(ix, iy):
    """Block holding tile (ix, iy) and the tile's (row, col) offset in pixels inside the block."""
    bx, by = ix // TILES_PER_BLOCK, iy // TILES_PER_BLOCK
    col = (ix - bx * TILES_PER_BLOCK) * TILE_PX
    row = ((by + 1) * TILES_PER_BLOCK - (iy + 1)) * TILE_PX
    return bx, by, row, col


def month_starts(region, ag_season):
    """First day of the 12 months of the agricultural year (start month from regions.py)."""
    start = pd.Timestamp(year=int(ag_season[:4]), month=settings(region)["start_month"], day=1)
    return [start + pd.DateOffset(months=k) for k in range(N_MONTHS)]


def month_bounds(month):
    return str(month.date()), str((month + pd.DateOffset(months=1)).date())


def month_names(bands):
    """Band names of a 12-month stack, month-major: <band>_m01 ... <band>_m12."""
    return [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in bands]


def write_block(path, array, bx, by, descriptions, crs_epsg, nodata=None, size_px=BLOCK_PX):
    """Write [bands, H, W] as a tiled, compressed GeoTIFF of one block (temp name, then renamed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    gsd = BLOCK_M / size_px
    profile = dict(driver="GTiff", width=size_px, height=size_px, count=array.shape[0], dtype=array.dtype,
                   crs=f"EPSG:{crs_epsg}", transform=Affine(gsd, 0, bx * BLOCK_M, 0, -gsd, (by + 1) * BLOCK_M),
                   nodata=nodata, compress="deflate", predictor=2, tiled=True,
                   blockxsize=min(256, size_px), blockysize=min(256, size_px), BIGTIFF="IF_SAFER")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array)
        for i, d in enumerate(descriptions, start=1):
            dst.set_band_description(i, d)
    tmp.replace(path)


def load_blocks(region, ag_season):
    """The block grid of 01_plan_blocks.py, with the training tiles of one agricultural year counted per block
    (n_training_tiles, has_training_tiles) from the current tile plan of data_training_prep/03_plan_tiles.py.
    Blocks with training tiles are always downloaded (02b, 02c), so every training tile can be cut out."""
    path = grid_file(region)
    if not path.exists():
        raise SystemExit(f"{path} not found: run data_download/01_plan_blocks.py first")
    blocks = gpd.read_parquet(path)
    plan = training_plan_file(region)
    n_train = {}
    if plan.exists():
        tp = pd.read_parquet(plan, columns=["ag_season", "ix", "iy"])
        tp = tp[tp["ag_season"] == ag_season]
        n_train = tp.groupby([tp["ix"] // TILES_PER_BLOCK, tp["iy"] // TILES_PER_BLOCK]).size().to_dict()
        outside = {k: n for k, n in n_train.items() if k not in set(zip(blocks["bx"], blocks["by"]))}
        if outside:
            print(f"  WARNING {ag_season}: {sum(outside.values())} training tiles in {len(outside)} blocks lie outside "
                  f"the block grid: {sorted(outside)[:10]} (check map_file / map_where in regions.py)")
    else:
        print(f"  {ag_season}: no training-tile plan yet ({plan}); blocks are selected by cropland only")
    blocks["n_training_tiles"] = [int(n_train.get((bx, by), 0)) for bx, by in zip(blocks["bx"], blocks["by"])]
    blocks["has_training_tiles"] = blocks["n_training_tiles"] > 0
    return blocks


# ------------------------------------------------------------ Earth Engine
def init_gee():
    import ee
    key = json.loads(KEY_FILE.read_text())
    ee.Initialize(ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key)), project=key["project_id"])
    print("Earth Engine initialised (service account)")
    return ee


def select_blocks(region, source, ag_season, require_cropland=False, only_blocks=None, n_test_blocks=None):
    """The blocks of one year that a source needs and does not have yet (None if 02a must run first).
    require_cropland: only blocks with >= MIN_CROP_PIXELS cropland pixels in the Dynamic World cropland file
    (02a_dynamic_world.py), plus every block with training tiles."""
    blocks = load_blocks(region, ag_season)
    if require_cropland:
        n = block_crop_pixels(region, ag_season, blocks)
        n_missing = sum(v is None for v in n)
        if n_missing == len(n):
            print(f"{source} {ag_season}: no Dynamic World cropland yet: run 02a_dynamic_world.py first")
            return None
        if n_missing:
            print(f"  WARNING {source} {ag_season}: {n_missing:,} blocks have no Dynamic World cropland yet and are "
                  f"skipped (unless they hold training tiles): run 02a_dynamic_world.py, then this script again")
        has_crop = np.array([v is not None and v >= MIN_CROP_PIXELS for v in n])
        keep = has_crop | blocks["has_training_tiles"].to_numpy()
        print(f"{source} {ag_season}: {int(keep.sum()):,} of {len(blocks):,} blocks hold cropland or training tiles")
        blocks = blocks[keep]
    if only_blocks:
        blocks = blocks[[(bx, by) in set(only_blocks) for bx, by in zip(blocks["bx"], blocks["by"])]]
    elif n_test_blocks:
        blocks = blocks.sort_values("has_training_tiles", ascending=False).head(n_test_blocks)
    blocks = blocks.sort_values("has_training_tiles", ascending=False, kind="stable")   # training blocks first
    todo = blocks[[not block_file(region, source, ag_season, bx, by).exists()
                   for bx, by in zip(blocks["bx"], blocks["by"])]]
    print(f"{source} {ag_season}: {len(blocks) - len(todo)} blocks done, {len(todo)} to go")
    return todo


def process_blocks(region, source, compute, names, seasons, dtype=np.int16, nodata=NODATA, workers=2,
                   n_test_blocks=None, only_blocks=None, require_cropland=False, size_px=BLOCK_PX, stop_after=20):
    """Compute and write one data source for every selected block that does not have it yet (resumable).

    compute     f(ag_season, bx, by) -> (array [len(names), size_px, size_px] or None, provenance dict or None);
                None = block failed (logged, not written: a rerun tries it again)
    source      folder name (s2, s1, ...); names = band descriptions; size_px = pixels per block side
    stop_after  stop after this many failed blocks in a row (e.g. quota used up or no connection): the blocks
                written so far are kept, and a rerun continues with the missing ones
    """
    crs = settings(region)["crs_epsg"]
    for ag_season in seasons:
        todo = select_blocks(region, source, ag_season, require_cropland, only_blocks, n_test_blocks)
        if todo is None or todo.empty:
            continue
        rows, t_start, failed, in_row = list(todo.itertuples()), time.time(), 0, 0

        def one(b):
            try:
                return compute(ag_season, b.bx, b.by)
            except Exception as err:                                # one block must not stop the others
                print(f"  block {block_name(b.bx, b.by)}: failed ({type(err).__name__}: {str(err)[:200]})")
                return None, None

        for start in range(0, len(rows), workers * 2):
            batch = rows[start:start + workers * 2]
            with ThreadPoolExecutor(workers) as pool:
                results = list(pool.map(one, batch))
            for b, (cube, info) in zip(batch, results):
                if cube is None:
                    failed += 1
                    in_row += 1
                    continue
                in_row = 0
                assert cube.shape == (len(names), size_px, size_px), cube.shape
                path = block_file(region, source, ag_season, b.bx, b.by)
                write_block(path, cube.astype(dtype), b.bx, b.by, names, crs, nodata=nodata, size_px=size_px)
                if info is not None:
                    path.with_name(f"{block_name(b.bx, b.by)}_scenes.json").write_text(json.dumps(info))
            done = min(start + len(batch), len(rows))
            rate = (time.time() - t_start) / done
            print(f"  {source} {ag_season}: {done}/{len(rows)} blocks ({rate:.0f} s/block, "
                  f"~{rate * (len(rows) - done) / 3600:.1f} h left)", flush=True)
            if stop_after and in_row >= stop_after:
                raise SystemExit(f"{source} {ag_season}: {in_row} blocks failed in a row; stopped. The finished "
                                 f"blocks are kept: run again later to continue with the missing ones")
        if failed:
            print(f"  {source} {ag_season}: {failed} block(s) failed: run again to fetch them")
