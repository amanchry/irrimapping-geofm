"""
download_common.py  —  block grid, file layout, Earth Engine login and the per-block download loop (not run alone)
================================================================================================================

Every data source is downloaded ONCE per region and agricultural year on one grid of 10.24 km blocks
(1024 x 1024 px at 10 m), in the region's projection (regions.py). The grid is the 128 px tile grid of the training
data, grouped 8 x 8: block (bx, by) holds the tiles ix = 8 bx ... 8 bx + 7, iy = 8 by ... 8 by + 7, so every training
tile and every map window is an exact cut-out of a block.

    outputs/<region>/data_download/
        grid/blocks.parquet, .geojson               the block grid over the whole mapping area, the same for every
                                                    year (01_plan_blocks.py); the training tiles per block are counted
                                                    when the blocks are loaded (load_blocks)
        dynamic_world/cropland_<season>.tif         binary cropland of the whole region, 10 m, on the block grid
                                                    (02a_dynamic_world.py; one download per year with geemap)
        <source>/<season>/<bx>_<by>.tif             one file per block, 12 months x bands of the source
        <source>/<season>/<bx>_<by>_scenes.json     scene IDs per month (provenance), where available

A new source (e.g. ERA5) is one more script 02x_<source>.py: it builds one Earth Engine image per agricultural
year and calls download_blocks(); nothing else changes. Earth Engine is reached with the service-account key in
secrets/ (KEY_FILE; never commit it). Earth Engine limits one computePixels request to 48 MB, so a block is fetched
in groups of months that stay below MAX_REQUEST_MB.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from rasterio.transform import Affine

from regions import dataset_dir, settings

ROOT = Path(__file__).resolve().parents[1]                          # project folder
# KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"     # Earth Engine key (never commit)
KEY_FILE = ROOT / "secrets" / "geonexa-service-account.json"     # Earth Engine key (never commit)

GSD_M = 10                     # pixel size (m)
TILE_PX = 128                  # model / training tile (1.28 km)
TILES_PER_BLOCK = 8            # a block is 8 x 8 tiles
BLOCK_PX = TILE_PX * TILES_PER_BLOCK                # 1024 px = 10.24 km
TILE_M, BLOCK_M = TILE_PX * GSD_M, BLOCK_PX * GSD_M
N_MONTHS = 12
NODATA = -32768                # int16 no-data of the image files
MAX_REQUEST_MB = 40            # stay below Earth Engine's 48 MB per computePixels request
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


DW_NODATA = 255                # no-data of the Dynamic World cropland file (uint8)


def dw_cropland_file(region, ag_season):
    """Region-wide binary Dynamic World cropland of one agricultural year (02a_dynamic_world.py)."""
    return download_dir(region) / "dynamic_world" / f"cropland_{season_tag(ag_season)}.tif"


def read_dw_window(src, x0, y1, size_px):
    """size_px x size_px pixels of the open cropland file with top-left corner (x0, y1); 255 outside the file."""
    col = int(round((x0 - src.transform.c) / GSD_M))
    row = int(round((src.transform.f - y1) / GSD_M))
    win = rasterio.windows.Window(col, row, size_px, size_px)
    return src.read(1, window=win, boundless=True, fill_value=DW_NODATA)


def block_crop_pixels(region, ag_season, blocks):
    """Cropland pixels per block (list, same order as blocks); None for all if 02a has not run yet."""
    path = dw_cropland_file(region, ag_season)
    if not path.exists():
        return [None] * len(blocks)
    with rasterio.open(path) as src:
        return [int((read_dw_window(src, bx * BLOCK_M, (by + 1) * BLOCK_M, BLOCK_PX) == 1).sum())
                for bx, by in zip(blocks["bx"], blocks["by"])]


def crop_pixels(region, ag_season, bx, by):
    """Cropland pixels of one block in the Dynamic World cropland file (02a); None if it is not there."""
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


def season_region(ee, blocks, crs_epsg):
    """Earth Engine rectangle around the blocks of one season (+ 2 km), used to filter the collections."""
    bounds = gpd.GeoSeries(blocks.geometry, crs=crs_epsg).to_crs(4326).total_bounds
    return ee.Geometry.Rectangle(list(bounds)).buffer(2000)


def block_geometry(ee, bx, by, crs_epsg):
    x0, y1 = bx * BLOCK_M, (by + 1) * BLOCK_M
    return ee.Geometry.Rectangle([float(x0), float(y1 - BLOCK_M), float(x0 + BLOCK_M), float(y1)],
                                 f"EPSG:{crs_epsg}", False)


def request_groups(names, size_px, bytes_per_value=2):
    """Split the band names into groups whose request stays below MAX_REQUEST_MB. Earth Engine counts more than
    the data type per pixel (an int16 band of 1024 x 1024 px counted as 3.1 MB, not 2.1 MB), so 2 bytes are added
    per value as a margin."""
    per_band = size_px * size_px * (bytes_per_value + 2)
    n = max(1, int(MAX_REQUEST_MB * 1e6 // per_band))
    return [names[i:i + n] for i in range(0, len(names), n)]


def compute_block(ee, image, bx, by, crs_epsg, names, dtype, size_px=BLOCK_PX, retries=4):
    """All bands `names` of one block, fetched in groups below the request limit -> [len(names), size, size]."""
    gsd = BLOCK_M / size_px
    grid = {"dimensions": {"width": size_px, "height": size_px},
            "affineTransform": {"scaleX": gsd, "shearX": 0, "translateX": float(bx * BLOCK_M),
                                "shearY": 0, "scaleY": -gsd, "translateY": float((by + 1) * BLOCK_M)},
            "crsCode": f"EPSG:{crs_epsg}"}
    parts = []
    for group in request_groups(names, size_px, np.dtype(dtype).itemsize):
        for attempt in range(1, retries + 1):
            try:
                arr = ee.data.computePixels({"expression": image.select(group), "fileFormat": "NUMPY_NDARRAY",
                                             "grid": grid})
                parts.append(np.stack([arr[n] for n in group]).astype(dtype))
                break
            except Exception as err:                                # network / quota / EE error
                if attempt == retries:
                    print(f"  block {block_name(bx, by)}: failed ({str(err)[:120]})")
                    return None
                time.sleep(10 * attempt)
    return np.concatenate(parts)


def download_blocks(region, source, make_image, names, seasons, dtype=np.int16, nodata=NODATA, workers=4,
                    n_test_blocks=None, scenes=None, size_px=BLOCK_PX, only_blocks=None, require_cropland=False):
    """Download one data source for every planned block that does not have it yet (resumable).

    source      folder name (s2, s1, dynamic_world, ...)
    make_image  f(ee, ag_season, ee_region) -> ee.Image with the bands `names` for one agricultural year
    scenes      optional f(ee, ag_season, ee_region, block_geometry) -> ee object of scene IDs (provenance)
    size_px     pixels per block side (1024 at 10 m; fewer for coarse sources such as weather data)
    only_blocks optional list of (bx, by) to download (tests)
    require_cropland  only blocks with >= MIN_CROP_PIXELS cropland pixels in the Dynamic World cropland file
                (02a_dynamic_world.py), plus every block with training tiles (S2, S1, ERA5, ...)
    """
    crs = settings(region)["crs_epsg"]
    ee = init_gee()
    for ag_season in seasons:
        blocks = load_blocks(region, ag_season)
        if require_cropland:
            n = block_crop_pixels(region, ag_season, blocks)
            has_crop = np.array([v is not None and v >= MIN_CROP_PIXELS for v in n])
            keep = has_crop | blocks["has_training_tiles"].to_numpy()
            if n[0] is None:
                print(f"{source} {ag_season}: no Dynamic World cropland yet ({dw_cropland_file(region, ag_season)}): "
                      f"run 02a_dynamic_world.py first")
                continue
            print(f"{source} {ag_season}: {int(keep.sum()):,} of {len(blocks):,} blocks hold cropland or training tiles")
            blocks = blocks[keep]
        if only_blocks:
            blocks = blocks[[(bx, by) in set(only_blocks) for bx, by in zip(blocks["bx"], blocks["by"])]]
        elif n_test_blocks:
            blocks = blocks.sort_values("has_training_tiles", ascending=False).head(n_test_blocks)
        todo = blocks[[not block_file(region, source, ag_season, bx, by).exists()
                       for bx, by in zip(blocks["bx"], blocks["by"])]]
        print(f"{source} {ag_season}: {len(blocks) - len(todo)} blocks done, {len(todo)} to go")
        if todo.empty:
            continue
        ee_region = season_region(ee, load_blocks(region, ag_season), crs)
        image = make_image(ee, ag_season, ee_region)
        rows, t_start, failed = list(todo.itertuples()), time.time(), 0

        def one(b):
            cube = compute_block(ee, image, b.bx, b.by, crs, names, dtype, size_px)
            ids = None
            if cube is not None and scenes is not None:
                try:
                    ids = scenes(ee, ag_season, ee_region, block_geometry(ee, b.bx, b.by, crs)).getInfo()
                except Exception:                                   # provenance only; the block is kept
                    ids = None
            return cube, ids

        for start in range(0, len(rows), workers * 2):
            batch = rows[start:start + workers * 2]
            with ThreadPoolExecutor(workers) as pool:
                results = list(pool.map(one, batch))
            for b, (cube, ids) in zip(batch, results):
                if cube is None:
                    failed += 1
                    continue
                path = block_file(region, source, ag_season, b.bx, b.by)
                write_block(path, cube, b.bx, b.by, names, crs, nodata=nodata, size_px=size_px)
                if ids is not None:
                    path.with_name(f"{block_name(b.bx, b.by)}_scenes.json").write_text(json.dumps(ids))
            done = min(start + len(batch), len(rows))
            rate = (time.time() - t_start) / done
            print(f"  {source} {ag_season}: {done}/{len(rows)} blocks ({rate:.0f} s/block, "
                  f"~{rate * (len(rows) - done) / 3600:.1f} h left)")
        if failed:
            print(f"  {failed} block(s) failed: run again to fetch them")
