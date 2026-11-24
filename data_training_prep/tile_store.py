"""
tile_store.py  —  paths and helpers of the training-tile store (not run on its own)
==================================================================================

The training tiles are 128 x 128 px cut-outs of the downloaded 10.24 km blocks (data_download/), around the labelled
parcels only. One folder per region:

    outputs/<region>/data_training_prep/03_tiles_128px/agyear/
        tiles/<tile_id>_lab.tif      label rasters (03_plan_tiles.py)
        tiles/<tile_id>_img.tif      Sentinel-2 + Sentinel-1, 228 bands: the file every model reads (04_cut_tiles.py)
        tiles/<tile_id>_dw.tif       binary Dynamic World cropland, 1 band uint8 (04_cut_tiles.py, where downloaded)
        parcels.csv, tiles.parquet, tiles_info.json, tiles_qa.jsonl

The grid, the block layout and regions.py live in data_download/ and are imported from there.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_download"))   # regions.py, download_common.py

import rasterio  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from download_common import GSD_M, N_MONTHS, NODATA, S2_BANDS, TILE_M, TILE_PX, season_tag  # noqa: E402,F401
from regions import dataset_dir  # noqa: E402

WINDOW = "agyear"


def out_dir(region):
    """outputs/<region>/data_training_prep/03_tiles_128px/agyear/"""
    return dataset_dir(region) / "03_tiles_128px" / WINDOW


def tile_dir(region):
    return out_dir(region) / "tiles"


def tile_file(region, tile_id, kind):
    """kind: lab, img, dw (see the module docstring)."""
    return tile_dir(region) / f"{tile_id}_{kind}.tif"


def tile_id(ag_season, ix, iy):
    return f"{season_tag(ag_season)}_{ix}_{iy}"


def tile_transform(ix, iy):
    """Top-left corner of tile (ix, iy) on the fixed grid; ix, iy = floor(x / TILE_M), floor(y / TILE_M)."""
    return Affine(GSD_M, 0, ix * TILE_M, 0, -GSD_M, (iy + 1) * TILE_M)


def write_tif(path, array, ix, iy, descriptions, crs_epsg, nodata=None):
    """Write [bands, H, W] as a compressed GeoTIFF on the fixed grid (written to a temp name, then renamed)."""
    tmp = path.with_name(path.name + ".part")
    profile = dict(driver="GTiff", width=TILE_PX, height=TILE_PX, count=array.shape[0], dtype=array.dtype,
                   crs=f"EPSG:{crs_epsg}", transform=tile_transform(ix, iy), nodata=nodata,
                   compress="deflate", predictor=2, tiled=True, blockxsize=TILE_PX, blockysize=TILE_PX)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array)
        for i, d in enumerate(descriptions, start=1):
            dst.set_band_description(i, d)
    tmp.replace(path)


def load_plan(region):
    """The training-tile plan of 03_plan_tiles.py (tiles.parquet) as a GeoDataFrame."""
    import geopandas as gpd
    path = out_dir(region) / "tiles.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found: run data_training_prep/03_plan_tiles.py first")
    return gpd.read_parquet(path)
