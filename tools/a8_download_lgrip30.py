"""
a8_download_lgrip30.py  —  LGRIP30 irrigated / rainfed cropland (30 m) for an area of interest (visualisation layer)
====================================================================================================================

What it does
------------
Finds the LGRIP30 tiles that cover an area of interest (AOI) in NASA's catalogue (CMR), downloads them from the
LP DAAC archive, mosaics them, clips them to the AOI and writes one GeoTIFF (and optionally a coarser copy). It is
an extra background layer for maps (e.g. in QGIS); no model or dataset script uses it.

Data
----
LGRIP30 v001: Landsat-derived global rainfed and irrigated cropland product, nominal year 2015, 30 m, EPSG:4326,
10 x 10 degree tiles (Teluguntla et al. 2023, doi:10.5067/Community/LGRIP/LGRIP30.001). Classes:
  0 water · 1 non-cropland · 2 irrigated cropland · 3 rainfed cropland

Account: the files are behind NASA Earthdata Login (free). Create a token at https://urs.earthdata.nasa.gov
(Profile -> Generate Token) and put it in secrets/earthdata_token.txt (one line; secrets/ is never committed) or in
the environment variable EARTHDATA_TOKEN. Tokens expire after 60 days. Never write the token into this script.

Steps
-----
1. Read the AOI polygon(s) (a boundary file and an optional attribute filter) and write them to one GeoJSON.
2. Ask NASA's Common Metadata Repository (CMR, public) for the LGRIP30 tiles inside the AOI's bounding box.
3. Download the tiles with the Earthdata token (skipped if already there).
4. Mosaic (VRT), clip to the AOI and write a compressed GeoTIFF; optionally a coarser copy (mode resampling).
5. Remove the downloaded tiles unless KEEP_TILES = True.

Output (OUT_DIR)
----------------
LGRIP30_2015_<AOI_NAME>.tif         30 m, clipped to the AOI, 255 = outside the AOI / no data
LGRIP30_2015_<AOI_NAME>_250m.tif    coarse copy (COARSE_RES_DEG), if set
aoi_<AOI_NAME>.geojson              the AOI used for the clip

Run (from the project folder; set the AOI below first)
-------------------------------------------------------
    python tools/a8_download_lgrip30.py
"""
import os
from pathlib import Path

os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")     # GAUL polygons are larger than the GDAL default limit

import geopandas as gpd
import pyogrio
import requests
from osgeo import gdal
from tqdm import tqdm

gdal.UseExceptions()

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder

# Area of interest: any polygon file, with an optional OGR SQL filter on its attributes. Examples:
#   Utah:    data/admin_boundaries/FAO_GAUL_2024_AdminL1.geojson, "gaul1_name = 'Utah'"
#   Morocco: data/admin_boundaries/FAO_GAUL_2015_AdminL0.geojson, "ADM0_NAME = 'Morocco'"
#   own AOI: any .geojson / .gpkg / .shp, AOI_WHERE = None

AOI_NAME = "Morocco"                                          
AOI_FILE = ROOT / "data" / "admin_boundaries" / "FAO_GAUL_2015_AdminL0.geojson"
AOI_WHERE = "ADM0_NAME = 'Morocco'"           


# AOI_NAME = "Utah"                                          # used in the output file names
# AOI_FILE = ROOT / "data" / "admin_boundaries" / "FAO_GAUL_2024_AdminL1.geojson"
# AOI_WHERE = "gaul1_name = 'Utah'"                          # None = all polygons of AOI_FILE

OUT_DIR = ROOT / "data" / "lgrip30"
COARSE_RES_DEG = 0.002245                                  # ~250 m copy for quick maps; None = no copy
KEEP_TILES = False                                         # True = keep the downloaded tiles in OUT_DIR/_tiles/
TOKEN_FILE = ROOT / "secrets" / "earthdata_token.txt"      # or environment variable EARTHDATA_TOKEN
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME, VERSION = "LGRIP30", "001"
NODATA = 255                                               # outside the AOI (class 0 is water, not no data)
# ==========================================================================


def read_aoi():
    """AOI polygons from AOI_FILE (filtered by AOI_WHERE), in EPSG:4326, dissolved into one feature."""
    gdf = pyogrio.read_dataframe(AOI_FILE, where=AOI_WHERE) if AOI_WHERE else gpd.read_file(AOI_FILE)
    if gdf.empty:
        raise SystemExit(f"No polygon in {AOI_FILE} matches {AOI_WHERE!r}")
    gdf = gdf.to_crs(4326)
    return gpd.GeoDataFrame({"name": [AOI_NAME]}, geometry=[gdf.geometry.union_all()], crs=4326)


def read_token():
    token = os.environ.get("EARTHDATA_TOKEN") or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if not token:
        raise SystemExit(f"No Earthdata token: create one at https://urs.earthdata.nasa.gov (Profile -> Generate "
                         f"Token) and save it in {TOKEN_FILE} or in the environment variable EARTHDATA_TOKEN")
    return token


def find_tiles(bbox):
    """(name, GeoTIFF URL) of every LGRIP30 tile that intersects the bounding box, from NASA CMR (no login)."""
    params = {"short_name": SHORT_NAME, "version": VERSION, "page_size": 2000,
              "bounding_box": ",".join(f"{v:.6f}" for v in bbox)}
    entries = requests.get(CMR_URL, params=params, timeout=60).json()["feed"]["entry"]
    tiles = []
    for e in entries:
        urls = [l["href"] for l in e.get("links", [])
                if l.get("rel", "").endswith("/data#") and l["href"].startswith("https") and l["href"].endswith(".tif")]
        if urls:
            tiles.append((e["title"], urls[0]))
    if not tiles:
        raise SystemExit(f"No {SHORT_NAME} tiles found in {bbox}")
    return tiles


def download(url, path, token):
    """Download one tile with the Earthdata token (temporary name first, so a broken download is never kept)."""
    tmp = path.with_suffix(".part")
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120) as r:
        if r.status_code in (401, 403) or "text/html" in r.headers.get("Content-Type", ""):
            raise SystemExit(f"Earthdata refused the download ({r.status_code}): the token is missing, expired or "
                             f"not valid. Create a new one at https://urs.earthdata.nasa.gov")
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tile_dir = OUT_DIR / "_tiles"
    tile_dir.mkdir(exist_ok=True)

    aoi = read_aoi()
    aoi_file = OUT_DIR / f"aoi_{AOI_NAME}.geojson"
    aoi.to_file(aoi_file, driver="GeoJSON")
    bbox = tuple(aoi.total_bounds)
    print(f"AOI {AOI_NAME}: bbox {tuple(round(v, 3) for v in bbox)}")

    tiles = find_tiles(bbox)
    print(f"{len(tiles)} LGRIP30 tile(s): {', '.join(name for name, _ in tiles)}")
    token = read_token()
    tifs = []
    for name, url in tqdm(tiles, desc="Downloading"):
        path = tile_dir / f"{name}.tif"
        if not path.exists():
            download(url, path, token)
        tifs.append(str(path))

    vrt = tile_dir / "mosaic.vrt"
    gdal.BuildVRT(str(vrt), tifs)
    out = OUT_DIR / f"LGRIP30_2015_{AOI_NAME}.tif"
    print(f"Clipping to the AOI -> {out}")
    gdal.Warp(str(out), str(vrt), options=gdal.WarpOptions(
        cutlineDSName=str(aoi_file), cropToCutline=True, dstNodata=NODATA, outputType=gdal.GDT_Byte,
        multithread=True, warpOptions=["NUM_THREADS=ALL_CPUS"],
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]))

    if COARSE_RES_DEG:
        coarse = OUT_DIR / f"LGRIP30_2015_{AOI_NAME}_{round(COARSE_RES_DEG * 111_320)}m.tif"
        print(f"Coarse copy ({COARSE_RES_DEG} deg, mode) -> {coarse}")
        gdal.Warp(str(coarse), str(out), options=gdal.WarpOptions(
            xRes=COARSE_RES_DEG, yRes=COARSE_RES_DEG, resampleAlg="mode", targetAlignedPixels=True,
            srcNodata=NODATA, dstNodata=NODATA, multithread=True,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES"]))

    if not KEEP_TILES:
        for f in tifs + [str(vrt)]:
            Path(f).unlink(missing_ok=True)
        tile_dir.rmdir()
    print("Done")


if __name__ == "__main__":
    main()
