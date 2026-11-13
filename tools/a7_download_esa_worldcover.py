"""
a7_download_esa_worldcover.py  —  ESA WorldCover 10 m land cover for an area of interest (visualisation layer)
================================================================================================================

What it does
------------
Downloads the ESA WorldCover tiles that cover an area of interest (AOI) from Microsoft Planetary Computer, mosaics
them, clips them to the AOI and writes one GeoTIFF (and optionally a coarser copy for quick maps). It is an extra
background layer for maps (e.g. in QGIS); no model or dataset script uses it.

Data
----
ESA WorldCover v100 (2020) / v200 (2021), 10 m, EPSG:4326 (Zanaga et al. 2021, 2022; https://esa-worldcover.org).
Planetary Computer collection "esa-worldcover", asset "map"; no account needed. Classes:
  10 tree cover · 20 shrubland · 30 grassland · 40 cropland · 50 built-up · 60 bare / sparse vegetation ·
  70 snow and ice · 80 permanent water · 90 herbaceous wetland · 95 mangroves · 100 moss and lichen · 0 no data

Steps
-----
1. Read the AOI polygon(s) (a boundary file and an optional attribute filter) and write them to one GeoJSON.
2. Search the Planetary Computer catalogue for the WorldCover tiles of YEAR inside the AOI's bounding box.
3. Download the tiles (skipped if already there).
4. Mosaic (VRT), clip to the AOI and write a compressed GeoTIFF; optionally a coarser copy (mode resampling,
   the right choice for classes).
5. Remove the downloaded tiles unless KEEP_TILES = True.

Output (OUT_DIR)
----------------
ESA_WorldCover_10m_<YEAR>_<AOI_NAME>.tif          10 m, clipped to the AOI, 0 = no data / outside the AOI
ESA_WorldCover_10m_<YEAR>_<AOI_NAME>_250m.tif     coarse copy (COARSE_RES_DEG), if set
aoi_<AOI_NAME>.geojson                            the AOI used for the clip

Run (from the project folder; set the AOI below first)
-------------------------------------------------------
    python tools/a7_download_esa_worldcover.py
"""
import os
from pathlib import Path

os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")     # GAUL polygons are larger than the GDAL default limit

import geopandas as gpd
import planetary_computer
import pyogrio
import requests
from osgeo import gdal
from pystac_client import Client
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
# AOI_WHERE = "gaul1_name = 'Utah'"   


YEAR = 2021                                                # 2020 (v100) or 2021 (v200)
OUT_DIR = ROOT / "data" / "worldcover"
COARSE_RES_DEG = 0.002245                                  # ~250 m copy for quick maps; None = no copy
KEEP_TILES = False                                         # True = keep the downloaded tiles in OUT_DIR/_tiles_<YEAR>/
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# ==========================================================================


def read_aoi():
    """AOI polygons from AOI_FILE (filtered by AOI_WHERE), in EPSG:4326, dissolved into one feature."""
    gdf = pyogrio.read_dataframe(AOI_FILE, where=AOI_WHERE) if AOI_WHERE else gpd.read_file(AOI_FILE)
    if gdf.empty:
        raise SystemExit(f"No polygon in {AOI_FILE} matches {AOI_WHERE!r}")
    gdf = gdf.to_crs(4326)
    return gpd.GeoDataFrame({"name": [AOI_NAME]}, geometry=[gdf.geometry.union_all()], crs=4326)


def find_tiles(bbox):
    """WorldCover items of YEAR that intersect the bounding box, signed for download."""
    catalog = Client.open(STAC_URL)
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox, datetime=f"{YEAR}-01-01/{YEAR}-12-31")
    items = list(search.items())
    if not items:
        raise SystemExit(f"No ESA WorldCover tiles found for {YEAR} in {bbox}")
    return [planetary_computer.sign(item) for item in items]


def download(url, path):
    """Stream a file to path (written to a temporary name first, so a broken download is never kept)."""
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tile_dir = OUT_DIR / f"_tiles_{YEAR}"
    tile_dir.mkdir(exist_ok=True)

    aoi = read_aoi()
    aoi_file = OUT_DIR / f"aoi_{AOI_NAME}.geojson"
    aoi.to_file(aoi_file, driver="GeoJSON")
    bbox = tuple(aoi.total_bounds)
    print(f"AOI {AOI_NAME}: bbox {tuple(round(v, 3) for v in bbox)}")

    items = find_tiles(bbox)
    print(f"{len(items)} WorldCover tile(s) for {YEAR}")
    tifs = []
    for item in tqdm(items, desc="Downloading"):
        path = tile_dir / f"{item.id}.tif"
        if not path.exists():
            download(item.assets["map"].href, path)
        tifs.append(str(path))

    vrt = tile_dir / f"mosaic_{YEAR}.vrt"
    gdal.BuildVRT(str(vrt), tifs)
    out = OUT_DIR / f"ESA_WorldCover_10m_{YEAR}_{AOI_NAME}.tif"
    print(f"Clipping to the AOI -> {out}")
    gdal.Warp(str(out), str(vrt), options=gdal.WarpOptions(
        cutlineDSName=str(aoi_file), cropToCutline=True, dstNodata=0, multithread=True,
        warpOptions=["NUM_THREADS=ALL_CPUS"],
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]))

    if COARSE_RES_DEG:
        coarse = OUT_DIR / f"ESA_WorldCover_10m_{YEAR}_{AOI_NAME}_{round(COARSE_RES_DEG * 111_320)}m.tif"
        print(f"Coarse copy ({COARSE_RES_DEG} deg, mode) -> {coarse}")
        gdal.Warp(str(coarse), str(out), options=gdal.WarpOptions(
            xRes=COARSE_RES_DEG, yRes=COARSE_RES_DEG, resampleAlg="mode", targetAlignedPixels=True,
            srcNodata=0, dstNodata=0, multithread=True,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES"]))

    if not KEEP_TILES:
        for f in tifs + [str(vrt)]:
            Path(f).unlink(missing_ok=True)
        tile_dir.rmdir()
    print("Done")


if __name__ == "__main__":
    main()
