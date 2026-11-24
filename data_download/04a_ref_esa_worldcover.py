"""
04a_ref_esa_worldcover.py  —  ESA WorldCover 10 m land cover for an area of interest (visualisation layer)
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
1. Read the AOI: the region's mapping area from regions.py (Morocco without Western Sahara, or Utah), the same
   area as the cropland mask; written to one GeoJSON.
2. Search the Planetary Computer catalogue for the WorldCover tiles of YEAR inside the AOI's bounding box.
3. Download the tiles (skipped if already there).
4. Mosaic (VRT), clip to the AOI and write a compressed GeoTIFF; optionally a coarser copy (mode resampling,
   the right choice for classes).
5. Remove the downloaded tiles unless KEEP_TILES = True.
6. PNG map for visualisation and documentation: WorldCover colours (the official ESA palette), AOI boundary,
   legend with each class's share of the AOI (MAKE_PNG). PNG_ONLY = True skips steps 2-5 and draws the PNG from
   the GeoTIFF that is already there.

Output (OUT_DIR = outputs/<region>/data_download/reference/esa_worldcover/)
-------------------------------------------------------------------
ESA_WorldCover_10m_<YEAR>_<AOI_NAME>.tif          10 m, clipped to the AOI, 0 = no data / outside the AOI
ESA_WorldCover_10m_<YEAR>_<AOI_NAME>_250m.tif     coarse copy (COARSE_RES_DEG), if set
aoi_<AOI_NAME>.geojson                            the AOI used for the clip
ESA_WorldCover_10m_<YEAR>_<AOI_NAME>.png          map with legend and AOI boundary (MAKE_PNG)

Run (from the project folder; set REGION below first)
------------------------------------------------------
    python data_download/04a_ref_esa_worldcover.py
"""
import os
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")                                      # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from osgeo import gdal
from rasterio.enums import Resampling

gdal.UseExceptions()
from regions import cropland_dir, map_area, settings  # noqa: E402

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder

REGION = "morocco"                                         # "morocco" | "utah": the AOI is the region's mapping area
                                                           # (regions.py: map_file / map_where), as for the mask
AOI_NAME = REGION.capitalize()                             # used in the output file names
YEAR = 2021                                                # 2020 (v100) or 2021 (v200)
OUT_DIR = cropland_dir(REGION) / "esa_worldcover"                 # outputs/<region>/data_download/reference/esa_worldcover/
COARSE_RES_DEG = 0.002245                                  # ~250 m copy for quick maps; None = no copy
KEEP_TILES = False                                         # True = keep the downloaded tiles in OUT_DIR/_tiles_<YEAR>/
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MAKE_PNG = True                                            # step 6: PNG map with legend and AOI boundary
PNG_ONLY = True                                           # True = only draw the PNG from the existing GeoTIFF
PNG_MAX_PX = 2500                                          # longest side of the map image (pixels, mode resampling)
# ==========================================================================

# ESA WorldCover classes and colours, as in the product metadata (Map_class_names, Map_class_palette)
CLASSES = {10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"),
           40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare / sparse vegetation", "#b4b4b4"),
           70: ("Snow and ice", "#f0f0f0"), 80: ("Permanent water bodies", "#0064c8"),
           90: ("Herbaceous wetland", "#0096a0"), 95: ("Mangroves", "#00cf75"), 100: ("Moss and lichen", "#fae6a0")}
NODATA = 0


def read_aoi():
    """The region's mapping area (regions.py) in EPSG:4326 as one feature."""
    return gpd.GeoDataFrame({"name": [AOI_NAME]}, geometry=[map_area(REGION)], crs=settings(REGION)["crs_epsg"]).to_crs(4326)


def find_tiles(bbox):
    """WorldCover items of YEAR that intersect the bounding box, signed for download."""
    import planetary_computer                              # download-only packages (not needed for PNG_ONLY)
    from pystac_client import Client
    catalog = Client.open(STAC_URL)
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox, datetime=f"{YEAR}-01-01/{YEAR}-12-31")
    items = list(search.items())
    if not items:
        raise SystemExit(f"No ESA WorldCover tiles found for {YEAR} in {bbox}")
    return [planetary_computer.sign(item) for item in items]


def download(url, path):
    """Stream a file to path (written to a temporary name first, so a broken download is never kept)."""
    import requests
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(path)


def make_png(tif, aoi, png):
    """Map of the GeoTIFF (read at <= PNG_MAX_PX, mode resampling) with the AOI boundary and a legend."""
    with rasterio.open(tif) as src:
        scale = max(src.width, src.height) / PNG_MAX_PX
        shape = (max(1, round(src.height / scale)), max(1, round(src.width / scale))) if scale > 1 else (src.height, src.width)
        arr = src.read(1, out_shape=shape, resampling=Resampling.mode)
        b = src.bounds
    codes = sorted(CLASSES)
    idx = np.full(arr.shape, -1)
    for k, c in enumerate(codes):
        idx[arr == c] = k
    valid = idx >= 0
    shares = {c: 100 * float((arr[valid] == c).mean()) for c in codes} if valid.any() else {}
    fig, ax = plt.subplots(figsize=(10, 10 * shape[0] / shape[1] + 1.2), dpi=150)
    ax.imshow(np.ma.masked_less(idx, 0), cmap=ListedColormap([CLASSES[c][1] for c in codes]), vmin=0,
              vmax=len(codes) - 1, interpolation="nearest", extent=[b.left, b.right, b.bottom, b.top])
    aoi.boundary.plot(ax=ax, color="#222222", linewidth=0.8)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"ESA WorldCover {YEAR} (10 m) · {AOI_NAME}", fontsize=12)
    handles = [Patch(facecolor=CLASSES[c][1], edgecolor="0.5", label=f"{c} {CLASSES[c][0]} ({shares[c]:.1f}%)")
               for c in codes if shares.get(c, 0) >= 0.05]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9,
              title="Class (share of the AOI)", title_fontsize=9)
    fig.text(0.01, 0.005, "Source: ESA WorldCover v200", fontsize=7, color="0.4")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG -> {png}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aoi = read_aoi()
    out = OUT_DIR / f"ESA_WorldCover_10m_{YEAR}_{AOI_NAME}.tif"
    if PNG_ONLY:
        if not out.exists():
            raise SystemExit(f"{out} not found: run once with PNG_ONLY = False")
        make_png(out, aoi, out.with_suffix(".png"))
        return
    from tqdm import tqdm
    tile_dir = OUT_DIR / f"_tiles_{YEAR}"
    tile_dir.mkdir(exist_ok=True)
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
    if MAKE_PNG:
        make_png(out, aoi, out.with_suffix(".png"))
    print("Done")


if __name__ == "__main__":
    main()
