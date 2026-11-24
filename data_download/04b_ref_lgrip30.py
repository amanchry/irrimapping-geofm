"""
04b_ref_lgrip30.py  —  LGRIP30 irrigated / rainfed cropland (30 m) for an area of interest (visualisation layer)
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
1. Read the AOI: the region's mapping area from regions.py (Morocco without Western Sahara, or Utah), the same
   area as the cropland mask; written to one GeoJSON.
2. Ask NASA's Common Metadata Repository (CMR, public) for the LGRIP30 tiles inside the AOI's bounding box.
3. Download the tiles with the Earthdata token (skipped if already there).
4. Mosaic (VRT), clip to the AOI and write a compressed GeoTIFF; optionally a coarser copy (mode resampling).
5. Remove the downloaded tiles unless KEEP_TILES = True.
6. PNG map for visualisation and documentation: class colours (CLASSES), AOI boundary, legend with each class's
   share of the AOI (MAKE_PNG). PNG_ONLY = True skips steps 2-5 and draws the PNG from the GeoTIFF that is
   already there.

Output (OUT_DIR = outputs/<region>/data_download/reference/lgrip30/)
------------------------------------------------------------
LGRIP30_2015_<AOI_NAME>.tif         30 m, clipped to the AOI, 255 = outside the AOI / no data
LGRIP30_2015_<AOI_NAME>_250m.tif    coarse copy (COARSE_RES_DEG), if set
aoi_<AOI_NAME>.geojson              the AOI used for the clip
LGRIP30_2015_<AOI_NAME>.png         map with legend and AOI boundary (MAKE_PNG)

Run (from the project folder; set REGION below first)
------------------------------------------------------
    python data_download/04b_ref_lgrip30.py
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
OUT_DIR = cropland_dir(REGION) / "lgrip30"                 # outputs/<region>/data_download/reference/lgrip30/
COARSE_RES_DEG = 0.002245                                  # ~250 m copy for quick maps; None = no copy
KEEP_TILES = False                                         # True = keep the downloaded tiles in OUT_DIR/_tiles/
TOKEN_FILE = ROOT / "secrets" / "earthdata_token.txt"      # or environment variable EARTHDATA_TOKEN
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME, VERSION = "LGRIP30", "001"
NODATA = 255                                               # outside the AOI (class 0 is water, not no data)
MAKE_PNG = True                                            # step 6: PNG map with legend and AOI boundary
PNG_ONLY = True                                           # True = only draw the PNG from the existing GeoTIFF
PNG_MAX_PX = 2500                                          # longest side of the map image (pixels, mode resampling)
# ==========================================================================

# LGRIP30 classes (product legend) and the colours used for the maps
CLASSES = {0: ("Ocean and water bodies", "#0050cb"), 1: ("Non-croplands", "#d58855"),
           2: ("Irrigated croplands", "#c2d30c"), 3: ("Rainfed croplands", "#379a4b")}


def read_aoi():
    """The region's mapping area (regions.py) in EPSG:4326 as one feature."""
    return gpd.GeoDataFrame({"name": [AOI_NAME]}, geometry=[map_area(REGION)], crs=settings(REGION)["crs_epsg"]).to_crs(4326)


def read_token():
    token = os.environ.get("EARTHDATA_TOKEN") or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if not token:
        raise SystemExit(f"No Earthdata token: create one at https://urs.earthdata.nasa.gov (Profile -> Generate "
                         f"Token) and save it in {TOKEN_FILE} or in the environment variable EARTHDATA_TOKEN")
    return token


def find_tiles(bbox):
    """(name, GeoTIFF URL) of every LGRIP30 tile that intersects the bounding box, from NASA CMR (no login)."""
    import requests                                        # download-only package (not needed for PNG_ONLY)
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
    import requests
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
    ax.set_title(f"LGRIP30 irrigated / rainfed cropland, nominal 2015 (30 m) · {AOI_NAME}", fontsize=12)
    handles = [Patch(facecolor=CLASSES[c][1], edgecolor="0.5", label=f"{c} {CLASSES[c][0]} ({shares.get(c, 0):.1f}%)")
               for c in codes]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9,
              title="Class (share of the AOI)", title_fontsize=9)
    fig.text(0.01, 0.005, "Source: LGRIP30 v001", fontsize=7, color="0.4")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG -> {png}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    aoi = read_aoi()
    out = OUT_DIR / f"LGRIP30_2015_{AOI_NAME}.tif"
    if PNG_ONLY:
        if not out.exists():
            raise SystemExit(f"{out} not found: run once with PNG_ONLY = False")
        make_png(out, aoi, out.with_suffix(".png"))
        return
    from tqdm import tqdm
    tile_dir = OUT_DIR / "_tiles"
    tile_dir.mkdir(exist_ok=True)
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
    if MAKE_PNG:
        make_png(out, aoi, out.with_suffix(".png"))
    print("Done")


if __name__ == "__main__":
    main()
