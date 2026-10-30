"""
a2_map_reference_products.py  —  maps of existing cropland / irrigation products over Morocco
=============================================================================================

What it does
------------
Draws two reference maps for the thesis, using Google Earth Engine:
  * ESA WorldCover v200 (2021, 10 m): cropland (class 40).
  * LGRIP30 (GFSAD Landsat-derived rainfed and irrigated cropland, nominal 2015, 30 m;
    Teluguntla et al. 2023, doi:10.5067/Community/LGRIP/LGRIP30.001): irrigated (class 2) and
    rainfed (class 3) cropland.

Steps
-----
1. Sign in to Earth Engine; read the Morocco boundary.
2. Mosaic each product over Morocco and keep the classes of interest.
3. Render each as a PNG through Earth Engine (getThumbURL) and download it.
4. Draw each PNG with the country outline, a legend and axes, and save it.

Output (outputs/analysis/reference_maps/)
-----------------------------------------
morocco_esa_cropland.png, morocco_lgrip_irrigated_rainfed.png

Run (from the project folder)
-----------------------------
    python tools/a2_map_reference_products.py
"""
import io
import json
from pathlib import Path

import ee
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"
BOUNDARY = ROOT / "data" / "admin" / "gadm41_MAR_0.json"
OUT_DIR = ROOT / "outputs" / "analysis" / "reference_maps"
PNG_WIDTH = 1400                                           # pixels of the downloaded rendering

ESA_CROP_COLOR = "#f096ff"
LGRIP_IRRIGATED = "#c2d30c"
LGRIP_RAINFED = "#379a4b"
LAND_COLOR = "#f4f1ea"
# ==========================================================================


def init_gee():
    key = json.loads(KEY_FILE.read_text())
    credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
    ee.Initialize(credentials, project=key["project_id"])
    print("Earth Engine initialised")


def paint(region, overlay, palette):
    """Country in a light land colour with the classes painted on top."""
    land = ee.Image.constant(1).clip(region).visualize(min=0, max=1, palette=[LAND_COLOR])
    painted = overlay.visualize(min=1, max=len(palette), palette=palette)
    return ee.ImageCollection([land, painted]).mosaic()


def download_png(image, region):
    url = image.getThumbURL({"region": region, "dimensions": PNG_WIDTH, "format": "png"})
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGBA")


def draw(image, extent, boundary, title, handles, path):
    fig, ax = plt.subplots(figsize=(8, 7.2), constrained_layout=True)
    ax.imshow(image, extent=extent, interpolation="nearest")
    boundary.boundary.plot(ax=ax, color="#333333", linewidth=0.6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=12)
    ax.legend(handles=handles + [Line2D([0], [0], color="#333333", linewidth=0.8, label="Morocco")],
              loc="lower left", frameon=True, fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    init_gee()
    boundary = gpd.read_file(BOUNDARY).to_crs(4326)
    morocco = ee.Geometry(boundary.union_all().__geo_interface__)
    bounds = morocco.bounds()
    xs, ys = zip(*bounds.coordinates().getInfo()[0])
    extent = [min(xs), max(xs), min(ys), max(ys)]

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(morocco).mosaic().select("Map").clip(morocco)
    esa_cropland = worldcover.eq(40).selfMask()
    lgrip = ee.ImageCollection("projects/sat-io/open-datasets/GFSAD/LGRIP30").mosaic().clip(morocco)
    lgrip = lgrip.select([lgrip.bandNames().get(0)])
    lgrip_crop = lgrip.remap([2, 3], [1, 2]).selfMask()        # 2 irrigated -> 1, 3 rainfed -> 2

    draw(download_png(paint(morocco, esa_cropland, [ESA_CROP_COLOR]), bounds), extent, boundary,
         "ESA WorldCover cropland, 2021", [Patch(facecolor=ESA_CROP_COLOR, label="Cropland")],
         OUT_DIR / "morocco_esa_cropland.png")
    draw(download_png(paint(morocco, lgrip_crop, [LGRIP_IRRIGATED, LGRIP_RAINFED]), bounds), extent, boundary,
         "LGRIP30 irrigated and rainfed, 2015",
         [Patch(facecolor=LGRIP_IRRIGATED, label="Irrigated"), Patch(facecolor=LGRIP_RAINFED, label="Rainfed")],
         OUT_DIR / "morocco_lgrip_irrigated_rainfed.png")


if __name__ == "__main__":
    main()
