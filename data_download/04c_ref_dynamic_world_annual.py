"""
04c_ref_dynamic_world_annual.py  —  annual Dynamic World 10 m land cover and cropland for a region (reference layer)
===================================================================================================================

What it does
------------
Dynamic World (Brown et al. 2022, Scientific Data 9, 251; https://dynamicworld.app) classifies every Sentinel-2
image into 9 land-cover classes ("label"). There is one layer per image, not per year, so this script builds an
ANNUAL layer for each calendar year (1 January - 31 December) in Earth Engine and downloads it for the whole region
as ONE 10 m GeoTIFF with geemap.download_ee_image (geedim splits the image into tiles below Earth Engine's request
limit, downloads them in parallel and joins them):

  label       most frequent class of the year (mode of the per-image labels), all 9 classes; cropland = label 4
              (crops), the same rule as 02a_dynamic_world.py
  crop_freq   optional (BANDS setting): share of the year's images labelled "crops" in percent (0-100); with it
              the cropland map uses crop_freq >= CROP_THRESHOLD instead of label 4
One band keeps the download small: Morocco is 117,760 x 93,184 px (the rectangle around the country), 11 GB per band
before compression; outside the country and over desert the compressed file is much smaller.

It is a reference layer, like 04a (ESA WorldCover) and 04b (LGRIP30): no model or dataset script uses it. The
download of the blocks uses its own cropland image (02a_dynamic_world.py).

Data
----
Earth Engine collection GOOGLE/DYNAMICWORLD/V1 (from 2015-06-27, Sentinel-2 L1C, 10 m). Classes (label values):
  0 water · 1 trees · 2 grass · 3 flooded vegetation · 4 crops · 5 shrub and scrub · 6 built · 7 bare ·
  8 snow and ice
Calendar years; the agricultural year of the labels is mostly in the second calendar year (Morocco September -
August: 2023/24 -> 2024; Utah October - September: 2023/24 -> 2024), so YEARS holds those years.
Earth Engine is reached with the service-account key in secrets/ (KEY_FILE; never commit it). Needs geemap and
geedim < 2 (geemap 0.37 calls the geedim 1.x API).

Steps
-----
1. Region: the mapping area of regions.py (map_file / map_where), simplified to about 100 m for the request.
2. Per year: Earth Engine computes the four bands, clipped to the region; pixels outside it or without an image are
   255. The image is downloaded on the project grid (region's projection, 10 m, origin on the 10.24 km block grid)
   to one GeoTIFF; overviews are added so the file opens quickly in QGIS. A finished year is skipped (OVERWRITE).
3. Two PNG maps per year (MAKE_PNG): the land cover (all 9 classes, Dynamic World colours, share of each class in the
   legend) and the binary cropland derived from it. PNG_ONLY = True only draws the PNGs from the GeoTIFFs already
   there.

Output (OUT_DIR = outputs/<region>/data_download/reference/dynamic_world/)
--------------------------------------------------------------------------
DynamicWorld_<year>_<AOI_NAME>.tif            band label (+ crop_freq if in BANDS); uint8; 10 m; NODATA 255
DynamicWorld_<year>_<AOI_NAME>.png            land cover map (all classes) with legend
DynamicWorld_cropland_<year>_<AOI_NAME>.png   binary cropland map with legend
info.json                                     settings, class codes, years

Run (from the project folder; set REGION below first)
------------------------------------------------------
    python data_download/04c_ref_dynamic_world_annual.py
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")                                      # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling

from regions import admin_boundaries, cropland_dir, map_area, settings  # noqa: E402

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
REGION = "morocco"                                         # "morocco" | "utah" (regions.py)
YEARS = {"morocco": [2024, 2025],                          # calendar years (1 Jan - 31 Dec); one annual layer each
         "utah": [2021, 2024]}[REGION]                     # (label years 2023/24, 2024/25 and 2020/21, 2023/24)
BANDS = ["label"]                                          # ["label"] (11 GB raw for Morocco) or ["label", "crop_freq"]
CROP_THRESHOLD = 0.50                                      # with crop_freq: cropland = share labelled crops >= this
AOI_NAME = REGION.capitalize()                             # used in the output file names
KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"     # Earth Engine key (never commit)
OUT_DIR = cropland_dir(REGION) / "dynamic_world"           # outputs/<region>/data_download/reference/dynamic_world/
GSD_M = 10                                                 # output pixel (m)
NUM_THREADS = 4                                            # tiles downloaded in parallel (geedim)
MAX_TILE_SIZE = 4                                          # MB per tile request (geedim; Earth Engine limit 32 MB)
MAX_TILE_DIM = 2048                                        # pixels per tile side (geedim)
OVERWRITE = False                                          # False: skip a year whose GeoTIFF exists
MAKE_PNG = True
PNG_ONLY = False                                           # True = only draw the PNGs from the existing GeoTIFFs
PNG_MAX_PX = 2500                                          # longest side of the map image (pixels, mode resampling)
# ==========================================================================

CRS_EPSG = settings(REGION)["crs_epsg"]
GRID_M = 10240                                             # the 10.24 km block grid of data_download/ (origin)
NODATA = 255
COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
CROPS = 4                                                  # Dynamic World class "crops"
# Dynamic World classes (label values) and the colours of the Dynamic World app
CLASSES = {0: ("Water", "#419bdf"), 1: ("Trees", "#397d49"), 2: ("Grass", "#88b053"),
           3: ("Flooded vegetation", "#7a87c6"), 4: ("Crops", "#e49635"), 5: ("Shrub and scrub", "#dfc35a"),
           6: ("Built", "#c4281b"), 7: ("Bare", "#a59b8f"), 8: ("Snow and ice", "#b39fe1")}


def tif_path(year):
    return OUT_DIR / f"DynamicWorld_{year}_{AOI_NAME}.tif"


# ------------------------------------------------------------ Earth Engine
def init_gee():
    """Earth Engine with the service-account key in secrets/ (KEY_FILE)."""
    import ee
    key = json.loads(KEY_FILE.read_text())
    ee.Initialize(ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key)), project=key["project_id"])
    print("Earth Engine initialised (service account)")
    return ee


def ee_region(ee, area):
    """The mapping area as an Earth Engine geometry (EPSG:4326, simplified to ~100 m to keep the request small)."""
    geom = gpd.GeoSeries([area], crs=CRS_EPSG).to_crs(4326).simplify(0.001).iloc[0]
    return ee.Geometry(geom.__geo_interface__)


def annual_image(ee, year, region):
    """label (mode) and, if in BANDS, crop_freq (% of the images labelled crops) of one calendar year."""
    dw = (ee.ImageCollection(COLLECTION).filterBounds(region)
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01").select("label"))
    n = dw.size().getInfo()
    print(f"  {year}: {n:,} Dynamic World images over the region")
    if n == 0:
        raise RuntimeError(f"no Dynamic World images for {year}")
    bands = {"label": lambda: dw.reduce(ee.Reducer.mode()),
             "crop_freq": lambda: dw.map(lambda img: img.eq(CROPS)).mean().multiply(100).round()}
    image = ee.Image.cat([bands[b]().rename(b) for b in BANDS])
    return image.clip(region).unmask(NODATA, sameFootprint=False).toUint8()


def grid(area):
    """Transform and shape of the output: 10 m pixels, origin snapped to the 10.24 km block grid."""
    minx, miny, maxx, maxy = area.bounds
    x0, y1 = np.floor(minx / GRID_M) * GRID_M, np.ceil(maxy / GRID_M) * GRID_M
    x1, y0 = np.ceil(maxx / GRID_M) * GRID_M, np.floor(miny / GRID_M) * GRID_M
    shape = (int(round((y1 - y0) / GSD_M)), int(round((x1 - x0) / GSD_M)))       # (height, width)
    return [GSD_M, 0, float(x0), 0, -GSD_M, float(y1)], shape


def download_year(ee, year, area, region):
    import geemap
    out = tif_path(year)
    transform, shape = grid(area)
    print(f"  {year}: {shape[1]:,} x {shape[0]:,} px at {GSD_M} m (EPSG:{CRS_EPSG}) -> {out}")
    geemap.download_ee_image(image=annual_image(ee, year, region), filename=str(out), crs=f"EPSG:{CRS_EPSG}",
                             crs_transform=transform, shape=shape, dtype="uint8", num_threads=NUM_THREADS,
                             max_tile_size=MAX_TILE_SIZE, max_tile_dim=MAX_TILE_DIM, overwrite=True)
    with rasterio.open(out, "r+") as dst:                           # band names, no-data, overviews for QGIS
        dst.nodata = NODATA
        for i, b in enumerate(BANDS, start=1):
            dst.set_band_description(i, b)
        dst.build_overviews([4, 16, 64, 256], Resampling.mode)
    print(f"  {year}: {out.stat().st_size / 1024 ** 3:.2f} GB")


# ------------------------------------------------------------ maps
def read_overview(tif, band, names=None):
    """One band at <= PNG_MAX_PX on the longest side (mode resampling) and the bounds."""
    with rasterio.open(tif) as src:
        names = names or list(src.descriptions)
        scale = max(src.width, src.height) / PNG_MAX_PX
        shape = (max(1, round(src.height / scale)), max(1, round(src.width / scale))) if scale > 1 else (src.height, src.width)
        return src.read(names.index(band) + 1, out_shape=shape, resampling=Resampling.mode), src.bounds, shape


def draw_map(arr, bounds, shape, colours, handles, title, png, legend_title):
    b = bounds
    fig, ax = plt.subplots(figsize=(10, 10 * shape[0] / shape[1] + 1.2), dpi=150)
    ax.imshow(np.ma.masked_equal(arr, NODATA), cmap=ListedColormap(colours), vmin=0, vmax=len(colours) - 1,
              interpolation="nearest", extent=[b.left, b.right, b.bottom, b.top])
    admin_boundaries(REGION).to_crs(CRS_EPSG).boundary.plot(ax=ax, color="#222222", linewidth=0.6)
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title, fontsize=12)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9,
              title=legend_title, title_fontsize=9)
    fig.text(0.01, 0.005, "Source: Dynamic World V1 (Brown et al. 2022), Google Earth Engine", fontsize=7, color="0.4")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG -> {png}")


def make_png(tif, year):
    """Land cover map (all classes) and binary cropland map of one year."""
    arr, bounds, shape = read_overview(tif, "label")
    codes = sorted(CLASSES)
    valid = np.isin(arr, codes)
    shares = {c: 100 * float((arr[valid] == c).mean()) for c in codes} if valid.any() else {}
    handles = [Patch(facecolor=CLASSES[c][1], edgecolor="0.5", label=f"{c} {CLASSES[c][0]} ({shares.get(c, 0):.1f}%)")
               for c in codes]
    draw_map(np.where(valid, arr, NODATA), bounds, shape, [CLASSES[c][1] for c in codes], handles,
             f"Dynamic World annual land cover {year} (most frequent class, 10 m) · {AOI_NAME}",
             OUT_DIR / f"DynamicWorld_{year}_{AOI_NAME}.png", "Class (share of the region)")
    with rasterio.open(tif) as src:
        names = list(src.descriptions)
    if "crop_freq" in names:
        arr, bounds, shape = read_overview(tif, "crop_freq", names)
        valid = arr != NODATA
        arr = np.where(valid, (arr >= 100 * CROP_THRESHOLD).astype(np.uint8), NODATA)
        rule = f"images labelled crops >= {CROP_THRESHOLD:.0%} of the year"
    else:
        arr, bounds, shape = read_overview(tif, "label", names)
        valid = np.isin(arr, sorted(CLASSES))
        arr = np.where(valid, (arr == CROPS).astype(np.uint8), NODATA)
        rule = "most frequent class = crops"
    share = 100 * float((arr[valid] == 1).mean()) if valid.any() else 0
    colours = ["#ece6d6", CLASSES[CROPS][1]]
    handles = [Patch(facecolor=colours[1], edgecolor="0.5", label=f"cropland ({share:.1f}%)"),
               Patch(facecolor=colours[0], edgecolor="0.5", label=f"other land cover ({100 - share:.1f}%)")]
    draw_map(np.where(valid, arr, NODATA), bounds, shape, colours, handles,
             f"Dynamic World cropland {year}: {rule} · {AOI_NAME}",
             OUT_DIR / f"DynamicWorld_cropland_{year}_{AOI_NAME}.png", "Share of the region")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PNG_ONLY:
        ee = init_gee()
        area = map_area(REGION)
        region = ee_region(ee, area)
        (OUT_DIR / "info.json").write_text(json.dumps({
            "region": REGION, "collection": COLLECTION, "crs_epsg": CRS_EPSG, "gsd_m": GSD_M,
            "nodata": NODATA, "bands": BANDS,
            "band_meaning": {"label": "mode of the per-image labels of the calendar year (cropland = 4)",
                             "crop_freq": f"share of images labelled crops (%); cropland map uses >= {CROP_THRESHOLD:.0%}"},
            "classes": {c: n for c, (n, _) in CLASSES.items()},
            "years": {y: [f"{y}-01-01", f"{y + 1}-01-01"] for y in YEARS}}, indent=2))   # [start, end)
        for year in YEARS:
            if tif_path(year).exists() and not OVERWRITE:
                print(f"  {year}: {tif_path(year).name} exists (OVERWRITE = True to download again)")
                continue
            download_year(ee, year, area, region)
    if MAKE_PNG or PNG_ONLY:
        for year in YEARS:
            if tif_path(year).exists():
                make_png(tif_path(year), year)
            else:
                print(f"  {year}: {tif_path(year).name} not found; no PNG")
    print("Done")


if __name__ == "__main__":
    main()
