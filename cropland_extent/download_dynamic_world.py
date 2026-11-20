"""
download_dynamic_world.py  —  annual Dynamic World 10 m land cover for a region (reference layer for the mask)
==================================================================================================================

What it does
------------
Dynamic World (Brown et al. 2022, Scientific Data 9, 251; https://dynamicworld.app) classifies every Sentinel-2
image into 9 land-cover classes, with one probability per class and the most likely class ("label"). There is one
layer per image, not per year, so this script builds an ANNUAL layer for each agricultural year of the region in
Earth Engine and downloads it on the 10 m grid of the project (the grid of the tiles and of the cropland mask):

  label    most frequent class of the year (mode of the per-image labels)
  crops    mean probability of the class "crops" over the year, in percent (0-100)
  n_obs    number of Dynamic World images of the pixel in the year (capped at 254)

It is a reference layer, like download_esa_worldcover.py and download_lgrip30.py: no model or dataset script uses it.
It is meant for the later choice of a cropland mask (e.g. Dynamic World alone, by class or by a threshold on the
crop probability, compared with the union of products of make_cropland_mask.py).

Data
----
Earth Engine collection GOOGLE/DYNAMICWORLD/V1 (from 2015-06-27, Sentinel-2 L1C, 10 m). Classes (label values):
  0 water · 1 trees · 2 grass · 3 flooded vegetation · 4 crops · 5 shrub and scrub · 6 built · 7 bare ·
  8 snow and ice
Agricultural year: START_MONTH of regions.py (Morocco 1 September - 31 August, Utah 1 October - 30 September).
Needs the Earth Engine service-account key in secrets/ (KEY_FILE; never commit it).

Steps
-----
1. Plan the download blocks (BLOCK_PX x BLOCK_PX pixels on the project grid) that touch the region's mapping area
   (regions.py: map_file / map_where, the same area as the cropland mask).
2. Per block: Earth Engine computes the annual layers (mode of the labels, mean crop probability, image count) and
   returns them with computePixels; pixels outside the mapping area get NODATA. One GeoTIFF per block; re-running
   skips finished blocks (resumable). A block that runs out of Earth Engine memory is fetched in four quarters.
3. Mosaic the blocks into one compressed GeoTIFF per year (3 bands, uint8).
4. PNG map of the annual label with the Dynamic World colours and each class's share of the region (MAKE_PNG).
   PNG_ONLY = True skips steps 1-3 and draws the PNG from the GeoTIFF that is already there.

Output (OUT_DIR = outputs/<region>/cropland_extent/dynamic_world/)
-------------------------------------------------------------------
DynamicWorld_<season>_<AOI_NAME>.tif     bands: label, crops, n_obs; uint8; NODATA 255 outside the region
DynamicWorld_<season>_<AOI_NAME>.png     map of the annual label with legend (MAKE_PNG)
blocks/<season>/<bx>_<by>.tif            the downloaded blocks (kept unless KEEP_BLOCKS = False)
info.json                                settings, class codes, years

Run (from the project folder; set REGION below first)
------------------------------------------------------
    python cropland_extent/download_dynamic_world.py
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                      # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixelwise_run"))   # regions.py is shared
from regions import admin_boundaries, cropland_dir, map_area, settings  # noqa: E402

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
REGION = "morocco"                                         # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2020/21", "2023/24"]}[REGION]         # agricultural years; one annual layer per year
AOI_NAME = REGION.capitalize()                             # used in the output file names
KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"     # Earth Engine key (never commit)
OUT_DIR = cropland_dir(REGION) / "dynamic_world"           # outputs/<region>/cropland_extent/dynamic_world/
GSD_M = 10
BLOCK_PX = 2048                                            # download block (20.48 km); a multiple of the 128 px tile
DOWNLOAD_WORKERS = 8                                       # parallel Earth Engine requests
RETRIES = 4
N_TEST_BLOCKS = None                                       # e.g. 2 blocks for a quick test; None = all
KEEP_BLOCKS = True                                         # False = delete the block files after the mosaic
MAKE_PNG = True
PNG_ONLY = False                                           # True = only draw the PNG from the existing GeoTIFF
PNG_MAX_PX = 2500                                          # longest side of the map image (pixels, mode resampling)
# ==========================================================================

CRS_EPSG = settings(REGION)["crs_epsg"]
START_MONTH = settings(REGION)["start_month"]
BLOCK_M = BLOCK_PX * GSD_M
NODATA = 255
BANDS = ["label", "crops", "n_obs"]
COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
# Dynamic World classes (label values) and the colours of the Dynamic World app
CLASSES = {0: ("Water", "#419bdf"), 1: ("Trees", "#397d49"), 2: ("Grass", "#88b053"),
           3: ("Flooded vegetation", "#7a87c6"), 4: ("Crops", "#e49635"), 5: ("Shrub and scrub", "#dfc35a"),
           6: ("Built", "#c4281b"), 7: ("Bare", "#a59b8f"), 8: ("Snow and ice", "#b39fe1")}


def season_tag(season):
    return season.replace("/", "-")


def season_dates(season):
    start = pd.Timestamp(year=int(season[:4]), month=START_MONTH, day=1)
    return start, start + pd.DateOffset(months=12)


# ------------------------------------------------------------ Earth Engine
def init_gee():
    import ee
    key = json.loads(KEY_FILE.read_text())
    ee.Initialize(ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key)), project=key["project_id"])
    print("Earth Engine initialised")
    return ee


def annual_image(ee, season, geom):
    """label (mode), crops (mean probability, %), n_obs (image count) of one agricultural year, uint8."""
    start, end = season_dates(season)
    dw = ee.ImageCollection(COLLECTION).filterBounds(geom).filterDate(str(start.date()), str(end.date()))
    label = dw.select("label").reduce(ee.Reducer.mode()).rename("label")
    crops = dw.select("crops").mean().multiply(100).round().rename("crops")
    n_obs = dw.select("label").count().min(254).rename("n_obs")
    return ee.Image.cat([label, crops, n_obs]).unmask(NODATA).toUint8()


def compute_grid(ee, season, x0, y0, size):
    """The three bands of one square of size x size pixels with top-left corner (x0, y0)."""
    geom = ee.Geometry.Rectangle([x0, y0 - size * GSD_M, x0 + size * GSD_M, y0], f"EPSG:{CRS_EPSG}", False)
    arr = ee.data.computePixels({
        "expression": annual_image(ee, season, geom),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": size, "height": size},
                 "affineTransform": {"scaleX": GSD_M, "shearX": 0, "translateX": float(x0),
                                     "shearY": 0, "scaleY": -GSD_M, "translateY": float(y0)},
                 "crsCode": f"EPSG:{CRS_EPSG}"},
    })
    return np.stack([arr[b].astype(np.uint8) for b in BANDS])


def download_block(ee, season, blk):
    """One block [3, BLOCK_PX, BLOCK_PX]; if Earth Engine runs out of memory, the block is fetched in 4 quarters."""
    x0, y0 = blk["bx"] * BLOCK_M, (blk["by"] + 1) * BLOCK_M
    split = False
    for attempt in range(1, RETRIES + 1):
        try:
            if not split:
                return compute_grid(ee, season, x0, y0, BLOCK_PX)
            h = BLOCK_PX // 2
            out = np.full((len(BANDS), BLOCK_PX, BLOCK_PX), NODATA, np.uint8)
            for r in range(2):
                for c in range(2):
                    out[:, r * h:(r + 1) * h, c * h:(c + 1) * h] = compute_grid(
                        ee, season, x0 + c * h * GSD_M, y0 - r * h * GSD_M, h)
            return out
        except Exception as err:                                            # noqa: BLE001
            split = split or "memory" in str(err).lower() or "limit" in str(err).lower()
            if attempt == RETRIES:
                print(f"  block {blk['bx']}_{blk['by']}: failed ({str(err)[:150]})")
                return None
            time.sleep(5 if split else 10 * attempt)


# ------------------------------------------------------------ local steps
def block_transform(bx, by):
    return Affine(GSD_M, 0, bx * BLOCK_M, 0, -GSD_M, (by + 1) * BLOCK_M)


def plan_blocks(area):
    from shapely.prepared import prep
    minx, miny, maxx, maxy = area.bounds
    pa = prep(area)
    blocks = []
    for bx in range(int(np.floor(minx / BLOCK_M)), int(np.floor(maxx / BLOCK_M)) + 1):
        for by in range(int(np.floor(miny / BLOCK_M)), int(np.floor(maxy / BLOCK_M)) + 1):
            b = box(bx * BLOCK_M, by * BLOCK_M, (bx + 1) * BLOCK_M, (by + 1) * BLOCK_M)
            if pa.intersects(b):
                blocks.append({"bx": bx, "by": by, "geometry": b})
    return blocks


def inside(area, blk):
    """True = pixel centre inside the mapping area."""
    clip = area.intersection(blk["geometry"])
    if clip.is_empty:
        return np.zeros((BLOCK_PX, BLOCK_PX), bool)
    return rasterize([clip], out_shape=(BLOCK_PX, BLOCK_PX), transform=block_transform(blk["bx"], blk["by"]),
                     fill=0, dtype="uint8").astype(bool)


def write_block(path, arr, blk):
    tmp = path.with_suffix(".part.tif")
    profile = dict(driver="GTiff", width=BLOCK_PX, height=BLOCK_PX, count=len(BANDS), dtype="uint8",
                   crs=f"EPSG:{CRS_EPSG}", transform=block_transform(blk["bx"], blk["by"]), nodata=NODATA,
                   compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr)
        for i, b in enumerate(BANDS, start=1):
            dst.set_band_description(i, b)
    tmp.replace(path)


def mosaic(block_dir, out):
    from osgeo import gdal
    gdal.UseExceptions()
    files = sorted(str(p) for p in block_dir.glob("*.tif") if not p.name.endswith(".part.tif"))
    vrt = block_dir / "mosaic.vrt"
    gdal.BuildVRT(str(vrt), files, srcNodata=NODATA, VRTNodata=NODATA)
    gdal.Translate(str(out), str(vrt), creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"])
    for i, b in enumerate(BANDS, start=1):
        with rasterio.open(out, "r+") as dst:
            dst.set_band_description(i, b)
    vrt.unlink()


def make_png(tif, png, season):
    """Map of the annual label (read at <= PNG_MAX_PX, mode resampling) with the region outline and a legend."""
    with rasterio.open(tif) as src:
        scale = max(src.width, src.height) / PNG_MAX_PX
        shape = (max(1, round(src.height / scale)), max(1, round(src.width / scale))) if scale > 1 else (src.height, src.width)
        arr = src.read(1, out_shape=shape, resampling=Resampling.mode)
        b = src.bounds
    codes = sorted(CLASSES)
    valid = np.isin(arr, codes)
    shares = {c: 100 * float((arr[valid] == c).mean()) for c in codes} if valid.any() else {}
    fig, ax = plt.subplots(figsize=(10, 10 * shape[0] / shape[1] + 1.2), dpi=150)
    ax.imshow(np.ma.masked_where(~valid, arr), cmap=ListedColormap([CLASSES[c][1] for c in codes]), vmin=0,
              vmax=len(codes) - 1, interpolation="nearest", extent=[b.left, b.right, b.bottom, b.top])
    admin_boundaries(REGION).to_crs(CRS_EPSG).boundary.plot(ax=ax, color="#222222", linewidth=0.6)
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Dynamic World annual land cover {season} (most frequent class, 10 m) · {AOI_NAME}", fontsize=12)
    handles = [Patch(facecolor=CLASSES[c][1], edgecolor="0.5", label=f"{c} {CLASSES[c][0]} ({shares.get(c, 0):.1f}%)")
               for c in codes]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9,
              title="Class (share of the region)", title_fontsize=9)
    fig.text(0.01, 0.005, "Source: Dynamic World V1 (Brown et al. 2022), Google Earth Engine", fontsize=7, color="0.4")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG -> {png}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PNG_ONLY:
        for season in SEASONS:
            tif = OUT_DIR / f"DynamicWorld_{season_tag(season)}_{AOI_NAME}.tif"
            if not tif.exists():
                raise SystemExit(f"{tif} not found: run once with PNG_ONLY = False")
            make_png(tif, tif.with_suffix(".png"), season)
        return
    ee = init_gee()
    area = map_area(REGION)
    blocks = plan_blocks(area)
    if N_TEST_BLOCKS:
        blocks = blocks[:N_TEST_BLOCKS]
    (OUT_DIR / "info.json").write_text(json.dumps({
        "region": REGION, "seasons": SEASONS, "collection": COLLECTION, "crs_epsg": CRS_EPSG, "gsd_m": GSD_M,
        "block_px": BLOCK_PX, "bands": {"label": "mode of the per-image labels of the agricultural year",
                                        "crops": "mean probability of crops (%)", "n_obs": "number of images (<= 254)"},
        "classes": {c: n for c, (n, _) in CLASSES.items()}, "nodata": NODATA,
        "years": {s: [str(d.date()) for d in season_dates(s)] for s in SEASONS}}, indent=2))
    for season in SEASONS:
        tag = season_tag(season)
        block_dir = OUT_DIR / "blocks" / tag
        block_dir.mkdir(parents=True, exist_ok=True)
        todo = [b for b in blocks if not (block_dir / f"{b['bx']}_{b['by']}.tif").exists()]
        print(f"== {REGION} {season}: {len(blocks):,} blocks of {BLOCK_M / 1000:.2f} km, {len(todo):,} to download")

        def one(blk):
            arr = download_block(ee, season, blk)
            if arr is None:
                return False
            arr[:, ~inside(area, blk)] = NODATA
            write_block(block_dir / f"{blk['bx']}_{blk['by']}.tif", arr, blk)
            return True

        done = 0
        with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
            for n, ok in enumerate(pool.map(one, todo), start=1):
                done += ok
                if n % 20 == 0 or n == len(todo):
                    print(f"  {n:,}/{len(todo):,} blocks ({done:,} written)")
        missing = [b for b in blocks if not (block_dir / f"{b['bx']}_{b['by']}.tif").exists()]
        if missing:
            print(f"  {len(missing)} block(s) failed: run again to fetch them; no mosaic yet")
            continue
        out = OUT_DIR / f"DynamicWorld_{tag}_{AOI_NAME}.tif"
        mosaic(block_dir, out)
        print(f"  mosaic -> {out}")
        if not KEEP_BLOCKS:
            for p in block_dir.glob("*.tif"):
                p.unlink()
        if MAKE_PNG:
            make_png(out, out.with_suffix(".png"), season)
    print("Done")


if __name__ == "__main__":
    main()
