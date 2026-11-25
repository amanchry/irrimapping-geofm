"""
02a_dynamic_world.py  —  binary Dynamic World cropland of every planned block, one 10 m GeoTIFF per block and year
================================================================================================================

What it does
------------
For each agricultural year and EVERY block of 01_plan_blocks.py, block by block,
computes the binary cropland from Dynamic World in Earth Engine (service account) and downloads it as one 1024 x 1024
px GeoTIFF. Finished blocks are kept, so a run that stops in the middle (quota, network, job time limit) continues where it stopped; after STOP_AFTER failed
blocks in a row the run stops by itself. Blocks with training tiles come first. At the end a VRT mosaic joins the
blocks of each year into one layer of the whole region (for QGIS and for reading any window):

  CROPLAND   1 = cropland, 0 = other, 255 = outside the mapping area or no Dynamic World image in the whole year
             cropland = "crops" (class 4) is the most frequent Dynamic World class in at least MIN_CROP_MONTHS (2)
             of the 12 months of the agricultural year (Morocco September - August, Utah October - September:
             the months of the Sentinel-2 / Sentinel-1 composites and of the labels)

Why monthly and not the class of the whole year: Dynamic World classifies every Sentinel-2 image (it has no annual
product). A rainfed cereal field is green for a few months (Morocco about December - April) and bare or dry grass
for the rest of the year, so its most frequent class of the year is often "bare" or "grass". Counting the months in
which "crops" dominates keeps short seasons, and two months guard against a single misclassified scene.

These blocks decide what comes next:
  - 02b_s2.py, 02c_s1.py (later ERA5) download only the blocks with at least MIN_CROP_PIXELS cropland pixels
    (download_common.py), plus every block with training tiles;
  - data_training_prep/04_cut_tiles.py cuts the training tiles (_dw.tif) out of them;
  - the maps classify only the cropland pixels of a block.

Output (outputs/<REGION>/data_download/dynamic_world/)
  <season>/<bx>_<by>.tif           uint8, 1 band CROPLAND, 10 m, NODATA 255, one file per block
  <season>/mosaic.vrt              VRT mosaic of all blocks of the year (rebuilt at the end of every run)
  cropland_map_<season>.png        cropland share of each 320 m cell (OVERVIEW_PX per block side), with legend
  cropland_blocks_<season>.png     the block extents (as in 01_plan_blocks.py): blocks with cropland highlighted,
                                   blocks without cropland, blocks not downloaded yet; blocks with training tiles
                                   outlined
The full Dynamic World land cover (all 9 classes) of each year is drawn by 04c_ref_dynamic_world_annual.py.

Run (from the project folder; after 01_plan_blocks.py; on the cluster for a whole region)
    python data_download/02a_dynamic_world.py
"""
import time

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from shapely.geometry import box
from shapely.prepared import prep

from download_common import (BLOCK_M, BLOCK_PX, DW_NODATA, DW_SOURCE, GSD_M, MIN_CROP_PIXELS, block_crop_pixels,
                             download_dir, dw_block_file, dw_mosaic_file, init_gee, load_blocks, month_bounds,
                             month_starts, process_blocks)
from regions import admin_boundaries, map_area, settings

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2023/24", "2020/21"]}[REGION]          # agricultural years
MIN_CROP_MONTHS = 2            # cropland = "crops" is the most frequent class in at least this many months
WORKERS = 4                    # blocks requested from Earth Engine in parallel
RETRIES = 4                    # attempts per block (waits 20, 40, 80 s between them)
STOP_AFTER = 20                # stop the run after this many failed blocks in a row (e.g. quota used up)
N_TEST_BLOCKS = None           # e.g. 2 blocks per year for a quick test; None = all
ONLY_BLOCKS = None             # e.g. [(48, 349)] for given blocks; None = all
DOWNLOAD = True                # False: only rebuild the mosaic and redraw the maps (no Earth Engine)
MAKE_PNG = True                # the two maps per agricultural year
OVERVIEW_PX = 32               # pixels per block side in the cropland map (32 -> 320 m)
# ==========================================================================

NAMES = ["CROPLAND"]
CROPS = 4                      # Dynamic World class "crops"
NODATA_U8 = DW_NODATA
CRS_EPSG = settings(REGION)["crs_epsg"]
EE = {}                        # Earth Engine module and the mapping area (set in main)


def make_image(ee, ag_season, bounds, area):
    """Binary cropland of one block and agricultural year: crops dominant in >= MIN_CROP_MONTHS of its 12 months;
    255 outside the mapping area (area = None: the block lies wholly inside) and where the year has no image."""
    proj = f"EPSG:{CRS_EPSG}"
    rect = ee.Geometry.Rectangle(list(bounds), proj, False)
    months = month_starts(REGION, ag_season)
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(rect)
          .filterDate(month_bounds(months[0])[0], month_bounds(months[-1])[1]).select("label"))
    empty = ee.ImageCollection([ee.Image.constant(0).rename("label").toUint8().updateMask(0)])
    crop_months = ee.Image.constant(0)
    for m in months:                                                # month without an image counts as not crops:
        mode = dw.filterDate(*month_bounds(m)).merge(empty).reduce(ee.Reducer.mode())   # masked dummy keeps 1 band
        crop_months = crop_months.add(mode.eq(CROPS).unmask(0))
    has_image = dw.count().gt(0)                                    # any Dynamic World image in the year
    cropland = crop_months.gte(MIN_CROP_MONTHS).updateMask(has_image)
    if area is not None:                                            # block on the border of the mapping area
        cropland = cropland.clip(ee.Geometry(area.__geo_interface__, proj, False))
    return cropland.unmask(NODATA_U8, sameFootprint=False).rename(NAMES).toUint8()


def compute(ag_season, bx, by):
    """CROPLAND of one block: [1, 1024, 1024] uint8 (one computePixels request on the block grid)."""
    ee = EE["ee"]
    x0, y0, x1, y1 = bx * BLOCK_M, by * BLOCK_M, (bx + 1) * BLOCK_M, (by + 1) * BLOCK_M
    blk = box(x0, y0, x1, y1)
    area = None if EE["inside"].contains(blk) else EE["area"].intersection(blk)
    request = {"expression": make_image(ee, ag_season, (x0, y0, x1, y1), area), "fileFormat": "NUMPY_NDARRAY",
               "grid": {"dimensions": {"width": BLOCK_PX, "height": BLOCK_PX},
                        "affineTransform": {"scaleX": GSD_M, "shearX": 0, "translateX": x0,
                                            "shearY": 0, "scaleY": -GSD_M, "translateY": y1},
                        "crsCode": f"EPSG:{CRS_EPSG}"}}
    for attempt in range(RETRIES):
        try:
            arr = ee.data.computePixels(request)
            break
        except Exception as err:                                    # rate limit, timeout, network
            if attempt == RETRIES - 1:
                raise
            print(f"  block {bx}_{by}: {str(err)[:120]}; retry in {20 * 2 ** attempt} s", flush=True)
            time.sleep(20 * 2 ** attempt)
    return np.asarray(arr[NAMES[0]], np.uint8)[None], None


def build_mosaic(ag_season):
    """VRT over all block files of the year (missing blocks read as 255)."""
    from osgeo import gdal
    gdal.UseExceptions()
    vrt = dw_mosaic_file(REGION, ag_season)
    files = sorted(str(p) for p in vrt.parent.glob("*.tif"))
    if not files:
        return
    gdal.BuildVRT(str(vrt), files, srcNodata=NODATA_U8, VRTNodata=NODATA_U8)
    print(f"  {ag_season}: mosaic of {len(files):,} blocks -> {vrt}")


def summary(ag_season):
    blocks = load_blocks(REGION, ag_season)
    n = block_crop_pixels(REGION, ag_season, blocks)
    blocks["crop_px"] = [np.nan if v is None else float(v) for v in n]
    done = blocks[blocks["crop_px"].notna()]
    print(f"{REGION} {ag_season}: Dynamic World cropland for {len(done):,} of {len(blocks):,} blocks | with cropland "
          f"(>= {MIN_CROP_PIXELS} px) {int((done['crop_px'] >= MIN_CROP_PIXELS).sum()):,}")
    return blocks


def background(ax, blocks):
    crs = settings(REGION)["crs_epsg"]
    try:
        admin_boundaries(REGION).to_crs(crs).plot(ax=ax, color="#f4f4f4", edgecolor="0.55", linewidth=0.5)
    except Exception:                                               # boundaries are only background
        pass
    gpd.GeoSeries([map_area(REGION)], crs=crs).boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=5)
    b = blocks.total_bounds
    pad = 0.03 * max(b[2] - b[0], b[3] - b[1])
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def cropland_map(blocks, ag_season, out):
    """Cropland share of every OVERVIEW_PX cell of the downloaded blocks, one mosaic over the mapping area."""
    bx0, by1 = int(blocks["bx"].min()), int(blocks["by"].max())
    w = (int(blocks["bx"].max()) - bx0 + 1) * OVERVIEW_PX
    h = (by1 - int(blocks["by"].min()) + 1) * OVERVIEW_PX
    share = np.full((h, w), np.nan, np.float32)                     # NaN = not downloaded
    noimg = np.zeros((h, w), bool)
    k = BLOCK_PX // OVERVIEW_PX                                     # block pixels per map cell
    cells = lambda m: m.reshape(OVERVIEW_PX, k, OVERVIEW_PX, k).sum(axis=(1, 3))
    for b in blocks[blocks["crop_px"].notna()].itertuples():
        with rasterio.open(dw_block_file(REGION, ag_season, b.bx, b.by)) as src:
            a = src.read(1)
            n_valid, n_crop = cells(a != NODATA_U8), cells(a == 1)
            r, c = (by1 - b.by) * OVERVIEW_PX, (b.bx - bx0) * OVERVIEW_PX
            share[r:r + OVERVIEW_PX, c:c + OVERVIEW_PX] = np.where(n_valid > 0, n_crop / np.maximum(n_valid, 1),
                                                                   np.nan)
            noimg[r:r + OVERVIEW_PX, c:c + OVERVIEW_PX] = n_valid == 0
    fig, ax = plt.subplots(figsize=(10, 10))
    background(ax, blocks)
    extent = (bx0 * BLOCK_M, (bx0 * BLOCK_M) + w / OVERVIEW_PX * BLOCK_M, (by1 + 1) * BLOCK_M - h / OVERVIEW_PX * BLOCK_M,
              (by1 + 1) * BLOCK_M)
    cmap = ListedColormap(["#ece6d6"] + [plt.get_cmap("YlOrBr")(v) for v in np.linspace(0.3, 1, 9)])
    im = ax.imshow(np.ma.masked_invalid(share), extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                   zorder=3)
    if noimg.any():
        ax.imshow(np.ma.masked_equal(noimg.astype(np.uint8), 0), extent=extent, cmap=ListedColormap(["#9e9e9e"]),
                  interpolation="nearest", zorder=4)
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label(f"cropland share of each {BLOCK_M / OVERVIEW_PX:.0f} m cell (0 = no cropland)")
    cb.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    n_done = int(blocks["crop_px"].notna().sum())
    total = blocks["crop_px"].sum() / (n_done * BLOCK_PX ** 2) if n_done else 0
    ax.legend(handles=[Patch(fc="#ece6d6", label="no cropland in the cell"),
                       Patch(fc="#9e9e9e", label="no Dynamic World image"),
                       Patch(fc="#f4f4f4", ec="0.55", label="block not downloaded yet")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=9, frameon=False)
    ax.set_title(f"{REGION.capitalize()} {ag_season}: Dynamic World cropland (crops the most frequent class in >= "
                 f"{MIN_CROP_MONTHS} months)\n"
                 f"{n_done:,} of {len(blocks):,} blocks downloaded; cropland {total:.1%} of their area", fontsize=11)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  map -> {out}")


def cropland_blocks(blocks, ag_season, out):
    """The block extents: blocks with cropland, without cropland, not downloaded; training blocks outlined."""
    fig, ax = plt.subplots(figsize=(10, 9))
    background(ax, blocks)
    done = blocks[blocks["crop_px"].notna()]
    crop = done[done["crop_px"] >= MIN_CROP_PIXELS]
    none = done[done["crop_px"] < MIN_CROP_PIXELS]
    todo = blocks[blocks["crop_px"].isna()]
    for part, fc in [(todo, "none"), (none, "#d9d9d9"), (crop, "#e49635")]:
        if len(part):
            part.plot(ax=ax, facecolor=fc, edgecolor="0.5", linewidth=0.15, zorder=3)
    train = blocks[blocks["has_training_tiles"]]
    if len(train):
        train.boundary.plot(ax=ax, color="#1f4e9c", linewidth=0.9, zorder=4)
    ax.legend(handles=[Patch(fc="#e49635", ec="0.5", label=f"block with cropland, >= {MIN_CROP_PIXELS} px ({len(crop):,}): "
                                                         f"S2 and S1 are downloaded"),
                       Patch(fc="#d9d9d9", ec="0.5", label=f"block without cropland ({len(none):,}): skipped"),
                       Patch(fc="none", ec="0.5", label=f"Dynamic World not downloaded yet ({len(todo):,})"),
                       Patch(fc="none", ec="#1f4e9c", lw=1.5, label=f"block with training tiles ({len(train):,}): "
                                                                    f"always downloaded")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9, frameon=False)
    ax.set_title(f"{REGION.capitalize()} {ag_season}: {len(blocks):,} blocks of {BLOCK_M / 1000:.2f} km; blocks with "
                 f"Dynamic World cropland", fontsize=11)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  map -> {out}")


if __name__ == "__main__":
    print(f"Dynamic World cropland for {REGION}, block by block")
    if DOWNLOAD:
        ee = init_gee()
        area = map_area(REGION)
        EE.update(ee=ee, area=area, inside=prep(area))
        process_blocks(REGION, DW_SOURCE, compute, NAMES, SEASONS, dtype=np.uint8, nodata=NODATA_U8,
                       workers=WORKERS, n_test_blocks=N_TEST_BLOCKS, only_blocks=ONLY_BLOCKS, stop_after=STOP_AFTER)
    for s in SEASONS:
        build_mosaic(s)
        blocks = summary(s)
        if MAKE_PNG and blocks["crop_px"].notna().any():
            folder = download_dir(REGION) / DW_SOURCE
            tag = s.replace("/", "-")
            cropland_map(blocks, s, folder / f"cropland_map_{tag}.png")
            cropland_blocks(blocks, s, folder / f"cropland_blocks_{tag}.png")
