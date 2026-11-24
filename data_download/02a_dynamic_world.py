"""
02a_dynamic_world.py  —  binary Dynamic World cropland of the whole region, one 10 m GeoTIFF per agricultural year
==================================================================================================================

What it does
------------
Downloads, for each agricultural year, one binary cropland image of the whole mapping area from Dynamic World
(Earth Engine, service account), with geemap.download_ee_image (geedim splits it into tiles below Earth Engine's
request limit, downloads them in parallel and joins them), like the reference maps 04a-04c. The file covers exactly
the blocks of 01_plan_blocks.py (10 m pixels on the block grid), so every block and every 128 px tile is a window of
it:

  CROPLAND   1 = cropland, 0 = other, 255 = no Dynamic World image in the whole year
             cropland = "crops" (class 4) is the most frequent Dynamic World class in at least MIN_CROP_MONTHS (2)
             of the 12 months of the agricultural year (Morocco September - August, Utah October - September:
             the months of the Sentinel-2 / Sentinel-1 composites and of the labels)

Why monthly and not the class of the whole year: Dynamic World classifies every Sentinel-2 image (it has no annual
product). A rainfed cereal field is green for a few months (Morocco about December - April) and bare or dry grass
for the rest of the year, so its most frequent class of the year is often "bare" or "grass". Counting the months in
which "crops" dominates keeps short seasons, and two months guard against a single misclassified scene.

This image decides what comes next:
  - 02b_s2.py, 02c_s1.py (later ERA5) download only the blocks with at least MIN_CROP_PIXELS cropland pixels
    (download_common.py), plus every block with training tiles;
  - data_training_prep/04_cut_tiles.py cuts it into the training tiles (_dw.tif);
  - the maps classify only the cropland pixels of a block.

Output (outputs/<REGION>/data_download/dynamic_world/)
  cropland_<season>.tif            uint8, 1 band CROPLAND, 10 m, NODATA 255, overviews for QGIS
  cropland_map_<season>.png        cropland share of each 320 m cell (OVERVIEW_PX per block side) over the mapping
                                   area, with legend
  cropland_blocks_<season>.png     the block extents (as in 01_plan_blocks.py): blocks with cropland highlighted,
                                   blocks without cropland; blocks with training tiles outlined
The full Dynamic World land cover (all 9 classes) of each year is drawn by 04c_ref_dynamic_world_annual.py.

Run (from the project folder; after 01_plan_blocks.py)
    python data_download/02a_dynamic_world.py
"""
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from matplotlib.patches import Patch

from download_common import (BLOCK_M, BLOCK_PX, DW_NODATA, GSD_M, MIN_CROP_PIXELS, block_crop_pixels, download_dir,
                             dw_cropland_file, init_gee, load_blocks, month_bounds, month_starts, read_dw_window)
from regions import admin_boundaries, map_area, settings

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2023/24", "2020/21"]}[REGION]          # agricultural years (block plan of each from 01)
MIN_CROP_MONTHS = 2            # cropland = "crops" is the most frequent class in at least this many months
NUM_THREADS = 4                # tiles downloaded in parallel (geedim)
MAX_TILE_SIZE = 4              # MB per tile request (geedim; Earth Engine limit 32 MB)
MAX_TILE_DIM = 2048            # pixels per tile side (geedim)
OVERWRITE = False              # False: skip a year whose GeoTIFF exists
DOWNLOAD = True                # False: only draw the maps from the GeoTIFFs already there (no Earth Engine)
MAKE_PNG = True                # the two maps per agricultural year
OVERVIEW_PX = 32               # pixels per block side in the cropland map (32 -> 320 m)
# ==========================================================================

NAMES = ["CROPLAND"]
CROPS = 4                      # Dynamic World class "crops"
NODATA_U8 = DW_NODATA


def make_image(ee, ag_season, region):
    """Binary cropland of one agricultural year: crops dominant in >= MIN_CROP_MONTHS of its 12 months."""
    months = month_starts(REGION, ag_season)
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
          .filterDate(month_bounds(months[0])[0], month_bounds(months[-1])[1]).select("label"))
    crop_months = ee.Image.constant(0)
    for m in months:                                                # month without an image counts as not crops
        mode = dw.filterDate(*month_bounds(m)).reduce(ee.Reducer.mode())
        crop_months = crop_months.add(mode.eq(CROPS).unmask(0))
    has_image = dw.count().gt(0)                                    # any Dynamic World image in the year
    cropland = crop_months.gte(MIN_CROP_MONTHS).updateMask(has_image)
    return cropland.clip(region).unmask(NODATA_U8, sameFootprint=False).rename(NAMES).toUint8()


def ee_region(ee):
    """The mapping area as an Earth Engine geometry (EPSG:4326, simplified to ~100 m to keep the request small)."""
    crs = settings(REGION)["crs_epsg"]
    geom = gpd.GeoSeries([map_area(REGION)], crs=crs).to_crs(4326).simplify(0.001).iloc[0]
    return ee.Geometry(geom.__geo_interface__)


def download_year(ee, ag_season, region):
    """One GeoTIFF over the extent of all blocks (10 m, origin on the block grid)."""
    import geemap
    blocks = load_blocks(REGION, ag_season)
    x0, y1 = int(blocks["bx"].min()) * BLOCK_M, (int(blocks["by"].max()) + 1) * BLOCK_M
    width = (int(blocks["bx"].max()) - int(blocks["bx"].min()) + 1) * BLOCK_PX
    height = (int(blocks["by"].max()) - int(blocks["by"].min()) + 1) * BLOCK_PX
    out = dw_cropland_file(REGION, ag_season)
    out.parent.mkdir(parents=True, exist_ok=True)
    crs = settings(REGION)["crs_epsg"]
    print(f"  {ag_season}: {width:,} x {height:,} px at {GSD_M} m (EPSG:{crs}) -> {out}")
    geemap.download_ee_image(image=make_image(ee, ag_season, region), filename=str(out), crs=f"EPSG:{crs}",
                             crs_transform=[GSD_M, 0, float(x0), 0, -GSD_M, float(y1)], shape=(height, width),
                             dtype="uint8", num_threads=NUM_THREADS, max_tile_size=MAX_TILE_SIZE,
                             max_tile_dim=MAX_TILE_DIM, overwrite=True)
    with rasterio.open(out, "r+") as dst:                           # band name, no-data, overviews for QGIS
        dst.nodata = NODATA_U8
        dst.set_band_description(1, NAMES[0])
        dst.build_overviews([4, 16, 64, 256], Resampling.mode)
    print(f"  {ag_season}: {out.stat().st_size / 1024 ** 3:.2f} GB")


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
    with rasterio.open(dw_cropland_file(REGION, ag_season)) as src:
        for b in blocks[blocks["crop_px"].notna()].itertuples():
            a = read_dw_window(src, b.bx * BLOCK_M, (b.by + 1) * BLOCK_M, BLOCK_PX)
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
    print(f"Dynamic World cropland for {REGION}")
    if DOWNLOAD:
        ee = init_gee()
        region = ee_region(ee)
        for s in SEASONS:
            if dw_cropland_file(REGION, s).exists() and not OVERWRITE:
                print(f"  {s}: {dw_cropland_file(REGION, s).name} exists (OVERWRITE = True to download again)")
                continue
            download_year(ee, s, region)
    for s in SEASONS:
        if not dw_cropland_file(REGION, s).exists():
            print(f"  {s}: {dw_cropland_file(REGION, s).name} not there yet; no maps")
            continue
        blocks = summary(s)
        if MAKE_PNG:
            folder = download_dir(REGION) / "dynamic_world"
            tag = s.replace("/", "-")
            cropland_map(blocks, s, folder / f"cropland_map_{tag}.png")
            cropland_blocks(blocks, s, folder / f"cropland_blocks_{tag}.png")
