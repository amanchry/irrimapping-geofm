"""
02a_dynamic_world.py  —  binary cropland image (Dynamic World) for every planned 10.24 km block
================================================================================================

What it does
------------
Downloads, for EVERY block of the plan (01_plan_blocks.py: the whole mapping area, no mask), one binary image from
Dynamic World (Earth Engine), block by block (blocks in parallel, resumable):

  CROPLAND   1 = cropland, 0 = other, 255 = no Dynamic World image
             cropland = the most frequent Dynamic World class over the calendar year in which the agricultural
             year ends (2023/24 -> 2024) is "crops" (class 4)

This image decides what comes next:
  - 02b_s2.py, 02c_s1.py (later ERA5) download only the blocks with at least MIN_CROP_PIXELS cropland pixels
    (download_common.py), plus every block with training tiles;
  - the maps classify only the cropland pixels of a block.

Output (outputs/<REGION>/data_download/dynamic_world/)
  <season>/<bx>_<by>.tif           uint8, 1 band, 1024 x 1024 px
  cropland_map_<season>.png        cropland share of each cell (OVERVIEW_PX cells per block side, 320 m) of all
                                   downloaded blocks over the mapping area, with legend
  cropland_blocks_<season>.png     the block extents (as in 01_plan_blocks.py): blocks with cropland highlighted,
                                   blocks without cropland, blocks not downloaded yet; blocks with training tiles
                                   outlined
The full Dynamic World land cover (all 9 classes) of each year is drawn by 04c_ref_dynamic_world_annual.py.

Run (from the project folder; after 01_plan_blocks.py; on the cluster for a whole region)
    python data_download/02a_dynamic_world.py
"""
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from download_common import (BLOCK_M, BLOCK_PX, MIN_CROP_PIXELS, block_file, crop_pixels, download_blocks,
                             download_dir, load_blocks)
from regions import admin_boundaries, map_area, settings

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2023/24", "2020/21"]}[REGION]          # agricultural years (block plan of each from 01)
DOWNLOAD_WORKERS = 4           # blocks fetched in parallel
N_TEST_BLOCKS = None           # e.g. 1 block per agricultural year for a quick test; None = all
ONLY_BLOCKS = None             # e.g. [(46, 349)] for a test on given blocks; None = all
DOWNLOAD = True                # False: only draw the maps from the blocks already downloaded (no Earth Engine)
MAKE_PNG = True                # the two maps per agricultural year
OVERVIEW_PX = 32               # pixels per block side in the cropland map (32 -> 320 m)
# ==========================================================================

NAMES = ["CROPLAND"]
CROPS = 4                      # Dynamic World class "crops"
NODATA_U8 = 255


def make_image(ee, ag_season, region):
    year = int(ag_season[:4]) + 1                                   # calendar year in which the agricultural year ends
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    label = dw.select("label").reduce(ee.Reducer.mode())
    return label.eq(CROPS).unmask(NODATA_U8).rename(NAMES).toUint8()


def summary(ag_season):
    blocks = load_blocks(REGION, ag_season)
    n = [crop_pixels(REGION, ag_season, bx, by) for bx, by in zip(blocks["bx"], blocks["by"])]
    blocks["crop_px"] = [np.nan if v is None else float(v) for v in n]
    done = blocks[blocks["crop_px"].notna()]
    print(f"{REGION} {ag_season}: Dynamic World in {len(done):,} of {len(blocks):,} blocks | with cropland "
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
    for b in blocks[blocks["crop_px"].notna()].itertuples():
        with rasterio.open(block_file(REGION, "dynamic_world", ag_season, b.bx, b.by)) as src:
            a = src.read(1)
        k = BLOCK_PX // OVERVIEW_PX                                 # block pixels per map cell
        cells = lambda m: m.reshape(OVERVIEW_PX, k, OVERVIEW_PX, k).sum(axis=(1, 3))
        n_valid, n_crop = cells(a != NODATA_U8), cells(a == 1)
        r, c = (by1 - b.by) * OVERVIEW_PX, (b.bx - bx0) * OVERVIEW_PX
        share[r:r + OVERVIEW_PX, c:c + OVERVIEW_PX] = np.where(n_valid > 0, n_crop / np.maximum(n_valid, 1), np.nan)
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
    year = int(ag_season[:4]) + 1
    ax.legend(handles=[Patch(fc="#ece6d6", label="no cropland in the cell"),
                       Patch(fc="#9e9e9e", label="no Dynamic World image"),
                       Patch(fc="#f4f4f4", ec="0.55", label="block not downloaded yet")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=9, frameon=False)
    ax.set_title(f"{REGION.capitalize()} {ag_season}: Dynamic World cropland (most frequent class of {year} = crops)\n"
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
        download_blocks(REGION, "dynamic_world", make_image, NAMES, SEASONS, dtype=np.uint8, nodata=NODATA_U8,
                        workers=DOWNLOAD_WORKERS, n_test_blocks=N_TEST_BLOCKS, only_blocks=ONLY_BLOCKS)
    for s in SEASONS:
        blocks = summary(s)
        if MAKE_PNG:
            folder = download_dir(REGION) / "dynamic_world"
            folder.mkdir(parents=True, exist_ok=True)
            tag = s.replace("/", "-")
            cropland_map(blocks, s, folder / f"cropland_map_{tag}.png")
            cropland_blocks(blocks, s, folder / f"cropland_blocks_{tag}.png")
    print("Next: 02b_s2.py and 02c_s1.py (blocks with cropland or training tiles)")
