"""
00_make_cropland_mask.py  —  one cropland mask per region and agricultural year, from several cropland products
=================================================================================================================

What it does
------------
Builds the area that the continuous (wall-to-wall) irrigated / rainfed maps cover: every 10 m pixel that at
least MIN_VOTES of the cropland products below call cropland (MIN_VOTES = 1: the UNION of the products).
Each product alone misses cropland (orchards, small fields, fallow, recent expansion); a pixel is kept if any
product sees cropland there. The cost is extra non-cropland in the mask, which the classifier has to handle
(3-class design: irrigated / rainfed / not cropland). The vote count and the flags of every product are
stored, so a stricter mask (e.g. >= 2 votes) or a different product set can be taken later without
downloading again.

The mask is made FIRST (script 00) so that the same mask is used everywhere:
  - 01_prepare_labels_*.py: reads the mask and records the share of each parcel inside it (flag, not drop)
  - 03_build_tiles.py: the continuous tiles = all 128 px tiles of the grid that contain cropland
  - the maps: final predictions are kept inside the mask
This script reads NO parcel or label data: the mask depends only on the cropland products, so it has the
same quality in labelled and unlabelled areas.

Products used (PRODUCTS; Earth Engine; year = the calendar year in which the agricultural year ends,
clipped to the years a product has)
---------------------------------------------------------------------------------------------------------
  bit  name        product, Earth Engine asset                                    res   year(s)    cropland =
   0   worldcover  ESA WorldCover v200                                            10 m  2021       class 40
                   ESA/WorldCover/v200
                   https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200
                   Zanaga et al. 2022, https://doi.org/10.5281/zenodo.7254221
                   (trees, incl. orchards and olive groves, are class 10, not cropland)
   1   esri_lulc   Esri / Impact Observatory 10 m annual land use / land cover    10 m  2017-2025  class 5 (crops)
                   projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS
                   https://gee-community-catalog.org/projects/S2TSLULC/
                   Karra et al. 2021, IGARSS 2021, "Global land use/land cover with Sentinel-2 and deep learning"
   4   gcep30      GFSAD Global Cropland Extent Product (GCEP30)                  30 m  2015       value 2
                   projects/sat-io/open-datasets/GFSAD/GCEP30
                   https://gee-community-catalog.org/projects/gcep30/
                   Thenkabail et al. 2021, USGS Professional Paper 1868, https://doi.org/10.3133/pp1868
   6   glad        GLAD global cropland extent (Potapov et al.)                   30 m  2003, 2007,  value 1
                   users/potapovpeter/Global_cropland_<year>                            2011, 2015,
                   https://glad.umd.edu/dataset/croplands                               2019
                   Potapov et al. 2022, Nature Food, https://doi.org/10.1038/s43016-021-00429-z
                   (annual and perennial herbaceous crops; tree crops, permanent pasture excluded)
Options (not in PRODUCTS; bit numbers fixed so they can be added later)
   2   dynamic_world  Dynamic World V1, GOOGLE/DYNAMICWORLD/V1                     10 m  the ag. year  crops = largest
                   https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1     mean probability
                   (slow: about 25 s per block, and Earth Engine memory errors on some blocks)
   3   worldcereal ESA WorldCereal 2021 temporary crops, ESA/WorldCereal/2021/MODELS/v100   10 m  2021  value 100
                   https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCereal_2021_MODELS_v100
   5   lgrip30     LGRIP30 irrigated / rainfed cropland, projects/sat-io/open-datasets/GFSAD/LGRIP30  30 m  2015  2 or 3
                   https://gee-community-catalog.org/projects/lgrip30/  (same extent as GCEP30 in the test)
   7   glc_fcs30d  GLC_FCS30D annual, projects/sat-io/open-datasets/GLC-FCS30D/annual   30 m  2000-2022  10, 11, 12, 20
                   https://gee-community-catalog.org/projects/glc_fcs/  (class 12 = orchards)
   8   cdl         USDA Cropland Data Layer (Utah), USDA/NASS/CDL               30 m  2008-2025  CDL_CROP_CODES
                   https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL
GCEP30 and LGRIP30 are stored as continental tiles that overlap (in Morocco the Africa and the Europe/Middle
East tiles); the overlapping tiles disagree, so the flags are combined with max() over the tiles, not with a
mosaic (the Europe tile alone has almost no cropland in Morocco).
WorldCover and GLAD do not count tree crops as cropland, so orchards depend on the other two products.
Not used: WRLU (Utah) and the Moroccan survey, because they are the label sources.

Hard exclusions (always removed from the mask): ESA WorldCover 2021 built-up (50), snow/ice (70),
permanent water (80). They are stored as a band, so they can be checked.

Steps
-----
1. Mapping area (regions.py: map_file / map_where).
2. Block plan: blocks of BLOCK_PX x BLOCK_PX pixels on the SAME grid as the tiles of 03_build_tiles.py
   (x = ix * 1280 m), so every block holds whole 128 px tiles.
3. Earth Engine: per block, the product flags (one bit per product) and the exclusion flag, downloaded with
   computePixels in parallel; votes and mask computed locally; one GeoTIFF per block. Re-running skips
   finished blocks.
4. One VRT per agricultural year (cropland_mask.vrt) over all blocks.
5. Tile table: every 128 px tile of the mapping area with its cropland pixel count -> input of the
   continuous tile download (03).
6. Summary: cropland area per product and of the union, area that ONLY one product calls cropland
   (what each product adds to the union), votes histogram; quicklook map.

Outputs (outputs/<region>/dataset/00_cropland_mask/<season>/)
-------------------------------------------------------------
blocks/<season>_<bx>_<by>.tif  uint16, 4 bands, NODATA 65535 outside the mapping area:
                               cropland (0/1) · votes (0-N) · products (bit flags, see table) · excluded (0/1)
cropland_mask.vrt              all blocks as one raster
tiles_cropland.parquet         tile_id, ix, iy, n_area_px, n_crop_px, crop_frac, n_crop_px_2votes, mean_votes
summary.json, product_areas.csv, votes_histogram.csv, cropland_mask_quicklook.png
mask_info.json                 products, years, rules and settings used
"""
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box

from regions import admin_boundaries, dataset_dir, map_area, settings

# ================================ SETTINGS ================================
REGION = "morocco"                      # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2020/21", "2023/24"]}[REGION]   # agricultural years; one mask per year
ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"     # Earth Engine key (never commit)

PRODUCTS = ["worldcover", "esri_lulc", "gcep30", "glad"]   # products used (all options in the table above;
                                        # the bit numbers stay fixed, so adding one later needs a new download only)
MIN_VOTES = 1                           # pixel is cropland if >= MIN_VOTES products say so (1 = union)
EXCLUDE_WORLDCOVER = [50, 70, 80]       # always removed: built-up, snow/ice, permanent water (WorldCover 2021)

CRS_EPSG = settings(REGION)["crs_epsg"]
START_MONTH = settings(REGION)["start_month"]
GSD_M = 10
TILE_PX = 128                           # tile of 03_build_tiles.py (1.28 km)
BLOCK_PX = 1024                         # download block = 8 x 8 tiles (10.24 km)

DOWNLOAD_WORKERS = 8                    # parallel Earth Engine requests
RETRIES = 4
N_TEST_BLOCKS = None                    # e.g. 6 blocks for a quick test; None = all
OUT_ROOT = dataset_dir(REGION) / "00_cropland_mask"
# ==========================================================================

BITS = {"worldcover": 0, "esri_lulc": 1, "dynamic_world": 2, "worldcereal": 3, "gcep30": 4, "lgrip30": 5,
        "glad": 6, "glc_fcs30d": 7, "cdl": 8}
REGION_ONLY = {"cdl": "utah"}           # products that exist for one region only
BANDS = ["cropland", "votes", "products", "excluded"]
NODATA = 65535
TILE_M = TILE_PX * GSD_M
BLOCK_M = BLOCK_PX * GSD_M
TILES_PER_BLOCK = BLOCK_PX // TILE_PX
PX_KM2 = (GSD_M * GSD_M) / 1e6

# CDL crop codes (NASS legend): 1-61 field crops incl. alfalfa (36), other hay (37) and fallow/idle (61);
# 66-77 tree crops and orchards; 204-254 double crops, vegetables, fruit. Not cropland: 62-65 (pasture/grass,
# forest, shrubland, barren), 81-195 (clouds, developed, water, natural vegetation, grassland/pasture 176, wetlands).
CDL_CROP_CODES = list(range(1, 62)) + list(range(66, 78)) + list(range(204, 255))
GLAD_YEARS = [2003, 2007, 2011, 2015, 2019]
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]


def season_tag(season):
    return season.replace("/", "-")


def season_dates(season):
    start = pd.Timestamp(year=int(season[:4]), month=START_MONTH, day=1)
    return start, start + pd.DateOffset(months=12)


def products_used():
    return [p for p in PRODUCTS if REGION_ONLY.get(p, REGION) == REGION]


def product_years(season):
    """Year of each product used for an agricultural year (end year of the season, clipped to what exists)."""
    y = int(season[:4]) + 1
    start, end = season_dates(season)
    years = {"worldcover": 2021, "esri_lulc": min(max(y, 2017), 2025), "dynamic_world": f"{start.date()} to {end.date()}",
             "worldcereal": 2021, "gcep30": 2015, "lgrip30": 2015,
             "glad": max([g for g in GLAD_YEARS if g <= y] or [GLAD_YEARS[0]]),
             "glc_fcs30d": min(max(y, 2000), 2022), "cdl": min(max(y, 2008), 2025)}
    return {p: years[p] for p in products_used()}


# ------------------------------------------------------------ Earth Engine
def init_gee():
    import ee
    key = json.loads(KEY_FILE.read_text())
    ee.Initialize(ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key)), project=key["project_id"])
    print("Earth Engine initialised")
    return ee


def product_flags(ee, season, geom):
    """One 0/1 image per product (1 = cropland), unmasked to 0 where a product has no data."""
    years = product_years(season)
    start, end = season_dates(season)
    flags = {}
    for p in products_used():
        y = years[p]
        if p == "worldcover":
            img = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(geom).mosaic().select("Map").eq(40)
        elif p == "esri_lulc":
            img = (ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
                   .filterDate(f"{y}-01-01", f"{y + 1}-01-01").filterBounds(geom).mosaic().eq(5))
        elif p == "dynamic_world":
            mean = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(geom)
                    .filterDate(str(start.date()), str(end.date())).select(DW_CLASSES).mean())
            others = mean.select([c for c in DW_CLASSES if c != "crops"]).reduce(ee.Reducer.max())
            img = mean.select("crops").gt(others)                   # crops = largest mean probability
        elif p == "worldcereal":
            img = (ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100").filter(ee.Filter.eq("product", "temporarycrops"))
                   .filterBounds(geom).mosaic().select("classification").eq(100))
        elif p == "gcep30":
            img = (ee.ImageCollection("projects/sat-io/open-datasets/GFSAD/GCEP30").filterBounds(geom)
                   .map(lambda i: i.eq(2)).max())
        elif p == "lgrip30":
            img = (ee.ImageCollection("projects/sat-io/open-datasets/GFSAD/LGRIP30").filterBounds(geom)
                   .map(lambda i: i.eq(2).Or(i.eq(3))).max())
        elif p == "glad":
            img = ee.ImageCollection(f"users/potapovpeter/Global_cropland_{y}").mosaic().eq(1)
        elif p == "glc_fcs30d":
            band = f"b{y - 1999}"                                   # b1 = 2000 ... b23 = 2022
            lc = (ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual").filterBounds(geom)
                  .select(band).mosaic())
            img = lc.remap([10, 11, 12, 20], [1, 1, 1, 1], 0)
        elif p == "cdl":
            cdl = ee.Image(f"USDA/NASS/CDL/{y}").select("cropland")
            img = cdl.remap(CDL_CROP_CODES, [1] * len(CDL_CROP_CODES), 0)
        else:
            raise ValueError(f"unknown product {p!r}")
        flags[p] = img.unmask(0).toUint16()
    return flags


def block_image(ee, season, geom):
    """products (bit flags) and excluded (0/1) as uint16."""
    flags = product_flags(ee, season, geom)
    bits = ee.Image.constant(0).toUint16()
    for p, img in flags.items():
        bits = bits.bitwiseOr(img.leftShift(BITS[p]))
    wc = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(geom).mosaic().select("Map")
    excluded = wc.remap(EXCLUDE_WORLDCOVER, [1] * len(EXCLUDE_WORLDCOVER), 0).unmask(0)
    return ee.Image.cat([bits.rename("products"), excluded.toUint16().rename("excluded")])


def compute_grid(ee, season, x0, y0, size):
    """products and excluded of one square of size x size pixels with top-left corner (x0, y0)."""
    geom = ee.Geometry.Rectangle([x0, y0 - size * GSD_M, x0 + size * GSD_M, y0], f"EPSG:{CRS_EPSG}", False)
    arr = ee.data.computePixels({
        "expression": block_image(ee, season, geom),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": size, "height": size},
                 "affineTransform": {"scaleX": GSD_M, "shearX": 0, "translateX": float(x0),
                                     "shearY": 0, "scaleY": -GSD_M, "translateY": float(y0)},
                 "crsCode": f"EPSG:{CRS_EPSG}"},
    })
    return arr["products"].astype(np.uint16), arr["excluded"].astype(np.uint16)


def download_block(ee, season, blk):
    """One block; if Earth Engine runs out of memory (long Dynamic World year), the block is fetched in 4 quarters."""
    x0, y0 = blk["bx"] * BLOCK_M, (blk["by"] + 1) * BLOCK_M
    split = False
    for attempt in range(1, RETRIES + 1):
        try:
            if not split:
                return compute_grid(ee, season, x0, y0, BLOCK_PX)
            h = BLOCK_PX // 2
            prod, excl = np.zeros((BLOCK_PX, BLOCK_PX), np.uint16), np.zeros((BLOCK_PX, BLOCK_PX), np.uint16)
            for r in range(2):
                for c in range(2):
                    p, e = compute_grid(ee, season, x0 + c * h * GSD_M, y0 - r * h * GSD_M, h)
                    prod[r * h:(r + 1) * h, c * h:(c + 1) * h], excl[r * h:(r + 1) * h, c * h:(c + 1) * h] = p, e
            return prod, excl
        except Exception as err:
            split = split or "memory" in str(err).lower()
            if attempt == RETRIES:
                print(f"  block {blk['bx']}_{blk['by']}: failed ({str(err)[:150]})")
                return None
            time.sleep(5 if split else 10 * attempt)


# ------------------------------------------------------------ local steps
def block_transform(bx, by):
    return Affine(GSD_M, 0, bx * BLOCK_M, 0, -GSD_M, (by + 1) * BLOCK_M)


def plan_blocks(area):
    minx, miny, maxx, maxy = area.bounds
    from shapely.prepared import prep
    pa = prep(area)
    blocks = []
    for bx in range(int(np.floor(minx / BLOCK_M)), int(np.floor(maxx / BLOCK_M)) + 1):
        for by in range(int(np.floor(miny / BLOCK_M)), int(np.floor(maxy / BLOCK_M)) + 1):
            b = box(bx * BLOCK_M, by * BLOCK_M, (bx + 1) * BLOCK_M, (by + 1) * BLOCK_M)
            if pa.intersects(b):
                blocks.append({"bx": bx, "by": by, "geometry": b})
    return blocks


def area_mask(area, blk):
    """1 = pixel centre inside the mapping area."""
    clip = area.intersection(blk["geometry"])
    if clip.is_empty:
        return np.zeros((BLOCK_PX, BLOCK_PX), bool)
    return rasterize([clip], out_shape=(BLOCK_PX, BLOCK_PX), transform=block_transform(blk["bx"], blk["by"]),
                     fill=0, dtype="uint8").astype(bool)


def popcount(bits):
    votes = np.zeros(bits.shape, np.uint16)
    for b in range(16):
        votes += (bits >> b) & 1
    return votes


def write_block(path, products, excluded, inside):
    votes = popcount(products)
    cropland = ((votes >= MIN_VOTES) & (excluded == 0)).astype(np.uint16)
    stack = np.stack([cropland, votes, products, excluded]).astype(np.uint16)
    stack[:, ~inside] = NODATA
    tmp = path.with_suffix(".part.tif")
    bx, by = (int(v) for v in path.stem.split("_")[-2:])
    profile = dict(driver="GTiff", width=BLOCK_PX, height=BLOCK_PX, count=len(BANDS), dtype="uint16",
                   crs=f"EPSG:{CRS_EPSG}", transform=block_transform(bx, by), nodata=NODATA,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(stack)
        for i, d in enumerate(BANDS, start=1):
            dst.set_band_description(i, d)
    tmp.replace(path)


def block_stats(path, season):
    """Per 128 px tile: pixels in the mapping area, cropland pixels; per product: cropland pixels."""
    with rasterio.open(path) as src:
        crop, votes, bits, excl = src.read()
    bx, by = (int(v) for v in path.stem.split("_")[-2:])
    inside = crop != NODATA
    n = TILES_PER_BLOCK
    def per_tile(a):
        return a.reshape(n, TILE_PX, n, TILE_PX).sum(axis=(1, 3))
    is_crop = inside & (crop == 1)
    n_area, n_crop = per_tile(inside), per_tile(is_crop)
    n_crop2 = per_tile(is_crop & (votes >= 2))
    sum_votes = per_tile(np.where(is_crop, votes, 0).astype(np.int64))
    rows = []
    for r in range(n):
        for c in range(n):
            if n_area[r, c] == 0:
                continue
            ix, iy = bx * n + c, (by + 1) * n - 1 - r                   # row 0 = top of the block
            rows.append({"tile_id": f"{season_tag(season)}_{ix}_{iy}", "ix": ix, "iy": iy,
                         "n_area_px": int(n_area[r, c]), "n_crop_px": int(n_crop[r, c]),
                         "n_crop_px_2votes": int(n_crop2[r, c]),
                         "mean_votes": round(float(sum_votes[r, c] / n_crop[r, c]), 2) if n_crop[r, c] else 0.0})
    ok = inside & (excl == 0)
    prod = {p: int((((bits >> BITS[p]) & 1).astype(bool) & ok).sum()) for p in products_used()}
    only = {p: int(((bits == (1 << BITS[p])) & ok).sum()) for p in products_used()}     # the only product saying cropland
    vh = np.bincount(votes[inside], minlength=len(products_used()) + 1)
    return rows, prod, only, vh, int(inside.sum()), int((inside & (excl == 1)).sum())


def quicklook(tiles, season, out):
    t = tiles.copy()
    t["frac"] = t["n_crop_px"] / t["n_area_px"]
    t["one_vote"] = np.where(t["n_crop_px"] > 0, 1 - t["n_crop_px_2votes"] / t["n_crop_px"].clip(lower=1), np.nan)
    x0, x1, y0, y1 = t["ix"].min(), t["ix"].max(), t["iy"].min(), t["iy"].max()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    adm = admin_boundaries(REGION).to_crs(CRS_EPSG)
    for ax, col, title, cmap in [(axes[0], "frac", f"Cropland share per 1.28 km tile (>= {MIN_VOTES} vote)", "YlGn"),
                                 (axes[1], "one_vote", "Share of mask pixels seen by ONE product only", "OrRd")]:
        grid = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan)
        grid[y1 - t["iy"], t["ix"] - x0] = t[col]
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                       extent=[x0 * TILE_M, (x1 + 1) * TILE_M, y0 * TILE_M, (y1 + 1) * TILE_M])
        adm.boundary.plot(ax=ax, color="0.3", linewidth=0.5)
        ax.set_xlim(x0 * TILE_M, (x1 + 1) * TILE_M)
        ax.set_ylim(y0 * TILE_M, (y1 + 1) * TILE_M)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle(f"{REGION.capitalize()} {season}: cropland mask ({len(products_used())} products)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ee = None
    for season in SEASONS:
        out_dir = OUT_ROOT / season_tag(season)
        blk_dir = out_dir / "blocks"
        blk_dir.mkdir(parents=True, exist_ok=True)
        years = product_years(season)
        print(f"\n== {REGION} {season} | products: " + ", ".join(f"{p} ({y})" for p, y in years.items()))
        (out_dir / "mask_info.json").write_text(json.dumps({
            "region": REGION, "season": season, "crs_epsg": CRS_EPSG, "gsd_m": GSD_M, "block_px": BLOCK_PX,
            "tile_px": TILE_PX, "grid_origin": "x = ix * tile_m, top y = (iy + 1) * tile_m (same grid as 03_build_tiles.py)",
            "bands": BANDS, "nodata": NODATA, "min_votes": MIN_VOTES,
            "products": {p: {"bit": BITS[p], "year": years[p]} for p in years},
            "exclude_worldcover_classes": EXCLUDE_WORLDCOVER, "cdl_crop_codes": CDL_CROP_CODES if "cdl" in years else None,
        }, indent=2))

        # 1-2. Mapping area and blocks.
        area = map_area(REGION)
        blocks = plan_blocks(area)
        if N_TEST_BLOCKS:
            blocks = blocks[len(blocks) // 2: len(blocks) // 2 + N_TEST_BLOCKS]
        print(f"Mapping area {area.area / 1e6:,.0f} km2 -> {len(blocks):,} blocks of {BLOCK_PX} px")

        # 3. Download.
        todo = [b for b in blocks if not (blk_dir / f"{season_tag(season)}_{b['bx']}_{b['by']}.tif").exists()]
        print(f"{len(blocks) - len(todo):,} blocks done, {len(todo):,} to go")
        if todo:
            ee = ee or init_gee()
            t0 = time.time()
            step = DOWNLOAD_WORKERS * 4
            for s in range(0, len(todo), step):
                batch = todo[s:s + step]
                with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
                    results = list(pool.map(lambda b: download_block(ee, season, b), batch))
                for b, res in zip(batch, results):
                    if res is not None:
                        write_block(blk_dir / f"{season_tag(season)}_{b['bx']}_{b['by']}.tif", res[0], res[1], area_mask(area, b))
                done = s + len(batch)
                rate = (time.time() - t0) / done
                print(f"  {done:,}/{len(todo):,} blocks ({rate:.1f} s/block, ~{rate * (len(todo) - done) / 60:.0f} min left)")

        # 4. VRT.
        files = sorted(str(p) for p in blk_dir.glob(f"{season_tag(season)}_*.tif") if ".part" not in p.name)
        if not files:
            print("No block downloaded.")
            continue
        vrt = out_dir / "cropland_mask.vrt"
        (out_dir / "block_list.txt").write_text("\n".join(files) + "\n")
        subprocess.run(["gdalbuildvrt", "-q", "-overwrite", "-input_file_list", str(out_dir / "block_list.txt"), str(vrt)], check=True)

        # 5-6. Tile table and summary.
        rows, n_area, n_excl = [], 0, 0
        prod_px, only_px, vhist = {p: 0 for p in years}, {p: 0 for p in years}, np.zeros(len(years) + 1, np.int64)
        for f in files:
            r, prod, only, vh, na, ne = block_stats(Path(f), season)
            rows += r
            for p in prod:
                prod_px[p] += prod[p]
                only_px[p] += only[p]
            vhist[:len(vh)] += vh[:len(vhist)]
            n_area += na
            n_excl += ne
        tiles = pd.DataFrame(rows)
        tiles["crop_frac"] = (tiles["n_crop_px"] / tiles["n_area_px"]).round(4)
        tiles.to_parquet(out_dir / "tiles_cropland.parquet")
        n_crop = int(tiles["n_crop_px"].sum())
        areas = pd.DataFrame([{"product": p, "bit": BITS[p], "year": years[p], "cropland_km2": round(prod_px[p] * PX_KM2, 1),
                               "share_of_union_pct": round(100 * prod_px[p] / max(n_crop, 1), 1),
                               "only_this_product_km2": round(only_px[p] * PX_KM2, 1)} for p in years])
        areas.to_csv(out_dir / "product_areas.csv", index=False)
        pd.DataFrame({"votes": np.arange(len(vhist)), "pixels": vhist, "km2": (vhist * PX_KM2).round(1)}).to_csv(
            out_dir / "votes_histogram.csv", index=False)
        summary = {
            "region": REGION, "season": season, "blocks": len(files), "mapped_area_km2": round(n_area * PX_KM2, 1),
            "cropland_km2": round(n_crop * PX_KM2, 1), "cropland_pct_of_area": round(100 * n_crop / max(n_area, 1), 2),
            "cropland_2votes_km2": round(int(tiles["n_crop_px_2votes"].sum()) * PX_KM2, 1),
            "excluded_km2": round(n_excl * PX_KM2, 1),
            "tiles_in_area": len(tiles), "tiles_with_cropland": int((tiles["n_crop_px"] > 0).sum()),
            "tiles_with_cropland_ge1pct": int((tiles["crop_frac"] >= 0.01).sum()),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        quicklook(tiles, season, out_dir / "cropland_mask_quicklook.png")

        print(f"Mapped {summary['mapped_area_km2']:,} km2 | cropland (>= {MIN_VOTES} vote) {summary['cropland_km2']:,} km2 "
              f"({summary['cropland_pct_of_area']}%), >= 2 votes {summary['cropland_2votes_km2']:,} km2, "
              f"excluded {summary['excluded_km2']:,} km2")
        print(f"Tiles: {summary['tiles_in_area']:,} in the area, {summary['tiles_with_cropland']:,} with cropland, "
              f"{summary['tiles_with_cropland_ge1pct']:,} with >= 1% cropland")
        print(areas.to_string(index=False))
        print(f"Saved in {out_dir}")


if __name__ == "__main__":
    main()
