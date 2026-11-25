"""
02b_s2.py  —  Sentinel-2 monthly composites for the 10.24 km blocks with cropland (open STAC catalogue)
======================================================================================================

What it does
------------
For each agricultural year and every block with cropland or training tiles (download_common.select_blocks: Dynamic
World cropland of 02a_dynamic_world.py, >= MIN_CROP_PIXELS, plus every block with training tiles), block by block
(blocks in parallel, resumable: finished blocks are skipped):
1. Searches the Sentinel-2 L2A scenes of the agricultural year over the block in Earth Search (Element 84, AWS;
   collection sentinel-2-c1-l2a, ESA Collection 1; no account), skipping scenes with eo:cloud_cover >=
   MAX_SCENE_CLOUD.
2. Month by month, reads the 12 bands and the scene classification (SCL) of every scene onto the block grid (region's
   projection, 10 m, nearest-neighbour resampling for the 20 m and 60 m bands), as raw integers.
3. Harmonises the reflectance: Collection 1 stores reflectance x 10,000 + 1000 (the asset metadata declares
   scale 0.0001 and offset -0.1); the offset is read from each scene's metadata and subtracted, values below 0 -> 0,
   so the values are reflectance x 10,000 as in Earth Engine's COPERNICUS/S2_SR_HARMONIZED.
4. Cloud mask per pixel and scene: SCL classes in SCL_INVALID removed (no data, saturated or defective, dark area,
   cloud shadow, cloud medium / high probability, thin cirrus), and pixels without data (B4 = 0).
5. Monthly composite = median of the clear observations per band; S2_NCLEAR = number of clear observations. Every
   scene counts, also where two Sentinel-2 granules overlap (the same acquisition then counts twice), as in Earth
   Engine.
Unreadable scenes are retried, then left out and listed; if more than MAX_FAILED_FRAC of a block's scenes cannot be
read, the block is not written (a rerun fetches it).

Per pixel and month (12 months of the agricultural year; m01 = September in Morocco, October in Utah)
  B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12   median of the clear observations (reflectance x 10,000)
  S2_NCLEAR                                number of clear observations
NODATA = -32768 where a month has no clear observation (S2_NCLEAR is 0 there). Band names, order, scaling and
NODATA are those of the earlier Earth Engine download, so data_training_prep/04_cut_tiles.py and every model read the
blocks unchanged. Difference: the cloud mask is the scene classification (SCL) instead of Cloud Score+ (Earth Engine
only).

Output (outputs/<REGION>/data_download/s2/<season>/)
  <bx>_<by>.tif               int16, 12 months x 13 bands = 156 bands (descriptions "B4_m03", ...)
  <bx>_<by>_scenes.json       per month: scene IDs used and not readable; source and cloud mask (provenance)

Run (from the project folder; after 02a_dynamic_world.py; on the cluster for a whole region, CPU and internet)
    python data_download/02b_s2.py
"""
import numpy as np
from pandas import DateOffset

from download_common import (N_MONTHS, NODATA, S2_BANDS, month_starts, process_blocks)
from regions import settings
from stac_common import block_bounds, in_month, lonlat_bbox, read_scenes, search

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2023/24", "2020/21"]}[REGION]          # agricultural years
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
MAX_SCENE_CLOUD = 60           # skip scenes with eo:cloud_cover >= this (%), as CLOUDY_PIXEL_PERCENTAGE < 60 before
SCL_INVALID = [0, 1, 2, 3, 8, 9, 10]   # no data, saturated/defective, dark area, cloud shadow, cloud medium/high, cirrus
MAX_FAILED_FRAC = 0.02         # more unreadable scenes than this share of a block's scenes -> block not written
WORKERS = 2                    # blocks in parallel
THREADS = 8                    # parallel reads within a block
N_TEST_BLOCKS = None           # e.g. 1 block per agricultural year for a quick test; None = all
ONLY_BLOCKS = None             # e.g. [(48, 349)] for a test on given blocks; None = all
# ==========================================================================

CRS_EPSG = settings(REGION)["crs_epsg"]
ASSETS = {"B1": "coastal", "B2": "blue", "B3": "green", "B4": "red", "B5": "rededge1", "B6": "rededge2",
          "B7": "rededge3", "B8": "nir", "B8A": "nir08", "B9": "nir09", "B11": "swir16", "B12": "swir22"}
BANDS = S2_BANDS + ["S2_NCLEAR"]
NAMES = [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in BANDS]
SOURCE = {"catalogue": STAC_URL, "collection": COLLECTION, "scene_filter": f"eo:cloud_cover < {MAX_SCENE_CLOUD}",
          "cloud_mask": f"SCL not in {SCL_INVALID} and B4 > 0", "resampling": "nearest",
          "harmonised": "offset of the asset metadata removed (reflectance x 10000), values < 0 -> 0"}


def dn_offset(item):
    """Integer offset of a scene's reflectance values (1000 in Collection 1): -offset / scale of the asset metadata."""
    band = item.assets["red"].extra_fields.get("raster:bands", [{}])[0]
    scale, offset = band.get("scale"), band.get("offset")
    if not scale or offset is None:
        return 0.0
    return float(round(-offset / scale))


class TooManyFailed(RuntimeError):
    pass


def compute(ag_season, bx, by, size_px=1024, bounds=None):
    """156 bands of one block (or of any square `bounds` of size_px pixels, for tests) and its provenance."""
    bounds = bounds or block_bounds(bx, by)
    months = month_starts(REGION, ag_season)
    end = months[-1] + DateOffset(months=1)                                   # end of the agricultural year
    items = search(STAC_URL, COLLECTION, lonlat_bbox(bounds, CRS_EPSG), months[0], end,
                   query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD}})
    cube = np.full((N_MONTHS, len(BANDS), size_px, size_px), NODATA, np.int16)
    cube[:, -1] = 0
    assets = [ASSETS[b] for b in S2_BANDS] + ["scl"]
    i_b4, i_scl = S2_BANDS.index("B4"), len(S2_BANDS)
    info, failed = {"source": SOURCE, "n_items_found": len(items), "months": {}}, []
    for k, m in enumerate(months):
        arr, kept, bad = read_scenes(in_month(items, m), assets, bounds, CRS_EPSG, "uint16", 0, size_px, THREADS)
        failed += bad
        if len(failed) > MAX_FAILED_FRAC * max(len(items), 1):
            raise TooManyFailed(f"{len(failed)} of {len(items)} scenes not readable")
        info["months"][f"m{k + 1:02d}"] = {"used": [it.id for it in kept], "not_readable": bad}
        if not len(kept):
            continue
        clear = ~np.isin(arr[:, i_scl], SCL_INVALID) & (arr[:, i_b4] > 0)       # [n, H, W]
        off = np.array([dn_offset(it) for it in kept], np.float32)[:, None, None]
        for j in range(len(S2_BANDS)):                                          # band by band: less memory
            refl = np.maximum(arr[:, j].astype(np.float32) - off, 0)
            refl[~clear] = np.nan
            with np.errstate(all="ignore"):
                med = np.nanmedian(refl, axis=0)
            cube[k, j] = np.where(np.isnan(med), NODATA, np.minimum(np.rint(med), 32767)).astype(np.int16)
        cube[k, -1] = np.minimum(clear.sum(0), 32767).astype(np.int16)
    return cube.reshape(N_MONTHS * len(BANDS), size_px, size_px), info


if __name__ == "__main__":
    print(f"Sentinel-2 for {REGION} from {STAC_URL} ({COLLECTION})")
    process_blocks(REGION, "s2", compute, NAMES, SEASONS, dtype=np.int16, nodata=NODATA, workers=WORKERS,
                   n_test_blocks=N_TEST_BLOCKS, only_blocks=ONLY_BLOCKS, require_cropland=True)
