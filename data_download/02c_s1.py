"""
02c_s1.py  —  Sentinel-1 monthly composites for the 10.24 km blocks with cropland (open STAC catalogue)
======================================================================================================

What it does
------------
For each agricultural year and every block with cropland or training tiles (download_common.select_blocks, the same
blocks as 02b_s2.py), block by block (blocks in parallel, resumable):
1. Searches the Sentinel-1 scenes of the agricultural year over the block in Microsoft Planetary Computer (collection
   sentinel-1-rtc: GRD radiometrically terrain corrected by Catalyst, linear gamma0 backscatter, 10 m; free, signed
   anonymous URLs), keeping interferometric wide swath (IW) scenes with both VV and VH.
2. Month by month and per orbit direction (ascending and descending kept apart: different incidence angles), reads
   VV and VH of every scene onto the block grid (region's projection, 10 m, nearest neighbour).
3. Monthly composite = mean of the valid observations in LINEAR power (values > 0), stored as 10 log10(mean) dB x 100;
   S1_N = number of scenes with a valid VV value at the pixel.
Unreadable scenes are retried, then left out and listed; if more than MAX_FAILED_FRAC of a block's scenes cannot be
read, the block is not written (a rerun fetches it).

Per pixel and month (12 months of the agricultural year)
  VV_ASC VH_ASC S1_N_ASC      ascending: mean in linear power, stored as dB x 100; number of scenes
  VV_DSC VH_DSC S1_N_DSC      descending: same
NODATA = -32768 where a month has no scene of that orbit (the count is 0 there). Band names, order, scaling and
NODATA are those of the earlier Earth Engine download, so data_training_prep/04_cut_tiles.py and every model read the
blocks unchanged. Difference: the backscatter is gamma0 with radiometric terrain flattening (RTC) instead of Earth
Engine's sigma0 (COPERNICUS/S1_GRD, orthorectified only); on flat land gamma0 is about 1 dB higher than sigma0
(gamma0 = sigma0 / cos(incidence angle)), so S1 values must not be mixed between the two sources.

Output (outputs/<REGION>/data_download/s1/<season>/)
  <bx>_<by>.tif               int16, 12 months x 6 bands = 72 bands (descriptions "VV_ASC_m03", ...)
  <bx>_<by>_scenes.json       per month and orbit: scene IDs used and not readable; source (provenance)

Run (from the project folder; after 02a_dynamic_world.py; on the cluster for a whole region, CPU and internet)
    python data_download/02c_s1.py
"""
import numpy as np
from pandas import DateOffset

from download_common import N_MONTHS, NODATA, month_starts, process_blocks
from regions import settings
from stac_common import block_bounds, in_month, lonlat_bbox, read_scenes, search

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2023/24", "2020/21"]}[REGION]          # agricultural years
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
MAX_FAILED_FRAC = 0.02         # more unreadable scenes than this share of a block's scenes -> block not written
WORKERS = 2                    # blocks in parallel
THREADS = 8                    # parallel reads within a block
N_TEST_BLOCKS = None           # e.g. 1 block per agricultural year for a quick test; None = all
ONLY_BLOCKS = None             # e.g. [(48, 349)] for a test on given blocks; None = all
# ==========================================================================

CRS_EPSG = settings(REGION)["crs_epsg"]
S1_SCALE = 100                 # dB stored as integer dB x 100
BANDS = ["VV_ASC", "VH_ASC", "S1_N_ASC", "VV_DSC", "VH_DSC", "S1_N_DSC"]
NAMES = [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in BANDS]
ORBITS = (("ascending", "ASC"), ("descending", "DSC"))
SOURCE = {"catalogue": STAC_URL, "collection": COLLECTION, "filter": "IW, VV and VH",
          "backscatter": "gamma0 RTC (linear), monthly mean in linear power, stored as dB x 100",
          "resampling": "nearest"}


class TooManyFailed(RuntimeError):
    pass


def usable(item):
    p = item.properties
    return p.get("sar:instrument_mode") == "IW" and {"VV", "VH"} <= set(p.get("sar:polarizations", []))


def compute(ag_season, bx, by, size_px=1024, bounds=None):
    """72 bands of one block (or of any square `bounds` of size_px pixels, for tests) and its provenance."""
    bounds = bounds or block_bounds(bx, by)
    months = month_starts(REGION, ag_season)
    items = search(STAC_URL, COLLECTION, lonlat_bbox(bounds, CRS_EPSG), months[0], months[-1] + DateOffset(months=1),
                   signed=True)
    items = [it for it in items if usable(it)]
    cube = np.full((N_MONTHS, len(BANDS), size_px, size_px), NODATA, np.int16)
    info, failed = {"source": SOURCE, "n_items_found": len(items), "months": {}}, []
    for k, m in enumerate(months):
        month_items, month_info = in_month(items, m), {}
        for o, (state, tag) in enumerate(ORBITS):
            sel = [it for it in month_items if it.properties.get("sat:orbit_state") == state]
            arr, kept, bad = read_scenes(sel, ["vv", "vh"], bounds, CRS_EPSG, "float32", np.nan, size_px, THREADS)
            failed += bad
            if len(failed) > MAX_FAILED_FRAC * max(len(items), 1):
                raise TooManyFailed(f"{len(failed)} of {len(items)} scenes not readable")
            month_info[tag.lower()] = {"used": [it.id for it in kept], "not_readable": bad}
            j = 3 * o                                                           # VV_tag, VH_tag, S1_N_tag
            cube[k, j + 2] = 0
            if not len(kept):
                continue
            valid = np.isfinite(arr) & (arr > 0)                                # [n, 2, H, W]; nodata is -32768
            for p in range(2):                                                  # VV, VH
                lin = np.where(valid[:, p], arr[:, p], np.nan)
                with np.errstate(all="ignore"):
                    mean = np.nanmean(lin, axis=0)
                    db = np.rint(10 * np.log10(mean) * S1_SCALE)
                cube[k, j + p] = np.where(np.isfinite(db), np.clip(db, -32767, 32767), NODATA).astype(np.int16)
            cube[k, j + 2] = np.minimum(valid[:, 0].sum(0), 32767).astype(np.int16)
        info["months"][f"m{k + 1:02d}"] = month_info
    return cube.reshape(N_MONTHS * len(BANDS), size_px, size_px), info


if __name__ == "__main__":
    print(f"Sentinel-1 for {REGION} from {STAC_URL} ({COLLECTION})")
    process_blocks(REGION, "s1", compute, NAMES, SEASONS, dtype=np.int16, nodata=NODATA, workers=WORKERS,
                   n_test_blocks=N_TEST_BLOCKS, only_blocks=ONLY_BLOCKS, require_cropland=True)
