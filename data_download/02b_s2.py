"""
02b_s2.py  —  Sentinel-2 monthly composites for every planned 10.24 km block (Earth Engine)
========================================================================================

What it does
------------
For each agricultural year of the block plan (01_plan_blocks.py) Earth Engine builds one image of 12 monthly
Sentinel-2 L2A composites, and every planned block (1024 x 1024 px) is downloaded with computePixels (in groups of months below the
request limit, blocks in parallel, resumable).

Per pixel and month (12 months of the agricultural year; m01 = September in Morocco, October in Utah)
  B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12   median of the clear pixels (reflectance x 10,000)
  S2_NCLEAR                                number of clear observations (Cloud Score+)
NODATA = -32768 where a month has no clear observation (S2_NCLEAR is 0 there).

Data: COPERNICUS/S2_SR_HARMONIZED, scenes with CLOUDY_PIXEL_PERCENTAGE < CLOUD_COVER_PERCENTAGE; pixels masked with
Cloud Score+ (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED, cs_cdf >= CS_THRESHOLD). Same compositing as the training tiles
downloaded before (former 03_build_tiles.py). Needs the Earth Engine service-account key in secrets/.

Output (outputs/<REGION>/data_download/s2/<season>/)
  <bx>_<by>.tif               int16, 12 months x 13 bands = 156 bands (descriptions "B4_m03", ...)
  <bx>_<by>_scenes.json       Sentinel-2 scene IDs per month (provenance)

Run (from the project folder; after 01_plan_blocks.py; on the cluster for a whole region)
    python data_download/02b_s2.py
"""
import numpy as np

from download_common import N_MONTHS, NODATA, S2_BANDS, download_blocks, month_bounds, month_starts

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2023/24", "2020/21"]}[REGION]          # agricultural years (block plan of each from 01)
CLOUD_COVER_PERCENTAGE = 60    # skip scenes cloudier than this (Cloud Score+ masks the rest)
CS_THRESHOLD = 0.60            # Cloud Score+ cs_cdf: keep pixels >= this (0.50-0.65 recommended)
DOWNLOAD_WORKERS = 4           # blocks fetched in parallel (each in several requests)
N_TEST_BLOCKS = None           # e.g. 1 block per agricultural year for a quick test; None = all
# ==========================================================================

BANDS = S2_BANDS + ["S2_NCLEAR"]
NAMES = [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in BANDS]


def collections(ee, ag_season, region):
    first = month_bounds(month_starts(REGION, ag_season)[0])[0]
    last = month_bounds(month_starts(REGION, ag_season)[-1])[1]
    raw = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(first, last)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_PERCENTAGE)))
    score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    masked = raw.linkCollection(score, ["cs_cdf"]).map(
        lambda img: img.select(S2_BANDS).updateMask(img.select("cs_cdf").gte(CS_THRESHOLD)))
    return raw, masked


def make_image(ee, ag_season, region):
    """12 months x 13 bands, int16."""
    _, s2 = collections(ee, ag_season, region)
    empty = ee.ImageCollection([ee.Image.constant([0] * len(S2_BANDS)).rename(S2_BANDS).toUint16()
                                .updateMask(ee.Image.constant(0))])        # keeps cloudy months valid
    months = []
    for k, m in enumerate(month_starts(REGION, ag_season), start=1):
        col = s2.filterDate(*month_bounds(m)).merge(empty)
        optical = col.median().select(S2_BANDS).min(32767).unmask(NODATA)
        n_clear = col.select("B4").count().unmask(0).rename("S2_NCLEAR")
        months.append(ee.Image.cat([optical, n_clear]).select(BANDS, [f"{b}_m{k:02d}" for b in BANDS]))
    return ee.Image.cat(months).toInt16()


def scene_ids(ee, ag_season, region, block_geom):
    raw, _ = collections(ee, ag_season, region)
    return ee.Dictionary({f"m{k:02d}": raw.filterBounds(block_geom).filterDate(*month_bounds(m)).aggregate_array("system:index")
                          for k, m in enumerate(month_starts(REGION, ag_season), start=1)})


if __name__ == "__main__":
    print(f"Sentinel-2 for {REGION}")
    download_blocks(REGION, "s2", make_image, NAMES, SEASONS, dtype=np.int16, nodata=NODATA,
                    workers=DOWNLOAD_WORKERS, n_test_blocks=N_TEST_BLOCKS, scenes=scene_ids, require_cropland=True)
