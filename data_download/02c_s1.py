"""
02c_s1.py  —  Sentinel-1 monthly composites for every planned 10.24 km block (Earth Engine)
========================================================================================

What it does
------------
For each agricultural year of the block plan (01_plan_blocks.py) Earth Engine builds one image of 12 monthly
Sentinel-1 composites, ascending and descending orbits kept apart (different incidence angles), and every planned
block (1024 x 1024 px) is downloaded with computePixels (in groups of months below the request limit,
blocks in parallel, resumable).

Per pixel and month (12 months of the agricultural year)
  VV_ASC VH_ASC S1_N_ASC      ascending: mean in LINEAR power, stored as dB x 100; number of scenes
  VV_DSC VH_DSC S1_N_DSC      descending: same
NODATA = -32768 where a month has no scene of that orbit (the count is 0 there).

Data: COPERNICUS/S1_GRD, IW mode, scenes with both VV and VH. Earth Engine stores S1_GRD in dB; the monthly mean is
taken in linear power and converted back to dB. Same compositing as the training tiles
downloaded before (former 03_build_tiles.py).
Needs the Earth Engine service-account key in secrets/.

Output (outputs/<REGION>/data_download/s1/<season>/)
  <bx>_<by>.tif               int16, 12 months x 6 bands = 72 bands (descriptions "VV_ASC_m03", ...)
  <bx>_<by>_scenes.json       scene IDs per month and orbit (provenance)

Run (from the project folder; after 01_plan_blocks.py; on the cluster for a whole region)
    python data_download/02c_s1.py
"""
import numpy as np

from download_common import N_MONTHS, NODATA, download_blocks, month_bounds, month_starts

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2023/24", "2020/21"]}[REGION]          # agricultural years (block plan of each from 01)
DOWNLOAD_WORKERS = 4           # blocks fetched in parallel (each in several requests)
N_TEST_BLOCKS = None           # e.g. 1 block per agricultural year for a quick test; None = all
# ==========================================================================

S1_SCALE = 100                 # dB stored as integer dB x 100
BANDS = ["VV_ASC", "VH_ASC", "S1_N_ASC", "VV_DSC", "VH_DSC", "S1_N_DSC"]
NAMES = [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in BANDS]
ORBITS = (("ASCENDING", "ASC"), ("DESCENDING", "DSC"))


def raw_collection(ee, ag_season, region):
    first = month_bounds(month_starts(REGION, ag_season)[0])[0]
    last = month_bounds(month_starts(REGION, ag_season)[-1])[1]
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(first, last)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def make_image(ee, ag_season, region):
    """12 months x 6 bands, int16."""
    raw = raw_collection(ee, ag_season, region)

    def to_linear(img):                                              # Earth Engine S1_GRD is in dB
        return ee.Image(ee.Image(10).pow(img.select(["VV", "VH"]).divide(10)).toFloat()
                        .copyProperties(img, ["system:time_start"]))

    s1 = {o: raw.filter(ee.Filter.eq("orbitProperties_pass", o)).map(to_linear) for o, _ in ORBITS}
    empty = ee.ImageCollection([ee.Image.constant([0, 0]).rename(["VV", "VH"]).toFloat()
                                .updateMask(ee.Image.constant(0))])
    months = []
    for k, m in enumerate(month_starts(REGION, ag_season), start=1):
        bands = []
        for orbit, tag in ORBITS:
            col = s1[orbit].filterDate(*month_bounds(m)).merge(empty)
            db = col.mean().log10().multiply(10 * S1_SCALE).unmask(NODATA).rename([f"VV_{tag}", f"VH_{tag}"])
            n = col.select("VV").count().unmask(0).rename(f"S1_N_{tag}")
            bands += [db, n]
        months.append(ee.Image.cat(bands).select(BANDS, [f"{b}_m{k:02d}" for b in BANDS]))
    return ee.Image.cat(months).toInt16()


def scene_ids(ee, ag_season, region, block_geom):
    raw = raw_collection(ee, ag_season, region).filterBounds(block_geom)
    out = {}
    for k, m in enumerate(month_starts(REGION, ag_season), start=1):
        col = raw.filterDate(*month_bounds(m))
        out[f"m{k:02d}"] = ee.Dictionary({tag.lower(): col.filter(ee.Filter.eq("orbitProperties_pass", o))
                                          .aggregate_array("system:index") for o, tag in ORBITS})
    return ee.Dictionary(out)


if __name__ == "__main__":
    print(f"Sentinel-1 for {REGION}")
    download_blocks(REGION, "s1", make_image, NAMES, SEASONS, dtype=np.int16, nodata=NODATA,
                    workers=DOWNLOAD_WORKERS, n_test_blocks=N_TEST_BLOCKS, scenes=scene_ids, require_cropland=True)
