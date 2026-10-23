"""
03_extract_s1s2_features.py  —  Sentinel-1 / Sentinel-2 features for every labelled parcel
============================================================================================

What it does
------------
For each parcel, it builds a time series of Sentinel-2 bands, three vegetation/water indices
and Sentinel-1 radar backscatter over the parcel's agricultural year, then stores the MEAN of
every band inside the parcel. The result is one table: one row per parcel, one column per
variable and period (for example NDVI_p05 = NDVI in the 5th period).

Steps
-----
1. Sign in to Google Earth Engine with the service-account key.
2. Read the labelled parcels (from 01_prepare_labels.py). Each parcel carries its
   survey season `ag_season` ("2023/24" or "2024/25").
3. Shrink every parcel by BUFFER_M metres so mixed edge pixels (roads, neighbours) are left out.
4. For each agricultural year separately:
     a. Sentinel-2 L2A: drop cloudy pixels with Cloud Score+, scale reflectance to 0-1,
        add NDVI, NDWI and EVI.
     b. Sentinel-1 GRD (IW mode, VV + VH, already in dB).
     c. Cut the year into periods of INTERVAL_DAYS and take the median of each period.
        Periods are aligned across years: p01 always starts on 1 September (START_MONTH).
     d. Average every band inside each parcel (reduceRegions) in small batches and append
        them to the output CSV.
5. A re-run skips parcels already in the CSV, so an interrupted run can simply be restarted.

Why the agricultural year
-------------------------
Labels were surveyed between December 2023 and January 2025, i.e. in two agricultural years.
Each parcel gets imagery from the year it was surveyed in, so label and image describe the
same season. Sept-Aug covers one rainfed season (sowing ~Nov, harvest May-Jul) plus the dry
summer, when only irrigated land stays green.

Output
------
outputs/03_s1s2_features/s1s2_features_<WINDOW>_<INTERVAL_DAYS>d.csv
    columns: B1_p01 ... VH_p12, parcel_id, label, ag_season
Empty periods (fully cloudy) stay empty here; 06_train_evaluate.py fills them by interpolation
in time.

Run (from the project folder)
-----------------------------
    python pipeline/03_extract_s1s2_features.py
All settings are in the SETTINGS block below (no command-line arguments).
"""
import json
import time
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                     # project folder
KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"  # Earth Engine service-account key (never commit)
PARCELS_FILE = ROOT / "outputs" / "01_labels" / "parcels_labelled.geojson"
OUT_DIR = ROOT / "outputs" / "03_s1s2_features"

# Time window inside each parcel's agricultural year (ag_season "2023/24" -> Sep 2023 - Aug 2024).
START_MONTH, N_MONTHS, WINDOW = 9, 12, "agyear"       # main run: whole agricultural year, 1 Sep - 31 Aug
# START_MONTH, N_MONTHS, WINDOW = 11, 7, "growing"    # ablation: growing season only, 1 Nov - 31 May
INTERVAL_DAYS = 30                                    # 30 -> 12 periods; 15 -> 24 periods

SCALE_M = 10                    # pixel size used when averaging inside parcels
BUFFER_M = -10                  # shrink parcels by one pixel to drop mixed edge pixels (0 = no buffer)
CLOUD_COVER_PERCENTAGE = 60     # skip Sentinel-2 scenes cloudier than this (Cloud Score+ masks the rest)
CS_THRESHOLD = 0.60             # Cloud Score+ cs_cdf: keep pixels >= this (0.50-0.65 recommended)
BATCH_SIZE = 100                # parcels per Earth Engine request
RETRIES = 3                     # attempts per batch before skipping it (re-run the script to retry)
N_TEST_PARCELS = None           # e.g. 50 for a quick test; None = all parcels

S2_RAW_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_BANDS = S2_RAW_BANDS + ["NDVI", "NDWI", "EVI"]     # 14 optical features per period
S1_BANDS = ["VV", "VH"]                               # 2 radar features per period
OUT_CSV = OUT_DIR / f"s1s2_features_{WINDOW}_{INTERVAL_DAYS}d.csv"
# ==========================================================================


def init_gee():
    """Sign in to Earth Engine with the local service-account key."""
    key = json.loads(KEY_FILE.read_text())
    credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
    ee.Initialize(credentials, project=key["project_id"])
    print("Earth Engine initialised")


def season_dates(ag_season):
    """'2023/24' -> (1 Sep 2023, 1 Sep 2024): first day and end (exclusive) of the window."""
    start = pd.Timestamp(year=int(ag_season[:4]), month=START_MONTH, day=1)
    return start, start + pd.DateOffset(months=N_MONTHS)


def n_periods():
    """Number of periods per window. Uses a year without 29 February so every year gets the same count."""
    start, end = season_dates("2022/23")
    return (end - start).days // INTERVAL_DAYS


def mask_s2(img):
    """Keep clear pixels (Cloud Score+) and scale reflectance from 0-10000 to 0-1."""
    clear = img.select("cs_cdf").gte(CS_THRESHOLD)
    reflectance = img.select(S2_RAW_BANDS).divide(10000)
    # divide() drops the acquisition time; copy it back or the period filters find nothing.
    return reflectance.updateMask(clear).copyProperties(img, ["system:time_start"])


def add_indices(img):
    """Add NDVI (greenness), NDWI (canopy water, NIR vs SWIR1) and EVI (greenness, less saturated)."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B8", "B11"]).rename("NDWI")
    evi = img.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
    ).rename("EVI")
    return img.addBands([ndvi, ndwi, evi]).copyProperties(img, ["system:time_start"])


def masked_median(collection, band_names):
    """Median of one period. A period with no images returns a fully masked image (-> empty value)."""
    empty = ee.Image.constant([0] * len(band_names)).rename(band_names).selfMask()
    median = collection.median().select(band_names)
    return ee.Image(ee.Algorithms.If(collection.size().gt(0), median, empty))


def build_image(region, ag_season):
    """One multi-band image: every feature for every period of this agricultural year."""
    season_start, season_end = season_dates(ag_season)
    start_ee = ee.Date(season_start.strftime("%Y-%m-%d"))
    end_ee = ee.Date(season_end.strftime("%Y-%m-%d"))

    # Sentinel-2 surface reflectance, joined with its Cloud Score+ layer, cloud-masked, plus indices.
    cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_ee, end_ee)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_PERCENTAGE))
        .linkCollection(cloud_score, ["cs_cdf"])
        .map(mask_s2)
        .map(add_indices)
    )
    # Sentinel-1 ground-range detected, interferometric wide swath, both polarisations (values in dB).
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_ee, end_ee)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(S1_BANDS)
    )

    period_images, bands = [], []
    for i in range(n_periods()):
        start = start_ee.advance(i * INTERVAL_DAYS, "day")
        end = start.advance(INTERVAL_DAYS, "day")
        tag = f"p{i + 1:02d}"                                     # p01, p02, ...
        optical = masked_median(s2.filterDate(start, end), S2_BANDS)
        radar = masked_median(s1.filterDate(start, end), S1_BANDS)
        names = [f"{b}_{tag}" for b in S2_BANDS + S1_BANDS]       # e.g. NDVI_p03
        period_images.append(optical.addBands(radar).rename(names))
        bands.extend(names)

    print(f"{ag_season}: {season_start.date()} to {season_end.date()} | {n_periods()} periods x "
          f"{len(S2_BANDS) + len(S1_BANDS)} variables = {len(bands)} columns")
    return ee.Image.cat(period_images).clip(region), bands


def shrink_parcels(parcels):
    """Negative buffer in metres (UTM 29N); a parcel that would vanish keeps its original outline."""
    if BUFFER_M == 0:
        return parcels
    utm = parcels.to_crs(32629)
    shrunk = utm.geometry.buffer(BUFFER_M)
    keep = ~shrunk.is_empty & shrunk.is_valid
    utm.loc[keep, "geometry"] = shrunk[keep]
    print(f"Inner buffer {BUFFER_M} m: {keep.sum()} parcels shrunk, {(~keep).sum()} too small (kept as is)")
    return utm.to_crs(4326)


def to_feature_collection(parcels):
    """Turn parcels into Earth Engine features carrying parcel_id and label (1 irrigated, 2 rainfed)."""
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(row.geometry.__geo_interface__),
                   {"parcel_id": str(row.parcel_id), "label": int(row.label)})
        for row in parcels.itertuples(index=False)
    ])


def sample_batch(image, bands, batch):
    """Mean of every band inside each parcel of one batch; returns one dict per parcel."""
    sampled = image.reduceRegions(
        collection=to_feature_collection(batch),
        reducer=ee.Reducer.mean(),
        scale=SCALE_M,
        tileScale=4,                   # smaller tiles on the server: avoids memory errors
    )
    rows = []
    for feature in sampled.getInfo()["features"]:
        props = feature["properties"]
        rows.append({b: props.get(b) for b in bands}            # missing key = empty period -> NaN
                    | {"parcel_id": str(props["parcel_id"]), "label": int(props["label"])})
    return rows


def sample_features(image, bands, parcels, ag_season, out_csv):
    """Sample in batches and append each batch to the CSV; parcels already in the CSV are skipped."""
    done = set(pd.read_csv(out_csv, usecols=["parcel_id"], dtype=str)["parcel_id"]) if out_csv.exists() else set()
    todo = parcels[~parcels["parcel_id"].isin(done)]
    print(f"  {len(parcels) - len(todo)} already sampled, {len(todo)} to go")
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[start:start + BATCH_SIZE]
        rows = None
        for attempt in range(1, RETRIES + 1):
            try:
                rows = sample_batch(image, bands, batch)
                break
            except Exception as err:                            # network or Earth Engine error
                print(f"  batch {start} failed (attempt {attempt}/{RETRIES}): {err}")
                time.sleep(10 * attempt)                        # wait longer after each failure
        if rows is None:
            print(f"  skipped batch {start}; re-run the script to retry it")
            continue
        table = pd.DataFrame(rows, columns=bands + ["parcel_id", "label"]).assign(ag_season=ag_season)
        table.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"  {ag_season}: {min(start + BATCH_SIZE, len(todo))}/{len(todo)} parcels sampled")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    init_gee()

    # 1. Labelled parcels (1 = irrigated, 2 = rainfed) with their survey season.
    parcels = gpd.read_file(PARCELS_FILE).to_crs(4326)
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    if N_TEST_PARCELS:
        parcels = parcels.groupby("ag_season", group_keys=False).head(N_TEST_PARCELS)
    print(f"{len(parcels)} parcels | " + str(pd.crosstab(parcels["ag_season"], parcels["label"]).to_dict()))

    # 2. Drop mixed edge pixels.
    parcels = shrink_parcels(parcels)

    # 3. One image per agricultural year; each parcel is sampled with the year it was surveyed in.
    for ag_season, group in parcels.groupby("ag_season"):
        region = ee.Geometry.Rectangle(list(group.total_bounds))
        image, bands = build_image(region, ag_season)
        sample_features(image, bands, group, ag_season, OUT_CSV)

    # 4. Report what was saved.
    table = pd.read_csv(OUT_CSV, dtype={"parcel_id": str}).drop_duplicates("parcel_id")
    features = [c for c in table.columns if c not in ("parcel_id", "label", "ag_season")]
    print(f"Saved {OUT_CSV.name}: {len(table)} parcels x {len(features)} features | "
          f"empty values {table[features].isna().to_numpy().mean():.1%}")


if __name__ == "__main__":
    main()
