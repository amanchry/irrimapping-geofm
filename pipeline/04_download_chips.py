"""
04_download_chips.py  —  one shared set of image chips for all foundation models
================================================================================

What it does
------------
Downloads, ONCE, a small image chip around every parcel for every month of the parcel's
agricultural year. All four foundation models (scripts 05a-05d: Clay, Prithvi, TerraMind,
AgriFM) read these same chips, so they see identical imagery and any difference in results
comes from the model, not from the data.

Each chip holds every band any of the models needs:
  Sentinel-2 L2A  B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12   (surface reflectance x 10000)
  Sentinel-1 GRD  VV VH                                     (backscatter in dB x 100)
Each model script picks its own bands, dates and normalisation from these.

Steps
-----
1. Sign in to Earth Engine and read the labelled parcels (01_prepare_labels.py).
2. For each agricultural year ("2023/24" = Sep 2023 - Aug 2024, ...), build one image with
   12 monthly composites x 14 bands:
     Sentinel-2: median of clear pixels (Cloud Score+ cs_cdf >= CS_THRESHOLD);
     Sentinel-1: mean of the month's IW scenes (dB).
   A month with no clear pixel is written as NODATA.
3. For every parcel, cut a CHIP_PX x CHIP_PX chip (default 64 x 64 px = 640 m at 10 m) centred on
   the parcel, in the parcel's own UTM zone, snapped to the 10 m grid, and download it with
   Earth Engine computePixels (several parcels in parallel).
4. Rasterise the parcel outline onto the same grid (1 = inside the parcel). The model scripts use
   this mask to average only the patches that cover the parcel.
5. Save in batches (part_*.npz); a re-run skips parcels that are already saved.

Output (outputs/04_chips/<WINDOW>/)
-----------------------------------
part_00000.npz ...   parcel_id [B], chips int16 [B, 12 months, 14 bands, H, W], mask uint8 [B, H, W]
index.csv            one row per parcel: parcel_id, label, ag_season, lat, lon, epsg, x0, y0, part
chips_info.json      band names, months, chip size, pixel size, NODATA value, scale factors
Size: about 1.4 MB per parcel before compression (~12 GB for 9,000 parcels; less on disk).

Run (from the project folder)
-----------------------------
    python pipeline/04_download_chips.py
Set N_TEST_PARCELS = 20 first for a quick check.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import Affine

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                     # project folder
KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"  # Earth Engine service-account key (never commit)
PARCELS_FILE = ROOT / "outputs" / "01_labels" / "parcels_labelled.geojson"

START_MONTH, N_MONTHS, WINDOW = 9, 12, "agyear"       # agricultural year 1 Sep - 31 Aug (same as script 03)
OUT_DIR = ROOT / "outputs" / "04_chips" / WINDOW

CHIP_PX = 64                   # chip width/height in pixels (64 x 10 m = 640 m); multiple of 16
GSD_M = 10.0                   # pixel size (m)
CLOUD_COVER_PERCENTAGE = 60    # skip Sentinel-2 scenes cloudier than this
CS_THRESHOLD = 0.60            # Cloud Score+ cs_cdf: keep pixels >= this (0.50-0.65 recommended)
BATCH_PARCELS = 64             # parcels per saved part file
DOWNLOAD_WORKERS = 8           # parallel Earth Engine requests
RETRIES = 3
N_TEST_PARCELS = None          # e.g. 20 per agricultural year for a quick test; None = all
# ==========================================================================

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
BANDS = S2_BANDS + S1_BANDS    # order of the band axis in the chips
S1_SCALE = 100                 # dB stored as integer dB x 100 (e.g. -12.34 dB -> -1234)
NODATA = -32768                # no clear observation in that month


# ------------------------------------------------------------ Earth Engine
def init_gee():
    """Sign in to Earth Engine with the local service-account key."""
    key = json.loads(KEY_FILE.read_text())
    credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
    ee.Initialize(credentials, project=key["project_id"])
    print("Earth Engine initialised")


def month_starts(ag_season):
    """First day of each month of the window: '2023/24' -> 2023-09-01 ... 2024-08-01."""
    start = pd.Timestamp(year=int(ag_season[:4]), month=START_MONTH, day=1)
    return [start + pd.DateOffset(months=k) for k in range(N_MONTHS)]


def monthly_stack(ag_season, region):
    """One int16 image with N_MONTHS x 14 bands named <band>_m01 ... <band>_m12."""
    months = month_starts(ag_season)
    first, last = months[0], months[-1] + pd.DateOffset(months=1)
    cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region).filterDate(str(first.date()), str(last.date()))
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_PERCENTAGE))
          .linkCollection(cloud_score, ["cs_cdf"])
          .map(lambda img: img.select(S2_BANDS).updateMask(img.select("cs_cdf").gte(CS_THRESHOLD))))
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(region).filterDate(str(first.date()), str(last.date()))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(S1_BANDS))
    images = []
    for k, m in enumerate(months, start=1):
        start, end = str(m.date()), str((m + pd.DateOffset(months=1)).date())
        optical = s2.filterDate(start, end).median().select(S2_BANDS).min(32767)        # int16 range
        radar = s1.filterDate(start, end).mean().select(S1_BANDS).multiply(S1_SCALE)
        names = [f"{b}_m{k:02d}" for b in BANDS]
        images.append(optical.addBands(radar).rename(names))
    return ee.Image.cat(images).unmask(NODATA).toInt16()


def utm_epsg(lon, lat):
    """EPSG code of the point's UTM zone (Morocco: 29N or 30N)."""
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def download_chip(image, row):
    """All months x bands of one parcel's chip -> int16 array [N_MONTHS, 14, CHIP_PX, CHIP_PX]."""
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": CHIP_PX, "height": CHIP_PX},
            "affineTransform": {"scaleX": GSD_M, "shearX": 0, "translateX": row.x0,
                                "shearY": 0, "scaleY": -GSD_M, "translateY": row.y0},
            "crsCode": f"EPSG:{row.epsg}",
        },
    }
    for attempt in range(1, RETRIES + 1):
        try:
            arr = ee.data.computePixels(request)                # structured array, one field per band
            cube = np.stack([arr[f"{b}_m{k:02d}"] for k in range(1, N_MONTHS + 1) for b in BANDS])
            return cube.reshape(N_MONTHS, len(BANDS), CHIP_PX, CHIP_PX).astype(np.int16)
        except Exception as err:
            if attempt == RETRIES:
                print(f"  parcel {row.parcel_id}: download failed ({err})")
                return None
            time.sleep(5 * attempt)


# ------------------------------------------------------------ parcels and masks
def prepare_parcels():
    """Parcels with their UTM zone, chip corner (x0, y0) and outline in that zone."""
    parcels = gpd.read_file(PARCELS_FILE).to_crs(4326)
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    if N_TEST_PARCELS:
        parcels = parcels.groupby("ag_season", group_keys=False).head(N_TEST_PARCELS)
    point = parcels.geometry.representative_point()
    parcels["lon"], parcels["lat"] = point.x, point.y
    parcels["epsg"] = [utm_epsg(x, y) for x, y in zip(parcels["lon"], parcels["lat"])]
    half = CHIP_PX * GSD_M / 2
    geoms, x0s, y0s = {}, {}, {}
    for epsg, group in parcels.groupby("epsg"):                 # each UTM zone separately
        utm = group.to_crs(epsg)
        for pid, geom in zip(utm["parcel_id"], utm.geometry):
            c = geom.centroid
            geoms[pid] = geom
            x0s[pid] = np.floor((c.x - half) / GSD_M) * GSD_M  # top-left corner on the 10 m grid
            y0s[pid] = np.ceil((c.y + half) / GSD_M) * GSD_M
    parcels["geom_utm"] = parcels["parcel_id"].map(geoms)
    parcels["x0"] = parcels["parcel_id"].map(x0s)
    parcels["y0"] = parcels["parcel_id"].map(y0s)
    return parcels


def parcel_mask(row):
    """1 where a chip pixel's centre lies inside the parcel; tiny parcels: every pixel the outline touches."""
    transform = Affine(GSD_M, 0, row.x0, 0, -GSD_M, row.y0)
    mask = rasterize([row.geom_utm], out_shape=(CHIP_PX, CHIP_PX), transform=transform, fill=0, dtype="uint8")
    if mask.sum() == 0:
        mask = rasterize([row.geom_utm], out_shape=(CHIP_PX, CHIP_PX), transform=transform, fill=0,
                         dtype="uint8", all_touched=True)
    return mask


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "chips_info.json").write_text(json.dumps({
        "bands": BANDS, "s2_bands": S2_BANDS, "s1_bands": S1_BANDS, "s1_scale": S1_SCALE,
        "s2_scale": "surface reflectance x 10000", "nodata": NODATA, "chip_px": CHIP_PX, "gsd_m": GSD_M,
        "start_month": START_MONTH, "n_months": N_MONTHS, "window": WINDOW,
        "cloud_mask": f"Cloud Score+ cs_cdf >= {CS_THRESHOLD}"}, indent=2))
    init_gee()
    parcels = prepare_parcels()
    print(f"{len(parcels)} parcels | chips {CHIP_PX} px x {N_MONTHS} months x {len(BANDS)} bands")

    # Resume: parcels already in a part file are skipped.
    index_file = OUT_DIR / "index.csv"
    index = pd.read_csv(index_file, dtype={"parcel_id": str}) if index_file.exists() else pd.DataFrame()
    done = set(index["parcel_id"]) if len(index) else set()
    n_part = len(list(OUT_DIR.glob("part_*.npz")))

    for ag_season, group in parcels.groupby("ag_season"):
        todo = group[~group["parcel_id"].isin(done)]
        print(f"{ag_season}: {len(group) - len(todo)} done, {len(todo)} to go")
        region = ee.Geometry.Rectangle(list(group.total_bounds)).buffer(2000)
        image = monthly_stack(ag_season, region)
        for start in range(0, len(todo), BATCH_PARCELS):
            t0 = time.time()
            rows = list(todo.iloc[start:start + BATCH_PARCELS].itertuples(index=False))
            with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
                cubes = list(pool.map(lambda r: download_chip(image, r), rows))
            kept = [(r, c) for r, c in zip(rows, cubes) if c is not None]
            if not kept:
                continue
            part = f"part_{n_part:05d}.npz"
            np.savez_compressed(OUT_DIR / part,
                                parcel_id=np.array([r.parcel_id for r, _ in kept]),
                                chips=np.stack([c for _, c in kept]),
                                mask=np.stack([parcel_mask(r) for r, _ in kept]))
            rows_index = pd.DataFrame([{"parcel_id": r.parcel_id, "label": r.label, "ag_season": r.ag_season,
                                        "lat": r.lat, "lon": r.lon, "epsg": r.epsg, "x0": r.x0, "y0": r.y0,
                                        "part": part} for r, _ in kept])
            rows_index.to_csv(index_file, mode="a", header=not index_file.exists(), index=False)
            n_part += 1
            print(f"  {ag_season}: {min(start + BATCH_PARCELS, len(todo))}/{len(todo)} parcels "
                  f"({time.time() - t0:.0f}s for {len(kept)} chips)")

    index = pd.read_csv(index_file, dtype={"parcel_id": str})
    print(f"Done: {len(index)} parcels in {index['part'].nunique()} part files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
