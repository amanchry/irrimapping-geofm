"""Sentinel-2 monthly features per parcel.

For every month in the window: cloud-masked (SCL) median surface reflectance of 10 bands, then
NDVI, EVI, NDWI, GCVI and NDRE computed from the monthly composite; each layer is averaged over
the pixels inside every parcel.

Columns: <VARIABLE>_m<MM>  (calendar month, e.g. NDVI_m09), plus n_px.
Source:  Microsoft Planetary Computer, collection sentinel-2-l2a (open, no account).

Edit the SETTINGS block, then run:   python extract_s2.py
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import (dask_init, items_in_month, load_parcels, merge_parts, month_starts, nan_layers, open_catalog,
                    parcel_means, run_extraction, search_items, setup_logging, tile_bbox, tile_geobox)

# ================================ SETTINGS ================================
HERE = Path(__file__).resolve().parent
PARCELS_FILE = HERE.parent / "outputs" / "inspect" / "morocco_parcels_labelled.geojson"   # needs 'parcel_id'
OUT_DIR = HERE.parent / "outputs" / "features"

START, END = "2023-09", "2024-08"     # months included (YYYY-MM, inclusive): the 2023/24 season
CRS = "EPSG:32629"                    # working grid (UTM 29N)
RESOLUTION = 10.0                     # metres
MAX_SCENE_CLOUD = 30                  # skip scenes with more cloud than this (%)

TILE_KM = 10.0                        # parcels are loaded tile by tile
WORKERS = 4                           # tiles processed in parallel (processes)
DASK_THREADS = 4                      # threads per worker
RETRIES = 3                           # attempts per tile before it is logged as failed
LIMIT_TILES = None                    # e.g. 3 for a quick test run; None = all parcels
MERGE_ONLY = False                    # True = only rebuild the output table from finished tiles
# ==========================================================================

COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
INDICES = ["NDVI", "EVI", "NDWI"]
VARIABLES = BANDS + INDICES
SCL_CLEAR = [4, 5, 6, 7]       # vegetation, bare soil, water, unclassified (clouds, shadows, snow removed)


def boa_offset(item) -> int:
    """Processing baseline >= 04.00 (from 25 Jan 2022) stores reflectance with a +1000 offset."""
    baseline = item.properties.get("s2:processing_baseline")
    if baseline is not None:
        return 1000 if float(baseline) >= 4.0 else 0
    return 1000 if pd.Timestamp(item.datetime).tz_convert(None) >= pd.Timestamp("2022-01-25") else 0


def monthly_composite(items: list, gbox) -> xr.Dataset:
    """Cloud-masked median reflectance and indices for one month (NaN where never clear)."""
    import odc.stac
    if not items:
        return nan_layers(gbox, VARIABLES)
    refl = []
    for offset in sorted({boa_offset(it) for it in items}):     # load each offset group separately
        group = [it for it in items if boa_offset(it) == offset]
        ds = odc.stac.load(group, bands=BANDS + ["SCL"], geobox=gbox, groupby="solar_day",
                           chunks={"x": 1024, "y": 1024}, resampling={"SCL": "nearest", "*": "bilinear"})
        clear = ds["SCL"].isin(SCL_CLEAR) & (ds["B04"] > 0)
        refl.append(((ds[BANDS].astype("float32") - offset) / 10_000).where(clear))
    med = (xr.concat(refl, "time") if len(refl) > 1 else refl[0]).median("time")
    b2, b3, b4, b5, b8, b8a, b11 = (med[b] for b in ["B02", "B03", "B04", "B05", "B08", "B8A", "B11"])
    med["NDVI"] = (b8 - b4) / (b8 + b4)
    med["EVI"] = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1)
    med["NDWI"] = (b8 - b11) / (b8 + b11)            # Gao (1996): NIR vs SWIR1, canopy water content
    med["GCVI"] = b8 / b3 - 1                          # green chlorophyll index
    med["NDRE"] = (b8a - b5) / (b8a + b5)              # red-edge NDVI
    return med[VARIABLES]


def tile_features(parcels, *, months, crs, resolution, max_cloud) -> pd.DataFrame:
    catalog = open_catalog()                          # fresh signed URLs for every tile
    bbox = tile_bbox(parcels)
    gbox = tile_geobox(bbox, crs, resolution)
    items = search_items(catalog, COLLECTION, bbox, months, query={"eo:cloud_cover": {"lt": max_cloud}})
    tables, n_px = [], None
    for month in months:
        stack = monthly_composite(items_in_month(items, month), gbox).to_array("feature").compute()
        table = parcel_means(stack, parcels)
        n_px = table.pop("n_px") if n_px is None else n_px
        table = table.drop(columns="n_px", errors="ignore")
        tables.append(table.add_suffix(f"_m{month.month:02d}"))
    out = pd.concat(tables, axis=1)
    out["n_px"] = n_px
    return out


def main():
    run = f"s2_{START}_{END}"
    log = setup_logging(OUT_DIR / "logs" / f"{run}.log")
    parcels = load_parcels(PARCELS_FILE, TILE_KM)
    if LIMIT_TILES:
        parcels = parcels[parcels["tile_id"].isin(sorted(parcels["tile_id"].unique())[:LIMIT_TILES])]
    months = month_starts(START, END)
    config = {"source": f"Planetary Computer {COLLECTION}", "months": [f"{m:%Y-%m}" for m in months],
              "variables": VARIABLES, "scl_clear": SCL_CLEAR, "max_scene_cloud": MAX_SCENE_CLOUD,
              "crs": CRS, "resolution_m": RESOLUTION, "tile_km": TILE_KM,
              "parcel_statistic": "mean of pixels inside the parcel (5 m inner buffer)"}
    if MERGE_ONLY:
        merge_parts(parcels, OUT_DIR / "_parts" / run, OUT_DIR, run, log, config)
        return
    fn = partial(tile_features, months=months, crs=CRS, resolution=RESOLUTION, max_cloud=MAX_SCENE_CLOUD)
    run_extraction(parcels, fn, OUT_DIR, run, log, workers=WORKERS, retries=RETRIES,
                   worker_init=partial(dask_init, DASK_THREADS), config=config)


if __name__ == "__main__":
    main()
