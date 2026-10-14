"""Sentinel-1 monthly backscatter features per parcel.

For every month in the window: median radiometrically terrain-corrected gamma0 (VV, VH) in linear
power, averaged over the pixels inside every parcel, then converted to dB. The VH/VV ratio is
computed from the parcel means (dB difference).

Columns: VV_m<MM>, VH_m<MM>, VHVV_m<MM> (dB), plus n_px.
Source:  Microsoft Planetary Computer, collection sentinel-1-rtc (open, no account).

Edit the SETTINGS block, then run:   python extract_s1.py
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

TILE_KM = 10.0                        # parcels are loaded tile by tile
WORKERS = 4                           # tiles processed in parallel (processes)
DASK_THREADS = 4                      # threads per worker
RETRIES = 3                           # attempts per tile before it is logged as failed
LIMIT_TILES = None                    # e.g. 3 for a quick test run; None = all parcels
MERGE_ONLY = False                    # True = only rebuild the output table from finished tiles
# ==========================================================================

COLLECTION = "sentinel-1-rtc"
POLS = ["vv", "vh"]


def monthly_composite(items: list, gbox) -> xr.Dataset:
    """Median gamma0 in linear power for one month (NaN where no acquisition)."""
    import odc.stac
    if not items:
        return nan_layers(gbox, POLS)
    ds = odc.stac.load(items, bands=POLS, geobox=gbox, groupby="solar_day",
                       chunks={"x": 1024, "y": 1024}, resampling="bilinear")
    return ds.where(ds > 0).median("time").astype("float32")      # <= 0 is no-data


def to_db(x):
    return 10 * np.log10(x.where(x > 0))


def tile_features(parcels, *, months, crs, resolution) -> pd.DataFrame:
    catalog = open_catalog()
    bbox = tile_bbox(parcels)
    gbox = tile_geobox(bbox, crs, resolution)
    items = search_items(catalog, COLLECTION, bbox, months)
    tables, n_px = [], None
    for month in months:
        stack = monthly_composite(items_in_month(items, month), gbox).to_array("feature").compute()
        lin = parcel_means(stack, parcels)                          # average in linear power ...
        n_px = lin["n_px"] if n_px is None else n_px
        vv, vh = to_db(lin["vv"]), to_db(lin["vh"])                 # ... then convert to dB
        mm = f"_m{month.month:02d}"
        tables.append(pd.DataFrame({f"VV{mm}": vv, f"VH{mm}": vh, f"VHVV{mm}": vh - vv}))
    out = pd.concat(tables, axis=1).astype("float32")
    out["n_px"] = n_px
    return out


def main():
    run = f"s1_{START}_{END}"
    log = setup_logging(OUT_DIR / "logs" / f"{run}.log")
    parcels = load_parcels(PARCELS_FILE, TILE_KM)
    if LIMIT_TILES:
        parcels = parcels[parcels["tile_id"].isin(sorted(parcels["tile_id"].unique())[:LIMIT_TILES])]
    months = month_starts(START, END)
    config = {"source": f"Planetary Computer {COLLECTION}", "months": [f"{m:%Y-%m}" for m in months],
              "variables": ["VV", "VH", "VHVV"], "units": "dB (mean taken in linear power)",
              "crs": CRS, "resolution_m": RESOLUTION, "tile_km": TILE_KM,
              "parcel_statistic": "mean of pixels inside the parcel (5 m inner buffer)"}
    if MERGE_ONLY:
        merge_parts(parcels, OUT_DIR / "_parts" / run, OUT_DIR, run, log, config)
        return
    fn = partial(tile_features, months=months, crs=CRS, resolution=RESOLUTION)
    run_extraction(parcels, fn, OUT_DIR, run, log, workers=WORKERS, retries=RETRIES,
                   worker_init=partial(dask_init, DASK_THREADS), config=config)


if __name__ == "__main__":
    main()
