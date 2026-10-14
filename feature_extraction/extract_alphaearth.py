"""AlphaEarth Foundations (AEF) annual embeddings per parcel.

The 64-dimensional annual embedding (10 m) is de-quantised, (q / 127.5)^2 * sign(q) with -128 as
no-data, and averaged over the pixels inside every parcel.

Columns: A00 .. A63, plus n_px.
Source:  AEF mosaic as cloud-native Zarr on Source Cooperative (open, anonymous S3; needs zarr>=3, s3fs).

Edit the SETTINGS block, then run:   python extract_alphaearth.py
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import dask_init, load_parcels, merge_parts, parcel_means, run_extraction, setup_logging, tile_bbox

# ================================ SETTINGS ================================
HERE = Path(__file__).resolve().parent
PARCELS_FILE = HERE.parent / "feature_extraction"/"morocco_parcels_labelled.geojson"   # needs 'parcel_id'
OUT_DIR = HERE.parent / "feature_extraction"/ "features"


YEAR = 2024                           # AEF annual embedding year (2017-2025)

TILE_KM = 10.0                        # parcels are loaded tile by tile
WORKERS = 4                           # tiles processed in parallel (processes)
DASK_THREADS = 4                      # threads per worker
RETRIES = 3                           # attempts per tile before it is logged as failed
LIMIT_TILES = 2                    # e.g. 3 for a quick test run; None = all parcels
MERGE_ONLY = False                    # True = only rebuild the output table from finished tiles
# ==========================================================================

AEF_ZARR = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/"
NODATA = -128
BANDS = [f"A{i:02d}" for i in range(64)]
_AEF = None                     # opened once per worker process


def open_aef() -> xr.Dataset:
    global _AEF
    if _AEF is None:
        try:
            import s3fs  # noqa: F401
            import zarr
        except ImportError as err:
            raise ImportError("needs zarr>=3 and s3fs: pip install 'zarr>=3' s3fs") from err
        if int(zarr.__version__.split(".")[0]) < 3:
            raise ImportError(f"zarr {zarr.__version__} found; the AEF mosaic is Zarr v3")
        ds = xr.open_zarr(AEF_ZARR, storage_options={"anon": True}, consolidated=False, chunks="auto", zarr_format=3)
        _AEF = ds.odc.assign_crs("EPSG:4326")
    return _AEF


def tile_features(parcels, *, year: int) -> pd.DataFrame:
    import odc.geo.xr
    from odc.geo.geom import box
    q = odc.geo.xr.crop(open_aef(), box(*tile_bbox(parcels), crs="EPSG:4326")).sel(time=year)["embeddings"]
    q = q.transpose("band", ...).compute()
    valid = (q != NODATA).any("band")
    emb = (((q.astype("float32") / 127.5) ** 2) * np.sign(q)).where(valid)
    emb = emb.rename(band="feature").assign_coords(feature=BANDS)
    return parcel_means(emb, parcels)


def main():
    run = f"alphaearth_{YEAR}"
    log = setup_logging(OUT_DIR / "logs" / f"{run}.log")
    parcels = load_parcels(PARCELS_FILE, TILE_KM)
    if LIMIT_TILES:
        parcels = parcels[parcels["tile_id"].isin(sorted(parcels["tile_id"].unique())[:LIMIT_TILES])]
    config = {"source": AEF_ZARR, "year": YEAR, "variables": BANDS,
              "dequantisation": "(q/127.5)^2 * sign(q), -128 = no-data", "tile_km": TILE_KM,
              "parcel_statistic": "mean of pixels inside the parcel (5 m inner buffer)"}
    if MERGE_ONLY:
        merge_parts(parcels, OUT_DIR / "_parts" / run, OUT_DIR, run, log, config)
        return
    run_extraction(parcels, partial(tile_features, year=YEAR), OUT_DIR, run, log, workers=WORKERS,
                   retries=RETRIES, worker_init=partial(dask_init, DASK_THREADS), config=config)


if __name__ == "__main__":
    main()
