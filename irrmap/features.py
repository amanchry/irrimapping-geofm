"""Parcel-level feature extraction from gridded xarray data.

The parcels are processed one spatial block at a time (the 20 km blocks of the split notebook).
For each block only the window around its parcels is read, so memory stays bounded and a failed
or interrupted run can resume from the per-block cache.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import odc.geo.xr  # noqa: F401  (registers the .odc accessor)
import xarray as xr
from rasterio.features import rasterize


def parcel_zonal_mean(
    da: xr.DataArray,
    parcels: gpd.GeoDataFrame,
    id_col: str = "parcel_id",
    inner_buffer_m: float = 5.0,
    metric_crs: int = 32629,
    std_features: list[str] | None = None,
) -> pd.DataFrame:
    """Mean of every feature layer of `da` inside each parcel (plus the std of `std_features`).

    `da` must have dims ("feature", <y>, <x>) and a geobox (odc-geo); the spatial dims may be named
    y/x or latitude/longitude. NaN marks missing values.
    Parcels are shrunk by `inner_buffer_m` to avoid mixed edge pixels; parcels too small to
    contain a pixel centre after shrinking fall back to all touched pixels of the full polygon.
    Returns one row per parcel with one column per feature, `<feature>_std` for each of
    `std_features` (within-parcel heterogeneity), and `n_px` (pixels used).
    """
    if da.ndim != 3 or da.dims[0] != "feature":
        raise ValueError(f"expected dims ('feature', y, x), got {da.dims}")
    gbox = da.odc.geobox
    shape, transform = gbox.shape, gbox.transform
    parcels = parcels.to_crs(gbox.crs)
    shrunk = parcels.to_crs(metric_crs).buffer(-inner_buffer_m).to_crs(gbox.crs)

    n = len(parcels)
    codes = np.arange(1, n + 1, dtype="int32")
    shapes = [(g, c) for g, c in zip(shrunk, codes) if g is not None and not g.is_empty]
    labels = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32") if shapes \
        else np.zeros(shape, dtype="int32")

    # Fallback for parcels without any pixel centre inside the shrunk polygon.
    has_px = np.bincount(labels.ravel(), minlength=n + 1)[1:] > 0
    if (~has_px).any():
        small = [(g, c) for g, c, ok in zip(parcels.geometry, codes, has_px) if not ok]
        extra = rasterize(small, out_shape=shape, transform=transform, fill=0, dtype="int32", all_touched=True)
        labels = np.where(labels == 0, extra, labels)

    flat = labels.ravel()
    inside = flat > 0
    lab = flat[inside]
    values = np.asarray(da.values, dtype="float32").reshape(da.sizes["feature"], -1)[:, inside]

    names = [str(f) for f in da["feature"].values]
    want_std = set(std_features or [])
    means = np.full((n, values.shape[0]), np.nan, dtype="float32")
    stds = {}
    for k, v in enumerate(values):
        ok = np.isfinite(v)
        total = np.bincount(lab[ok], weights=v[ok], minlength=n + 1)[1:]
        count = np.bincount(lab[ok], minlength=n + 1)[1:]
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        means[:, k] = mean
        if names[k] in want_std:
            sq = np.bincount(lab[ok], weights=v[ok].astype("float64") ** 2, minlength=n + 1)[1:]
            var = np.where(count > 1, sq / np.maximum(count, 1) - mean ** 2, np.nan)
            stds[f"{names[k]}_std"] = np.sqrt(np.clip(var, 0, None)).astype("float32")

    out = pd.DataFrame(means, columns=names, index=parcels[id_col].values)
    for col, v in stds.items():
        out[col] = v
    out.index.name = id_col
    out["n_px"] = np.bincount(lab, minlength=n + 1)[1:]
    return out


def block_bbox(parcels: gpd.GeoDataFrame, pad_m: float = 50.0, metric_crs: int = 32629) -> tuple[float, ...]:
    """Bounding box (lon/lat) of a group of parcels, padded by `pad_m` metres."""
    x0, y0, x1, y1 = parcels.to_crs(metric_crs).total_bounds
    padded = gpd.GeoSeries.from_xy([x0 - pad_m, x1 + pad_m], [y0 - pad_m, y1 + pad_m], crs=metric_crs).to_crs(4326)
    return (padded.x.min(), padded.y.min(), padded.x.max(), padded.y.max())


def extract_by_block(
    parcels: gpd.GeoDataFrame,
    block_fn: Callable[[gpd.GeoDataFrame], pd.DataFrame],
    cache_dir: Path,
    block_col: str = "block_id",
    retries: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """Run `block_fn(parcels_in_block)` for every block, caching one parquet file per block.

    `block_fn` returns the feature table of the parcels it receives (one row per parcel), typically
    by loading the data window around them and calling `parcel_zonal_mean`. Blocks already in the
    cache are skipped, so an interrupted run resumes where it stopped.
    Returns the concatenated feature table and the blocks that still failed after `retries` attempts.
    """
    from tqdm.auto import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for block, group in tqdm(parcels.groupby(block_col), desc="blocks"):
        target = cache_dir / f"{block}.parquet"
        if target.exists():
            continue
        for attempt in range(1, retries + 1):
            try:
                block_fn(group).to_parquet(target)
                break
            except Exception as err:  # network hiccups are common when streaming cloud data
                if attempt == retries:
                    print(f"block {block} failed: {err!r}")
                    failed.append(block)
                else:
                    time.sleep(5 * attempt)
    parts = [pd.read_parquet(p) for p in sorted(cache_dir.glob("*.parquet"))]
    table = pd.concat(parts) if parts else pd.DataFrame()
    return table, failed
