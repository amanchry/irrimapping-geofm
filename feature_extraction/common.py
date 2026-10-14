"""Shared machinery for parcel-level feature extraction.

Parcels are grouped into square tiles (default 10 km). For each tile the data window around its
parcels is loaded once and every parcel's mean is computed in one vectorised pass. Results are
written per tile, so a run can be stopped and resumed at any time, tiles can run in parallel, and
a failing tile (e.g. a network error) is retried, logged and skipped without stopping the run.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401  (registers the .odc accessor)
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from tqdm.auto import tqdm

ID_COL = "parcel_id"
METRIC_CRS = 32629          # UTM 29N: metric CRS for tiling, buffers and padding (Morocco)


# --------------------------------------------------------------------------- setup
def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("features")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(log_file), logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def load_parcels(path: Path, tile_km: float, id_col: str = ID_COL) -> gpd.GeoDataFrame:
    """Parcels with a valid polygon, a unique id and a tile id (labels are not needed here)."""
    gdf = gpd.read_file(path)
    if id_col not in gdf:
        raise KeyError(f"{path} has no '{id_col}' column")
    gdf = gdf[[id_col, "geometry"]].copy()
    gdf[id_col] = gdf[id_col].astype(str)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf["geometry"] = gdf.geometry.make_valid()
    dup = gdf[id_col].duplicated()
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicated parcel ids, e.g. {gdf.loc[dup, id_col].iloc[0]}")
    gdf = gdf.to_crs(4326)
    c = gdf.to_crs(METRIC_CRS).geometry.centroid
    size = tile_km * 1000
    gdf["tile_id"] = (np.floor(c.x / size).astype(int).astype(str) + "_" +
                      np.floor(c.y / size).astype(int).astype(str))
    return gdf.reset_index(drop=True)


def tile_bbox(parcels: gpd.GeoDataFrame, pad_m: float = 50.0) -> tuple[float, float, float, float]:
    """Lon/lat bounding box of the parcels, padded by `pad_m` metres."""
    x0, y0, x1, y1 = parcels.to_crs(METRIC_CRS).total_bounds
    pts = gpd.GeoSeries.from_xy([x0 - pad_m, x1 + pad_m], [y0 - pad_m, y1 + pad_m], crs=METRIC_CRS).to_crs(4326)
    return float(pts.x.min()), float(pts.y.min()), float(pts.x.max()), float(pts.y.max())


# --------------------------------------------------------------------------- zonal mean
def parcel_means(da: xr.DataArray, parcels: gpd.GeoDataFrame, inner_buffer_m: float = 5.0,
                 id_col: str = ID_COL) -> pd.DataFrame:
    """Mean of every layer of `da` (dims: feature, y, x) over the pixels inside each parcel.

    Parcels are shrunk by `inner_buffer_m` to drop mixed edge pixels; a parcel with no pixel centre
    left falls back to all pixels touched by its full polygon. NaN pixels are ignored.
    Returns one row per parcel, one column per layer, plus `n_px` (pixels used).
    """
    if da.ndim != 3 or da.dims[0] != "feature":
        raise ValueError(f"expected dims ('feature', y, x), got {da.dims}")
    gbox = da.odc.geobox
    parcels = parcels.to_crs(gbox.crs)
    shrunk = parcels.to_crs(METRIC_CRS).buffer(-inner_buffer_m).to_crs(gbox.crs)
    n = len(parcels)
    codes = np.arange(1, n + 1, dtype="int32")

    shapes = [(g, c) for g, c in zip(shrunk, codes) if g is not None and not g.is_empty]
    labels = (rasterize(shapes, out_shape=gbox.shape, transform=gbox.transform, fill=0, dtype="int32")
              if shapes else np.zeros(gbox.shape, dtype="int32"))
    has_px = np.bincount(labels.ravel(), minlength=n + 1)[1:] > 0
    if (~has_px).any():
        small = [(g, c) for g, c, ok in zip(parcels.geometry, codes, has_px) if not ok]
        extra = rasterize(small, out_shape=gbox.shape, transform=gbox.transform, fill=0,
                          dtype="int32", all_touched=True)
        labels = np.where(labels == 0, extra, labels)

    flat = labels.ravel()
    inside = flat > 0
    lab = flat[inside]
    values = np.asarray(da.values, dtype="float32").reshape(da.sizes["feature"], -1)[:, inside]
    means = np.full((n, values.shape[0]), np.nan, dtype="float32")
    for k, v in enumerate(values):
        ok = np.isfinite(v)
        total = np.bincount(lab[ok], weights=v[ok], minlength=n + 1)[1:]
        count = np.bincount(lab[ok], minlength=n + 1)[1:]
        means[:, k] = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    out = pd.DataFrame(means, columns=[str(f) for f in da["feature"].values], index=parcels[id_col].values)
    out.index.name = id_col
    out["n_px"] = np.bincount(lab, minlength=n + 1)[1:].astype("int32")
    return out


# --------------------------------------------------------------------------- runner
def _run_tile(tile_fn: Callable, tile_id: str, parcels: gpd.GeoDataFrame, parts_dir: Path,
              retries: int) -> dict:
    """Process one tile with retries; write its part file or a failure record. Runs in a worker."""
    started = time.time()
    fail_file = parts_dir / f"{tile_id}.failed.json"
    last_err, last_tb = None, None
    for attempt in range(1, retries + 1):
        try:
            table = tile_fn(parcels)
            missing = set(parcels[ID_COL]) - set(table.index)
            if missing:
                raise RuntimeError(f"{len(missing)} parcels missing from tile output")
            tmp = parts_dir / f"{tile_id}.parquet.tmp"
            table.to_parquet(tmp)
            tmp.replace(parts_dir / f"{tile_id}.parquet")      # atomic: a part is either complete or absent
            fail_file.unlink(missing_ok=True)
            return {"tile": tile_id, "status": "ok", "n": len(parcels), "seconds": round(time.time() - started, 1),
                    "empty": int((table["n_px"] == 0).sum()), "attempts": attempt}
        except Exception as err:  # noqa: BLE001 - every error is recorded, the run continues
            last_err, last_tb = repr(err), traceback.format_exc()
            if attempt < retries:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    fail_file.write_text(json.dumps({
        "tile": tile_id, "n_parcels": len(parcels), "parcel_ids": parcels[ID_COL].tolist(),
        "error": last_err, "traceback": last_tb, "time": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return {"tile": tile_id, "status": "failed", "n": len(parcels), "error": last_err,
            "seconds": round(time.time() - started, 1)}


def run_extraction(parcels: gpd.GeoDataFrame, tile_fn: Callable, out_dir: Path, run_name: str,
                   logger: logging.Logger, workers: int = 1, retries: int = 3,
                   worker_init: Callable | None = None, config: dict | None = None) -> pd.DataFrame:
    """Run `tile_fn` over all tiles (resumable, parallel), then merge the parts into one table.

    `tile_fn(parcels_in_tile) -> DataFrame` must be a picklable top-level function (or partial).
    """
    parts_dir = out_dir / "_parts" / run_name
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = {p.name.removesuffix(".parquet") for p in parts_dir.glob("*.parquet")}
    groups = {t: g for t, g in parcels.groupby("tile_id")}
    todo = [t for t in groups if t not in done]
    n_done = int(sum(len(groups[t]) for t in groups if t in done))
    logger.info(f"{run_name}: {len(parcels):,} parcels in {len(groups)} tiles | "
                f"already done {len(done)} tiles ({n_done:,} parcels) | to do {len(todo)} tiles | workers {workers}")

    results = []
    bar = tqdm(total=len(parcels), initial=n_done, unit="parcel", desc=run_name, dynamic_ncols=True)

    def record(res):
        results.append(res)
        bar.update(res["n"])
        failed = sum(r["status"] == "failed" for r in results)
        bar.set_postfix(tiles=f"{len(results)}/{len(todo)}", failed=failed)
        if res["status"] == "ok":
            logger.info(f"tile {res['tile']}: {res['n']} parcels in {res['seconds']} s"
                        + (f" ({res['empty']} without valid pixels)" if res["empty"] else ""))
        else:
            logger.error(f"tile {res['tile']} FAILED ({res['n']} parcels): {res['error']}")

    if workers <= 1:
        if worker_init:
            worker_init()
        for t in todo:
            record(_run_tile(tile_fn, t, groups[t], parts_dir, retries))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
            futures = [pool.submit(_run_tile, tile_fn, t, groups[t], parts_dir, retries) for t in todo]
            for fut in as_completed(futures):
                try:
                    record(fut.result())
                except Exception as err:  # worker crash (e.g. out of memory)
                    logger.error(f"worker crashed: {err!r}")
    bar.close()
    return merge_parts(parcels, parts_dir, out_dir, run_name, logger, config)


def merge_parts(parcels: gpd.GeoDataFrame, parts_dir: Path, out_dir: Path, run_name: str,
                logger: logging.Logger, config: dict | None = None) -> pd.DataFrame:
    """Concatenate tile parts, write the feature table and a summary of what is missing."""
    parts = sorted(parts_dir.glob("*.parquet"))
    table = pd.concat([pd.read_parquet(p) for p in parts]) if parts else pd.DataFrame()
    failed = [json.loads(p.read_text()) for p in sorted(parts_dir.glob("*.failed.json"))]
    failed_ids = sorted({i for f in failed for i in f["parcel_ids"]} - set(table.index))
    not_run = sorted(set(parcels[ID_COL]) - set(table.index) - set(failed_ids))

    summary = {
        "run": run_name, "created": datetime.now(timezone.utc).isoformat(),
        "parcels_total": int(len(parcels)), "parcels_extracted": int(len(table)),
        "parcels_without_valid_pixels": int((table["n_px"] == 0).sum()) if len(table) else 0,
        "parcels_failed": len(failed_ids), "parcels_not_run": len(not_run),
        "failed_tiles": {f["tile"]: f["error"] for f in failed},
        "n_features": int(table.shape[1] - 1) if len(table) else 0,
        "config": config or {},
    }
    if len(table):
        table = table.sort_index()
        table.to_parquet(out_dir / f"{run_name}.parquet")
    (out_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    pd.Series(failed_ids, name=ID_COL).to_csv(out_dir / f"{run_name}_failed_parcels.csv", index=False)

    logger.info(f"Merged {len(table):,}/{len(parcels):,} parcels -> {out_dir / (run_name + '.parquet')}")
    if failed_ids:
        logger.warning(f"{len(failed_ids)} parcels in {len(failed)} failed tiles; re-run the script to retry them")
    return table


def dask_init(threads: int):
    """Configure dask inside each worker process."""
    import dask
    dask.config.set(scheduler="threads", num_workers=threads)


# --------------------------------------------------------------------------- STAC helpers (Sentinel scripts)
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    """First day of every month from `start` to `end` inclusive ('YYYY-MM')."""
    return list(pd.date_range(f"{start}-01", f"{end}-01", freq="MS"))


def open_catalog(url: str = PC_STAC):
    """Planetary Computer STAC client; asset URLs are signed at search time (no account needed)."""
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(url, modifier=planetary_computer.sign_inplace)


def search_items(catalog, collection: str, bbox, months: list[pd.Timestamp], **kwargs) -> list:
    end = months[-1] + pd.offsets.MonthEnd(0)
    window = f"{months[0]:%Y-%m-%d}/{end:%Y-%m-%d}"
    return list(catalog.search(collections=[collection], bbox=bbox, datetime=window, **kwargs).items())


def items_in_month(items: list, month: pd.Timestamp) -> list:
    nxt = month + pd.offsets.MonthBegin(1)
    return [it for it in items if month <= pd.Timestamp(it.datetime).tz_convert(None) < nxt]


def tile_geobox(bbox, crs: str, resolution: float):
    """Fixed analysis grid for one tile, shared by every month."""
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import box
    return GeoBox.from_geopolygon(box(*bbox, crs="EPSG:4326").to_crs(crs), resolution=resolution)


def nan_layers(gbox, names: list[str]) -> xr.Dataset:
    from odc.geo.xr import xr_zeros
    nan = xr_zeros(gbox, dtype="float32") * np.nan
    return xr.Dataset({n: nan for n in names})
