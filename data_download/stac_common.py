"""
stac_common.py  —  search an open STAC catalogue and read scenes onto a block of the grid (not run alone)
=======================================================================================================

Used by 02b_s2.py (Sentinel-2 L2A, Earth Search) and 02c_s1.py (Sentinel-1 RTC, Microsoft Planetary Computer).
The scenes are cloud-optimised GeoTIFFs: only the window of the block is read over HTTPS (stackstac + GDAL), and
every scene is resampled onto the block grid (the region's projection, 10 m, nearest neighbour), so a block file is
an exact cut-out of the project grid. No account and no quota are needed; Planetary Computer URLs are signed with
anonymous, short-lived tokens (planetary_computer), so the items are searched again for every block.

Reading is robust: a failed read is retried; if it still fails, the scenes are read one by one, and scenes that
cannot be read are reported (never treated as cloud), so the caller can skip the block and a rerun fetches it.
"""
import os
import time

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.enums import Resampling

from download_common import BLOCK_M, BLOCK_PX

# GDAL settings for many small range reads of cloud-optimised GeoTIFFs over HTTPS (cluster: through the UT proxy,
# which GDAL takes from http_proxy / https_proxy)
for _k, _v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "3",
               "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "VSI_CACHE": "TRUE",
               "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff"}.items():
    os.environ.setdefault(_k, _v)

SEARCH_RETRIES = 5
READ_RETRIES = 3


def block_bounds(bx, by):
    """(minx, miny, maxx, maxy) of block (bx, by) in the region's projection."""
    return (bx * BLOCK_M, by * BLOCK_M, (bx + 1) * BLOCK_M, (by + 1) * BLOCK_M)


def lonlat_bbox(bounds, crs_epsg):
    """Longitude / latitude box around a box in the region's projection (for the STAC search)."""
    minx, miny, maxx, maxy = bounds
    t = Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    xs, ys = t.transform([minx, minx, maxx, maxx], [miny, maxy, miny, maxy])
    return [min(xs), min(ys), max(xs), max(ys)]


def search(url, collection, bbox, start, end, query=None, signed=False):
    """All items of a collection over bbox in [start, end) (UTC), sorted by time."""
    from pystac_client import Client
    modifier = None
    if signed:                                                      # Planetary Computer: signed asset URLs
        import planetary_computer
        modifier = planetary_computer.sign_inplace
    end_incl = (pd.Timestamp(end) - pd.Timedelta(seconds=1)).isoformat()
    for attempt in range(SEARCH_RETRIES):
        try:
            client = Client.open(url, modifier=modifier)
            items = list(client.search(collections=[collection], bbox=bbox,
                                       datetime=f"{pd.Timestamp(start).isoformat()}Z/{end_incl}Z",
                                       query=query or {}).items())
            for it in items:                                        # older stackstac reads proj:epsg only
                code = it.properties.get("proj:code")
                if "proj:epsg" not in it.properties and code and code.upper().startswith("EPSG:"):
                    it.properties["proj:epsg"] = int(code.split(":")[1])
            return sorted(items, key=lambda it: it.datetime)
        except Exception as err:                                    # network / catalogue errors
            if attempt == SEARCH_RETRIES - 1:
                raise
            print(f"    search failed ({str(err)[:100]}), retrying", flush=True)
            time.sleep(10 * (attempt + 1))


def item_time(item):
    """Acquisition time of an item as a naive UTC timestamp (to compare with the month bounds)."""
    return pd.Timestamp(item.datetime).tz_convert("UTC").tz_localize(None)


def in_month(items, month):
    """Items acquired in the calendar month that starts at `month` (naive UTC timestamp)."""
    end = month + pd.DateOffset(months=1)
    return [it for it in items if month <= item_time(it) < end]


def _stack(items, assets, bounds, crs_epsg, dtype, fill, size_px, threads):
    """[n, len(assets), size_px, size_px] on the block grid and the items of its n scenes (same order).
    stackstac leaves out scenes whose footprint does not reach the block (the search box is larger than the block),
    so the scenes are matched to the items by id, never by position."""
    import stackstac
    fill = np.dtype(dtype).type(fill)                               # stackstac needs a fill value of the dtype
    res = (bounds[2] - bounds[0]) / size_px
    da = stackstac.stack(items, assets=assets, epsg=crs_epsg, resolution=res, bounds=bounds,
                         resampling=Resampling.nearest, dtype=dtype, fill_value=fill, rescale=False,
                         snap_bounds=False, chunksize=size_px)
    if da.sizes["time"] == 0:
        return np.zeros((0, len(assets), size_px, size_px), dtype), []
    if list(da["band"].values) != list(assets):                     # stackstac drops an asset missing everywhere
        raise RuntimeError(f"bands {list(da['band'].values)} instead of {assets}")
    arr = np.asarray(da.compute(scheduler="threads", num_workers=threads))
    if arr.shape[-2:] != (size_px, size_px):
        raise RuntimeError(f"unexpected block shape {arr.shape}")
    by_id = {it.id: it for it in items}
    return arr, [by_id[i] for i in np.atleast_1d(da["id"].values)]


def read_scenes(items, assets, bounds, crs_epsg, dtype, fill, size_px=BLOCK_PX, threads=8):
    """Read the scenes onto the block grid. Returns (array [n, assets, H, W], items of the n scenes, ids of scenes
    that could not be read). A failed read is retried, then the scenes are read one by one."""
    empty = np.zeros((0, len(assets), size_px, size_px), dtype)
    if not items:
        return empty, [], []
    for attempt in range(READ_RETRIES):
        try:
            arr, kept = _stack(items, assets, bounds, crs_epsg, dtype, fill, size_px, threads)
            return arr, kept, []
        except Exception as err:                                    # network / read errors
            print(f"    read failed ({str(err)[:120]}), retry {attempt + 1}/{READ_RETRIES}", flush=True)
            time.sleep(15 * (attempt + 1))
    arrs, kept, failed = [], [], []
    for it in items:
        for attempt in range(READ_RETRIES):
            try:
                a, k = _stack([it], assets, bounds, crs_epsg, dtype, fill, size_px, threads)
                arrs.append(a)
                kept += k
                break
            except Exception as err:                                # noqa: BLE001
                if attempt == READ_RETRIES - 1:
                    print(f"    scene {it.id} not readable: {str(err)[:120]}", flush=True)
                    failed.append(it.id)
                else:
                    time.sleep(10 * (attempt + 1))
    return (np.concatenate(arrs) if arrs else empty), kept, failed
