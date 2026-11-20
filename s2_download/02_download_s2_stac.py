"""
02_download_s2_stac.py  —  monthly Sentinel-2 composites per 128 px cropland tile from a STAC catalogue
=======================================================================================================

What it does
------------
For every download block (1024 x 1024 px = 10.24 km, 8 x 8 model tiles; 01_s2_grid_overview.py) and agricultural
year:
1. Searches the Sentinel-2 L2A scenes of the year over the block (Earth Search, collection sentinel-2-c1-l2a; no
   account needed). Scenes with more than MAX_SCENE_CLOUD % cloud are skipped.
2. Month by month, reads the 12 bands and the scene classification (SCL) of every scene onto the block grid
   (region's UTM CRS, 10 m, nearest-neighbour resampling), as raw integers.
3. Harmonises the reflectance: scenes of processing baseline 04.00 or later (from 25 Jan 2022) carry an offset of
   +1000 in Collection 1; it is subtracted (values below 0 -> 0), so all years are on one scale.
   A failed read is retried, then the month is read scene by scene; scenes that still fail are left out and
   logged (never treated as cloud). If more than MAX_FAILED_FRAC of a block's scenes fail, the block is skipped
   and logged, and no tiles are written, so a rerun downloads it again.
4. Cloud mask per pixel and scene: SCL classes in SCL_INVALID are removed (no data, defective, dark, cloud shadow,
   cloud medium / high, cirrus); a value of 0 is no data.
5. Monthly composite = median of the clear observations per band; S2_NCLEAR = number of clear observations.
   Every scene counts, also where two Sentinel-2 tiles overlap (the same acquisition then counts twice).
6. Cuts the block into its 8 x 8 tiles and writes only the selected tiles: int16, 156 bands month-major
   (<band>_m01 ... <band>_m12 with B1 ... B12 and S2_NCLEAR), NODATA -32768 where a month has no clear observation,
   tile_id and grid of the cropland mask.

Which tiles
-----------
Every tile with >= MIN_CROP_FRAC cropland in the cropland mask of the year (tiles_cropland.parquet of
cropland_extent/make_cropland_mask.py), grouped into blocks of 8 x 8 tiles.

Outputs (outputs/<region>/s2_download/cropland/<season>/)
--------------------------------------------------------
tiles/<tile_id>_img.tif   int16, 156 bands, NODATA -32768, deflate
tiles_qa.jsonl            per tile: s2_nodata_pct, months_with_data, mean_nclear (on the tile's cropland pixels)
blocks_done.jsonl         per block: scenes found / used / not readable (ids), seconds, tiles written
blocks_failed.jsonl       blocks skipped because more than MAX_FAILED_FRAC of the scenes could not be read
tiles_info.json           grid, bands, months, source, cloud mask
logs/<name>_p<part>.jsonl the same three logs, written by each part to its own file while it runs

Many parts on several nodes write to the same folder (NFS), where appends to one shared file are not safe, so each
part logs to its own files. Once every tile of the season exists, the part that finishes last merges them into the
three files above: one QA row per tile (rows still missing are recomputed from the tiles), one entry per block in
blocks_done.jsonl, and only the blocks that still miss tiles in blocks_failed.jsonl (none). Merged files are written
to a temporary file and renamed, never appended.

Run (from the project folder; CPU, internet; resumable: finished tiles are skipped)
----------------------------------------------------------------------------------
    python s2_download/02_download_s2_stac.py
    # cluster, one array task per share of the blocks (compute nodes go through the proxy):
    #   EXPERIMENT_REGION=morocco S2_SEASONS=2023/24 S2_PART=<k> S2_N_PARTS=<n> \
    #       python s2_download/02_download_s2_stac.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixelwise_run"))   # regions.py is shared
from regions import cropland_dir, settings  # noqa: E402

# GDAL settings for many small range reads of cloud-optimised GeoTIFFs over HTTPS
for k, v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "3",
             "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "VSI_CACHE": "TRUE",
             "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff"}.items():
    os.environ.setdefault(k, v)


def env(name, default):
    """Setting from the environment (cluster jobs), else the default below."""
    v = os.environ.get(name)
    return default if v is None else type(default)(v) if not isinstance(default, bool) else v.lower() in ("1", "true")


# ================================ SETTINGS ================================
REGION = env("EXPERIMENT_REGION", "morocco")        # "morocco" | "utah"
SEASONS = {"morocco": ["2023/24", "2024/25"], "utah": ["2023/24"]}[REGION]   # the training years
SEASONS = [x for x in env("S2_SEASONS", ",".join(SEASONS)).split(",") if x]  # e.g. S2_SEASONS=2024/25 (one job set per year)
MIN_CROP_FRAC = 0.01                                # tile kept if >= 1% cropland (as 01_s2_grid_overview)
N_TEST_BLOCKS = env("S2_TEST_BLOCKS", 0)            # e.g. 3 blocks for a quick test; 0 = all
PART, N_PARTS = env("S2_PART", 0), env("S2_N_PARTS", 1)   # this job handles blocks PART, PART + N_PARTS, ...

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
MAX_SCENE_CLOUD = 60                                # scene cloud cover (%), as CLOUDY_PIXEL_PERCENTAGE < 60 in EE
SCL_INVALID = [0, 1, 2, 3, 8, 9, 10]                # no data, defective, dark, shadow, cloud medium/high, cirrus
RESAMPLING = "nearest"                              # 20 m / 60 m bands to 10 m
OFFSET_FROM_BASELINE = "04.00"                      # baseline >= this: subtract 1000 (harmonised reflectance)
THREADS = 8                                         # parallel reads (dask threads)
READ_RETRIES = 3                                    # retries of a failed read (month, then scene by scene)
MAX_FAILED_FRAC = 0.02                              # more unreadable scenes than this share -> block skipped,
                                                    # logged in blocks_failed.jsonl, no tiles written (rerun later)
# ==========================================================================

ROOT = Path(__file__).resolve().parents[1]
CRS_EPSG = settings(REGION)["crs_epsg"]
START_MONTH = settings(REGION)["start_month"]
GSD_M, TILE_PX, BLOCK_TILES, N_MONTHS = 10, 128, 8, 12
TILE_M, BLOCK_PX = GSD_M * TILE_PX, TILE_PX * BLOCK_TILES
BLOCK_M = BLOCK_PX * GSD_M
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
ASSETS = {"B1": "coastal", "B2": "blue", "B3": "green", "B4": "red", "B5": "rededge1", "B6": "rededge2",
          "B7": "rededge3", "B8": "nir", "B8A": "nir08", "B9": "nir09", "B11": "swir16", "B12": "swir22"}
MONTH_BANDS = S2_BANDS + ["S2_NCLEAR"]
NODATA = -32768
MASK_ROOT = cropland_dir(REGION) / "cropland_mask"


def band_names():
    return [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in MONTH_BANDS]


def season_tag(season):
    return season.replace("/", "-")


def month_starts(season):
    start = pd.Timestamp(year=int(season[:4]), month=START_MONTH, day=1)
    return [start + pd.DateOffset(months=k) for k in range(N_MONTHS)]


# ------------------------------------------------------------ tile selection
def select_tiles(season):
    tag = season_tag(season)
    mask = pd.read_parquet(MASK_ROOT / tag / "tiles_cropland.parquet")
    t = mask[(mask["n_crop_px"] > 0) & (mask["crop_frac"] >= MIN_CROP_FRAC)]
    t = t.assign(bx=t["ix"] // BLOCK_TILES, by=t["iy"] // BLOCK_TILES)
    return t[["tile_id", "ix", "iy", "bx", "by", "n_crop_px"]].reset_index(drop=True)


# ------------------------------------------------------------ STAC
def search(bounds_ll, season):
    from pystac_client import Client
    months = month_starts(season)
    a, b = months[0], months[-1] + pd.DateOffset(months=1) - pd.Timedelta(seconds=1)
    for attempt in range(5):
        try:
            items = list(Client.open(STAC_URL).search(
                collections=[COLLECTION], bbox=list(bounds_ll), datetime=f"{a.isoformat()}Z/{b.isoformat()}Z",
                query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD}}).items())
            return sorted(items, key=lambda it: it.datetime)
        except Exception as err:                                            # noqa: BLE001
            if attempt == 4:
                raise
            print(f"    search failed ({str(err)[:100]}), retrying")
            time.sleep(10 * (attempt + 1))


def baseline_offset(item):
    """1000 for scenes of processing baseline >= 04.00 (Collection 1 stores the raw values with the offset)."""
    pb = item.properties.get("s2:processing_baseline", "00.00")
    return 1000 if float(pb) >= float(OFFSET_FROM_BASELINE) else 0


def empty_stack():
    return np.zeros((0, len(S2_BANDS) + 1, BLOCK_PX, BLOCK_PX), np.uint16)


def stack(items, bounds):
    """[n, 13, H, W] uint16 (12 bands + SCL) on the block grid and the n items of its scenes, in the same order;
    raises if any read fails.
    stackstac leaves out scenes whose footprint (asset proj:bbox) does not reach the block: the STAC search uses the
    lon/lat box around the block, which is larger than the block. The scenes kept are therefore matched to the items
    by id, never by position; n = 0 if no scene reaches the block."""
    import stackstac
    assets = [ASSETS[b] for b in S2_BANDS] + ["scl"]
    da = stackstac.stack(items, assets=assets, epsg=CRS_EPSG, resolution=GSD_M, bounds=bounds,
                         resampling=getattr(Resampling, RESAMPLING), dtype="uint16", fill_value=np.uint16(0),
                         rescale=False, snap_bounds=False, chunksize=BLOCK_PX)
    if da.sizes["time"] == 0:
        return empty_stack(), []
    if list(da["band"].values) != assets:          # stackstac also drops an asset missing from every scene
        raise RuntimeError(f"bands {list(da['band'].values)} instead of {assets}")
    arr = np.asarray(da.compute(scheduler="threads", num_workers=THREADS))
    if arr.shape[-2:] != (BLOCK_PX, BLOCK_PX):
        raise RuntimeError(f"unexpected block shape {arr.shape}")
    by_id = {it.id: it for it in items}
    return arr, [by_id[i] for i in np.atleast_1d(da["id"].values)]


def read_scenes(items, bounds):
    """All scenes of one month that reach the block. A failed read is retried; if the month still fails, the scenes
    are read one by one and the ones that cannot be read are left out and returned as failed (never silently
    treated as cloud). Returns the array, the items of its scenes in the same order, and the failed ids."""
    for attempt in range(READ_RETRIES):
        try:
            arr, kept = stack(items, bounds)
            return arr, kept, []
        except Exception as err:                                            # noqa: BLE001
            print(f"    read failed ({str(err)[:120]}), retry {attempt + 1}/{READ_RETRIES}")
            time.sleep(15 * (attempt + 1))
    ok, arrs, failed = [], [], []
    for it in items:
        for attempt in range(READ_RETRIES):
            try:
                a, kept = stack([it], bounds)
                arrs.append(a)
                ok += kept
                break
            except Exception as err:                                        # noqa: BLE001
                if attempt == READ_RETRIES - 1:
                    print(f"    scene {it.id} not readable: {str(err)[:120]}")
                    failed.append(it.id)
                else:
                    time.sleep(10 * (attempt + 1))
    arr = np.concatenate(arrs) if arrs else empty_stack()
    return arr, ok, failed


def read_month(items, bounds):
    """[n_items, 12 bands, H, W] float32 reflectance x 10000 with NaN where not clear, for one month;
    plus the clear mask and the ids of scenes that could not be read."""
    arr, items, failed = read_scenes(items, bounds)
    if len(items) != len(arr):
        raise RuntimeError(f"{len(arr)} scenes stacked but {len(items)} items matched")
    scl = arr[:, -1]
    refl = arr[:, :-1].astype(np.float32)
    clear = ~np.isin(scl, SCL_INVALID) & (arr[:, S2_BANDS.index("B4")] > 0)
    off = np.array([baseline_offset(it) for it in items], np.float32)[:, None, None, None]
    refl = np.maximum(refl - off, 0)
    refl[~np.broadcast_to(clear[:, None], refl.shape)] = np.nan
    return refl, clear, failed


class TooManyFailed(RuntimeError):
    pass


def block_cube(items, bounds, months):
    """[156, 1024, 1024] int16 monthly composites of one block, scenes used, ids of scenes that failed."""
    cube = np.full((N_MONTHS, len(MONTH_BANDS), BLOCK_PX, BLOCK_PX), NODATA, np.int16)
    cube[:, -1] = 0
    used, failed = 0, []
    for k, m in enumerate(months):
        sel = [it for it in items if m <= pd.Timestamp(it.datetime).tz_localize(None) < m + pd.DateOffset(months=1)]
        if not sel:
            continue
        refl, clear, bad = read_month(sel, bounds)
        failed += bad
        if len(failed) > MAX_FAILED_FRAC * len(items):
            raise TooManyFailed(f"{len(failed)} of {len(items)} scenes not readable")
        used += len(refl)                                                   # scenes read that reach the block
        if not len(refl):
            continue
        n = clear.sum(0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(refl, axis=0)                                # [12, H, W]
        med = np.where(np.isnan(med), NODATA, np.minimum(np.rint(med), 32767))
        cube[k, :-1] = med.astype(np.int16)
        cube[k, -1] = np.minimum(n, 32767).astype(np.int16)
    return cube.reshape(N_MONTHS * len(MONTH_BANDS), BLOCK_PX, BLOCK_PX), used, failed


# ------------------------------------------------------------ outputs
def write_tif(path, cube, ix, iy):
    tmp = path.with_suffix(".part.tif")
    profile = dict(driver="GTiff", width=TILE_PX, height=TILE_PX, count=cube.shape[0], dtype="int16",
                   crs=f"EPSG:{CRS_EPSG}", transform=Affine(GSD_M, 0, ix * TILE_M, 0, -GSD_M, (iy + 1) * TILE_M),
                   nodata=NODATA, compress="deflate", predictor=2, tiled=True, blockxsize=TILE_PX, blockysize=TILE_PX)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(cube)
        for i, d in enumerate(band_names(), start=1):
            dst.set_band_description(i, d)
    tmp.replace(path)


def tile_qa(cube, crop):
    c = cube.reshape(N_MONTHS, len(MONTH_BANDS), TILE_PX, TILE_PX)
    on = crop == 1
    if not on.any():
        on = np.ones_like(on, bool)                                         # no cropland pixel in the window
    missing = c[:, S2_BANDS.index("B4")][:, on] == NODATA
    return {"s2_nodata_pct": round(100 * float(missing.mean()), 2),
            "months_with_data": int(((~missing).mean(axis=1) >= 0.5).sum()),
            "mean_nclear": round(float(c[:, -1][:, on].mean()), 2)}


def main():
    import geopandas as gpd
    from shapely.geometry import box
    for season in SEASONS:
        tag = season_tag(season)
        out = ROOT / "outputs" / REGION / "s2_download" / "cropland" / tag
        (out / "tiles").mkdir(parents=True, exist_ok=True)
        (out / "tiles_info.json").write_text(json.dumps({
            "region": REGION, "season": season, "crs_epsg": CRS_EPSG, "gsd_m": GSD_M,
            "tile_px": TILE_PX, "block_px": BLOCK_PX, "start_month": START_MONTH, "n_months": N_MONTHS,
            "month_bands": MONTH_BANDS, "s2_scale": "surface reflectance x 10000 (harmonised)", "nodata": NODATA,
            "band_order": "month-major: <band>_m01 ... <band>_m12",
            "source": f"{STAC_URL} {COLLECTION}", "resampling": RESAMPLING,
            "cloud_mask": f"SCL not in {SCL_INVALID}; scenes with eo:cloud_cover < {MAX_SCENE_CLOUD}",
            "offset": f"-1000 for processing baseline >= {OFFSET_FROM_BASELINE}"}, indent=2))

        tiles = select_tiles(season)
        tiles["path"] = [out / "tiles" / f"{t}_img.tif" for t in tiles["tile_id"]]
        blocks = tiles.groupby(["bx", "by"], sort=True)
        keys = [k for k in blocks.groups][PART::N_PARTS]
        if N_TEST_BLOCKS:
            keys = keys[:N_TEST_BLOCKS]
        todo = [k for k in keys if not all(p.exists() for p in blocks.get_group(k)["path"])]
        print(f"== {REGION} {season}: {len(tiles):,} tiles in {blocks.ngroups:,} blocks; this job "
              f"{len(keys):,} blocks (part {PART + 1}/{N_PARTS}), {len(todo):,} to go")

        months = month_starts(season)
        mask_vrt = rasterio.open(MASK_ROOT / tag / "cropland_mask.vrt")
        t_start = time.time()
        for n, (bx, by) in enumerate(todo, start=1):
            t0 = time.time()
            bounds = (bx * BLOCK_M, by * BLOCK_M, (bx + 1) * BLOCK_M, (by + 1) * BLOCK_M)
            ll = gpd.GeoSeries([box(*bounds)], crs=CRS_EPSG).to_crs(4326).total_bounds
            items = search(ll, season)
            try:
                cube, used, failed = block_cube(items, bounds, months)
            except TooManyFailed as err:
                print(f"  block {bx}_{by}: SKIPPED, {err}; no tiles written (rerun later)")
                append(part_log(out, "blocks_failed"), {"bx": int(bx), "by": int(by), "scenes": len(items),
                                                        "error": str(err), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                continue
            members = blocks.get_group((bx, by))
            for t in members.itertuples():
                c, r = t.ix - bx * BLOCK_TILES, (by + 1) * BLOCK_TILES - 1 - t.iy
                tc = cube[:, r * TILE_PX:(r + 1) * TILE_PX, c * TILE_PX:(c + 1) * TILE_PX]
                write_tif(t.path, tc, t.ix, t.iy)
                win = from_bounds(t.ix * TILE_M, t.iy * TILE_M, (t.ix + 1) * TILE_M, (t.iy + 1) * TILE_M,
                                  mask_vrt.transform)
                qa = tile_qa(tc, mask_vrt.read(1, window=win, boundless=True, fill_value=65535))
                append(part_log(out, "tiles_qa"), {"tile_id": t.tile_id, **qa})
            sec = time.time() - t0
            append(part_log(out, "blocks_done"), {"bx": int(bx), "by": int(by), "scenes": len(items),
                                                  "scenes_used": used, "scenes_failed": len(failed),
                                                  "failed_ids": failed, "tiles": len(members),
                                                  "seconds": round(sec, 1)})
            rate = (time.time() - t_start) / n
            print(f"  block {bx}_{by}: {len(items)} scenes ({len(failed)} not readable), {len(members)} tiles, {sec:.0f} s "
                  f"({n}/{len(todo)}, ~{rate * (len(todo) - n) / 3600:.1f} h left)")
        done = sum(p.exists() for p in tiles["path"])
        print(f"Tiles written: {done:,} of {len(tiles):,} | {out}")
        if done == len(tiles):
            finalise(out, tiles, blocks, mask_vrt)
        mask_vrt.close()


def part_log(out, name):
    """This part's own log file (one writer per file; see the docstring)."""
    return out / "logs" / f"{name}_p{PART:03d}.jsonl"


def append(path, row):
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def write_jsonl(path, rows):
    tmp = path.with_suffix(f".p{PART:03d}.tmp")              # own temporary file per part
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


def merged(out, name):
    """Rows of <name>.jsonl (merged earlier, or appended by older runs) and of every part's log file."""
    rows = read_jsonl(out / f"{name}.jsonl")
    for f in sorted((out / "logs").glob(f"{name}_p*.jsonl")):
        rows += read_jsonl(f)
    return rows


def read_jsonl(path):
    """Rows of a jsonl file; lines that do not parse (e.g. cut by a crash) are left out."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def finalise(out, tiles, blocks, mask_vrt):
    """Once every tile of the season exists (whichever part finishes last): merges the part logs into one QA row per
    tile (recomputed from the tile for any tile without a row), one blocks_done entry per block (the latest) and the
    blocks that still miss tiles in blocks_failed.jsonl (none). Each file is written once, atomically."""
    qa = {r["tile_id"]: r for r in merged(out, "tiles_qa") if "tile_id" in r}
    missing = [t for t in tiles.itertuples() if t.tile_id not in qa]
    for t in missing:
        with rasterio.open(t.path) as src:
            tc = src.read()
        win = from_bounds(t.ix * TILE_M, t.iy * TILE_M, (t.ix + 1) * TILE_M, (t.iy + 1) * TILE_M, mask_vrt.transform)
        qa[t.tile_id] = {"tile_id": t.tile_id,
                         **tile_qa(tc, mask_vrt.read(1, window=win, boundless=True, fill_value=65535))}
    keep = set(tiles["tile_id"])
    rows = [qa[k] for k in sorted(qa) if k in keep]
    write_jsonl(out / "tiles_qa.jsonl", rows)
    done = {(r["bx"], r["by"]): r for r in merged(out, "blocks_done") if "bx" in r}
    write_jsonl(out / "blocks_done.jsonl", [done[k] for k in sorted(done)])
    failed = merged(out, "blocks_failed")
    still = [r for r in failed if (r["bx"], r["by"]) in blocks.groups
             and not all(p.exists() for p in blocks.get_group((r["bx"], r["by"]))["path"])]
    write_jsonl(out / "blocks_failed.jsonl", still)
    print(f"Merged logs: {len(rows):,} QA rows ({len(missing)} recomputed), {len(done):,} blocks done, "
          f"{len(still)} failed ({len(failed) - len(still)} resolved)")


if __name__ == "__main__":
    main()
