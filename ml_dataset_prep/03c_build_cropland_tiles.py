"""
03c_build_cropland_tiles.py  —  wall-to-wall tile store of monthly Sentinel-2 images over the cropland mask
===========================================================================================================

What it does
------------
03_build_tiles.py downloads only the tiles around the labelled parcels (training). The continuous annual maps
need imagery for EVERY cropland pixel of the region, so this script downloads every tile of the same grid that
contains cropland in the mask of 00_make_cropland_mask.py. 03_build_tiles.py is unchanged.

Grid (identical to 03_build_tiles.py and the cropland mask)
-----------------------------------------------------------
Tiles of 128 x 128 pixels at 10 m (1.28 km) in the region's UTM zone (regions.py), tile (ix, iy) with
x0 = ix * 1280 m, top y = (iy + 1) * 1280 m, tile_id = "<season>_<ix>_<iy>". The same tile_id therefore
means the same pixels in the parcel store (03), the cropland mask (00) and this store, so labels, mask and
images line up without resampling. 128 px is kept because it is a whole number of every model's patches
(16 px Prithvi / AgriFM, 8 px Clay) and of 64 px model chips.

Which tiles
-----------
tiles_cropland.parquet of 00_make_cropland_mask.py: every tile with >= MIN_CROP_PX cropland pixels in the
mask of that agricultural year. MIN_CROP_PX = 1 keeps every cropland pixel (a complete map); the script prints
how many tiles and how much cropland a higher threshold would keep.

Per tile and month (12 months of the agricultural year; m01 = September in Morocco, October in Utah)
------------------------------------------------------------------------------------------------------
  B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12   Sentinel-2 L2A median of clear pixels (reflectance x 10000)
  S2_NCLEAR                                number of clear Sentinel-2 observations (Cloud Score+)
NODATA = -32768 where a month has no clear observation (S2_NCLEAR = 0 there). int16, 156 bands
(month-major: <band>_m01 ... <band>_m12). Same compositing and cloud settings as 03_build_tiles.py, so a tile
of this store equals the Sentinel-2 bands of the same tile in the parcel store. No Sentinel-1.

Download
--------
computePixels in requests of REQUEST_TILES x REQUEST_TILES tiles (2 x 2 = 256 px, about 20 MB; Earth Engine
allows 48 MB per request), cut into single tiles locally and written only for the selected tiles; a request
that fails with a memory / size error is retried tile by tile. Parcel-store tiles of the same season are
copied (Sentinel-2 bands only) instead of downloaded when their cloud settings match (REUSE_PARCEL_TILES).
Re-running skips finished tiles (resumable).

QA per tile (on the tile's cropland pixels of the mask)
-------------------------------------------------------
  s2_nodata_pct     share of month x cropland pixel values without a clear observation
  months_with_data  months with a clear observation on at least half of the cropland pixels
  mean_nclear       mean number of clear observations per month on cropland pixels

Outputs (outputs/<region>/dataset/03c_cropland_tiles_128px/<season>/)
---------------------------------------------------------------------
tiles/<tile_id>_img.tif   int16, 156 bands, NODATA -32768, deflate
tiles.parquet             tile index: tile_id, ix, iy, n_crop_px, crop_frac, downloaded, source, QA, geometry
tiles_qa.jsonl            QA per finished tile
tiles_info.json           grid, bands, months, cloud settings, mask settings
"""
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from shapely.geometry import box

from regions import dataset_dir, settings

# ================================ SETTINGS ================================
REGION = "morocco"                         # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24"], "utah": ["2023/24"]}[REGION]   # agricultural years to download
ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"     # Earth Engine key (never commit)

MIN_CROP_PX = 1                            # tile is downloaded if it has >= this many cropland pixels (1 = all)
REQUEST_TILES = 2                          # tiles per side in one Earth Engine request (2 -> 256 px, ~20 MB)
REUSE_PARCEL_TILES = True                  # copy the S2 bands of tiles already in the parcel store (03)

CLOUD_COVER_PERCENTAGE = 60                # same as 03_build_tiles.py
CS_THRESHOLD = 0.60                        # same as 03_build_tiles.py (Cloud Score+ cs_cdf >= this)

DOWNLOAD_WORKERS = 8                       # parallel Earth Engine requests
RETRIES = 4
N_TEST_TILES = None                        # e.g. 16 tiles for a quick test; None = all
# ==========================================================================

CRS_EPSG = settings(REGION)["crs_epsg"]
START_MONTH = settings(REGION)["start_month"]
GSD_M, TILE_PX, N_MONTHS = 10, 128, 12
TILE_M = TILE_PX * GSD_M
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
MONTH_BANDS = S2_BANDS + ["S2_NCLEAR"]
NODATA = -32768
MASK_ROOT = dataset_dir(REGION) / "00_cropland_mask"
PARCEL_STORE = dataset_dir(REGION) / "03_tiles_128px" / "agyear"
OUT_ROOT = dataset_dir(REGION) / "03c_cropland_tiles_128px"


def band_names():
    return [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in MONTH_BANDS]


def season_tag(season):
    return season.replace("/", "-")


def tile_transform(ix, iy):
    return Affine(GSD_M, 0, ix * TILE_M, 0, -GSD_M, (iy + 1) * TILE_M)


def month_starts(season):
    start = pd.Timestamp(year=int(season[:4]), month=START_MONTH, day=1)
    return [start + pd.DateOffset(months=k) for k in range(N_MONTHS)]


# ------------------------------------------------------------ tile selection
def select_tiles(season):
    path = MASK_ROOT / season_tag(season) / "tiles_cropland.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 00_make_cropland_mask.py for {season} first")
    t = pd.read_parquet(path)
    total = t["n_crop_px"].sum()
    print("Tiles kept and cropland covered for a minimum share of cropland per tile:")
    for f in (0.0, 0.01, 0.05, 0.10, 0.25):
        k = t[(t["n_crop_px"] > 0) & (t["crop_frac"] >= f)]
        print(f"  crop_frac >= {f:4.2f}: {len(k):7,} tiles, {100 * k['n_crop_px'].sum() / total:5.1f}% of the cropland")
    t = t[t["n_crop_px"] >= MIN_CROP_PX].sort_values(["iy", "ix"], ascending=[False, True]).reset_index(drop=True)
    return t


# ------------------------------------------------------------ Earth Engine
def init_gee():
    import ee
    key = json.loads(KEY_FILE.read_text())
    ee.Initialize(ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key)), project=key["project_id"])
    print("Earth Engine initialised")
    return ee


def monthly_stack(ee, season, region):
    """One int16 image: 12 months x (12 S2 bands + S2_NCLEAR), compositing as in 03_build_tiles.py."""
    months = month_starts(season)
    first, last = str(months[0].date()), str((months[-1] + pd.DateOffset(months=1)).date())
    s2_raw = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(first, last)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_PERCENTAGE)))
    cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (s2_raw.linkCollection(cloud_score, ["cs_cdf"])
          .map(lambda img: img.select(S2_BANDS).updateMask(img.select("cs_cdf").gte(CS_THRESHOLD))))
    empty = ee.ImageCollection([ee.Image.constant([0] * len(S2_BANDS)).rename(S2_BANDS).toUint16()
                                .updateMask(ee.Image.constant(0))])
    images = []
    for k, m in enumerate(months, start=1):
        a, b = str(m.date()), str((m + pd.DateOffset(months=1)).date())
        s2m = s2.filterDate(a, b).merge(empty)                       # dummy keeps empty months valid
        optical = s2m.median().select(S2_BANDS).min(32767).unmask(NODATA)
        n_clear = s2m.select("B4").count().unmask(0).rename("S2_NCLEAR")
        images.append(ee.Image.cat([optical, n_clear]).select(MONTH_BANDS, [f"{b}_m{k:02d}" for b in MONTH_BANDS]))
    return ee.Image.cat(images).toInt16()


def compute(ee, image, x0, y0, size):
    """[156, size, size] int16 for the square with top-left corner (x0, y0)."""
    arr = ee.data.computePixels({
        "expression": image, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": size, "height": size},
                 "affineTransform": {"scaleX": GSD_M, "shearX": 0, "translateX": float(x0),
                                     "shearY": 0, "scaleY": -GSD_M, "translateY": float(y0)},
                 "crsCode": f"EPSG:{CRS_EPSG}"}})
    return np.stack([arr[n] for n in band_names()]).astype(np.int16)


def download_group(ee, image, gx, gy, tiles):
    """All selected tiles of one request square -> {tile_id: cube}; falls back to single tiles on size errors."""
    g = REQUEST_TILES
    out, single = {}, g == 1
    for attempt in range(1, RETRIES + 1):
        try:
            if not single:
                cube = compute(ee, image, gx * g * TILE_M, (gy + 1) * g * TILE_M, g * TILE_PX)
                for t in tiles:
                    c, r = t.ix - gx * g, (gy + 1) * g - 1 - t.iy
                    out[t.tile_id] = cube[:, r * TILE_PX:(r + 1) * TILE_PX, c * TILE_PX:(c + 1) * TILE_PX]
            else:
                for t in tiles:
                    if t.tile_id not in out:
                        out[t.tile_id] = compute(ee, image, t.ix * TILE_M, (t.iy + 1) * TILE_M, TILE_PX)
            return out
        except Exception as err:
            msg = str(err).lower()
            single = single or any(k in msg for k in ("memory", "too large", "limit"))
            if attempt == RETRIES:
                print(f"  request {gx}_{gy}: failed ({str(err)[:150]})")
                return out
            time.sleep(5 if single else 10 * attempt)


# ------------------------------------------------------------ local steps
def write_tif(path, cube, ix, iy):
    tmp = path.with_suffix(".part.tif")
    profile = dict(driver="GTiff", width=TILE_PX, height=TILE_PX, count=cube.shape[0], dtype="int16",
                   crs=f"EPSG:{CRS_EPSG}", transform=tile_transform(ix, iy), nodata=NODATA,
                   compress="deflate", predictor=2, tiled=True, blockxsize=TILE_PX, blockysize=TILE_PX)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(cube)
        for i, d in enumerate(band_names(), start=1):
            dst.set_band_description(i, d)
    tmp.replace(path)


def parcel_store_ok():
    """Parcel-store tiles can be reused if they were built with the same cloud settings."""
    info = PARCEL_STORE / "tiles_info.json"
    if not (REUSE_PARCEL_TILES and info.exists()):
        return False
    i = json.loads(info.read_text())
    same = (i.get("cloud_mask") == f"Cloud Score+ cs_cdf >= {CS_THRESHOLD}; scenes with CLOUDY_PIXEL_PERCENTAGE < "
                                   f"{CLOUD_COVER_PERCENTAGE}" and i.get("crs_epsg") == CRS_EPSG
            and i.get("tile_px") == TILE_PX and i.get("start_month") == START_MONTH)
    if not same:
        print("Parcel store built with other settings: its tiles are not reused")
    return same


def copy_parcel_tile(src_path):
    """S2 bands of a parcel-store tile (12 months x 19 bands) in the band order of this store."""
    with rasterio.open(src_path) as src:
        desc = list(src.descriptions)
        idx = [desc.index(n) + 1 for n in band_names()]
        return src.read(idx).astype(np.int16)


def tile_qa(cube, crop):
    c = cube.reshape(N_MONTHS, len(MONTH_BANDS), TILE_PX, TILE_PX)
    on = crop == 1
    if not on.any():
        return {"s2_nodata_pct": None, "months_with_data": None, "mean_nclear": None}
    missing = c[:, S2_BANDS.index("B4")][:, on] == NODATA                      # months x cropland pixels
    n_clear = c[:, MONTH_BANDS.index("S2_NCLEAR")][:, on]
    return {"s2_nodata_pct": round(100 * float(missing.mean()), 2),
            "months_with_data": int(((~missing).mean(axis=1) >= 0.5).sum()),
            "mean_nclear": round(float(n_clear.mean()), 2)}


def mask_window(src, ix, iy):
    win = from_bounds(ix * TILE_M, iy * TILE_M, (ix + 1) * TILE_M, (iy + 1) * TILE_M, src.transform)
    return src.read(1, window=win, boundless=True, fill_value=65535)


def main():
    for season in SEASONS:
        tag = season_tag(season)
        out_dir = OUT_ROOT / tag
        tile_dir = out_dir / "tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)
        qa_file = out_dir / "tiles_qa.jsonl"
        mask_info = json.loads((MASK_ROOT / tag / "mask_info.json").read_text())
        (out_dir / "tiles_info.json").write_text(json.dumps({
            "region": REGION, "season": season, "crs_epsg": CRS_EPSG, "gsd_m": GSD_M, "tile_px": TILE_PX,
            "grid_origin": "x = ix * tile_m, top y = (iy + 1) * tile_m (same grid as 03_build_tiles.py and the mask)",
            "start_month": START_MONTH, "n_months": N_MONTHS, "month_bands": MONTH_BANDS, "s2_bands": S2_BANDS,
            "s2_scale": "surface reflectance x 10000", "nodata": NODATA,
            "band_order": "month-major: <band>_m01 ... <band>_m12",
            "cloud_mask": f"Cloud Score+ cs_cdf >= {CS_THRESHOLD}; scenes with CLOUDY_PIXEL_PERCENTAGE < {CLOUD_COVER_PERCENTAGE}",
            "selection": f"tiles with >= {MIN_CROP_PX} cropland pixels in the mask",
            "mask": {"products": list(mask_info["products"]), "min_votes": mask_info["min_votes"]},
        }, indent=2))

        print(f"\n== {REGION} {season}")
        tiles = select_tiles(season)
        if N_TEST_TILES:
            mid = len(tiles) // 2
            tiles = tiles.iloc[mid:mid + N_TEST_TILES].reset_index(drop=True)
        tiles["path"] = [tile_dir / f"{t}_img.tif" for t in tiles["tile_id"]]
        todo = tiles[[not p.exists() for p in tiles["path"]]]
        print(f"{len(tiles):,} tiles selected (>= {MIN_CROP_PX} cropland px), {len(tiles) - len(todo):,} done, "
              f"{len(todo):,} to go (~{len(todo) * 2.8 / 1000:,.0f} GB at ~2.8 MB per tile)")

        mask_vrt = rasterio.open(MASK_ROOT / tag / "cropland_mask.vrt")
        sources = {}

        def finish(t, cube, source):
            write_tif(t.path, cube, t.ix, t.iy)
            qa = tile_qa(cube, mask_window(mask_vrt, t.ix, t.iy))
            with open(qa_file, "a") as fh:
                fh.write(json.dumps({"tile_id": t.tile_id, "source": source, **qa}) + "\n")

        # Reuse parcel-store tiles.
        if not todo.empty and parcel_store_ok():
            n = 0
            for t in todo.itertuples():
                src = PARCEL_STORE / "tiles" / f"{t.tile_id}_img.tif"
                if src.exists():
                    finish(t, copy_parcel_tile(src), "parcel_store")
                    n += 1
            print(f"{n:,} tiles copied from the parcel store")
            todo = todo[[not p.exists() for p in todo["path"]]]

        # Download the rest.
        if not todo.empty:
            ee = init_gee()
            b = gpd.GeoSeries([box(todo["ix"].min() * TILE_M, todo["iy"].min() * TILE_M,
                                   (todo["ix"].max() + 1) * TILE_M, (todo["iy"].max() + 1) * TILE_M)],
                              crs=CRS_EPSG).to_crs(4326).total_bounds
            image = monthly_stack(ee, season, ee.Geometry.Rectangle(list(b)))
            g = REQUEST_TILES
            groups = [(k, list(grp.itertuples())) for k, grp in todo.groupby([todo["ix"] // g, todo["iy"] // g], sort=False)]
            t0, done_tiles = time.time(), 0
            step = DOWNLOAD_WORKERS * 4
            for s in range(0, len(groups), step):
                batch = groups[s:s + step]
                with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
                    results = list(pool.map(lambda kg: download_group(ee, image, kg[0][0], kg[0][1], kg[1]), batch))
                for (_, members), cubes in zip(batch, results):
                    for t in members:
                        if t.tile_id in cubes:
                            finish(t, cubes[t.tile_id], "earth_engine")
                    done_tiles += len(members)
                rate = (time.time() - t0) / done_tiles
                print(f"  {done_tiles:,}/{len(todo):,} tiles ({rate:.2f} s/tile, ~{rate * (len(todo) - done_tiles) / 3600:.1f} h left)")
        mask_vrt.close()

        # Index.
        qa = pd.read_json(qa_file, lines=True).drop_duplicates("tile_id", keep="last") if qa_file.exists() else pd.DataFrame(columns=["tile_id"])
        index = tiles.drop(columns="path").merge(qa, on="tile_id", how="left")
        index["downloaded"] = [p.exists() for p in tiles["path"]]
        index = gpd.GeoDataFrame(index, geometry=[box(x * TILE_M, y * TILE_M, (x + 1) * TILE_M, (y + 1) * TILE_M)
                                                  for x, y in zip(index["ix"], index["iy"])], crs=CRS_EPSG)
        index.to_parquet(out_dir / "tiles.parquet")
        size_gb = sum(p.stat().st_size for p in tile_dir.glob("*_img.tif")) / 1e9
        print(f"Tiles: {len(index):,} selected, {int(index['downloaded'].sum()):,} downloaded ({size_gb:.1f} GB) | "
              f"median S2 no-data on cropland {index['s2_nodata_pct'].median():.1f}% | saved in {out_dir}")


if __name__ == "__main__":
    main()
