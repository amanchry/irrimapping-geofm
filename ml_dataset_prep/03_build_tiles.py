"""
03_build_tiles.py  —  one tile store of monthly Sentinel-1/2 images + label rasters (stage 1 of 2)
==================================================================================================

What it does
------------
Builds the single, non-overlapping image store that every model reads from. The study area is cut
into a fixed grid of 128 x 128-pixel tiles (1.28 km at 10 m) in one projection (UTM 29N). Only
tiles needed by the labelled parcels are built: each parcel's own tiles plus the neighbouring
"context" tiles a 64-pixel model chip around the parcel can reach. Each tile is downloaded ONCE,
so neighbouring parcels no longer get overlapping, duplicated chips.

Every derived input (parcel features for the random forest, pixel sets, patch-aligned chips for
the foundation models, full tiles for segmentation models) is cut from this store locally by
04_make_views.py, so all models see exactly the same pixels.

Design follows the ML-ready crop datasets: fixed non-overlapping tiles with dense, partly empty
label rasters and a parcel-ID raster (PASTIS, Garnot & Landrieu 2021; Sen4AgriNet, Sykas et al.
2022), "unlabelled" pixels kept as an ignore class (Fields of The World, Kerner et al. 2025),
and documented QA counts per tile (ClarkCGA / Prithvi crop chips).

Per tile and month (12 months of the parcel's agricultural year, p01 = September)
-------------------------------------------------------------------------------
  B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12   Sentinel-2 L2A median of clear pixels (reflectance x 10000)
  S2_NCLEAR                                number of clear Sentinel-2 observations (Cloud Score+)
  VV_ASC VH_ASC S1_N_ASC                   Sentinel-1 ascending: mean in LINEAR power -> dB x 100, scene count
  VV_DSC VH_DSC S1_N_DSC                   Sentinel-1 descending: same
NODATA = -32768 where a month has no observation (count bands are 0 there).
Ascending and descending orbits are kept apart (different incidence angles); the views script
chooses which to use.

Label raster per tile (int32, 3 bands)
--------------------------------------
  label        0 = unlabelled / ignore, 1 = irrigated, 2 = rainfed
  parcel_idx   0 = no parcel, otherwise the parcel's row in parcels.csv
  edge         1 = parcel pixel within INNER_BUFFER_M of the outline (mixed edge pixel), else 0
A pixel is a parcel pixel if its centre lies inside the polygon. Parcels too small to contain a
pixel centre are burnt with all touched pixels (flag `all_touched` in parcels.csv). Pixels claimed
by two overlapping parcels are set to 0 (ignore) and counted.
Only parcels of the tile's own agricultural year are burnt in; the same place in another year is
another tile (tile_id starts with the season).

Steps
-----
1. Read the parcels with their folds (02_make_spatial_folds.py); project to UTM 29N.
2. Parcel table: index, centroid, area, perimeter, pixel counts, home tile.
3. Tile plan: home tiles + tiles touched by the parcel outline + context for model chips.
4. Label rasters for all planned tiles (local, fast).
5. Earth Engine: one monthly image per agricultural year; each tile downloaded with
   computePixels (in parallel), written as a compressed GeoTIFF together with the IDs of the
   Sentinel-1/2 scenes used per month.
6. Tile index (GeoParquet) with folds and QA numbers.
Re-running skips finished tiles (resumable). DOWNLOAD = False stops after step 4 (no Earth Engine).

Output (outputs/dataset/03_tiles/<WINDOW>/)
-----------------------------------
tiles/<tile_id>_img.tif     int16, 12 months x 19 bands = 228 bands (descriptions "B4_m03", ...)
tiles/<tile_id>_lab.tif     int32, 3 bands (label, parcel_idx, edge)
tiles/<tile_id>_scenes.json scene IDs per month (S2, S1 ascending, S1 descending)
parcels.csv                 one row per parcel (index, ids, fold, geometry stats, home tile, tiles needed)
parcel_overlaps.csv         overlapping parcel pairs (same season) with their labels (label QA)
tiles.parquet               one row per tile: geometry, season, fold, labelled pixels, QA
tiles_qa.jsonl              QA lines written as tiles finish (merged into tiles.parquet)
tiles_info.json             grid, bands, months, NODATA, cloud settings

Run (from the project folder)
-----------------------------
    python ml_dataset_prep/03_build_tiles.py
Set N_TEST_TILES = 20 for a quick check first.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                         # project folder
KEY_FILE = ROOT / "secrets" / "waterinfor-service-account.json"      # Earth Engine key (never commit)
PARCELS_FILE = ROOT / "outputs" / "dataset" / "02_folds" / "parcels_folds.geojson"   # parcels + folds (ml_dataset_prep/02_make_spatial_folds.py)

START_MONTH, N_MONTHS, WINDOW = 9, 12, "agyear"    # agricultural year 1 Sep - 31 Aug
OUT_DIR = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW

CRS_EPSG = 32629               # one projection for the whole study area (UTM 29N; 30N parcels are resampled)
GSD_M = 10                     # pixel size (m)
TILE_PX = 128                  # tile width/height in pixels (1.28 km)
CHIP_PX = 64                   # largest model chip cut later by 04_make_views.py (pixels)
MAX_SHIFT_PX = 8               # largest patch-alignment shift used by 04_make_views.py (half of a 16-px patch)
INNER_BUFFER_M = 10            # edge band: parcel pixels within this distance of the outline

CLOUD_COVER_PERCENTAGE = 60    # skip Sentinel-2 scenes cloudier than this (Cloud Score+ masks the rest)
CS_THRESHOLD = 0.60            # Cloud Score+ cs_cdf: keep pixels >= this (0.50-0.65 recommended)

DOWNLOAD = True                # False = plan + label rasters only (no Earth Engine)
DOWNLOAD_WORKERS = 8           # parallel Earth Engine requests
RETRIES = 4
N_TEST_TILES = None            # e.g. 20 tiles per agricultural year for a quick test; None = all
# ==========================================================================

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
MONTH_BANDS = S2_BANDS + ["S2_NCLEAR", "VV_ASC", "VH_ASC", "S1_N_ASC", "VV_DSC", "VH_DSC", "S1_N_DSC"]
COUNT_BANDS = ["S2_NCLEAR", "S1_N_ASC", "S1_N_DSC"]
S1_SCALE = 100                 # dB stored as integer dB x 100
NODATA = -32768
TILE_M = TILE_PX * GSD_M
TILE_DIR = OUT_DIR / "tiles"


def band_names():
    return [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in MONTH_BANDS]


def season_tag(ag_season):
    return ag_season.replace("/", "-")                          # "2023/24" -> "2023-24"


def tile_id(ag_season, ix, iy):
    return f"{season_tag(ag_season)}_{ix}_{iy}"


def tile_transform(ix, iy):
    """Top-left corner of tile (ix, iy) on the fixed grid; ix, iy = floor(x / TILE_M), floor(y / TILE_M)."""
    return Affine(GSD_M, 0, ix * TILE_M, 0, -GSD_M, (iy + 1) * TILE_M)


def month_starts(ag_season):
    start = pd.Timestamp(year=int(ag_season[:4]), month=START_MONTH, day=1)
    return [start + pd.DateOffset(months=k) for k in range(N_MONTHS)]


# ------------------------------------------------------------ steps 1-2: parcels
def pixel_count(geom, all_touched=False):
    """Pixels of the fixed 10 m grid whose centre lies inside geom (or that it touches)."""
    minx, miny, maxx, maxy = geom.bounds
    c0, r0 = int(np.floor(minx / GSD_M)), int(np.ceil(maxy / GSD_M))
    w = max(1, int(np.ceil(maxx / GSD_M)) - c0)
    h = max(1, r0 - int(np.floor(miny / GSD_M)))
    m = rasterize([geom], out_shape=(h, w), transform=Affine(GSD_M, 0, c0 * GSD_M, 0, -GSD_M, r0 * GSD_M),
                  fill=0, dtype="uint8", all_touched=all_touched)
    return int(m.sum()), h * w


def prepare_parcels():
    parcels = gpd.read_file(PARCELS_FILE)
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    lonlat = parcels.geometry.representative_point()
    parcels["lon"], parcels["lat"] = lonlat.x, lonlat.y
    parcels = parcels.to_crs(CRS_EPSG).reset_index(drop=True)
    parcels["parcel_idx"] = np.arange(1, len(parcels) + 1)          # 0 is reserved for "no parcel"
    cent = parcels.geometry.centroid
    parcels["cx"], parcels["cy"] = cent.x, cent.y
    b = parcels.geometry.bounds
    parcels["minx"], parcels["miny"], parcels["maxx"], parcels["maxy"] = b.minx, b.miny, b.maxx, b.maxy
    parcels["area_ha"] = parcels.geometry.area / 10_000
    parcels["perimeter_m"] = parcels.geometry.length
    counts = [pixel_count(g) for g in parcels.geometry]
    parcels["n_px"] = [c for c, _ in counts]
    parcels["cover_ratio"] = [c / b for c, b in counts]             # parcel pixels / bounding-box pixels
    parcels["all_touched"] = parcels["n_px"] == 0                   # too small for a pixel centre
    for i in np.flatnonzero(parcels["all_touched"]):
        parcels.loc[i, "n_px"] = pixel_count(parcels.geometry[i], all_touched=True)[0]
    inner = parcels.geometry.buffer(-INNER_BUFFER_M)
    parcels["n_px_inner"] = [0 if g.is_empty else pixel_count(g)[0] for g in inner]
    parcels["home_ix"] = np.floor(parcels["cx"] / TILE_M).astype(int)
    parcels["home_iy"] = np.floor(parcels["cy"] / TILE_M).astype(int)
    parcels["home_tile"] = [tile_id(s, x, y) for s, x, y in zip(parcels["ag_season"], parcels["home_ix"], parcels["home_iy"])]
    parcels["tiles_needed"] = [";".join(tile_id(r.ag_season, x, y) for x, y in parcel_tiles(r)) for r in parcels.itertuples()]
    return parcels


def overlapping_parcels(parcels):
    """Pairs of parcels of the same season that overlap (> 1 m2): share of the smaller one, labels."""
    rows = []
    left, right = parcels.sindex.query(parcels.geometry, predicate="intersects")
    for i, j in zip(left, right):
        if i >= j or parcels.at[i, "ag_season"] != parcels.at[j, "ag_season"]:
            continue
        inter = parcels.geometry[i].intersection(parcels.geometry[j]).area
        if inter > 1:
            small = min(parcels.geometry[i].area, parcels.geometry[j].area)
            rows.append({"parcel_id_a": parcels.at[i, "parcel_id"], "parcel_id_b": parcels.at[j, "parcel_id"],
                         "ag_season": parcels.at[i, "ag_season"], "label_a": parcels.at[i, "label"],
                         "label_b": parcels.at[j, "label"], "overlap_m2": round(inter, 1),
                         "share_of_smaller": round(inter / small, 3)})
    return pd.DataFrame(rows, columns=["parcel_id_a", "parcel_id_b", "ag_season", "label_a", "label_b",
                                       "overlap_m2", "share_of_smaller"])


# ------------------------------------------------------------ step 3: tile plan
def parcel_tiles(row):
    """Tiles one parcel needs: its outline plus a (CHIP_PX/2 + MAX_SHIFT_PX) pixel margin around its centroid."""
    reach = (CHIP_PX // 2 + MAX_SHIFT_PX) * GSD_M
    minx, miny = min(row.minx, row.cx - reach), min(row.miny, row.cy - reach)
    maxx, maxy = max(row.maxx, row.cx + reach), max(row.maxy, row.cy + reach)
    return [(ix, iy) for ix in range(int(np.floor(minx / TILE_M)), int(np.floor((maxx - 1e-6) / TILE_M)) + 1)
            for iy in range(int(np.floor(miny / TILE_M)), int(np.floor((maxy - 1e-6) / TILE_M)) + 1)]


def plan_tiles(parcels):
    """All tiles needed by the parcels (see parcel_tiles)."""
    need = {(row.ag_season, ix, iy) for row in parcels.itertuples() for ix, iy in parcel_tiles(row)}
    tiles = pd.DataFrame(sorted(need), columns=["ag_season", "ix", "iy"])
    tiles["tile_id"] = [tile_id(s, x, y) for s, x, y in zip(tiles["ag_season"], tiles["ix"], tiles["iy"])]
    tiles["x0"], tiles["y0"] = tiles["ix"] * TILE_M, (tiles["iy"] + 1) * TILE_M   # top-left corner
    geom = [box(x, y - TILE_M, x + TILE_M, y) for x, y in zip(tiles["x0"], tiles["y0"])]
    return gpd.GeoDataFrame(tiles, geometry=geom, crs=CRS_EPSG)


# ------------------------------------------------------------ step 4: label rasters
def label_raster(tile, parcels):
    """label, parcel_idx, edge rasters of one tile -> int32 [3, TILE_PX, TILE_PX] and the overlap pixel count."""
    transform = tile_transform(tile.ix, tile.iy)
    shape = (TILE_PX, TILE_PX)
    sub = parcels.iloc[parcels.sindex.query(tile.geometry, predicate="intersects")]
    sub = sub[sub["ag_season"] == tile.ag_season]
    out = np.zeros((3, TILE_PX, TILE_PX), dtype=np.int32)
    if sub.empty:
        return out, 0
    groups = [sub[~sub["all_touched"]], sub[sub["all_touched"]]]
    hits = np.zeros(shape, dtype=np.int32)
    for grp, touched in zip(groups, (False, True)):
        if grp.empty:
            continue
        geoms = list(grp.geometry)
        hits += rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0,
                          dtype="int32", all_touched=touched, merge_alg=MergeAlg.add)
        idx = rasterize(list(zip(geoms, grp["parcel_idx"])), out_shape=shape, transform=transform,
                        fill=0, dtype="int32", all_touched=touched)
        lab = rasterize(list(zip(geoms, grp["label"])), out_shape=shape, transform=transform,
                        fill=0, dtype="int32", all_touched=touched)
        new = idx > 0
        out[1][new], out[0][new] = idx[new], lab[new]
    inner_geoms = [g for g in sub.geometry.buffer(-INNER_BUFFER_M) if not g.is_empty]
    inner = rasterize([(g, 1) for g in inner_geoms], out_shape=shape, transform=transform, fill=0,
                      dtype="int32") if inner_geoms else np.zeros(shape, dtype=np.int32)
    overlap = hits > 1
    out[0][overlap] = 0
    out[1][overlap] = 0
    out[2] = ((out[1] > 0) & (inner == 0)).astype(np.int32)
    return out, int(overlap.sum())


def write_tif(path, array, ix, iy, descriptions, nodata=None):
    """Write [bands, H, W] as a compressed GeoTIFF on the fixed grid (written to a temp name, then renamed)."""
    tmp = path.with_name(path.name + ".part")
    profile = dict(driver="GTiff", width=TILE_PX, height=TILE_PX, count=array.shape[0], dtype=array.dtype,
                   crs=f"EPSG:{CRS_EPSG}", transform=tile_transform(ix, iy), nodata=nodata,
                   compress="deflate", predictor=2, tiled=True, blockxsize=TILE_PX, blockysize=TILE_PX)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(array)
        for i, d in enumerate(descriptions, start=1):
            dst.set_band_description(i, d)
    tmp.replace(path)


# ------------------------------------------------------------ step 5: Earth Engine
def init_gee():
    import ee
    key = json.loads(KEY_FILE.read_text())
    credentials = ee.ServiceAccountCredentials(key["client_email"], key_data=json.dumps(key))
    ee.Initialize(credentials, project=key["project_id"])
    print("Earth Engine initialised")
    return ee


def season_collections(ee, ag_season, region):
    """Sentinel-2 (cloud-masked) and Sentinel-1 (linear power, per orbit) collections of one agricultural year."""
    months = month_starts(ag_season)
    first, last = str(months[0].date()), str((months[-1] + pd.DateOffset(months=1)).date())
    s2_raw = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region).filterDate(first, last)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_PERCENTAGE)))
    cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (s2_raw.linkCollection(cloud_score, ["cs_cdf"])
          .map(lambda img: img.select(S2_BANDS).updateMask(img.select("cs_cdf").gte(CS_THRESHOLD))))
    s1_raw = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(region).filterDate(first, last)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

    def to_linear(img):                                              # GEE S1_GRD is in dB
        linear = ee.Image(10).pow(img.select(["VV", "VH"]).divide(10)).toFloat()
        return ee.Image(linear.copyProperties(img, ["system:time_start"]))

    s1 = {o: s1_raw.filter(ee.Filter.eq("orbitProperties_pass", o)).map(to_linear) for o in ("ASCENDING", "DESCENDING")}
    s1_ids = {o: s1_raw.filter(ee.Filter.eq("orbitProperties_pass", o)) for o in ("ASCENDING", "DESCENDING")}
    return months, s2, s2_raw, s1, s1_ids


def monthly_stack(ee, ag_season, region):
    """One int16 image: N_MONTHS x 19 bands named like band_names()."""
    months, s2, _, s1, _ = season_collections(ee, ag_season, region)
    off = ee.Image.constant(0)
    empty_s2 = ee.ImageCollection([ee.Image.constant([0] * len(S2_BANDS)).rename(S2_BANDS).toUint16().updateMask(off)])
    empty_s1 = ee.ImageCollection([ee.Image.constant([0, 0]).rename(["VV", "VH"]).toFloat().updateMask(off)])
    images = []
    for k, m in enumerate(months, start=1):
        a, b = str(m.date()), str((m + pd.DateOffset(months=1)).date())
        s2m = s2.filterDate(a, b).merge(empty_s2)                     # dummy keeps empty months valid
        optical = s2m.median().select(S2_BANDS).min(32767).unmask(NODATA)
        n_clear = s2m.select("B4").count().unmask(0).rename("S2_NCLEAR")
        bands = [optical, n_clear]
        for orbit, tag in (("ASCENDING", "ASC"), ("DESCENDING", "DSC")):
            col = s1[orbit].filterDate(a, b).merge(empty_s1)
            db = col.mean().log10().multiply(10 * S1_SCALE).unmask(NODATA).rename([f"VV_{tag}", f"VH_{tag}"])
            n = col.select("VV").count().unmask(0).rename(f"S1_N_{tag}")
            bands += [db, n]
        names = [f"{b}_m{k:02d}" for b in MONTH_BANDS]
        images.append(ee.Image.cat(bands).select(MONTH_BANDS, names))
    return ee.Image.cat(images).toInt16()


def scene_ids_request(ee, ag_season, region, tile_geom):
    """Earth Engine list of [S2, S1 asc, S1 desc] scene IDs per month for one tile."""
    months, _, s2_raw, _, s1_ids = season_collections(ee, ag_season, region)
    out = []
    for m in months:
        a, b = str(m.date()), str((m + pd.DateOffset(months=1)).date())
        out.append([c.filterBounds(tile_geom).filterDate(a, b).aggregate_array("system:index")
                    for c in (s2_raw, s1_ids["ASCENDING"], s1_ids["DESCENDING"])])
    return ee.List(out)


def download_tile(ee, image, ag_season, region, tile):
    """All months x bands of one tile -> int16 [228, TILE_PX, TILE_PX] + scene IDs, or None after RETRIES."""
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": TILE_PX, "height": TILE_PX},
                 "affineTransform": {"scaleX": GSD_M, "shearX": 0, "translateX": float(tile.x0),
                                     "shearY": 0, "scaleY": -GSD_M, "translateY": float(tile.y0)},
                 "crsCode": f"EPSG:{CRS_EPSG}"},
    }
    tile_geom = ee.Geometry.Rectangle([float(tile.x0), float(tile.y0 - TILE_M), float(tile.x0 + TILE_M), float(tile.y0)],
                                      f"EPSG:{CRS_EPSG}", False)
    for attempt in range(1, RETRIES + 1):
        try:
            arr = ee.data.computePixels(request)
            cube = np.stack([arr[n] for n in band_names()]).astype(np.int16)
            scenes = scene_ids_request(ee, ag_season, region, tile_geom).getInfo()
            return cube, scenes
        except Exception as err:                                      # network / quota / EE error
            if attempt == RETRIES:
                print(f"  tile {tile.tile_id}: failed ({str(err)[:120]})")
                return None
            time.sleep(10 * attempt)


def tile_qa(cube, lab):
    """QA numbers of one tile: share of NODATA, clear Sentinel-2 observations on labelled pixels."""
    c = cube.reshape(N_MONTHS, len(MONTH_BANDS), TILE_PX, TILE_PX)
    s2_missing = (c[:, S2_BANDS.index("B4")] == NODATA)
    n_clear = c[:, MONTH_BANDS.index("S2_NCLEAR")]
    on = lab[1] > 0
    qa = {"s2_nodata_pct": round(100 * float(s2_missing.mean()), 2),
          "s1_asc_months": int((c[:, MONTH_BANDS.index("S1_N_ASC")].max(axis=(1, 2)) > 0).sum()),
          "s1_dsc_months": int((c[:, MONTH_BANDS.index("S1_N_DSC")].max(axis=(1, 2)) > 0).sum())}
    if on.any():
        qa["s2_nodata_parcel_pct"] = round(100 * float(s2_missing[:, on].mean()), 2)
        qa["s2_nclear_parcel_median"] = [float(np.median(n_clear[k][on])) for k in range(N_MONTHS)]
    return qa


# ------------------------------------------------------------ step 6: index
def tile_index(tiles, parcels, qa_file):
    """Fold of every tile = fold holding most of its labelled pixels; mixed tiles are flagged."""
    fold_of = parcels.set_index("parcel_idx")["fold"]
    rows = []
    for t in tiles.itertuples():
        f = TILE_DIR / f"{t.tile_id}_lab.tif"
        row = {"tile_id": t.tile_id, "n_irrigated_px": 0, "n_rainfed_px": 0, "n_parcels_home": 0,
               "fold": -1, "folds_present": "", "mixed_folds": False}
        if f.exists():
            with rasterio.open(f) as src:
                lab = src.read()
            row["n_irrigated_px"] = int((lab[0] == 1).sum())
            row["n_rainfed_px"] = int((lab[0] == 2).sum())
            ids, counts = np.unique(lab[1][lab[1] > 0], return_counts=True)
            if len(ids):
                folds = pd.Series(counts, index=fold_of.loc[ids].to_numpy()).groupby(level=0).sum()
                row["fold"] = int(folds.idxmax())
                row["folds_present"] = ",".join(str(int(x)) for x in folds.index)
                row["mixed_folds"] = len(folds) > 1
        rows.append(row)
    out = tiles.merge(pd.DataFrame(rows), on="tile_id")
    out["n_parcels_home"] = out["tile_id"].map(parcels["home_tile"].value_counts()).fillna(0).astype(int)
    out["role"] = np.where(out["n_parcels_home"] > 0, "home", np.where(out["n_irrigated_px"] + out["n_rainfed_px"] > 0,
                                                                       "outline", "context"))
    if qa_file.exists():
        qa = pd.read_json(qa_file, lines=True).drop_duplicates("tile_id", keep="last")
        out = out.merge(qa, on="tile_id", how="left")
    out["downloaded"] = [(TILE_DIR / f"{t}_img.tif").exists() for t in out["tile_id"]]
    return out


def main():
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    qa_file = OUT_DIR / "tiles_qa.jsonl"
    (OUT_DIR / "tiles_info.json").write_text(json.dumps({
        "crs_epsg": CRS_EPSG, "gsd_m": GSD_M, "tile_px": TILE_PX, "grid_origin": "x = ix * tile_m, top y = (iy + 1) * tile_m",
        "start_month": START_MONTH, "n_months": N_MONTHS, "window": WINDOW, "month_bands": MONTH_BANDS,
        "s2_bands": S2_BANDS, "count_bands": COUNT_BANDS, "s2_scale": "surface reflectance x 10000",
        "s1": "Sentinel-1 GRD IW, mean in linear power per orbit direction, stored as dB x 100",
        "s1_scale": S1_SCALE, "nodata": NODATA, "band_order": "month-major: <band>_m01 ... <band>_m12",
        "label_bands": ["label (0 ignore, 1 irrigated, 2 rainfed)", "parcel_idx (parcels.csv)", "edge"],
        "inner_buffer_m": INNER_BUFFER_M, "chip_px": CHIP_PX, "max_shift_px": MAX_SHIFT_PX,
        "cloud_mask": f"Cloud Score+ cs_cdf >= {CS_THRESHOLD}; scenes with CLOUDY_PIXEL_PERCENTAGE < {CLOUD_COVER_PERCENTAGE}",
    }, indent=2))

    # 1-2. Parcels.
    parcels = prepare_parcels()
    cols = ["parcel_idx", "parcel_id", "ag_season", "label", "class_name", "zone", "fold", "random_fold", "block_id",
            "lon", "lat", "cx", "cy", "minx", "miny", "maxx", "maxy", "area_ha", "perimeter_m", "n_px", "n_px_inner", "cover_ratio", "all_touched",
            "home_tile", "tiles_needed"]
    parcels[[c for c in cols if c in parcels.columns]].to_csv(OUT_DIR / "parcels.csv", index=False)
    print(f"{len(parcels):,} parcels | {int(parcels['all_touched'].sum())} smaller than one pixel centre "
          f"(burnt with all touched pixels) | {int((parcels['n_px_inner'] == 0).sum())} with no pixel after the "
          f"{INNER_BUFFER_M} m inner buffer")
    overlaps = overlapping_parcels(parcels)
    overlaps.to_csv(OUT_DIR / "parcel_overlaps.csv", index=False)
    conflict = overlaps["label_a"] != overlaps["label_b"]
    print(f"Overlapping parcel pairs: {len(overlaps)} ({int(conflict.sum())} with different labels; "
          f"{int((overlaps['share_of_smaller'] > 0.5).sum())} cover > 50% of the smaller parcel) -> parcel_overlaps.csv; "
          f"their shared pixels are set to ignore")

    # 3. Tile plan.
    tiles = plan_tiles(parcels)
    homes = set(parcels["home_tile"])
    if N_TEST_TILES:
        keep = []
        for s, grp in tiles.groupby("ag_season"):
            first = [t for t in grp["tile_id"] if t in homes][:N_TEST_TILES]
            sub = parcels[parcels["home_tile"].isin(first)]
            keep += list(plan_tiles(sub)["tile_id"])
        tiles = tiles[tiles["tile_id"].isin(keep)].reset_index(drop=True)
    print(f"Tile plan: {len(tiles):,} tiles of {TILE_PX} px ({tiles['tile_id'].isin(homes).sum():,} home tiles) | "
          + ", ".join(f"{s}: {n}" for s, n in tiles["ag_season"].value_counts().sort_index().items()))

    # 4. Label rasters.
    n_overlap = 0
    for t in tiles.itertuples():
        path = TILE_DIR / f"{t.tile_id}_lab.tif"
        if path.exists():
            continue
        lab, ov = label_raster(t, parcels)
        n_overlap += ov
        write_tif(path, lab, t.ix, t.iy, ["label", "parcel_idx", "edge"])
    print(f"Label rasters written ({n_overlap} pixels claimed by overlapping parcels set to ignore)")

    # 5. Images.
    if DOWNLOAD:
        ee = init_gee()
        names = band_names()
        for ag_season, group in tiles.groupby("ag_season"):
            todo = group[[not (TILE_DIR / f"{t}_img.tif").exists() for t in group["tile_id"]]]
            print(f"{ag_season}: {len(group) - len(todo)} tiles done, {len(todo)} to go")
            if todo.empty:
                continue
            bounds = gpd.GeoSeries(group.geometry, crs=CRS_EPSG).to_crs(4326).total_bounds
            region = ee.Geometry.Rectangle(list(bounds)).buffer(2000)
            image = monthly_stack(ee, ag_season, region)
            rows = list(todo.itertuples())
            t_start = time.time()
            for start in range(0, len(rows), DOWNLOAD_WORKERS * 4):
                batch = rows[start:start + DOWNLOAD_WORKERS * 4]
                with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
                    results = list(pool.map(lambda t: download_tile(ee, image, ag_season, region, t), batch))
                for t, res in zip(batch, results):
                    if res is None:
                        continue
                    cube, scenes = res
                    write_tif(TILE_DIR / f"{t.tile_id}_img.tif", cube, t.ix, t.iy, names, nodata=NODATA)
                    (TILE_DIR / f"{t.tile_id}_scenes.json").write_text(json.dumps(
                        {f"m{k + 1:02d}": {"s2": s[0], "s1_asc": s[1], "s1_dsc": s[2]} for k, s in enumerate(scenes)}))
                    with rasterio.open(TILE_DIR / f"{t.tile_id}_lab.tif") as src:
                        qa = tile_qa(cube, src.read())
                    with open(qa_file, "a") as fh:
                        fh.write(json.dumps({"tile_id": t.tile_id, **qa}) + "\n")
                done = min(start + len(batch), len(rows))
                rate = (time.time() - t_start) / done
                print(f"  {ag_season}: {done}/{len(rows)} tiles ({rate:.1f} s/tile, ~{rate * (len(rows) - done) / 60:.0f} min left)")

    # 6. Index.
    index = tile_index(tiles, parcels, qa_file)
    index.to_parquet(OUT_DIR / "tiles.parquet")
    print(f"\nTiles: {len(index):,} ({index['downloaded'].sum():,} downloaded) | roles "
          f"{index['role'].value_counts().to_dict()} | tiles with parcels from two folds: {int(index['mixed_folds'].sum())}")
    print(f"Saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
