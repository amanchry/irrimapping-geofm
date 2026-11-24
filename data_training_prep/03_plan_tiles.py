"""
03_plan_tiles.py  —  tile plan, label rasters and tile index (step 1 of the tile store; no download)
===================================================================================================

What it does
------------
Cuts the study area into a fixed grid of 128 x 128-pixel tiles (1.28 km at 10 m) in one projection per region
(regions.py: Morocco UTM 29N, Utah UTM 12N) and plans the tiles the labelled parcels need: each parcel's own tiles
plus the neighbouring "context" tiles a 64-pixel model chip around the parcel can reach. It writes the label raster
of every planned tile and the tile index. The images are not downloaded per tile: data_download/ downloads the
10.24 km blocks of the whole area (which hold all planned tiles), and 04_cut_tiles.py cuts the tiles out.

Design follows the ML-ready crop datasets: fixed non-overlapping tiles with dense, partly empty label rasters and a
parcel-ID raster (PASTIS, Garnot & Landrieu 2021; Sen4AgriNet, Sykas et al. 2022), "unlabelled" pixels kept as an
ignore class (Fields of The World, Kerner et al. 2025), and documented QA counts per tile.

Label raster per tile (int32, 3 bands)
--------------------------------------
  label        0 = unlabelled / ignore, 1 = irrigated, 2 = rainfed
  parcel_idx   0 = no parcel, otherwise the parcel's row in parcels.csv
  edge         1 = parcel pixel within INNER_BUFFER_M of the outline (mixed edge pixel), else 0
A pixel is a parcel pixel if its centre lies inside the polygon. Parcels too small to contain a pixel centre are
burnt with all touched pixels (flag `all_touched` in parcels.csv). Pixels claimed by two overlapping parcels are set
to 0 (ignore) and counted. Only parcels of the tile's own agricultural year are burnt in; the same place in another
year is another tile (tile_id starts with the season).

Steps
-----
1. Read the parcels with their folds (02_make_spatial_folds.py); project to the region's UTM zone.
2. Parcel table: index, centroid, area, perimeter, pixel counts, home tile; overlapping parcel pairs.
3. Tile plan: home tiles + tiles touched by the parcel outline + context for model chips.
4. Label rasters for all planned tiles (re-running skips finished ones).
5. Tile index (GeoParquet) with folds, roles, which tiles are cut and the QA of 04_cut_tiles.py (which refreshes
the index).

Output (outputs/<REGION>/data_training_prep/03_tiles_128px/agyear/)
-----------------------------------------------------
tiles/<tile_id>_lab.tif     int32, 3 bands (label, parcel_idx, edge)
parcels.csv                 one row per parcel (index, ids, fold, geometry stats, home tile, tiles needed)
parcel_overlaps.csv         overlapping parcel pairs (same season) with their labels (label QA)
tiles.parquet               one row per tile: geometry, season, fold, labelled pixels, role, tiles cut, QA

Run (from the project folder; no Earth Engine needed)
------------------------------------------------------
    python data_training_prep/03_plan_tiles.py
Set N_TEST_TILES = 20 for a quick check first.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box

from tile_store import (GSD_M, N_MONTHS, TILE_M, TILE_PX, WINDOW, out_dir, tile_dir, tile_file, tile_id,
                        tile_transform, write_tif)
from regions import dataset_dir, settings  # noqa: E402  (data_download/regions.py, via tile_store)

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
PARCELS_FILE = dataset_dir(REGION) / "02_folds" / "parcels_folds.geojson"   # parcels + folds (02_make_spatial_folds.py)
CHIP_PX = 64                   # largest model chip cut later (pixels)
MAX_SHIFT_PX = 8               # largest patch-alignment shift of a chip (half of a 16-px patch)
INNER_BUFFER_M = 10            # edge band: parcel pixels within this distance of the outline
N_TEST_TILES = None            # e.g. 20 home tiles per agricultural year for a quick test; None = all
# ==========================================================================

CRS_EPSG = settings(REGION)["crs_epsg"]
START_MONTH = settings(REGION)["start_month"]
OUT_DIR = out_dir(REGION)
TILE_DIR = tile_dir(REGION)
SOURCES = ["img", "dw"]                                             # per-tile files of 04_cut_tiles.py


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
    for k in SOURCES:
        out[f"has_{k}"] = [tile_file(REGION, t, k).exists() for t in out["tile_id"]]
    out["downloaded"] = out["has_img"]                              # tile cut from the blocks (04_cut_tiles.py)
    return out


def main():
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    qa_file = OUT_DIR / "tiles_qa.jsonl"
    info_file = OUT_DIR / "tiles_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    info.update({"region": REGION, "crs_epsg": CRS_EPSG, "gsd_m": GSD_M, "tile_px": TILE_PX,
                 "grid_origin": "x = ix * tile_m, top y = (iy + 1) * tile_m", "start_month": START_MONTH,
                 "n_months": N_MONTHS, "window": WINDOW,
                 "label_bands": ["label (0 ignore, 1 irrigated, 2 rainfed)", "parcel_idx (parcels.csv)", "edge"],
                 "inner_buffer_m": INNER_BUFFER_M, "chip_px": CHIP_PX, "max_shift_px": MAX_SHIFT_PX})
    info_file.write_text(json.dumps(info, indent=2))

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
        write_tif(path, lab, t.ix, t.iy, ["label", "parcel_idx", "edge"], CRS_EPSG)
    print(f"Label rasters written ({n_overlap} pixels claimed by overlapping parcels set to ignore)")

    # 5. Index.
    index = tile_index(tiles, parcels, qa_file)
    index.to_parquet(OUT_DIR / "tiles.parquet")
    have = ", ".join(f"{k} {int(index[f'has_{k}'].sum()):,}" for k in SOURCES)
    print(f"\nTiles: {len(index):,} | downloaded: {have} | roles {index['role'].value_counts().to_dict()} | "
          f"tiles with parcels from two folds: {int(index['mixed_folds'].sum())}")
    print(f"Saved in {OUT_DIR}. Next: data_download/01_plan_blocks.py and 02a-02d, then data_training_prep/04_cut_tiles.py")


if __name__ == "__main__":
    main()
