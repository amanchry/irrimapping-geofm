"""
04_make_pixel_samples.py  —  labelled PIXELS for pixel-wise classification 
===================================================================================================

What it does
------------
Pixel-wise models classify every 10 m pixel on its own, so they are trained and tested on pixels instead of
parcels. This script draws the labelled pixels from the parcel tile store (03_build_tiles.py) and writes one
pixel table per agricultural year: the 12-month time series of every sampled pixel plus its label, parcel,
fold and position. Every pixel-wise experiment (Sentinel-2 time series, embeddings sampled at the same
positions, ...) starts from this table, so all models see the same pixels.

Classes (3-class design: the map later covers the union cropland mask, which contains non-cropland)
---------------------------------------------------------------------------------------------------
  1 irrigated      pixels of irrigated parcels (label raster of 03)
  2 rainfed        pixels of rainfed parcels
  3 not cropland   pixels in the same tiles that are outside every labelled parcel, that NONE of the cropland
                   products calls cropland (votes = 0 in the mask of cropland_extent/make_cropland_mask.py), that are not
                   built-up / water / snow (those are never mapped) and that lie >= NONCROP_BUFFER_PX pixels from
                   any parcel pixel
Unlabelled cropland pixels (cropland in the mask but in no surveyed parcel) are not used: their class is unknown.

Sampling
--------
Parcel pixels: inner pixels only (edge = 0 in the label raster: not within INNER_BUFFER_M of the outline, so
no mixed boundary pixels); parcels with no inner pixel fall back to all their pixels (edge_fallback = True).
At most MAX_PX_PER_PARCEL pixels per parcel, drawn at random (SEED), so large parcels do not dominate and the
pixels of one parcel stay a small, fixed group.
Not-cropland pixels: at most NONCROP_PX_PER_TILE per tile at random, from every downloaded tile of the year.

Folds (the spatial folds of 02_make_spatial_folds.py)
-----------------------------------------------------
Parcel pixels take their parcel's fold (all pixels of a parcel are in the same fold). Not-cropland pixels take
the fold of the 20 km block they fall in (same block grid as 02); pixels in a block without any labelled
parcel get fold -1 (kept, but not used in cross-validation).

Outputs (outputs/<region>/dataset/04c_pixel_samples/<season>/)
--------------------------------------------------------------
pixels_X.npy        int16 [N, 12 months, 19 bands]: the bands of the tile store (12 Sentinel-2 bands,
                    S2_NCLEAR, VV/VH and scene counts per Sentinel-1 orbit), NODATA = -32768, same scaling
pixels.parquet      one row per pixel (same order as pixels_X): label, class_name, parcel_idx, parcel_id, fold,
                    block_id, zone, tile_id, row, col, x, y (UTM centre of the pixel), lon, lat, edge,
                    edge_fallback, mask_cropland, mask_votes
summary.csv         pixels and parcels per class and fold
samples_info.json   settings, band order, class codes, CRS
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation

from regions import cropland_dir, dataset_dir, settings

# ================================ SETTINGS ================================
REGION = "morocco"                         # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2020/21", "2023/24"]}[REGION]   # label years (each needs its tiles (03) and cropland mask)
MAX_PX_PER_PARCEL = 100                    # random inner pixels per parcel (None = all)
NONCROP_PX_PER_TILE = 100                  # random not-cropland pixels per tile (0 = 2-class samples only)
NONCROP_BUFFER_PX = 2                      # not-cropland pixels at least this many pixels from any parcel pixel
SEED = 42
TILE_ROOT = dataset_dir(REGION) / "03_tiles_128px" / "agyear"          # parcel tile store (03_build_tiles.py)
MASK_ROOT = cropland_dir(REGION) / "cropland_mask"                     # cropland mask (cropland_extent/make_cropland_mask.py)
OUT_ROOT = dataset_dir(REGION) / "04c_pixel_samples"
# ==========================================================================

CLASS_NAMES = {1: "Irrigated", 2: "Rainfed", 3: "Not cropland"}
CRS_EPSG = settings(REGION)["crs_epsg"]
BLOCK_M = settings(REGION)["block_km"] * 1000
INFO = json.loads((TILE_ROOT / "tiles_info.json").read_text())
TILE_PX, GSD, NODATA = INFO["tile_px"], INFO["gsd_m"], INFO["nodata"]
N_MONTHS, MB = INFO["n_months"], INFO["month_bands"]
TILE_M = TILE_PX * GSD
MASK_NODATA = 65535


def season_tag(season):
    return season.replace("/", "-")


def block_id(x, y):
    """Name of the BLOCK_M square of 02_make_spatial_folds.py, e.g. '23_167'."""
    return f"{int(np.floor(x / BLOCK_M))}_{int(np.floor(y / BLOCK_M))}"


def read_mask(vrt, ix, iy):
    """cropland, votes, excluded of one tile from the mask VRT (NODATA outside the mapping area)."""
    win = from_bounds(ix * TILE_M, iy * TILE_M, (ix + 1) * TILE_M, (iy + 1) * TILE_M, vrt.transform)
    m = vrt.read([1, 2, 4], window=win, boundless=True, fill_value=MASK_NODATA)
    return m[0], m[1], m[2]


def candidates(tiles, vrt, rng):
    """Pass 1 (label rasters + mask only): parcel pixel candidates and the not-cropland sample per tile."""
    parcel_rows, noncrop_rows, n_noncrop_tiles = [], [], 0
    for k, t in enumerate(tiles.itertuples()):
        with rasterio.open(TILE_ROOT / "tiles" / f"{t.tile_id}_lab.tif") as src:
            label, pidx, edge = src.read()
        crop, votes, excl = read_mask(vrt, t.ix, t.iy)
        r, c = np.nonzero((label > 0) & (pidx > 0))
        if r.size:
            parcel_rows.append(np.column_stack([np.full(r.size, k), r, c, pidx[r, c], label[r, c], edge[r, c],
                                                crop[r, c], votes[r, c]]))
        if NONCROP_PX_PER_TILE:
            near = binary_dilation(pidx > 0, iterations=NONCROP_BUFFER_PX) if NONCROP_BUFFER_PX else pidx > 0
            ok = (pidx == 0) & (label == 0) & ~near & (crop != MASK_NODATA) & (votes == 0) & (excl == 0)
            r, c = np.nonzero(ok)
            if r.size:
                n_noncrop_tiles += 1
                pick = rng.choice(r.size, min(NONCROP_PX_PER_TILE, r.size), replace=False)
                r, c = r[pick], c[pick]
                noncrop_rows.append(np.column_stack([np.full(r.size, k), r, c, np.zeros(r.size, int), np.full(r.size, 3),
                                                     np.zeros(r.size, int), crop[r, c], votes[r, c]]))
        if (k + 1) % 500 == 0:
            print(f"  pass 1: {k + 1:,}/{len(tiles):,} tiles")
    cols = ["tile_k", "row", "col", "parcel_idx", "label", "edge", "mask_cropland", "mask_votes"]
    par = pd.DataFrame(np.vstack(parcel_rows), columns=cols) if parcel_rows else pd.DataFrame(columns=cols)
    non = pd.DataFrame(np.vstack(noncrop_rows), columns=cols) if noncrop_rows else pd.DataFrame(columns=cols)
    return par, non, n_noncrop_tiles


def sample_parcels(par, rng):
    """Inner pixels (edge = 0) of each parcel, at most MAX_PX_PER_PARCEL; all pixels if a parcel has no inner pixel."""
    out = []
    for p, g in par.groupby("parcel_idx"):
        inner = g[g["edge"] == 0]
        use, fallback = (inner, False) if len(inner) else (g, True)
        if MAX_PX_PER_PARCEL and len(use) > MAX_PX_PER_PARCEL:
            use = use.iloc[rng.choice(len(use), MAX_PX_PER_PARCEL, replace=False)]
        out.append(use.assign(edge_fallback=fallback))
    return pd.concat(out, ignore_index=True)


def read_pixels(samples, tiles):
    """Pass 2: the time series of the sampled pixels, one tile at a time."""
    X = np.empty((len(samples), N_MONTHS, len(MB)), dtype=np.int16)
    for n, (k, g) in enumerate(samples.groupby("tile_k")):
        with rasterio.open(TILE_ROOT / "tiles" / f"{tiles.iloc[k].tile_id}_img.tif") as src:
            img = src.read().reshape(N_MONTHS, len(MB), TILE_PX, TILE_PX)
        X[g.index.to_numpy()] = img[:, :, g["row"].to_numpy(), g["col"].to_numpy()].transpose(2, 0, 1)
        if (n + 1) % 500 == 0:
            print(f"  pass 2: {n + 1:,} tiles read")
    return X


def main():
    parcels = pd.read_csv(TILE_ROOT / "parcels.csv")
    block_fold = parcels.drop_duplicates("block_id").set_index("block_id")["fold"].to_dict()
    tiles_all = pd.read_parquet(TILE_ROOT / "tiles.parquet")
    for season in SEASONS:
        tag = season_tag(season)
        vrt_path = MASK_ROOT / tag / "cropland_mask.vrt"
        if not vrt_path.exists():
            raise FileNotFoundError(f"{vrt_path} not found: run cropland_extent/make_cropland_mask.py for {season} first")
        out_dir = OUT_ROOT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(SEED)
        tiles = tiles_all[(tiles_all["ag_season"] == season) & tiles_all["downloaded"]].reset_index(drop=True)
        tiles = tiles[[(TILE_ROOT / "tiles" / f"{t}_lab.tif").exists() for t in tiles["tile_id"]]].reset_index(drop=True)
        print(f"\n== {REGION} {season}: {len(tiles):,} downloaded tiles")

        # Pass 1: candidates and sample.
        with rasterio.open(vrt_path) as vrt:
            par, non, n_noncrop_tiles = candidates(tiles, vrt, rng)
        n_parcels = par["parcel_idx"].nunique()
        par = sample_parcels(par, rng)
        non = non.assign(edge_fallback=False)
        samples = pd.concat([par, non], ignore_index=True)
        for c in ["tile_k", "row", "col", "parcel_idx", "label", "edge", "mask_cropland", "mask_votes"]:
            samples[c] = samples[c].astype(int)
        print(f"Parcel pixels: {len(par):,} from {n_parcels:,} parcels | not-cropland pixels: {len(non):,} "
              f"from {n_noncrop_tiles:,} tiles")

        # Attributes.
        t = tiles.iloc[samples["tile_k"].to_numpy()]
        samples["tile_id"] = t["tile_id"].to_numpy()
        samples["x"] = (t["ix"].to_numpy() * TILE_PX + samples["col"].to_numpy() + 0.5) * GSD
        samples["y"] = ((t["iy"].to_numpy() + 1) * TILE_PX - samples["row"].to_numpy() - 0.5) * GSD
        info = parcels.set_index("parcel_idx")
        is_p = samples["parcel_idx"] > 0
        samples["parcel_id"] = samples["parcel_idx"].map(info["parcel_id"]).where(is_p, "")
        samples["zone"] = samples["parcel_idx"].map(info["zone"]).where(is_p, "")
        samples["block_id"] = [block_id(x, y) for x, y in zip(samples["x"], samples["y"])]
        samples["block_id"] = samples["parcel_idx"].map(info["block_id"]).where(is_p, samples["block_id"])
        samples["fold"] = samples["parcel_idx"].map(info["fold"]).where(is_p, samples["block_id"].map(block_fold))
        samples["fold"] = samples["fold"].fillna(-1).astype(int)
        samples["class_name"] = samples["label"].map(CLASS_NAMES)
        ll = gpd.GeoSeries(gpd.points_from_xy(samples["x"], samples["y"]), crs=CRS_EPSG).to_crs(4326)
        samples["lon"], samples["lat"] = ll.x.round(6), ll.y.round(6)

        # Pass 2: time series.
        X = read_pixels(samples, tiles)
        np.save(out_dir / "pixels_X.npy", X)
        cols = ["label", "class_name", "parcel_idx", "parcel_id", "fold", "block_id", "zone", "tile_id", "row", "col",
                "x", "y", "lon", "lat", "edge", "edge_fallback", "mask_cropland", "mask_votes"]
        samples[cols].to_parquet(out_dir / "pixels.parquet", index=False)

        summary = (samples.groupby(["class_name", "fold"])
                   .agg(pixels=("label", "size"), parcels=("parcel_idx", lambda v: v[v > 0].nunique()))
                   .reset_index())
        summary.to_csv(out_dir / "summary.csv", index=False)
        (out_dir / "samples_info.json").write_text(json.dumps({
            "region": REGION, "season": season, "crs_epsg": CRS_EPSG, "classes": CLASS_NAMES,
            "X_shape": list(X.shape), "X_axes": ["pixel", "month", "band"], "month_bands": MB, "nodata": NODATA,
            "s2_scale": INFO.get("s2_scale"), "s1_scale": INFO.get("s1_scale"), "start_month": INFO["start_month"],
            "max_px_per_parcel": MAX_PX_PER_PARCEL, "noncrop_px_per_tile": NONCROP_PX_PER_TILE,
            "noncrop_buffer_px": NONCROP_BUFFER_PX, "seed": SEED,
            "noncrop_rule": "no parcel, mask votes = 0, not excluded (built-up/water/snow), inside the mapping area",
            "fold_rule": "parcel pixels: parcel fold; not cropland: fold of the 20 km block, -1 if no labelled parcel there",
            "tile_store": str(TILE_ROOT.relative_to(TILE_ROOT.parents[4])),
        }, indent=2))

        crop = samples[samples["label"] < 3]
        print(f"Pixels per class: {samples['class_name'].value_counts().to_dict()}")
        print(f"Parcel pixels with edge fallback (no inner pixel): {int(crop['edge_fallback'].sum()):,}; "
              f"parcel pixels outside the cropland mask: {100 * (crop['mask_cropland'] != 1).mean():.1f}%")
        print(f"Not-cropland pixels without a fold (block has no labelled parcel): "
              f"{int(((samples['label'] == 3) & (samples['fold'] < 0)).sum()):,}")
        print(pd.crosstab(samples["fold"], samples["class_name"]).to_string())
        print(f"Saved {out_dir} ({X.nbytes / 1e6:,.0f} MB pixel series)")


if __name__ == "__main__":
    main()
