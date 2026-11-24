"""
04_make_pixel_samples.py  —  labelled PIXELS of the surveyed parcels for pixel-wise classification
=====================================================================================================

What it does
------------
Pixel-wise models classify every 10 m pixel on its own, so they are trained and tested on pixels instead of
parcels. This script draws the labelled pixels from the parcel tile store (data_training_prep/03_plan_tiles.py ... 05_stack_tiles.py) and writes one
pixel table per agricultural year: the 12-month time series of every sampled pixel plus its label, parcel,
fold and position. Every pixel-wise experiment starts from this table, so all models see the same pixels.

Classes (2 classes: the experiments compare models on the surveyed cropland parcels only)
-----------------------------------------------------------------------------------------
  1 irrigated      pixels of irrigated parcels (label raster of 03)
  2 rainfed        pixels of rainfed parcels
Pixels outside the surveyed parcels are not used: their class is unknown. No cropland mask is used; where
cropland is (Dynamic World per tile, data_download/02a_dynamic_world.py) is a question of the mapping phase.

Sampling
--------
Inner pixels only (edge = 0 in the label raster: not within INNER_BUFFER_M of the outline, so no mixed boundary
pixels); parcels with no inner pixel fall back to all their pixels (edge_fallback = True). At most
MAX_PX_PER_PARCEL pixels per parcel, drawn at random (SEED), so large parcels do not dominate and the pixels of
one parcel stay a small, fixed group.

Folds (the spatial folds of data_training_prep/02_make_spatial_folds.py)
-----------------------------------------------------
Every pixel takes its parcel's fold and 20 km block (all pixels of a parcel are in the same fold, so no parcel is
split between training and test).

Outputs (outputs/<region>/data_training_prep/04_pixel_samples/<season>/)
-------------------------------------------------------------
pixels_X.npy        int16 [N, 12 months, 19 bands]: the bands of the tile store (12 Sentinel-2 bands,
                    S2_NCLEAR, VV/VH and scene counts per Sentinel-1 orbit), NODATA = -32768, same scaling
pixels.parquet      one row per pixel (same order as pixels_X): label, class_name, parcel_idx, parcel_id, fold,
                    block_id, zone, tile_id, row, col, x, y (UTM centre of the pixel), lon, lat, edge,
                    edge_fallback
summary.csv         pixels and parcels per class and fold
samples_info.json   settings, band order, class codes, CRS
(The earlier 3-class samples with not-cropland pixels are kept unchanged in 04c_pixel_samples/.)

Run (from the project folder; needs the tiles of 03 for every season)
---------------------------------------------------------------------
    python pixelwise_run/04_make_pixel_samples.py
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_download"))   # regions.py (data_download/)
from regions import dataset_dir, settings

# ================================ SETTINGS ================================
REGION = "morocco"                         # "morocco" | "utah" (regions.py)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2020/21", "2023/24"]}[REGION]   # label years (each needs its tiles of 03)
MAX_PX_PER_PARCEL = 100                    # random inner pixels per parcel (None = all)
SEED = 42
TILE_ROOT = dataset_dir(REGION) / "03_tiles_128px" / "agyear"          # parcel tile store (data_training_prep/03_plan_tiles.py ... 05_stack_tiles.py)
OUT_ROOT = dataset_dir(REGION) / "04_pixel_samples"
# ==========================================================================

CLASS_NAMES = {1: "Irrigated", 2: "Rainfed"}
CRS_EPSG = settings(REGION)["crs_epsg"]
INFO = json.loads((TILE_ROOT / "tiles_info.json").read_text())
TILE_PX, GSD, NODATA = INFO["tile_px"], INFO["gsd_m"], INFO["nodata"]
N_MONTHS, MB = INFO["n_months"], INFO["month_bands"]


def season_tag(season):
    return season.replace("/", "-")


def candidates(tiles):
    """Pass 1 (label rasters only): every labelled parcel pixel of the tiles."""
    rows = []
    for k, t in enumerate(tiles.itertuples()):
        with rasterio.open(TILE_ROOT / "tiles" / f"{t.tile_id}_lab.tif") as src:
            label, pidx, edge = src.read()
        r, c = np.nonzero((label > 0) & (pidx > 0))
        if r.size:
            rows.append(np.column_stack([np.full(r.size, k), r, c, pidx[r, c], label[r, c], edge[r, c]]))
        if (k + 1) % 500 == 0:
            print(f"  pass 1: {k + 1:,}/{len(tiles):,} tiles")
    cols = ["tile_k", "row", "col", "parcel_idx", "label", "edge"]
    return pd.DataFrame(np.vstack(rows), columns=cols) if rows else pd.DataFrame(columns=cols)


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
    tiles_all = pd.read_parquet(TILE_ROOT / "tiles.parquet")
    for season in SEASONS:
        tag = season_tag(season)
        rng = np.random.default_rng(SEED)
        tiles = tiles_all[(tiles_all["ag_season"] == season) & tiles_all["downloaded"]].reset_index(drop=True)
        tiles = tiles[[(TILE_ROOT / "tiles" / f"{t}_lab.tif").exists() for t in tiles["tile_id"]]].reset_index(drop=True)
        print(f"\n== {REGION} {season}: {len(tiles):,} downloaded tiles")
        if tiles.empty:
            print("  no tiles with label rasters: run data_training_prep/03_plan_tiles.py and 04_cut_tiles.py for this season first")
            continue
        out_dir = OUT_ROOT / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: candidates and sample.
        par = candidates(tiles)
        n_parcels = par["parcel_idx"].nunique()
        samples = sample_parcels(par, rng)
        for c in ["tile_k", "row", "col", "parcel_idx", "label", "edge"]:
            samples[c] = samples[c].astype(int)
        print(f"Parcel pixels: {len(samples):,} from {n_parcels:,} parcels")

        # Attributes.
        t = tiles.iloc[samples["tile_k"].to_numpy()]
        samples["tile_id"] = t["tile_id"].to_numpy()
        samples["x"] = (t["ix"].to_numpy() * TILE_PX + samples["col"].to_numpy() + 0.5) * GSD
        samples["y"] = ((t["iy"].to_numpy() + 1) * TILE_PX - samples["row"].to_numpy() - 0.5) * GSD
        info = parcels.set_index("parcel_idx")
        samples["parcel_id"] = samples["parcel_idx"].map(info["parcel_id"])
        samples["zone"] = samples["parcel_idx"].map(info["zone"])
        samples["block_id"] = samples["parcel_idx"].map(info["block_id"])
        samples["fold"] = samples["parcel_idx"].map(info["fold"]).fillna(-1).astype(int)
        samples["class_name"] = samples["label"].map(CLASS_NAMES)
        ll = gpd.GeoSeries(gpd.points_from_xy(samples["x"], samples["y"]), crs=CRS_EPSG).to_crs(4326)
        samples["lon"], samples["lat"] = ll.x.round(6), ll.y.round(6)

        # Pass 2: time series.
        X = read_pixels(samples, tiles)
        np.save(out_dir / "pixels_X.npy", X)
        cols = ["label", "class_name", "parcel_idx", "parcel_id", "fold", "block_id", "zone", "tile_id", "row", "col",
                "x", "y", "lon", "lat", "edge", "edge_fallback"]
        samples[cols].to_parquet(out_dir / "pixels.parquet", index=False)

        summary = (samples.groupby(["class_name", "fold"])
                   .agg(pixels=("label", "size"), parcels=("parcel_idx", "nunique"))
                   .reset_index())
        summary.to_csv(out_dir / "summary.csv", index=False)
        (out_dir / "samples_info.json").write_text(json.dumps({
            "region": REGION, "season": season, "crs_epsg": CRS_EPSG, "classes": CLASS_NAMES,
            "X_shape": list(X.shape), "X_axes": ["pixel", "month", "band"], "month_bands": MB, "nodata": NODATA,
            "s2_scale": INFO.get("s2_scale"), "s1_scale": INFO.get("s1_scale"), "start_month": INFO["start_month"],
            "max_px_per_parcel": MAX_PX_PER_PARCEL, "seed": SEED,
            "fold_rule": "every pixel takes its parcel's fold and block",
            "tile_store": str(TILE_ROOT.relative_to(TILE_ROOT.parents[4])),
        }, indent=2))

        print(f"Pixels per class: {samples['class_name'].value_counts().to_dict()}")
        print(f"Pixels with edge fallback (parcel without an inner pixel): {int(samples['edge_fallback'].sum()):,}; "
              f"pixels without a fold: {int((samples['fold'] < 0).sum()):,}")
        print(pd.crosstab(samples["fold"], samples["class_name"]).to_string())
        print(f"Saved {out_dir} ({X.nbytes / 1e6:,.0f} MB pixel series)")


if __name__ == "__main__":
    main()
