"""
01_s2_grid_overview.py  —  Sentinel-2 tiles, download blocks and model tiles over the cropland mask
====================================================================================================

What it does
------------
1. Reads the cropland tiles of the mask (cropland_extent/make_cropland_mask.py: tiles_cropland.parquet) and keeps the
   128 px tiles with at least MIN_CROP_FRAC cropland.
2. Groups them into download blocks of BLOCK_TILES x BLOCK_TILES tiles (8 x 8 = 1024 px = 10.24 km) on the same grid.
3. Finds the Sentinel-2 tiles (MGRS, 109.8 km) over the region in Earth Search and their full extent (from the
   asset transform), and counts which of them touch a cropland block.
4. Writes the block list (used by 02_download_s2_stac.py) and the overview figure: region -> one Sentinel-2 tile
   -> one block -> its 8 x 8 model tiles.

Three units
-----------
  Sentinel-2 tile  109.8 km (10,980 px)  how the images are published; one file per tile, date and band
  block            10.24 km (1,024 px)   one download job: all scenes of the year for this square
  model tile        1.28 km (128 px)     what is stored and predicted; same tile_id as the mask and the training tiles

Outputs (outputs/<region>/s2_download/grid/<season>/)
-----------------------------------------------------
blocks.gpkg        one row per block: bx, by, n_tiles (cropland tiles in the block), n_crop_px, n_s2_tiles
s2_tiles.gpkg      Sentinel-2 tiles touching a cropland block (code, EPSG of the tile, geometry in the region's CRS)
summary.json       counts
figure: s2_download/figures/<region>_s2_blocks_tiles.png

Run (Mac or cluster login node; needs internet for Earth Search and the basemap)
-------------------------------------------------------------------------------
    python s2_download/01_s2_grid_overview.py
"""
import json
import sys
import warnings
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pystac_client import Client
from rasterio.windows import from_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixelwise_run"))   # regions.py is shared
from regions import cropland_dir, settings  # noqa: E402

warnings.filterwarnings("ignore")

# ================================ SETTINGS ================================
REGIONS = ["morocco", "utah"]
SEASON = "2023/24"                       # agricultural year of the cropland mask used for the block list
MIN_CROP_FRAC = 0.01                     # model tile kept if >= 1% cropland (164 px); 99.9% / 99.7% of the cropland
BLOCK_TILES = 8                          # block = 8 x 8 model tiles = 1024 px = 10.24 km
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
SEARCH_WINDOWS = ["2024-07-01/2024-07-12", "2024-08-01/2024-08-12"]   # two 12-day windows: every tile has a scene
FIGURE = True
# ==========================================================================

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent / "figures"
GSD_M, TILE_PX = 10, 128
TILE_M = GSD_M * TILE_PX
BLOCK_M = TILE_M * BLOCK_TILES
NAME = {"morocco": "Morocco", "utah": "Utah"}


def out_dir(region, season):
    d = ROOT / "outputs" / region / "s2_download" / "grid" / season.replace("/", "-")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cropland_blocks(region, season, epsg):
    t = pd.read_parquet(cropland_dir(region) / "cropland_mask" / season.replace("/", "-") / "tiles_cropland.parquet")
    t = t[(t["n_crop_px"] > 0) & (t["crop_frac"] >= MIN_CROP_FRAC)]
    b = (t.assign(bx=t["ix"] // BLOCK_TILES, by=t["iy"] // BLOCK_TILES)
         .groupby(["bx", "by"]).agg(n_tiles=("tile_id", "size"), n_crop_px=("n_crop_px", "sum")).reset_index())
    geom = [box(x * BLOCK_M, y * BLOCK_M, (x + 1) * BLOCK_M, (y + 1) * BLOCK_M) for x, y in zip(b["bx"], b["by"])]
    return gpd.GeoDataFrame(b, geometry=geom, crs=epsg), len(t)


def s2_tiles(blocks, epsg):
    """Full extent of every Sentinel-2 tile with a scene in the search windows, in the region's CRS."""
    bbox = list(blocks.to_crs(4326).total_bounds)
    cat = Client.open(STAC_URL)
    found = {}
    for win in SEARCH_WINDOWS:
        for it in cat.search(collections=[COLLECTION], bbox=bbox, datetime=win).items():
            code = it.properties["grid:code"].replace("MGRS-", "")
            if code in found:
                continue
            a = it.assets["red"].extra_fields
            tr, (h, w) = a["proj:transform"], a["proj:shape"]
            tile_epsg = int(str(it.properties["proj:code"]).split(":")[1])
            g = gpd.GeoSeries([box(tr[2], tr[5] - h * GSD_M, tr[2] + w * GSD_M, tr[5])], crs=tile_epsg).to_crs(epsg)
            found[code] = (tile_epsg, g.iloc[0])
    return gpd.GeoDataFrame({"code": list(found), "tile_epsg": [v[0] for v in found.values()]},
                            geometry=[v[1] for v in found.values()], crs=epsg)


def figure(region, season, epsg, blocks, s2):
    inside = gpd.sjoin(blocks, s2, predicate="within")
    code = inside.groupby("code").size().idxmax()                       # Sentinel-2 tile with most whole blocks
    T = s2.loc[s2["code"] == code, "geometry"].iloc[0]
    cand = blocks[blocks.within(T) & blocks["n_tiles"].between(35, 58)]
    blk = (cand if len(cand) else blocks[blocks.within(T)]).sort_values("n_crop_px", ascending=False).iloc[0]
    basemap = ctx.providers.Esri.WorldGrayCanvas

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    a = ax[0]
    blocks.plot(ax=a, column="n_tiles", cmap="YlOrBr", vmin=0, vmax=BLOCK_TILES ** 2, lw=0, alpha=0.85, legend=True,
                legend_kwds=dict(label=f"cropland tiles in the block (of {BLOCK_TILES ** 2})", shrink=0.6))
    s2.boundary.plot(ax=a, color="#1f4e9c", lw=0.8)
    gpd.GeoSeries([T], crs=epsg).boundary.plot(ax=a, color="red", lw=2)
    for _, s in s2.iterrows():
        c = s.geometry.centroid
        a.text(c.x, c.y, s["code"], fontsize=5.5, ha="center", va="center", color="#1f4e9c")
    a.set_title(f"{NAME[region]}: {len(s2)} Sentinel-2 tiles (blue, 109.8 km) over cropland\n"
                f"{len(blocks):,} blocks of {BLOCK_M / 1000:.2f} km with cropland (coloured squares)", fontsize=10)
    ctx.add_basemap(a, crs=f"EPSG:{epsg}", source=basemap, attribution_size=5)

    a = ax[1]
    x0, y0, x1, y1 = T.bounds
    blocks[blocks.intersects(T)].plot(ax=a, column="n_tiles", cmap="YlOrBr", vmin=0, vmax=BLOCK_TILES ** 2,
                                      edgecolor="#333", lw=0.5, alpha=0.6)
    gpd.GeoSeries([T], crs=epsg).boundary.plot(ax=a, color="red", lw=2)
    a.add_patch(Rectangle(blk.geometry.bounds[:2], BLOCK_M, BLOCK_M, fill=False, ec="black", lw=2.2))
    a.set_xlim(x0 - 3000, x1 + 3000)
    a.set_ylim(y0 - 3000, y1 + 3000)
    a.set_title(f"Sentinel-2 tile {code} (red): {int(blocks.intersects(T).sum())} cropland blocks\n"
                f"black square = one {BLOCK_M / 1000:.2f} km block = one download job", fontsize=10)
    ctx.add_basemap(a, crs=f"EPSG:{epsg}", source=basemap, attribution_size=5)

    a = ax[2]
    bx0, by0, bx1, by1 = blk.geometry.bounds
    with rasterio.open(cropland_dir(region) / "cropland_mask" / season.replace("/", "-") / "cropland_mask.vrt") as src:
        m = src.read(1, window=from_bounds(bx0, by0, bx1, by1, src.transform), boundless=True, fill_value=65535)
    crop = (m == 1).astype(float)
    crop[m == 65535] = np.nan
    a.imshow(crop, extent=(bx0, bx1, by0, by1), cmap=ListedColormap(["#f2efe6", "#5a9e3a"]), interpolation="nearest")
    n_px, kept = BLOCK_TILES * TILE_PX, 0
    for i in range(BLOCK_TILES):
        for j in range(BLOCK_TILES):
            sub = crop[n_px - (j + 1) * TILE_PX:n_px - j * TILE_PX, i * TILE_PX:(i + 1) * TILE_PX]
            ok = np.nansum(sub) >= MIN_CROP_FRAC * TILE_PX ** 2
            kept += ok
            a.add_patch(Rectangle((bx0 + i * TILE_M, by0 + j * TILE_M), TILE_M, TILE_M, fill=False,
                                  ec="#d62728" if ok else "#999", lw=1.6 if ok else 0.6, ls="-" if ok else ":"))
    a.set_xlim(bx0, bx1)
    a.set_ylim(by0, by1)
    a.set_title(f"One block = {BLOCK_TILES} x {BLOCK_TILES} tiles of {TILE_PX} px ({TILE_M / 1000:.2f} km)\n"
                f"green = cropland mask; red = tile kept (>= {MIN_CROP_FRAC:.0%} cropland): {kept} of {BLOCK_TILES ** 2}",
                fontsize=10)
    for q in ax:
        q.set_xticks([])
        q.set_yticks([])
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{region}_s2_blocks_tiles.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure: {path}")


def main():
    for region in REGIONS:
        epsg = settings(region)["crs_epsg"]
        blocks, n_tiles = cropland_blocks(region, SEASON, epsg)
        s2 = s2_tiles(blocks, epsg)
        s2 = s2[s2.intersects(blocks.union_all())].reset_index(drop=True)
        blocks["n_s2_tiles"] = gpd.sjoin(blocks, s2, predicate="intersects").groupby(level=0).size()
        blocks["inside_one_s2_tile"] = gpd.sjoin(blocks, s2, predicate="within").groupby(level=0).size() \
            .reindex(blocks.index, fill_value=0) > 0
        d = out_dir(region, SEASON)
        blocks.to_file(d / "blocks.gpkg")
        s2.to_file(d / "s2_tiles.gpkg")
        summary = {"region": region, "season": SEASON, "min_crop_frac": MIN_CROP_FRAC, "block_px": BLOCK_TILES * TILE_PX,
                   "model_tiles": n_tiles, "blocks": len(blocks), "s2_tiles": len(s2),
                   "utm_zones": sorted({c[:2] for c in s2["code"]}),
                   "blocks_inside_one_s2_tile": int(blocks["inside_one_s2_tile"].sum()),
                   "s2_tiles_per_block_mean": round(float(blocks["n_s2_tiles"].mean()), 2)}
        (d / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{region}: {n_tiles:,} model tiles, {len(blocks):,} blocks, {len(s2)} Sentinel-2 tiles "
              f"(zones {summary['utm_zones']}), {summary['blocks_inside_one_s2_tile']:,} blocks inside one S2 tile")
        if FIGURE:
            figure(region, SEASON, epsg, blocks, s2)


if __name__ == "__main__":
    main()
