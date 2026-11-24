"""
01_plan_blocks.py  —  the grid of 10.24 km blocks over the whole mapping area (no mask, no download)
====================================================================================================

What it does
------------
Plans, once per region, every block of 1024 x 1024 px (10.24 km at 10 m) that touches the region's mapping area
(regions.py: map_file / map_where). No cropland mask and no training data are used here: the grid is the same for
every agricultural year and only the data in the blocks change. The training tiles per block are counted when the
downloads load the grid (download_common.load_blocks, from data_training_prep/03_plan_tiles.py), so 02b/02c always
download the blocks of the current training data.
Everything after this works block by block:
  02a_dynamic_world.py   binary cropland image (Dynamic World) for every block
  02b_s2.py, 02c_s1.py   (later ERA5) only the blocks with cropland (>= MIN_CROP_PIXELS) or a training tile
  maps                   each block classified on its own (cropland pixels only), block maps merged at the end

The grid is the 128 px tile grid of the training data grouped 8 x 8 (download_common.py): block (bx, by) covers
x = bx * 10240 m ... + 10240 m and y = by * 10240 m ... + 10240 m in the region's projection (regions.py).

Output (outputs/<REGION>/data_download/grid/)
-----------------------------------------
blocks.parquet            one row per block: bx, by, block, x0, y0 (top-left), area_frac (share inside the mapping
                          area), geometry
blocks.geojson            the same in EPSG:4326 (to view in QGIS)
blocks.png                map of the block extents over the mapping area (the cropland of each year is drawn by
                          02a_dynamic_world.py)

Run (from the project folder; first step of data_download/, independent of data_training_prep/)
-------------------------------------------------------------------------------------------
    python data_download/01_plan_blocks.py
"""
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.prepared import prep

from download_common import BLOCK_M, block_name, download_dir, grid_file
from regions import admin_boundaries, map_area, settings

# ================================ SETTINGS ================================
REGION = "utah"             # "morocco" | "utah" (regions.py)
MAKE_PNG = True                # map of the block extents (blocks.png)
# ==========================================================================

CRS_EPSG = settings(REGION)["crs_epsg"]


def area_blocks(area):
    """Every block that intersects the mapping area, with the share of the block inside it."""
    minx, miny, maxx, maxy = area.bounds
    parea = prep(area)
    rows = {}
    for bx in range(int(np.floor(minx / BLOCK_M)), int(np.floor(maxx / BLOCK_M)) + 1):
        for by in range(int(np.floor(miny / BLOCK_M)), int(np.floor(maxy / BLOCK_M)) + 1):
            b = box(bx * BLOCK_M, by * BLOCK_M, (bx + 1) * BLOCK_M, (by + 1) * BLOCK_M)
            if parea.intersects(b):
                rows[(bx, by)] = 1.0 if parea.contains(b) else round(area.intersection(b).area / BLOCK_M ** 2, 4)
    return rows


def plan_grid(in_area):
    """The block grid: extents and share of each block inside the mapping area."""
    df = pd.DataFrame([{"bx": bx, "by": by, "area_frac": v} for (bx, by), v in sorted(in_area.items())])
    df["block"] = [block_name(bx, by) for bx, by in zip(df["bx"], df["by"])]
    df["x0"], df["y0"] = df["bx"] * BLOCK_M, (df["by"] + 1) * BLOCK_M                 # top-left corner
    geom = [box(x, y - BLOCK_M, x + BLOCK_M, y) for x, y in zip(df["x0"], df["y0"])]
    return gpd.GeoDataFrame(df, geometry=geom, crs=CRS_EPSG)


def make_png(grid):
    """The block extents over the mapping area."""
    fig, ax = plt.subplots(figsize=(10, 9))
    try:
        admin_boundaries(REGION).to_crs(CRS_EPSG).plot(ax=ax, color="#eef3e8", edgecolor="0.3", linewidth=0.8)
    except Exception:                                               # boundaries are only background
        pass
    gpd.GeoSeries([map_area(REGION)], crs=CRS_EPSG).boundary.plot(ax=ax, color="black", linewidth=1.0)
    grid.boundary.plot(ax=ax, color="#d95f02", linewidth=0.25)
    b = grid.total_bounds
    pad = 0.03 * max(b[2] - b[0], b[3] - b[1])
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{REGION.capitalize()}: {len(grid):,} blocks of {BLOCK_M / 1000:.2f} km (1024 px, 8 x 8 tiles of "
                 f"128 px) over the mapping area\n{grid['area_frac'].sum() * (BLOCK_M / 1000) ** 2:,.0f} km²; ", fontsize=11)
    out = grid_file(REGION).with_suffix(".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  map -> {out}")


def main():
    path = grid_file(REGION)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = plan_grid(area_blocks(map_area(REGION)))
    grid.to_parquet(path)
    grid.to_crs(4326).to_file(path.with_suffix(".geojson"), driver="GeoJSON")
    print(f"{REGION}: {len(grid):,} blocks of {BLOCK_M / 1000:.2f} km "
          f"({grid['area_frac'].sum() * (BLOCK_M / 1000) ** 2:,.0f} km² of mapping area), the same for every year")
    if MAKE_PNG:
        make_png(grid)
    print(f"Saved in {download_dir(REGION) / 'grid'}. Next: 02a_dynamic_world.py (binary cropland image per block), "
          f"then 02b_s2.py and 02c_s1.py (only blocks with cropland or training tiles)")


if __name__ == "__main__":
    main()
