"""
03b_tile_quicklooks.py  —  quick-look figures of tiles from the tile store (visual check of 03_build_tiles.py)
=============================================================================================================

What it does
------------
Draws one PNG per tile: the 12 monthly composites as true colour (B4/B3/B2), NDVI and Sentinel-1 VH,
the three label rasters (label, parcel_idx, edge) and the mean NDVI and VH per month of the irrigated and
rainfed parcel pixels. It shows at a glance whether a tile is complete, whether the labels sit on the
fields, and whether irrigated and rainfed parcels differ through the agricultural year.

Which tiles
-----------
TILE_IDS = None picks tiles automatically from tiles.parquet (downloaded "home" tiles with < 1 % missing
Sentinel-2): per agricultural year the N_MIXED tiles with the most pixels of BOTH classes, plus the tile with
the most irrigated and the tile with the most rainfed pixels. Without tiles.parquet (e.g. a folder of copied
sample tiles) every *_img.tif in the folder is drawn. Or list tile IDs, e.g. ["2023-24_330_3651"].

Sentinel-1 orbit: ORBIT = "auto" takes the orbit direction with more observations in the tile (ascending
and descending are stored separately; Utah 2023/24 is descending only).

Output
------
<TILE_STORE>/quicklooks/<tile_id>.png

Run (from the project folder)
-----------------------------
    python ml_dataset_prep/03b_tile_quicklooks.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from regions import dataset_dir  # noqa: E402

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
TILE_STORE = dataset_dir(REGION) / "03_tiles_128px" / "agyear"   # folder with tiles/ and tiles_info.json
TILE_IDS = None                # None = pick automatically (see above); or a list of tile IDs
SEASONS = None                 # agricultural years to pick from, e.g. ["2023/24"]; None = all downloaded
N_MIXED = 3                    # automatic pick: tiles with most pixels of both classes, per year
ORBIT = "auto"                 # Sentinel-1 orbit to draw: "auto" | "ASC" | "DSC"
RGB_MAX = 3000                 # reflectance x 10000 shown as white in the true-colour images
NDVI_RANGE = (0.0, 0.9)
VH_RANGE_DB = (-28, -10)
DPI = 80
# ==========================================================================

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CLASSES = ((1, "irrigated", "#1f77b4"), (2, "rainfed", "#e6a118"))


def pick_tiles(tile_dir):
    """Tile IDs to draw (TILE_IDS, else automatic pick from tiles.parquet, else all tiles in the folder)."""
    if TILE_IDS:
        return list(TILE_IDS)
    index = TILE_STORE / "tiles.parquet"
    if not index.exists():
        return sorted(p.name[:-8] for p in tile_dir.glob("*_img.tif"))
    t = pd.read_parquet(index)
    t = t[t["downloaded"] & (t["role"] == "home") & (t["s2_nodata_pct"] < 1)]
    if SEASONS:
        t = t[t["ag_season"].isin(SEASONS)]
    t = t.assign(both=t[["n_irrigated_px", "n_rainfed_px"]].min(axis=1))
    pick = []
    for _, grp in t.groupby("ag_season"):
        pick += grp.sort_values("both", ascending=False).head(N_MIXED)["tile_id"].tolist()
        pick += grp.sort_values("n_irrigated_px", ascending=False).head(1)["tile_id"].tolist()
        pick += grp.sort_values("n_rainfed_px", ascending=False).head(1)["tile_id"].tolist()
    return list(dict.fromkeys(pick))


def quicklook(tid, tile_dir, info, out_dir):
    bands, nodata = info["month_bands"], info["nodata"]
    n_months, start = info["n_months"], info["start_month"]
    months = [MONTH_ABBR[(start - 1 + k) % 12] for k in range(n_months)]

    with rasterio.open(tile_dir / f"{tid}_img.tif") as src:
        a = src.read().astype(float).reshape(n_months, len(bands), src.height, src.width)
    with rasterio.open(tile_dir / f"{tid}_lab.tif") as src:
        lab = src.read()
    a[a == nodata] = np.nan

    def b(name):
        return a[:, bands.index(name)]

    ndvi = (b("B8") - b("B4")) / (b("B8") + b("B4"))
    orbit = ORBIT
    if orbit == "auto":
        orbit = "DSC" if np.isfinite(b("VH_DSC")).mean() >= np.isfinite(b("VH_ASC")).mean() else "ASC"
    vh = b(f"VH_{orbit}") / info.get("s1_scale", 100)

    fig, ax = plt.subplots(4, n_months, figsize=(2 * n_months, 9))
    for k in range(n_months):
        rgb = np.dstack([b("B4")[k], b("B3")[k], b("B2")[k]]) / RGB_MAX
        ax[0, k].imshow(np.clip(np.nan_to_num(rgb), 0, 1))
        ax[0, k].set_title(f"m{k + 1:02d} {months[k]}", fontsize=9)
        ax[1, k].imshow(ndvi[k], cmap="RdYlGn", vmin=NDVI_RANGE[0], vmax=NDVI_RANGE[1])
        ax[2, k].imshow(vh[k], cmap="gray", vmin=VH_RANGE_DB[0], vmax=VH_RANGE_DB[1])
    ax[0, 0].set_ylabel("RGB")
    ax[1, 0].set_ylabel("NDVI")
    ax[2, 0].set_ylabel(f"VH {orbit} dB")

    ax[3, 0].imshow(lab[0], cmap=ListedColormap(["#eeeeee", "#1f77b4", "#e6a118"]), vmin=0, vmax=2,
                    interpolation="nearest")
    ax[3, 0].set_title("label: grey ignore\nblue irrigated, orange rainfed", fontsize=8)
    ax[3, 1].imshow(np.where(lab[1] > 0, lab[1] % 20, np.nan), cmap="tab20", interpolation="nearest")
    ax[3, 1].set_title("parcel_idx", fontsize=8)
    ax[3, 2].imshow(lab[2], cmap="gray_r", interpolation="nearest")
    ax[3, 2].set_title("edge pixels", fontsize=8)

    curves = (ax[3, 4], ax[3, 8])
    for c, name, colour in CLASSES:
        m = lab[0] == c
        if m.any():
            curves[0].plot(range(1, n_months + 1), [np.nanmean(ndvi[k][m]) for k in range(n_months)],
                           "-o", color=colour, label=name, ms=3)
            curves[1].plot(range(1, n_months + 1), [np.nanmean(vh[k][m]) for k in range(n_months)],
                           "-o", color=colour, label=name, ms=3)
    for x, title in zip(curves, ("mean NDVI per month", f"mean VH {orbit} (dB)")):
        x.set_title(title, fontsize=8)
        if x.lines:
            x.legend(fontsize=7)
        x.set_xticks(range(1, n_months + 1))
        x.set_xticklabels([m[0] for m in months], fontsize=7)

    for x in ax.flat:
        if x not in curves:
            x.set_xticks([])
            x.set_yticks([])
    for j in range(3, n_months):
        if ax[3, j] not in curves:
            ax[3, j].axis("off")
    for x, shift in zip(curves, (0.04, 0.06)):        # widen the two curve panels over the empty cells
        x.set_position(x.get_position().expanded(3.6, 1.0).translated(shift, 0))

    fig.suptitle(f"{REGION} {tid}  |  irrigated px {(lab[0] == 1).sum()}, rainfed px {(lab[0] == 2).sum()}"
                 f"  |  m01 = {months[0]}", fontsize=12)
    fig.savefig(out_dir / f"{tid}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return orbit


def main():
    tile_dir = TILE_STORE / "tiles"
    info = json.loads((TILE_STORE / "tiles_info.json").read_text())
    out_dir = TILE_STORE / "quicklooks"
    out_dir.mkdir(exist_ok=True)
    tiles = [t for t in pick_tiles(tile_dir) if (tile_dir / f"{t}_img.tif").exists()]
    print(f"{REGION}: {len(tiles)} tiles -> {out_dir}")
    for tid in tiles:
        orbit = quicklook(tid, tile_dir, info, out_dir)
        print(f"  {tid}  (S1 {orbit})")


if __name__ == "__main__":
    main()
