"""
a7_pixel_samples_explainer.py  —  six-panel figure of what 04_make_pixel_samples.py does, on one tile
=======================================================================================================

Two-class samples (irrigated, rainfed) of the surveyed parcels; no cropland mask is used.
Panels: (1) Sentinel-2 tile with the surveyed parcels, (2) label raster of 03 with the edge pixels, (3) inner pixels
and the sampled pixels (at most 100 per parcel), (4) the sampled pixels on the image, coloured by their parcel's fold,
(5) NDVI of the sampled pixels per class, (6) the output table.
Writes docs/phase1_data/figures/pixel_samples_explainer.png (Pixel samples page).

Run (Mac, needs the Morocco tiles and the pixel samples of 2023/24 in 04_pixel_samples/):
    python tools/a7_pixel_samples_explainer.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation

R = Path(__file__).resolve().parents[1]

# ================================ SETTINGS ================================
TID = "2023-24_398_2810"                     # example tile (Morocco 2023/24, Doukkala): both classes, 5 parcels cut to 100 px
MONTH = 6                                    # month shown as true colour (index in the agricultural year: 6 = March)
# ==========================================================================

T = R / "outputs/morocco/data_training_prep/03_tiles_128px/agyear"
S = R / "outputs/morocco/data_training_prep/04_pixel_samples/2023-24"
info = json.loads((T / "tiles_info.json").read_text())
MB = info["month_bands"]
with rasterio.open(T / "tiles" / f"{TID}_img.tif") as src:
    img = src.read().reshape(12, len(MB), 128, 128).astype(float)
with rasterio.open(T / "tiles" / f"{TID}_lab.tif") as src:
    label, pidx, edge = src.read()
m = pd.read_parquet(S / "pixels.parquet")
X = np.load(S / "pixels_X.npy", mmap_mode="r")
sel = m[m.tile_id == TID]
parcels = pd.read_csv(T / "parcels.csv").set_index("parcel_idx")
months = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
COL = {"Irrigated": "#1f77b4", "Rainfed": "#d62728"}
FOLD_COL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
ext = [0, 128, 128, 0]
c = np.arange(128) + 0.5                                   # pixel centres; rows run downward as in imshow


def rgb(k):
    a = img[k][[MB.index("B4"), MB.index("B3"), MB.index("B2")]].transpose(1, 2, 0)
    return np.clip(a / 2500, 0, 1) ** 0.8


def outlines(ax, color="white", lw=0.8):
    ax.contour(c, c, (pidx > 0).astype(float), levels=[0.5], colors=color, linewidths=lw)
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)


outline = np.zeros_like(pidx, bool)
for p in np.unique(pidx[pidx > 0]):
    q = pidx == p
    outline |= q & binary_dilation(~q)

fig = plt.figure(figsize=(18, 11.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.18)
below = dict(loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False)

# 1 · image
ax = fig.add_subplot(gs[0, 0])
a = rgb(MONTH).copy()
a[outline] = [1, 1, 0]
ax.imshow(a, extent=ext)
ax.set_title("1 · Input from 03: Sentinel-2 tile (128 × 128 px = 1.28 km)\n"
             f"{months[MONTH]} true colour, surveyed parcels outlined", fontsize=10)

# 2 · label raster with edge pixels
ax = fig.add_subplot(gs[0, 1])
lab_show = np.where(label == 0, 0, np.where(edge == 1, label + 2, label))   # 1 irr, 2 rain, 3 irr edge, 4 rain edge
ax.imshow(lab_show, cmap=ListedColormap(["#eeeeee", COL["Irrigated"], COL["Rainfed"], "#9ec5e8", "#f0a0a0"]),
          norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], 5), interpolation="nearest", extent=ext)
ax.legend(handles=[Patch(fc=COL["Irrigated"], label="irrigated parcel (inner pixels)"),
                   Patch(fc=COL["Rainfed"], label="rainfed parcel (inner pixels)"),
                   Patch(fc="#9ec5e8", label="irrigated edge pixels (not sampled)"),
                   Patch(fc="#f0a0a0", label="rainfed edge pixels (not sampled)"),
                   Patch(fc="#eeeeee", ec="0.6", label="no label (outside the surveyed parcels): not used")],
          ncol=2, fontsize=7.5, **below)
ax.set_title("2 · Label raster from 03: ONE label per parcel, copied to all its pixels\n"
             "edge pixels (≤ 10 m from the outline) are mixed and skipped", fontsize=10)

# 3 · inner pixels and the sample
ax = fig.add_subplot(gs[0, 2])
inner = np.where((label > 0) & (edge == 0), label, 0)
ax.imshow(inner, cmap=ListedColormap(["#ffffff", "#c6dbef", "#fcbba1"]), norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3),
          interpolation="nearest", extent=ext)
for cls, g in sel.groupby("class_name"):
    ax.scatter(g.col + 0.5, g.row + 0.5, s=7, c=COL[cls], linewidths=0)
outlines(ax, color="0.4", lw=0.6)
n_inner = {p: int(((pidx == p) & (edge == 0)).sum()) for p in np.unique(pidx[pidx > 0])}
n_kept = sel.groupby("parcel_idx").size().to_dict()
capped = sum(1 for p, n in n_inner.items() if n > 100)
ax.legend(handles=[Patch(fc="#c6dbef", label="inner pixels, irrigated"), Patch(fc="#fcbba1", label="inner pixels, rainfed"),
                   Patch(fc=COL["Irrigated"], label="sampled, irrigated"), Patch(fc=COL["Rainfed"], label="sampled, rainfed")],
          ncol=2, fontsize=7.5, **below)
ax.set_title("3 · Sample: at most 100 random inner pixels per parcel\n"
             f"{len(n_inner)} parcels here; {capped} have more than 100 inner pixels and are cut to 100", fontsize=10)

# 4 · folds
ax = fig.add_subplot(gs[1, 0])
ax.imshow(rgb(MONTH) * 0.7, extent=ext)
for f, g in sel.groupby("fold"):
    ax.scatter(g.col + 0.5, g.row + 0.5, s=8, c=FOLD_COL[int(f)], edgecolors="white", linewidths=0.25,
               label=f"fold {int(f)}: {len(g)} px")
outlines(ax, color="yellow", lw=0.6)
blocks = sorted(sel["block_id"].unique())
ax.legend(ncol=3, fontsize=8, markerscale=1.8, **below)
ax.set_title("4 · Every pixel takes its parcel's fold and 20 km block\n"
             f"here block{'s' if len(blocks) > 1 else ''} {', '.join(blocks)}: a parcel is never split between training and test",
             fontsize=10)

# 5 · NDVI
ax = fig.add_subplot(gs[1, 1])
x = np.asarray(X[sel.index.to_numpy()])[:, :, [MB.index("B4"), MB.index("B8")]].astype(float)
x[x == info["nodata"]] = np.nan
nd = (x[..., 1] - x[..., 0]) / (x[..., 1] + x[..., 0])
for cls in ["Irrigated", "Rainfed"]:
    k = (sel.class_name == cls).to_numpy()
    if k.any():
        q = np.nanpercentile(nd[k], [25, 50, 75], axis=0)
        ax.plot(months, q[1], color=COL[cls], lw=2.2, label=f"{cls} (median, IQR; {k.sum()} px)")
        ax.fill_between(months, q[0], q[2], color=COL[cls], alpha=0.2)
ax.set_ylabel("NDVI")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("5 · Each sampled pixel keeps its own 12-month time series\n"
             "(19 bands per month; NDVI of this tile's samples shown)", fontsize=10)

# 6 · output table
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ex = sel.groupby("class_name").head(2)[["label", "class_name", "parcel_id", "fold", "block_id", "row", "col"]]
ex["parcel_id"] = ex["parcel_id"].astype(str).str.slice(0, 20)
n1, n2 = int((m.label == 1).sum()), int((m.label == 2).sum())
ax.text(0, 1, "6 · Output: ONE ROW PER PIXEL (not per parcel)\n\n"
        "pixels.parquet  (label, parcel, fold, block, position)\n" + ex.to_string(index=False) +
        "\n\npixels_X.npy  int16 [pixels, 12 months, 19 bands]\n"
        f"  Morocco 2023/24: {len(m):,} pixels\n"
        f"  {n1:,} irrigated · {n2:,} rainfed\n"
        f"  from {m.parcel_idx.nunique():,} parcels in {m.tile_id.nunique():,} tiles\n\n"
        "• two classes: irrigated, rainfed (no cropland mask)\n"
        "• label of a pixel = label of its parcel (from the survey)\n"
        "• fold of a pixel = fold of its parcel → no parcel is split\n"
        "  between training and test",
        family="monospace", fontsize=8.6, va="top")
fig.suptitle(f"04_make_pixel_samples.py: from labelled parcels to labelled pixels · example tile {TID} (Morocco 2023/24)",
             fontsize=13, y=0.995)
out = R / "docs/phase1_data/figures/pixel_samples_explainer.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(out)
