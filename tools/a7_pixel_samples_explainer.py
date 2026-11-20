"""
a7_pixel_samples_explainer.py  —  six-panel figure of what 04_make_pixel_samples.py does, on one tile
=======================================================================================================

Panels: (1) Sentinel-2 tile with the surveyed parcels, (2) label raster of 03, (3) votes of the cropland mask with
the parcel outlines, (4) the sampled pixels, (5) NDVI of the sampled pixels per class, (6) the output table.
Writes docs/phase1_data/figures/pixel_samples_explainer.png (Pixel samples page).

Run (Mac, needs the Morocco tiles, cropland mask and pixel samples of 2023/24):
    python tools/a7_pixel_samples_explainer.py
"""
import json, numpy as np, pandas as pd, rasterio, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation
from pathlib import Path
R = Path(__file__).resolve().parents[1]

# ================================ SETTINGS ================================
TID = "2023-24_391_2832"                     # example tile (Morocco 2023/24, Doukkala)
# ==========================================================================

_, ix, iy = TID.split("_"); ix, iy = int(ix), int(iy)
T = R / "outputs/morocco/dataset/03_tiles_128px/agyear"
S = R / "outputs/morocco/dataset/04c_pixel_samples/2023-24"
info = json.loads((T / "tiles_info.json").read_text()); MB = info["month_bands"]
with rasterio.open(T / "tiles" / f"{TID}_img.tif") as src: img = src.read().reshape(12, len(MB), 128, 128).astype(float)
with rasterio.open(T / "tiles" / f"{TID}_lab.tif") as src: label, pidx, edge = src.read()
with rasterio.open(R / "outputs/morocco/cropland_extent/cropland_mask/2023-24/cropland_mask.vrt") as v:
    w = from_bounds(ix * 1280, iy * 1280, (ix + 1) * 1280, (iy + 1) * 1280, v.transform)
    crop, votes, bits, excl = v.read(window=w)
m = pd.read_parquet(S / "pixels.parquet"); X = np.load(S / "pixels_X.npy", mmap_mode="r")
sel = m[m.tile_id == TID]
months = ["Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
def rgb(k):
    a = img[k][[MB.index("B4"), MB.index("B3"), MB.index("B2")]].transpose(1, 2, 0)
    return np.clip(a / 2500, 0, 1) ** 0.8
outline = np.zeros_like(pidx, bool)
for p in np.unique(pidx[pidx > 0]):
    q = pidx == p; outline |= q & binary_dilation(~q)
COL = {"Irrigated": "#1f77b4", "Rainfed": "#d62728", "Not cropland": "#ff9f1c"}
fig = plt.figure(figsize=(18, 11.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.18)
ext = [0, 128, 128, 0]
# A
ax = fig.add_subplot(gs[0, 0]); a = rgb(6).copy(); a[outline] = [1, 1, 0]; ax.imshow(a, extent=ext)
ax.set_title("1 · Input from 03: Sentinel-2 tile (128 × 128 px = 1.28 km)\nMarch 2024 true colour, surveyed parcels outlined", fontsize=10)
# B
ax = fig.add_subplot(gs[0, 1])
lab_show = np.where(label == 0, 0, np.where(edge == 1, label + 2, label))   # 1 irr, 2 rain, 3 irr edge, 4 rain edge
cm = ListedColormap(["#eeeeee", COL["Irrigated"], COL["Rainfed"], "#9ec5e8", "#f0a0a0"])
ax.imshow(lab_show, cmap=cm, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], 5), interpolation="nearest", extent=ext)
ax.legend(handles=[Patch(fc=COL["Irrigated"], label="irrigated parcel (inner pixels)"), Patch(fc=COL["Rainfed"], label="rainfed parcel (inner pixels)"),
                   Patch(fc="#9ec5e8", label="irrigated edge pixels (not sampled)"), Patch(fc="#f0a0a0", label="rainfed edge pixels (not sampled)"),
                   Patch(fc="#eeeeee", ec="0.6", label="no label (outside surveyed parcels)")], loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=7.5, frameon=False)
ax.set_title("2 · Label raster from 03: ONE label per parcel, copied to all its pixels\nedge pixels (≤ 10 m from the outline) are mixed and skipped", fontsize=10)
# C
ax = fig.add_subplot(gs[0, 2])
vv = np.where(excl == 1, 5, votes).astype(float)
cm2 = ListedColormap(["#ff9f1c", "#e5f5e0", "#a1d99b", "#41ab5d", "#00441b", "#555555"])
ax.imshow(vv, cmap=cm2, norm=BoundaryNorm(np.arange(-.5, 6.5), 6), interpolation="nearest", extent=ext)
c = np.arange(128) + 0.5                                   # pixel centres; rows run downward as in imshow
ax.contour(c, c, (pidx > 0).astype(float), levels=[0.5], colors="white", linewidths=0.8)
ax.set_xlim(0, 128); ax.set_ylim(128, 0)
ax.legend(handles=[Patch(fc=c, label=l) for c, l in zip(["#ff9f1c", "#e5f5e0", "#a1d99b", "#41ab5d", "#00441b", "#555555"],
          ["0 products say cropland → candidate 'not cropland'", "1 product", "2 products", "3 products", "4 products", "built-up / water (excluded)"])],
          loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=7.5, frameon=False)
ax.set_title("3 · Cropland mask (cropland_extent): votes of the 4 products\nwhite = surveyed parcels", fontsize=10)
# D
ax = fig.add_subplot(gs[1, 0]); ax.imshow(rgb(6) * 0.75, extent=ext)
for cls, g in sel.groupby("class_name"):
    ax.scatter(g.col + 0.5, g.row + 0.5, s=9, c=COL[cls], edgecolors="white", linewidths=0.3, label=f"{cls}: {len(g)} px")
ax.set_xlim(0, 128); ax.set_ylim(128, 0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8, frameon=False, markerscale=1.8)
ax.set_title("4 · What 04 picks: ≤ 100 random inner pixels per parcel\n+ ≤ 100 random 'not cropland' pixels per tile (0 votes, ≥ 2 px from any parcel)", fontsize=10)
# E
ax = fig.add_subplot(gs[1, 1])
x = np.asarray(X[sel.index.to_numpy()])[:, :, [MB.index("B4"), MB.index("B8")]].astype(float)
nd = (x[..., 1] - x[..., 0]) / (x[..., 1] + x[..., 0])
for cls in ["Irrigated", "Rainfed", "Not cropland"]:
    k = (sel.class_name == cls).to_numpy()
    if k.any():
        q = np.nanpercentile(nd[k], [25, 50, 75], axis=0)
        ax.plot(months, q[1], color=COL[cls], lw=2.2, label=f"{cls} (median, IQR)"); ax.fill_between(months, q[0], q[2], color=COL[cls], alpha=0.2)
ax.set_ylabel("NDVI"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title("5 · Each sampled pixel keeps its own 12-month time series\n(19 bands per month; NDVI of this tile's samples shown)", fontsize=10)
# F
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
ex = sel.groupby("class_name").head(2)[["label", "class_name", "parcel_id", "fold", "row", "col", "mask_votes"]]
ex["parcel_id"] = ex["parcel_id"].str.slice(0, 22).replace("", "—")
txt = ex.to_string(index=False)
ax.text(0, 1, "6 · Output: ONE ROW PER PIXEL (not per parcel)\n\n"
        "pixels.parquet  (label, parcel, fold, position, mask)\n" + txt +
        f"\n\npixels_X.npy  int16 [pixels, 12 months, 19 bands]\n"
        f"  Morocco 2023/24: {len(m):,} pixels = {X.shape}\n"
        f"  {int((m.label==1).sum()):,} irrigated · {int((m.label==2).sum()):,} rainfed · {int((m.label==3).sum()):,} not cropland\n\n"
        "• label of a pixel = label of its parcel (from the survey)\n"
        "• fold of a pixel = fold of its parcel → no parcel is split\n"
        "  between training and test\n"
        "• 'not cropland' pixels: fold of their 20 km block",
        family="monospace", fontsize=8.6, va="top")
fig.suptitle(f"04_make_pixel_samples.py: from labelled parcels to labelled pixels · example tile {TID} (Morocco 2023/24)", fontsize=13, y=0.995)
out = R / "docs/phase1_data/figures/pixel_samples_explainer.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print(out)