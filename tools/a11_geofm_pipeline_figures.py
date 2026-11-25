"""
a11_geofm_pipeline_figures.py  —  how one 128 px tile goes through each foundation model and the trained head
============================================================================================================

One figure per foundation model, one row per step, with the shape of the data after every step, for the set-up of
this study (pixelwise_run/encoders.py and tile_common.GeoFMSegmenter):

  image ViTs (Prithvi-EO-2.0, Clay v1.5, TerraMind)   input -> encoder (frozen) -> L-TAE -> pyramid -> UPerNet -> upsampling
  AgriFM (video Swin)                                 input -> 3-D patches -> Swin stages (its own 4 maps) -> UPerNet -> upsampling
  pixel models (Presto, TESSERA)                      input -> per-pixel encoder -> pooled pyramid -> UPerNet at 10 m

Sizes and weight counts were checked by running the modules of tile_common.py (L-TAE, ViTPyramid, PixelPyramid,
UPerHead) and AgriFM's encoder on a zero tile. Schematic: the maps are made-up illustrations (no data is read).

Output: docs/foundation_models/figures/<model>_pipeline_steps.png

Run (Mac or cluster; no data needed):
    python tools/a11_geofm_pipeline_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter, zoom

R = Path(__file__).resolve().parents[1]
OUT_DIR = R / "docs/foundation_models/figures"

# ================================ SETTINGS ================================
MODELS_TO_DRAW = ["prithvi", "clay", "terramind", "agrifm", "presto", "tessera"]
SEED = 7
DPI = 130
# ==========================================================================

MONTHS = list("SONDJFMAMJJA")                      # Morocco: September - August (Utah: October - September)
C_LOCK, C_TRAIN = "#95a5a6", "#e67e22"            # frozen, trained (as in the E04 explainers)
C_IRR, C_RAIN = "#1f77b4", "#d62728"
GREY = "0.35"

# Per model: what the code does (pixelwise_run/encoders.py, tile_common.py); sizes checked on a zero tile.
VIT_PYRAMID = "x 4 and x 2: transposed convolutions;\nx 1: 1 x 1 convolution; x 1/2: max-pool"
SPEC = {
    "prithvi": dict(
        title="Prithvi-EO-2.0", color="#1f77b4", kind="vit",
        bands=["B2", "B3", "B4", "B8A", "B11", "B12"], band_note="Prithvi's HLS bands",
        norm="minus mean, divided by\nstd (Prithvi's pretraining\nstatistics); cloudy pixels\n-> 0 and masked",
        patch=16, n=8, dim=1024, groups=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        proj="linear projection\n1,536 -> 1,024", extra="+ date (year, day)\n+ location (lat, lon)",
        tf="24 transformer layers\n(ViT-Large)\nevery token attends to\nall 256 tokens of its pass",
        enc_w="304 M", ltae_w="0.36 M", pyr_w="0.39 M", head_w="1.72 M",
        levels=[32, 16, 8, 4], up=4),
    "clay": dict(
        title="Clay v1.5", color="#2ca02c", kind="vit",
        bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"], band_note="10 bands",
        norm="minus mean, divided by\nstd (Clay's statistics,\nmetadata.yaml); cloudy\npixels -> 0 and masked",
        patch=8, n=16, dim=1024, groups=[[m] for m in range(12)],
        proj="projection\n640 -> 1,024 (weights\nfrom wavelengths)",
        extra="+ week, location,\npixel size (10 m)",
        tf="24 transformer layers\n(ViT-Large)\nevery token attends to\nall 256 tokens of its month",
        enc_w="311 M", ltae_w="0.36 M", pyr_w="0.39 M", head_w="1.72 M",
        levels=[64, 32, 16, 8], up=2),
    "terramind": dict(
        title="TerraMind 1.0 base", color="#9467bd", kind="vit",
        bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"], band_note="modality S2L2A",
        norm="minus mean, divided by\nstd (TerraMind's\npretraining statistics);\ncloudy pixels -> 0, masked",
        patch=16, n=8, dim=768, groups=[[m] for m in range(12)],
        proj="linear projection\n3,072 -> 768", extra="no date or\nlocation input",
        tf="12 transformer layers\n(ViT-Base)\nevery token attends to\nall 64 tokens of its month",
        enc_w="87 M", ltae_w="0.30 M", pyr_w="0.39 M", head_w="1.72 M",
        levels=[32, 16, 8, 4], up=4),
    "agrifm": dict(
        title="AgriFM", color="#d62728", kind="agrifm",
        bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"], band_note="10 bands",
        norm="minus mean, divided by\nstd of the REGION's\ntiles (D15); cloudy\npixels -> 0",
        enc_w="88 M", head_w="3.23 M", levels=[32, 16, 8, 4], up=4),
    "presto": dict(
        title="Presto", color="#ff7f0e", kind="pixel",
        bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"], band_note="10 bands + NDVI",
        norm="reflectance / 10,000;\ncloudy months filled by\nlinear interpolation in\ntime (not masked)",
        enc_w="0.4 M", head_w="1.72 M", levels=[128, 64, 32, 16], up=1),
    "tessera": dict(
        title="TESSERA v2", color="#8c564b", kind="pixel",
        bands=["B4", "B2", "B3", "B8", "B8A", "B5", "B6", "B7", "B11", "B12"], band_note="TESSERA's band order",
        norm="minus mean, divided by\nstd (TESSERA's statistics);\nonly the clear months,\neach dated the 15th",
        enc_w="21 M", head_w="1.72 M", levels=[128, 64, 32, 16], up=1),
}
INPUT_NOTES = {
    "prithvi": ["Each training sample is one 128 x 128 px tile with its 12 monthly composites.",
                "Prithvi reads 6 bands: those of its HLS pretraining (Blue, Green, Red, narrow NIR, SWIR 1, SWIR 2).",
                "Bands are standardised with Prithvi's own statistics, so the values look like its pretraining data.",
                "Pixels without a clear observation in a month are set to 0 and remembered in a mask (used by the L-TAE)."],
    "clay": ["Each training sample is one 128 x 128 px tile with its 12 monthly composites.",
             "Clay reads 10 bands (B1 and B9, the 60 m atmospheric bands, are left out).",
             "Bands are standardised with the Sentinel-2 statistics of Clay's metadata.yaml; each band's central wavelength is also passed.",
             "Pixels without a clear observation in a month are set to 0 and remembered in a mask (used by the L-TAE)."],
    "terramind": ["Each training sample is one 128 x 128 px tile with its 12 monthly composites.",
                  "TerraMind reads all 12 Sentinel-2 L2A bands (modality S2L2A); Sentinel-1 is not given (D14).",
                  "Bands are standardised with TerraMind's pretraining statistics.",
                  "Pixels without a clear observation in a month are set to 0 and remembered in a mask (used by the L-TAE)."],
    "agrifm": ["Each training sample is one 128 x 128 px tile with its 12 monthly composites, read as one 12-frame video.",
               "AgriFM reads 10 bands (no B1, B9).",
               "Bands are standardised with the mean and std of 200 tiles of the region itself (D15), not with fixed pretraining statistics.",
               "Pixels without a clear observation are set to 0; AgriFM has no mask input, so the network sees them as zeros."],
    "presto": ["Each training sample is one 128 x 128 px tile with its 12 monthly composites; Presto reads every pixel's series.",
               "Presto reads 10 bands as reflectance / 10,000, plus NDVI computed from B4 and B8.",
               "Cloudy months are filled by linear interpolation between the neighbouring clear months (first / last month: nearest).",
               "Location (latitude, longitude) and the first month of the year are also given."],
    "tessera": ["Each training sample is one 128 x 128 px tile with its 12 monthly composites; TESSERA reads every pixel's series.",
                "TESSERA reads 10 bands in its own order (B4 B2 B3 B8 B8A B5 B6 B7 B11 B12), standardised with its statistics.",
                "Only the pixel's clear months are kept (hatched: cloudy, left out); each is dated by its day of year (the 15th).",
                "TESSERA was trained on single acquisitions; here each observation is a monthly composite."],
}
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- drawing helpers
def grid(ax, x, y, s, n, ec, fc="#f3efe6", lw=0.6, hi=None, hc=None, z=2):
    """Square of side s at (x, y) cut into n x n cells (lines drawn up to 64 cells); hi = highlighted (row, col)."""
    ax.add_patch(Rectangle((x, y), s, s, fc=fc, ec=ec, lw=lw * 1.8, zorder=z))
    c = s / n
    if n <= 64:
        for k in range(1, n):
            ax.plot([x + k * c, x + k * c], [y, y + s], color=ec, lw=lw * 0.6, zorder=z + 0.1)
            ax.plot([x, x + s], [y + k * c, y + k * c], color=ec, lw=lw * 0.6, zorder=z + 0.1)
    if hi is not None:
        r, cc = hi
        ax.add_patch(Rectangle((x + cc * c, y + s - (r + 1) * c), c, c, fc=hc or ec, ec=ec, lw=lw, zorder=z + 0.2))


def image(ax, x, y, s, arr, cmap, vmin=0, vmax=1, z=2, ec="0.3"):
    ax.imshow(arr, extent=(x, x + s, y, y + s), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", zorder=z,
              origin="upper")
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec=ec, lw=1.0, zorder=z + 0.1))


def arrow(ax, p, q, color=GREY, lw=1.4, rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def box(ax, x, y, w, h, text, fc, ec, fs=9, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=1.2,
                                zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=z + 0.1)


def label(ax, x, y, text, fs=8.5, color=GREY, ha="center", va="top", **kw):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, **kw)


def smooth_field(n, k=4):
    """Made-up smooth map in [0, 1], size n x n."""
    up = np.kron(rng.random((k, k)), np.ones((n // k, n // k)))
    return np.clip(gaussian_filter(up, n / (k * 2.5)) * 1.4 - 0.2, 0, 1)


def vector(ax, x, y, w, h, n, cmap, ec, hatch=None, z=3):
    """A feature vector drawn as a column of n coloured cells."""
    c = h / n
    for i, v in enumerate(rng.random(n)):
        ax.add_patch(Rectangle((x, y + i * c), w, c, fc=plt.get_cmap(cmap)(0.25 + 0.6 * v), ec="none", zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=1.0, hatch=hatch, zorder=z + 0.1))


def canvas(ax, w=100, h=40):
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")


def header(ax, nx, num, title, part, shape):
    col = {"frozen": C_LOCK, "trained": C_TRAIN, "data": "0.25", "fixed": "0.25"}[part]
    ax.text(0.0, 1.02, f"{num}", transform=ax.transAxes, ha="left", va="bottom", fontsize=15, weight="bold",
            color="white", bbox=dict(boxstyle="circle,pad=0.25", fc=col, ec="none"))
    ax.text(0.045, 1.02, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, weight="bold")
    tag = {"frozen": "frozen in E04", "trained": "trained", "data": "data", "fixed": "no weights"}[part]
    nx.text(0.02, 1.02, f"{tag}  |  output {shape}", transform=nx.transAxes, ha="left", va="bottom", fontsize=11,
            weight="bold", color=col, family="monospace")


def notes(ax, lines):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.97
    for ln in lines:
        sub = ln.startswith("  ")
        ax.text(0.06 if sub else 0.02, y, ("– " if sub else "• ") + ln.strip(), ha="left", va="top", fontsize=9.6,
                wrap=True, transform=ax.transAxes)
        y -= 0.105 if sub else 0.115


# ---------------------------------------------------------------- steps shared by all models
def step_input(ax, nx, M, num):
    canvas(ax)
    nb = len(M["bands"])
    header(ax, nx, num, "Input: one tile, 12 monthly Sentinel-2 composites", "data", f"[12, {nb}, 128, 128]")
    s = 17
    cloudy = {2, 3}
    for k in range(12):
        x, y = 3 + k * 2.6, 6 + k * 1.1
        image(ax, x, y, s, smooth_field(32), "YlGn", z=2 + k * 0.01)
        if M["key"] == "tessera" and k in cloudy:
            ax.add_patch(Rectangle((x, y), s, s, fc="0.85", ec="0.4", hatch="////", alpha=0.9, zorder=2 + k * 0.01 + 0.005))
        label(ax, x + 1.2, y - 0.3, MONTHS[k], fs=7, color="0.2")
    label(ax, 25, 4.2, "12 months of the agricultural year (S ... A), 128 x 128 px = 1.28 km", fs=9)
    bx, h = 62, min(3.8, 25.0 / nb)
    for j, b in enumerate(M["bands"]):
        y = 9 + j * h
        ax.add_patch(Rectangle((bx, y), 14, h * 0.85, fc=plt.get_cmap("Blues")(0.25 + 0.6 * j / nb), ec="0.3", lw=0.8))
        ax.text(bx + 7, y + h * 0.42, b, ha="center", va="center", fontsize=8.5 if nb <= 8 else 7.2)
    label(ax, bx + 7, 8.3, f"{nb} bands per month\n({M['band_note']})", fs=8.5)
    arrow(ax, (47, 20), (bx - 1, 20))
    ax.text(80, 22, M["norm"], ha="left", va="center", fontsize=8.2, color="0.25")
    notes(nx, INPUT_NOTES[M["key"]])


def step_upernet(ax, nx, M, num):
    canvas(ax)
    lv, out = M["levels"], M["levels"][0]
    header(ax, nx, num, "UPerNet head: combine the four sizes -> class scores", "trained", f"[2, {out}, {out}]")
    s, xs = 10, dict(zip(lv, [4, 22, 40, 58]))
    for n, x in xs.items():
        grid(ax, x, 24, s, n, C_TRAIN, fc="#fef5e7", lw=0.3 if n >= 32 else 0.4)
        label(ax, x + s / 2, 23.5, f"{n} x {n}", fs=8.5, color="black")
    box(ax, 72, 25, 14, 8, f"pyramid pooling\naverage to 1 x 1,\n2 x 2, 4 x 4 -> back\nto {lv[3]} x {lv[3]}, concatenate",
        "#fdebd0", C_TRAIN, fs=7.6)
    arrow(ax, (68.3, 29), (71.7, 29), color=C_TRAIN)
    label(ax, 79, 24.3, f"= new {lv[3]} x {lv[3]} level, start\nof the top-down path", fs=7.6, color=C_TRAIN)
    for a, b in zip(lv[::-1][:-1], lv[::-1][1:]):
        arrow(ax, (xs[a] + 1, 36.5), (xs[b] + s - 1, 36.5), color=C_TRAIN, rad=0.35)
        ax.text((xs[a] + xs[b] + s) / 2, 39.2, "upsample + add", ha="center", va="center", fontsize=7.5, color=C_TRAIN)
    for x in xs.values():
        arrow(ax, (x + s / 2, 22.5), (30, 14.5), color=C_TRAIN, lw=1.0)
    box(ax, 22, 8, 17, 6, f"all 4 upsampled to {out} x {out},\nconcatenated (4 x 128),\n3 x 3 conv -> 128", "#fdebd0",
        C_TRAIN, fs=7.8 if out < 128 else 7.2)
    box(ax, 46, 8.5, 13, 5, "1 x 1 conv\n128 -> 2 classes", "#fdebd0", C_TRAIN, fs=8)
    arrow(ax, (39.5, 11), (45.8, 11), color=C_TRAIN)
    p = smooth_field(32, k=8)
    image(ax, 66, 2.5, 10, p, "Blues")
    image(ax, 79, 2.5, 10, 1 - p, "Reds")
    label(ax, 71, 2.0, "score irrigated", fs=8)
    label(ax, 84, 2.0, "score rainfed", fs=8)
    arrow(ax, (59.5, 11), (65.5, 9), color=C_TRAIN)
    m = 1280 // out
    label(ax, 91, 9.5, f"{out} x {out}\n= {m} m\ncells", fs=8.5, ha="left", va="center", color="black")
    coarse = {32: "4 x 4", 64: "8 x 8", 128: "16 x 16"}[out]
    notes(nx, ["UPerNet (Xiao et al. 2018) merges the four sizes in three moves:",
               f"  pyramid pooling: the coarsest map ({coarse}) is averaged to 1 x 1, 2 x 2 and 4 x 4 cells, giving tile-wide context;",
               f"  top-down path: each coarser map is upsampled and added to the next finer one, so fine maps get context;",
               f"  fusion: all four are upsampled to {out} x {out}, stacked and mixed by a 3 x 3 convolution.",
               f"A 1 x 1 convolution turns the 128 values of each {m} m cell into 2 class scores (irrigated, rainfed).",
               f"Trained: {M['head_w']} weights" + (" (UPerNet only; AgriFM brings its own pyramid)." if M["kind"] == "agrifm"
                                                    else ".")])


def step_output(ax, nx, M, num):
    canvas(ax)
    up, out = M["up"], M["levels"][0]
    title = "Upsampling to 10 m and the loss" if up > 1 else "Class per 10 m pixel and the loss"
    header(ax, nx, num, title, "fixed", "[2, 128, 128]")
    p = smooth_field(32, k=8)
    full = np.clip(zoom(p, 4, order=1), 0, 1)
    if up > 1:
        coarse = full[::up, ::up]
        image(ax, 4, 8, 22, coarse, "RdBu")
        label(ax, 15, 7.4, f"scores on {out} x {out} ({1280 // out} m)", fs=8.5)
        arrow(ax, (27, 19), (35.5, 19))
        ax.text(31.2, 21.5, f"bilinear\nx {up}", ha="center", va="bottom", fontsize=8.5, color=GREY)
    else:
        ax.text(15, 19, "no upsampling:\nthe head already\nworks on the\n128 x 128 pixels", ha="center", va="center",
                fontsize=9, color=GREY)
        arrow(ax, (27, 19), (35.5, 19))
    image(ax, 36, 8, 22, full, "RdBu")
    label(ax, 47, 7.4, "scores on 128 x 128 (10 m)", fs=8.5)
    cls = (full > 0.5).astype(float)
    image(ax, 68, 8, 22, cls, "RdBu", vmin=-0.3, vmax=1.3)
    for _ in range(40):
        r, c = rng.integers(10, 118, 2)
        ax.plot(68 + (c + 0.5) * 22 / 128, 8 + 22 - (r + 0.5) * 22 / 128, "o", ms=3.2, mfc="white",
                mec=C_IRR if cls[r, c] > 0.5 else C_RAIN, mew=1.1, zorder=5)
    arrow(ax, (59, 19), (67.5, 19))
    label(ax, 79, 7.4, "class per pixel; circles = labelled\npixels used in the loss", fs=8.5)
    for k, (c, t) in enumerate([(C_IRR, "irrigated"), (C_RAIN, "rainfed")]):
        ax.add_patch(Rectangle((92, 26 - k * 3), 1.6, 1.6, fc=c, ec="none"))
        ax.text(94.2, 26.8 - k * 3, t, ha="left", va="center", fontsize=8.5)
    first = (f"The {out} x {out} scores are interpolated bilinearly (no weights) to 128 x 128: one score pair per 10 m pixel."
             if up > 1 else "The head's output already has one score pair per 10 m pixel; nothing is interpolated.")
    last = ("Decisions are made per 10 m pixel, but each pixel's features came from its own series only: the head adds the neighbours."
            if up == 1 else f"Because decisions are made on {1280 // out} m cells and smoothed, field edges rely on the head.")
    notes(nx, [first, "Training: class-weighted cross-entropy only at the labelled parcel pixels of the training folds.",
               "The encoder stays frozen (E04): only the head learns.",
               "Prediction: the larger score gives the class of each pixel; the map is the classes of all tiles.", last])


# ---------------------------------------------------------------- image ViTs: encoder, L-TAE, pyramid
def step_vit_encoder(ax, nx, M, num):
    canvas(ax)
    col, n, P, nb, D = M["color"], M["n"], M["patch"], len(M["bands"]), M["dim"]
    header(ax, nx, num, f"{M['title']} encoder: patches -> tokens -> transformer", "frozen", f"[12, {D}, {n}, {n}]")
    groups = M["groups"]
    light = matplotlib.colors.to_rgba(col, 0.12)
    if len(groups) == 3:
        for p, g in enumerate(groups):
            x0 = 2 + p * 20.5
            ax.add_patch(FancyBboxPatch((x0, 22), 19, 15.5, boxstyle="round,pad=0.2,rounding_size=0.8", fc=light, ec=col))
            ax.text(x0 + 9.5, 36.6, f"pass {p + 1}: {' '.join(MONTHS[g[0]:g[-1] + 1])}", ha="center", va="center",
                    fontsize=8.5, color=col, weight="bold")
            for k in range(4):
                grid(ax, x0 + 0.8 + k * 4.5, 27.5, 4, n, col, hi=(0, 0) if (p == 0 and k == 0) else None, lw=0.4)
            label(ax, x0 + 9.5, 27, f"4 months x {n} x {n} patches\n= {4 * n * n} tokens", fs=7.5)
    else:
        for k in range(12):
            x0 = 2 + k * 5.1
            ax.add_patch(FancyBboxPatch((x0, 25), 4.4, 10.5, boxstyle="round,pad=0.15,rounding_size=0.5", fc=light,
                                        ec=col, lw=0.8))
            ax.text(x0 + 2.2, 34.4, MONTHS[k], ha="center", va="center", fontsize=8, color=col, weight="bold")
            grid(ax, x0 + 0.3, 27.5, 3.8, n, col, hi=(0, 0) if k == 0 else None, lw=0.3)
        label(ax, 32, 24.4, f"12 passes, one month each: {n} x {n} patches = {n * n} tokens per pass", fs=8)
    for j in range(min(nb, 6) - 1, -1, -1):                        # band layers of one patch
        ax.add_patch(Rectangle((3 + 0.5 * j, 7 - 0.5 * j), 8, 8, fc=col if j == 0 else light, ec=col,
                               lw=1 if j == 0 else 0.6, alpha=0.7 if j == 0 else 1, zorder=2 + (5 - j) * 0.01))
    label(ax, 8, 4.0, f"one patch: {P} x {P} px x {nb} bands\n= {P * P * nb:,} numbers", fs=8)
    arrow(ax, (3.4, 27.3 if len(groups) == 3 else 25.2), (6, 15.3), color=col, rad=0.25)
    box(ax, 16, 6, 12, 5.5, M["proj"], "#ecf0f1", C_LOCK, fs=7.8)
    arrow(ax, (12.5, 8.7), (15.8, 8.7))
    vector(ax, 31.5, 2.5, 2.2, 13, 24, "Blues", col)
    label(ax, 32.6, 2.2, f"token\n{D:,}", fs=8)
    box(ax, 30, 17, 13, 3.4, M["extra"], "#ecf0f1", C_LOCK, fs=7.8)
    arrow(ax, (28.3, 8.7), (31.3, 8.7))
    box(ax, 38, 5, 18, 8, M["tf"], "#ecf0f1", C_LOCK, fs=8.1)
    arrow(ax, (34.2, 8.8), (37.8, 8.8))
    for k in range(12):
        grid(ax, 64 + (k % 6) * 5.6, 18 + (1 - k // 6) * 7.2, 4.6, n, col, fc="#f4f6f8", lw=0.3)
        label(ax, 64 + (k % 6) * 5.6 + 2.3, 17.7 + (1 - k // 6) * 7.2, MONTHS[k], fs=7, color="0.2")
    arrow(ax, (56.5, 9), (63.5, 17), rad=-0.2)
    label(ax, 80.5, 13.0, f"12 maps of {n} x {n} tokens:\none {D:,}-value vector per\n{P * 10} m patch and month\n(the class token is dropped)",
          fs=8.5)
    lines = {
        "prithvi": ["Prithvi-EO-2.0-300M-TL was pretrained on 4 dates per sample, so the 12 months go in as 3 passes of 4.",
                    "Each month is cut into 8 x 8 patches of 16 x 16 px (160 m); each patch of one month is one token.",
                    "A token starts as the patch's 1,536 numbers (16 x 16 px x 6 bands), linearly projected to 1,024: the model's width (ViT-Large).",
                    "Date and location embeddings are added; 24 transformer layers let each token look at all 256 tokens of its pass (all patches, 4 months).",
                    "Output: per month an 8 x 8 map, each cell a 1,024-value vector -> 12 x 1,024 x 8 x 8."],
        "clay": ["Clay v1.5 encodes one date per pass, so the 12 months are 12 separate passes.",
                 "Each month is cut into 16 x 16 patches of 8 x 8 px (80 m): the finest patch grid of the image models.",
                 "A token starts as the patch's 640 numbers (8 x 8 px x 10 bands). The projection weights are generated from each band's central wavelength, so any sensor's bands can be read.",
                 "Week of year, location and pixel size (10 m) are encoded and added; 24 transformer layers (ViT-Large, width 1,024) let each token attend to the 256 tokens of its month.",
                 "Output: per month a 16 x 16 map of 1,024-value vectors -> 12 x 1,024 x 16 x 16."],
        "terramind": ["TerraMind can read nine modalities; here only Sentinel-2 L2A (S2L2A) is given (D14), one month per pass.",
                      "Each month is cut into 8 x 8 patches of 16 x 16 px (160 m).",
                      "A token starts as the patch's 3,072 numbers (16 x 16 px x 12 bands), linearly projected to 768: the width of ViT-Base.",
                      "No date or location is given: the encoder does not know which month it sees; only the L-TAE orders the months.",
                      "12 transformer layers; output per month an 8 x 8 map of 768-value vectors -> 12 x 768 x 8 x 8."],
    }[M["key"]]
    notes(nx, lines + [f"  Frozen: {M['enc_w']} encoder weights. One vector describes all {P * P} pixels of its patch."])


def step_ltae(ax, nx, M, num):
    canvas(ax)
    col, n, D = M["color"], M["n"], M["dim"]
    header(ax, nx, num, "L-TAE: the 12 months of each patch -> one vector", "trained", f"[256, {n}, {n}]")
    grid(ax, 2, 18, 15, n, col, fc="#f4f6f8", hi=(2, 5), hc=C_TRAIN, lw=0.5)
    label(ax, 9.5, 17.3, f"the same patch position\nin all 12 months\n(done for all {n * n})", fs=8)
    cloudy = {2, 3}
    for k in range(12):
        x = 22 + k * 2.6
        vector(ax, x, 18, 1.8, 15, 20, "Blues", col, hatch="////" if k in cloudy else None)
        label(ax, x + 0.9, 17.5, MONTHS[k], fs=7.5, color="0.2")
    label(ax, 37, 39.5, f"12 vectors of {D:,} (hatched = no clear pixel in the patch that month: masked)", fs=8)
    arrow(ax, (17.5, 25), (21.5, 25), color=C_TRAIN)
    box(ax, 22, 6.5, 30, 6.5, f"{D:,} -> 256 (linear) + month position\n16 heads, each with a learned query",
        "#fdebd0", C_TRAIN, fs=8.3)
    arrow(ax, (37, 17), (37, 13.3), color=C_TRAIN)
    w = np.array([0.02, 0.05, 0, 0, 0.10, 0.22, 0.25, 0.14, 0.07, 0.05, 0.06, 0.04])
    ax0, ay0 = 58, 18
    for k, v in enumerate(w / w.sum()):
        ax.add_patch(Rectangle((ax0 + k * 2.2, ay0), 1.6, v * 50, fc=C_TRAIN, ec="none"))
        label(ax, ax0 + k * 2.2 + 0.8, ay0 - 0.4, MONTHS[k], fs=7, color="0.2")
    ax.plot([ax0 - 0.5, ax0 + 26.5], [ay0, ay0], color="0.3", lw=0.8)
    label(ax, ax0 + 13, 36.5, "attention weights of one head\n(sum to 1; masked months get 0)", fs=8)
    arrow(ax, (52.5, 10), (57.5, 18), color=C_TRAIN, rad=0.2)
    box(ax, 60, 4.5, 18, 6.5, "weighted sum per head\n16 heads x 16 = 256\n-> MLP 256 -> 256", "#fdebd0", C_TRAIN, fs=8)
    arrow(ax, (72, 17.5), (70, 11.3), color=C_TRAIN)
    vector(ax, 81, 2.5, 2.2, 12, 16, "Oranges", C_TRAIN)
    label(ax, 82.1, 2.1, "256", fs=8)
    arrow(ax, (78.3, 8), (80.7, 8), color=C_TRAIN)
    grid(ax, 88, 4, 11, n, C_TRAIN, fc="#fdebd0", hi=(2, 5), hc=C_TRAIN, lw=0.4)
    arrow(ax, (83.6, 8.5), (87.6, 9.5), color=C_TRAIN)
    label(ax, 93.5, 3.4, f"{n} x {n} map,\n256 per cell", fs=8)
    notes(nx, [f"The L-TAE (lightweight temporal attention encoder, Garnot & Landrieu 2020) runs separately at each of the {n * n} patch positions.",
               f"Each month's {D:,}-value vector is reduced to 256 values and tagged with its month's position in the year.",
               "16 heads each hold one learned query; a head scores the 12 months and turns the scores into weights (softmax).",
               "Months without any clear pixel in the patch are masked (weight 0), so clouds cannot leak in.",
               "Each head takes the weighted sum of its 16 values; the 16 heads give 256 values, refined by a small MLP.",
               f"  Trained: {M['ltae_w']} weights. The weights are learned from the labels (e.g. summer, when only irrigated fields stay green)."])


def step_vit_pyramid(ax, nx, M, num):
    canvas(ax)
    n, lv = M["n"], M["levels"]
    header(ax, nx, num, f"Feature pyramid: one {n} x {n} map -> four map sizes", "trained", "4 maps, 128 channels each")
    s = 15
    grid(ax, 3, 13, s, n, C_TRAIN, fc="#fdebd0", lw=0.5)
    label(ax, 3 + s / 2, 12.4, f"L-TAE map\n{n} x {n} x 256", fs=8.5)
    hows = ["x 4: two transposed\nconvolutions (2 x 2)", "x 2: one transposed\nconvolution (2 x 2)",
            "x 1: 1 x 1 convolution", "x 1/2: 2 x 2 max-pool\n+ 1 x 1 convolution"]
    for i, (c, how) in enumerate(zip(lv, hows)):
        x = 30 + i * 18
        grid(ax, x, 13, s, c, C_TRAIN, fc="#fef5e7", lw=0.45 if c < 32 else 0.3)
        label(ax, x + s / 2, 12.4, f"{c} x {c} cells = {1280 // c} m", fs=8.5, color="black")
        label(ax, x + s / 2, 36.5, how, fs=8)
        arrow(ax, (18.5, 20.5), (x - 0.5, 29.5 if i else 28.5), color=C_TRAIN, rad=-0.25 + 0.06 * i)
    label(ax, 66, 6.5, "all four maps cover the same 1.28 km tile; only the cell size differs", fs=8.5)
    notes(nx, [f"A plain ViT gives one map size only ({n} x {n}). Segmentation heads work best with several sizes: fine maps for edges, coarse maps for context.",
               "The simple pyramid of ViTDet (Li et al. 2022), as used for ViTs in PANGAEA, builds four sizes from the one map:",
               "  x 4 and x 2 by learned transposed convolutions (they learn how to spread a cell over finer cells),",
               "  x 1 by a 1 x 1 convolution, x 1/2 by max-pooling 2 x 2 cells into one.",
               f"All four get 128 channels; the finest ({lv[0]} x {lv[0]}, {1280 // lv[0]} m) is where the head decides. Trained: {M['pyr_w']} weights.",
               "  The upsampled maps are learnt from the coarse map: they add no new spatial detail."])


# ---------------------------------------------------------------- AgriFM
def step_agrifm_patches(ax, nx, M, num):
    canvas(ax)
    col = M["color"]
    header(ax, nx, num, "AgriFM: 3-D patches of 4 months x 4 x 4 px", "frozen", "3 x 32 x 32 tokens of 128")
    for t in range(3):                                              # 3 blocks of 4 months
        x0 = 3 + t * 19
        ax.add_patch(FancyBboxPatch((x0, 21), 17, 16.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc=matplotlib.colors.to_rgba(col, 0.08), ec=col))
        ax.text(x0 + 8.5, 36.6, f"months {' '.join(MONTHS[4 * t:4 * t + 4])}", ha="center", va="center", fontsize=8.5,
                color=col, weight="bold")
        for k in range(4):
            grid(ax, x0 + 2 + k * 0.9, 23.5 + k * 0.9, 9, 32, col, fc="#fbf3f3", lw=0.25, z=2 + k * 0.01)
        label(ax, x0 + 8.5, 23.0, "32 x 32 blocks of\n4 months x 4 x 4 px", fs=7.5)
    for j in range(5, -1, -1):                                      # one 3-D patch
        ax.add_patch(Rectangle((4 + 0.6 * j, 6 - 0.6 * j), 7, 7, fc=col if j == 0 else "#f6dada", ec=col,
                               lw=0.8, alpha=0.7 if j == 0 else 1, zorder=2 + (5 - j) * 0.01))
    label(ax, 8, 2.6, "one token: 4 months x 4 x 4 px\nx 10 bands = 640 numbers", fs=8)
    arrow(ax, (5, 22.5), (6, 13.5), color=col, rad=0.25)
    box(ax, 19, 5.5, 13, 5, "3-D projection\n640 -> 128", "#ecf0f1", C_LOCK, fs=8.3)
    arrow(ax, (14.5, 8), (18.8, 8))
    vector(ax, 35, 2.5, 2.2, 12, 16, "Reds", col)
    label(ax, 36.1, 2.1, "128", fs=8)
    arrow(ax, (32.5, 8), (34.8, 8))
    for t in range(3):
        grid(ax, 66 + t * 1.6, 12 + t * 1.6, 14, 32, col, fc="#fbf3f3", lw=0.25, z=2 + t * 0.01)
    arrow(ax, (38, 9), (65, 15), rad=-0.15)
    label(ax, 74, 11.3, "3 time steps x 32 x 32 tokens\n(3,072 tokens, 128 values each)", fs=8.5)
    notes(nx, ["AgriFM is a video Swin transformer: it reads the whole year in one pass, as a short video of 12 frames.",
               "Its pretrained patch embedding cuts the stack into 3-D blocks of 4 months x 4 x 4 px (40 m), each 640 numbers (with 10 bands).",
               "Each block is projected to 128 values: 3 time steps x 32 x 32 positions = 3,072 tokens.",
               "Bands are standardised with the statistics of the region's own tiles (D15), as AgriFM was fine-tuned that way.",
               f"  Frozen: {M['enc_w']} encoder weights; no L-TAE is needed, because time is reduced inside the network."])


def step_agrifm_stages(ax, nx, M, num):
    canvas(ax)
    col = M["color"]
    header(ax, nx, num, "Swin stages: space and time shrink -> four maps", "frozen",
           "384 x 32², 512 x 16², 512 x 8², 1,024 x 4²")
    specs = [(32, "patch embedding\n(no transformer yet)", "3 steps x 128\n= 384 channels"),
             (16, "stage 1: 2 blocks,\nthen merge 2 x 2 x 2", "2 steps x 256\n= 512 channels"),
             (8, "stage 2: 2 blocks,\nthen merge 2 x 2 x 2", "1 step x 512\n= 512 channels"),
             (4, "stage 3: 18 blocks,\nthen merge 2 x 2 x 2", "1 step x 1,024\n= 1,024 channels")]
    for i, (c, how, ch) in enumerate(specs):
        x = 4 + i * 23
        grid(ax, x, 13, 15, c, col, fc="#fbf3f3", lw=0.3 if c >= 32 else 0.45)
        label(ax, x + 7.5, 12.4, f"{c} x {c} = {1280 // c} m\n{ch}", fs=8.2, color="black")
        label(ax, x + 7.5, 36.8, how, fs=8)
        if i:
            arrow(ax, (x - 7.5, 20.5), (x - 0.5, 20.5), color=col)
    label(ax, 50, 3.2, "each map: the remaining time steps are stacked as channels; stage 4 (2 more blocks at 4 x 4) runs but is not used",
          fs=8)
    notes(nx, ["Each stage applies Swin blocks: attention inside local windows (8 time steps x 7 x 7 tokens), shifted between blocks.",
               "After each stage, 2 x 2 x 2 neighbouring tokens are merged: space halves (32 -> 16 -> 8 -> 4) and time halves (3 -> 2 -> 1), averaging the time steps.",
               "The four maps that go to the head are the patch embedding and the outputs of stages 1-3, with their time steps stacked as channels.",
               "So AgriFM delivers its own four map sizes: no L-TAE and no extra pyramid are needed.",
               "  The finest map (32 x 32, 40 m) is the patch embedding itself, before any attention: fine detail comes with little context."])


# ---------------------------------------------------------------- pixel models
def step_pixel_encoder(ax, nx, M, num):
    canvas(ax)
    col, presto = M["color"], M["key"] == "presto"
    header(ax, nx, num, f"{M['title']}: one encoder run per pixel", "frozen", "[128, 128, 128]")
    grid(ax, 2, 20, 15, 16, "0.6", fc="#f3efe6", hi=(5, 9), hc=col, lw=0.25)
    label(ax, 9.5, 19.3, "one pixel of the tile\n(16,384 runs per tile)", fs=8)
    arrow(ax, (17.5, 27), (21.5, 27), color=col)
    if presto:
        groups = ["RGB", "red edge", "NIR 10 m", "NIR 20 m", "SWIR", "NDVI"]
        for g, name in enumerate(groups):
            y = 34 - g * 2.4
            ax.text(29.5, y + 1, name, ha="right", va="center", fontsize=7.3)
            for k in range(12):
                ax.add_patch(Rectangle((30 + k * 2.2, y), 2.0, 2.1, fc=matplotlib.colors.to_rgba(col, 0.25 + 0.5 * rng.random()),
                                       ec=col, lw=0.4))
        for k in range(12):
            label(ax, 31 + k * 2.2, 19.4, MONTHS[k], fs=7, color="0.2")
        ax.add_patch(Rectangle((58, 29), 2.4, 2.1, fc="0.75", ec="0.3", lw=0.6))
        label(ax, 59.2, 28.6, "location\ntoken", fs=7.3)
        label(ax, 43, 39.8, "6 band groups x 12 months = 72 tokens (+ 1 location)", fs=8)
        box(ax, 22, 5.5, 28, 7, "each token: linear embedding of its\ngroup's bands + month, group and\nposition encodings (128 values)",
            "#ecf0f1", C_LOCK, fs=7.8)
        box(ax, 54, 5.5, 18, 7, "2 transformer layers\n(width 128, 8 heads)\nall 73 tokens attend\nto each other", "#ecf0f1",
            C_LOCK, fs=7.8)
        pool = "mean of the\ntokens -> 128"
    else:
        cloudy = {2, 3}
        clear = [k for k in range(12) if k not in cloudy]
        n_c, size = len(clear), 16                                  # _pad_pattern of pixelwise_run/encoders.py
        fill = [g[len(g) // 2] for g in np.array_split(np.arange(n_c), size - n_c)]
        slots = clear + [clear[i] for i in fill]
        for i, k in enumerate(slots):
            x = 22 + i * 2.35
            rep = i >= len(clear)
            ax.add_patch(Rectangle((x, 24), 2.1, 9, fc=matplotlib.colors.to_rgba(col, 0.25 if rep else 0.6), ec=col,
                                   lw=0.6, ls="--" if rep else "-"))
            ax.text(x + 1.05, 28.5, MONTHS[k], ha="center", va="center", fontsize=7.5)
        label(ax, 22 + 8 * 2.35, 39.8, "10 clear months (N, D cloudy: left out), padded to 16 by repeating months (dashed)",
              fs=8)
        label(ax, 22 + 8 * 2.35, 23.4, "one token per observation: 10 bands + day of year", fs=8)
        box(ax, 22, 5.5, 26, 7, "each token: MLP 10 bands -> 256\n+ day-of-year encoding", "#ecf0f1", C_LOCK, fs=8)
        box(ax, 52, 5.5, 20, 7, "4 transformer layers\n(width 256, 4 heads),\nthen attention pooling\n-> 256", "#ecf0f1",
            C_LOCK, fs=7.8)
        pool = "join empty S1\nbranch: 512\n-> MLP -> 128"
    arrow(ax, (36, 18.6), (36, 12.8))
    arrow(ax, (50.4, 9), (53.8, 9))
    box(ax, 75, 5.5, 11, 7, pool, "#ecf0f1", C_LOCK, fs=7.3)
    arrow(ax, (72.4, 9), (74.8, 9))
    vector(ax, 88, 3, 2.2, 12, 16, "Oranges" if presto else "copper", col)
    label(ax, 89.1, 2.6, "128", fs=8)
    image(ax, 84, 22, 13, smooth_field(64, k=8), "YlOrBr")
    arrow(ax, (89, 15.5), (89.5, 21.5))
    label(ax, 90.5, 39.8, "all pixels:\n128 x 128 map,\n128 per pixel", fs=8)
    if presto:
        lines = ["Presto encodes each 10 m pixel on its own: 16,384 runs per tile; neighbouring pixels are never seen together.",
                 "The pixel's 12 months are cut into tokens by band group and month: 6 groups x 12 months = 72 tokens, plus one location token.",
                 "The Sentinel-1, ERA5, SRTM and Dynamic World tokens Presto was pretrained with are masked (D14) and removed.",
                 "Cloudy months are filled by linear interpolation in time, because Presto needs the same tokens for every pixel of a batch.",
                 "2 transformer layers (width 128); the mean of the tokens is the pixel's 128-value embedding.",
                 f"  Frozen: {M['enc_w']} encoder weights. Output: a 128 x 128 map with 128 values per 10 m pixel."]
    else:
        lines = ["TESSERA encodes each 10 m pixel on its own: 16,384 runs per tile.",
                 "Only the pixel's clear months are used, each as one token: 10 bands plus its day of year (the 15th of the month).",
                 "The tokens are padded by repeating months, or subsampled, to the size TESSERA was trained with: 8 (1-8 clear months) or 16 (9-12).",
                 "4 transformer layers (width 256); attention pooling gives one 256-value vector. The Sentinel-1 branch gets an empty input (D14).",
                 "Both branches are joined and reduced by an MLP to the pixel's 128-value embedding.",
                 f"  Frozen: {M['enc_w']} encoder weights. Output: a 128 x 128 map with 128 values per 10 m pixel."]
    notes(nx, lines)


def step_pixel_pyramid(ax, nx, M, num):
    canvas(ax)
    header(ax, nx, num, "Pyramid by average pooling: 10 m map -> four sizes", "fixed", "128², 64², 32², 16² (128 channels)")
    s = 15
    base = smooth_field(128, k=8)
    image(ax, 3, 13, s, base, "YlOrBr")
    label(ax, 3 + s / 2, 12.4, "encoder map\n128 x 128 x 128", fs=8.5)
    for i, (c, how) in enumerate(zip(M["levels"], ["x 1: as it is", "average of 2 x 2 pixels", "average of 4 x 4 pixels",
                                                   "average of 8 x 8 pixels"])):
        x = 30 + i * 18
        f = base.reshape(c, 128 // c, c, 128 // c).mean(axis=(1, 3))
        image(ax, x, 13, s, f, "YlOrBr")
        label(ax, x + s / 2, 12.4, f"{c} x {c} = {1280 // c} m", fs=8.5, color="black")
        label(ax, x + s / 2, 36.5, how, fs=8)
        arrow(ax, (18.5, 20.5), (x - 0.5, 29.5 if i else 28.5), color=GREY, rad=-0.25 + 0.06 * i)
    notes(nx, ["The pixel encoders give one map at 10 m, so the four sizes are made by plain average pooling (no weights):",
               "  x 1 (10 m), x 1/2 (20 m), x 1/4 (40 m), x 1/8 (80 m).",
               "The coarser maps give each pixel the context of its neighbourhood, which the per-pixel encoder never saw.",
               "Unlike the image ViTs, the finest level here is real 10 m information, not a learnt upsampling."])


# ---------------------------------------------------------------- assembling one figure
STEPS = {"vit": [step_input, step_vit_encoder, step_ltae, step_vit_pyramid, step_upernet, step_output],
         "agrifm": [step_input, step_agrifm_patches, step_agrifm_stages, step_upernet, step_output],
         "pixel": [step_input, step_pixel_encoder, step_pixel_pyramid, step_upernet, step_output]}


def draw(key):
    M = dict(SPEC[key], key=key)
    steps = STEPS[M["kind"]]
    fig = plt.figure(figsize=(17, 4.85 * len(steps)))
    gs = fig.add_gridspec(len(steps), 2, width_ratios=[1.55, 1], hspace=0.32, wspace=0.04,
                          top=1 - 0.26 / len(steps), bottom=0.015, left=0.02, right=0.98)
    fig.suptitle(f"{M['title']} in this study: one 128 x 128 px tile (1.28 km), 12 months, step by step",
                 fontsize=17, weight="bold", y=0.995)
    fig.text(0.5, 1 - 0.13 / len(steps), "Shapes are [channels or months, height, width] for one tile; grey = frozen "
             "encoder, orange = trained head; the maps are made-up illustrations (pixelwise_run/tile_common.py, "
             "GeoFMSegmenter)", ha="center", fontsize=10.5, color="0.3")
    for i, f in enumerate(steps):
        f(fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), M, i)
    out = OUT_DIR / f"{key}_pipeline_steps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    for k in MODELS_TO_DRAW:
        draw(k)
