"""
a8_geofm_tokens_figure.py  —  how each foundation model cuts one 128 px tile and its 12 months into tokens
=========================================================================================================

One row per model: (left) the 128 x 128 px tile with the unit the encoder sees (patch grid or single pixel),
(right) the 12 months and how they are grouped per encoder pass. Values are those of pixelwise_run/encoders.py
and tile_common.GeoFMSegmenter. Writes docs/figures/geofm_tokens.png (Foundation models page).

Run (Mac):
    python tools/a8_geofm_tokens_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = Path(__file__).resolve().parents[1]
OUT = R / "docs" / "figures" / "geofm_tokens.png"

# ================================ SETTINGS ================================
# name, patch (px; 1 = pixel), months per pass (groups), colour, bands, text of the token panel, text of the month panel
MODELS = [
    ("Prithvi-EO-2.0", 16, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], "#1f77b4",
     "6 bands (B2 B3 B4 B8A B11 B12)",
     "16 px patches → 8 × 8 = 64 tokens per month\n1024 values per token",
     "3 passes of 4 months (with date + location);\none token set per month → L-TAE"),
    ("Clay v1.5", 8, [[m] for m in range(12)], "#2ca02c",
     "10 bands (B2 B3 B4 B5 B6 B7 B8 B8A B11 B12)",
     "8 px patches → 16 × 16 = 256 tokens per month\n1024 values per token; each band embedded\nby its wavelength",
     "12 passes of 1 month (with week, location, GSD);\nmonths combined by L-TAE"),
    ("TerraMind", 16, [[m] for m in range(12)], "#9467bd",
     "12 bands (B1 B2 B3 B4 B5 B6 B7 B8 B8A B9\nB11 B12)",
     "16 px patches → 8 × 8 = 64 tokens per month\n768 values per token (modality S2L2A)",
     "12 passes of 1 month (no date input);\nmonths combined by L-TAE"),
    ("AgriFM", 4, [list(range(12))], "#d62728",
     "10 bands (B2 B3 B4 B5 B6 B7 B8 B8A B11 B12)",
     "4 px × 4 months 3-D patches → 32 × 32 × 3 tokens\nmerged by Swin stages to 16², 8², 4² maps",
     "1 pass of all 12 months; time is reduced inside\nthe network (no L-TAE needed)"),
    ("Presto", 1, [list(range(12))], "#ff7f0e",
     "10 bands + NDVI (B2 B3 B4 B5 B6 B7 B8 B8A\nB11 B12 + NDVI)",
     "1 pixel at a time: 16,384 pixels per tile\n6 band groups × 12 months = 72 tokens\n128 values per pixel",
     "1 pass of the pixel's 12 months (tokens per\nband group and month)"),
    ("TESSERA v2", 1, [list(range(12))], "#8c564b",
     "10 bands (B4 B2 B3 B8 B8A B5 B6 B7 B11 B12)",
     "1 pixel at a time: 16,384 pixels per tile\n128 values per pixel",
     "1 pass of the pixel's clear months, each dated\nby day of year (cloudy months left out)"),
]
MONTHS = ["S", "O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A"]   # Morocco agricultural year
# ==========================================================================


def tile(ax, patch, color):
    ax.add_patch(Rectangle((0, 0), 128, 128, fc="#f2efe6", ec="0.3", lw=1))
    if patch > 1:
        for k in range(0, 129, patch):
            ax.plot([k, k], [0, 128], color=color, lw=0.5 if patch < 8 else 0.8, alpha=0.8)
            ax.plot([0, 128], [k, k], color=color, lw=0.5 if patch < 8 else 0.8, alpha=0.8)
        ax.add_patch(Rectangle((0, 128 - patch), patch, patch, fc=color, alpha=0.6))
        n = 128 // patch
        ax.annotate(f"one token =\n{patch} × {patch} px", (patch / 2, 128 - patch / 2), (140, 112), fontsize=7.5,
                    color=color, va="center", arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                    annotation_clip=False)
        ax.text(64, -9, f"{n} patches × {patch} px = 128 px", ha="center", va="top", fontsize=7.5, color="0.3")
    else:
        for k in range(0, 129, 8):                             # pixels too small to draw: a light 8 px grid
            ax.plot([k, k], [0, 128], color="0.85", lw=0.3)
            ax.plot([0, 128], [k, k], color="0.85", lw=0.3)
        ax.add_patch(Rectangle((0, 127), 1.6, 1.6, fc=color))
        ax.annotate("one token =\none pixel (10 m)", (1, 127), (140, 112), fontsize=7.5, color=color, va="center",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8), annotation_clip=False)
        ax.text(64, -9, "128 × 128 pixels = 128 px", ha="center", va="top", fontsize=7.5, color="0.3")
    ax.set_xlim(-2, 130)
    ax.set_ylim(-22, 130)
    ax.set_aspect("equal")
    ax.axis("off")


def months(ax, groups, color):
    for g, grp in enumerate(groups):
        x0, x1 = grp[0], grp[-1] + 1
        ax.add_patch(Rectangle((x0 + 0.08, 0.45), x1 - x0 - 0.16, 0.5, fc=color, alpha=0.25 + 0.5 * (g % 2), ec=color))
    for m, name in enumerate(MONTHS):
        ax.text(m + 0.5, 0.7, name, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.6, 1)
    ax.axis("off")


fig = plt.figure(figsize=(14, 1.95 * len(MODELS)))
gs = fig.add_gridspec(len(MODELS), 4, width_ratios=[1.0, 1.25, 3.9, 3.0], hspace=0.25, wspace=0.08)
for r, (name, patch, groups, color, t_bands, t_tile, t_month) in enumerate(MODELS):
    a = fig.add_subplot(gs[r, 0])
    a.axis("off")
    a.text(0, 0.5, name, fontsize=12, fontweight="bold", color=color, va="center")
    a = fig.add_subplot(gs[r, 1])
    tile(a, patch, color)
    a = fig.add_subplot(gs[r, 2])
    a.axis("off")
    a.text(0.12, 0.6, t_bands, fontsize=9.2, va="bottom", fontweight="bold", color=color)
    a.text(0.12, 0.55, t_tile, fontsize=9, va="top")
    a = fig.add_subplot(gs[r, 3])
    months(a, groups, color)
    a.text(6, 0.3, t_month, fontsize=8, ha="center", va="top")
fig.suptitle("What each foundation model sees of one 128 × 128 px tile (1.28 km) and its 12 monthly composites\n"
             "(tile: grid = how the tile is cut, one cell = one token, the shaded cell is labelled; months: one box = one encoder pass)",
             fontsize=11, y=0.995)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(OUT)
