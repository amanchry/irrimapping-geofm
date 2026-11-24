"""
a10_geofm_pretraining_figure.py  —  the six foundation models as published: what each was pretrained on, and how
=================================================================================================================

Left: a matrix of the data sources each model saw in pretraining (sensors and auxiliary layers). Right: the spatial
unit of a token, the time handling, the pretraining objective and the size of the encoder, as given by the papers,
model cards and code (see the Foundation models pages). This is the ORIGINAL design of each model, not the set-up of
this study. Writes docs/foundation_models/figures/geofm_pretraining.png.

Run (Mac; no data needed):
    python tools/a10_geofm_pretraining_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = Path(__file__).resolve().parents[1]
OUT = R / "docs" / "foundation_models" / "figures" / "geofm_pretraining.png"

# ================================ SETTINGS ================================
SOURCES = ["Sentinel-2", "Landsat (HLS)", "Sentinel-1", "MODIS", "Aerial (NAIP, LINZ)", "Elevation (DEM / SRTM)",
           "Weather (ERA5)", "Land cover", "NDVI", "Coordinates", "Text captions"]
# model: colour, sources used in pretraining, token unit, time, objective, encoder size
MODELS = {
    "Prithvi-EO-2.0": ("#1f77b4", {"Sentinel-2", "Landsat (HLS)", "Coordinates"},
                       "16 × 16 px patch\n(30 m HLS → 480 m)", "4 dates per sample,\ndated (year, day)",
                       "masked autoencoder\n(75% of patches hidden)", "ViT-L, 300 M"),
    "Clay v1.5": ("#2ca02c", {"Sentinel-2", "Landsat (HLS)", "Sentinel-1", "MODIS", "Aerial (NAIP, LINZ)", "Coordinates"},
                  "8 × 8 px patch,\nany sensor (0.5–500 m)", "1 date per sample,\ndated (week, hour)",
                  "masked autoencoder\n+ DINOv2 teacher", "ViT-L, 311 M"),
    "TerraMind": ("#9467bd", {"Sentinel-2", "Sentinel-1", "Elevation (DEM / SRTM)", "Land cover", "NDVI", "Coordinates",
                              "Text captions"},
                  "16 × 16 px patch\n(10 m → 160 m)", "1 date per sample,\nno date input",
                  "any-to-any masked\ntoken prediction", "base ViT, ≈ 87 M"),
    "AgriFM": ("#d62728", {"Sentinel-2", "Landsat (HLS)", "MODIS", "Land cover"},
               "3-D block:\ntime × 4 × 4 px", "image time series\n(3–32 frames)",
               "supervised: land-cover\nfractions (GLC_FCS30D)", "Video Swin, ≈ 88 M"),
    "Presto": ("#ff7f0e", {"Sentinel-2", "Sentinel-1", "Elevation (DEM / SRTM)", "Weather (ERA5)", "Land cover", "NDVI",
                           "Coordinates"},
               "one pixel:\nband group × month", "12 monthly steps",
               "masked autoencoder\n(4 masking strategies)", "2 blocks, 0.4 M"),
    "TESSERA": ("#8c564b", {"Sentinel-2", "Sentinel-1"},
                "one pixel:\none acquisition", "all acquisitions\nof a year, dated",
                "Barlow Twins\n(+ distillation in v2)", "student 1–44 M\n(teacher 2 B)"),
}
# ==========================================================================

names = list(MODELS)
fig = plt.figure(figsize=(17, 7.6))
ax = fig.add_axes([0.13, 0.08, 0.36, 0.78])
for j, name in enumerate(names):
    col, used = MODELS[name][0], MODELS[name][1]
    for i, src in enumerate(SOURCES):
        on = src in used
        ax.add_patch(FancyBboxPatch((j + 0.12, len(SOURCES) - 1 - i + 0.12), 0.76, 0.76,
                                    boxstyle="round,pad=0,rounding_size=0.12", fc=col if on else "#f2f3f4",
                                    ec="white", alpha=0.85 if on else 1))
        if on:
            ax.text(j + 0.5, len(SOURCES) - 1 - i + 0.5, "✓", ha="center", va="center", color="white", fontsize=12,
                    fontweight="bold")
    ax.text(j + 0.5, len(SOURCES) + 0.15, name, ha="left", va="bottom", rotation=35, fontsize=10.5,
            color=col, fontweight="bold")
for i, src in enumerate(SOURCES):
    ax.text(-0.1, len(SOURCES) - 1 - i + 0.5, src, ha="right", va="center", fontsize=10)
ax.set_xlim(-0.05, len(names))
ax.set_ylim(0, len(SOURCES) + 1.6)
ax.axis("off")
ax.set_title("Data seen in pretraining", fontsize=12, loc="left", x=-0.33, y=1.0)

ax2 = fig.add_axes([0.52, 0.08, 0.47, 0.78])
cols = ["one token", "time", "pretraining objective", "encoder"]
xs = [1.7, 4.1, 6.3, 8.8]
for k, c in enumerate(cols):
    ax2.text(xs[k], len(names) + 0.15, c, fontsize=11, fontweight="bold", va="bottom")
for j, name in enumerate(names):
    col, _, unit, time, obj, size = MODELS[name]
    y = len(names) - 1 - j
    ax2.text(0.0, y + 0.5, name, fontsize=10, va="center", color=col, fontweight="bold")
    ax2.add_patch(FancyBboxPatch((-0.15, y + 0.06), 11.0, 0.88, boxstyle="round,pad=0,rounding_size=0.1",
                                 fc=col, alpha=0.08, ec=col, lw=1))
    for k, t in enumerate([unit, time, obj, size]):
        ax2.text(xs[k], y + 0.5, t, fontsize=9.2, va="center")
ax2.set_xlim(-0.2, 11.0)
ax2.set_ylim(0, len(names) + 0.9)
ax2.axis("off")
ax2.set_title("How each model was built", fontsize=12, loc="left")

fig.suptitle("The six foundation models as published (original design, not the set-up of this study)", fontsize=14,
             y=0.99)
fig.text(0.5, 0.01, "Sources: Szwarcman et al. 2024 (Prithvi-EO-2.0); Clay specification v1.5; Jakubik et al. 2025 "
         "(TerraMind); Li et al. 2025 (AgriFM); Tseng et al. 2023 (Presto); Feng et al. 2025, 2026 (TESSERA). "
         "Land cover: ESRI LULC (TerraMind), Dynamic World (Presto), GLC_FCS30D labels (AgriFM).",
         ha="center", fontsize=8.5, color="0.35")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(OUT)
