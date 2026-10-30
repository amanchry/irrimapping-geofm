"""
01_prepare_labels.py  —  irrigated / rainfed reference parcels from the Moroccan field survey
=============================================================================================

What it does
------------
Turns the raw field survey (Ouassanouan et al. 2026, Scientific Data; ~10,000 parcels surveyed
Dec 2023 - Jan 2025 in Gharb, Tadla, Doukkala, El Haouz, Souss) into one clean file of labelled
cropland parcels: label 1 = irrigated, 2 = rainfed. Every later step starts from this file.

Steps
-----
1. Read the survey shapefile.
2. Clean geometry: drop parcels without geometry, repair invalid polygons, drop exact duplicates.
3. Label: any irrigation method (drip, flood, sprinkler, basin, pivot) = irrigated (1);
   "rainfed" = rainfed (2). Spelling variants are normalised (lower case, trimmed).
4. Keep cropland only (Seasonal Crop, Trees, Trees + Seasonal); bare soil, weeds and water
   are dropped, as are parcels without an irrigation value.
5. Add attributes used later:
     area_ha     parcel area (UTM 29N)
     ag_season   agricultural year of the survey date, 1 Sep - 31 Aug ("2023/24", "2024/25");
                 each parcel later gets satellite data from this year
     nn_dist_m   distance to the nearest other parcel (spatial clustering)
     wc_cropland_frac, wc_major  share of the parcel that ESA WorldCover 2021 calls cropland,
                 and its main WorldCover class (optional, CHECK_WORLDCOVER = True)
6. Save the parcels, a summary table and figures (map, class balance, parcel size, survey timing).

Output (outputs/dataset/01_labels/)
---------------------------
parcels_labelled.geojson   one polygon per parcel: parcel_id, date, ag_season, label, class_name,
                           crop, zone, area_ha, nn_dist_m, wc_cropland_frac, wc_major
summary.csv                parcels and area per class and per agricultural year
*.png                      figures

Run (from the project folder)
-----------------------------
    python ml_dataset_prep/01_prepare_labels.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")                      # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from shapely.ops import unary_union

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
FIELDS_FILE = ROOT / "data" / "field_survey" / "lulc_fielddata.shp"
COUNTRY_FILE = ROOT / "data"  / "admin" / "gadm41_MAR_0.json"
WORLDCOVER_FILE = ROOT / "data" / "worldcover" / "ESA_WorldCover_10m_2021.tif"
OUT_DIR = ROOT / "outputs" / "dataset" / "01_labels"

UTM = 32629                                # metric CRS (UTM 29N) for areas and distances
SEASON_START_MONTH = 9                     # agricultural year starts on 1 September
CHECK_WORLDCOVER = True                    # False = skip the WorldCover check (faster)
CROP_KEEP = ["Seasonal Crop", "Trees", "Trees + Seasonal"]
IRRIGATED_TEXT = ["drip irrigation", "flood irrigation", "flood irrigationflood irrigation",
                  "sprinkler irrigation", "basin irrigation", "pivot irrigation"]
RAINFED_TEXT = ["rainfed"]
# ==========================================================================

CLASS_NAMES = {1: "Irrigated", 2: "Rainfed"}
CLASS_COLORS = {"Irrigated": "#1f77b4", "Rainfed": "#d62728"}
WC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
            60: "Bare / sparse", 70: "Snow / ice", 80: "Water", 90: "Herbaceous wetland",
            95: "Mangroves", 100: "Moss / lichen"}


def save_fig(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def repair(geom):
    """Make an invalid polygon valid; keep only its polygon parts."""
    if geom.is_valid:
        return geom
    fixed = geom.buffer(0)
    if fixed.geom_type in ("Polygon", "MultiPolygon"):
        return fixed
    return unary_union([p for p in getattr(fixed, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")])


def agricultural_season(date):
    """Survey date -> agricultural year starting on 1 September, e.g. 2024-04-25 -> '2023/24'."""
    year = date.year if date.month >= SEASON_START_MONTH else date.year - 1
    return f"{year}/{str(year + 1)[-2:]}"


def worldcover_agreement(parcels):
    """Share of each parcel's 10 m pixels that WorldCover 2021 calls cropland (class 40), and the main class."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    frac, major = [], []
    with rasterio.open(WORLDCOVER_FILE) as src:
        for geom in parcels.to_crs(src.crs).geometry:
            try:
                arr, _ = rio_mask(src, [geom], crop=True, filled=False)
                values = arr.compressed()
                values = values[values > 0]
            except ValueError:                                     # parcel outside the raster
                values = np.array([], dtype=np.uint8)
            if values.size == 0:
                frac.append(np.nan)
                major.append(None)
                continue
            frac.append((values == 40).mean())
            classes, counts = np.unique(values, return_counts=True)
            major.append(WC_NAMES.get(int(classes[counts.argmax()]), str(classes[counts.argmax()])))
    return frac, major


# ------------------------------------------------------------ figures
def plot_map(lab, country):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    country.plot(ax=ax, facecolor="#f0ebe0", edgecolor="#3a3a3a", linewidth=0.8, zorder=1)
    pts = lab.geometry.representative_point()
    for name, color in CLASS_COLORS.items():
        m = (lab["class_name"] == name).to_numpy()
        ax.scatter(pts.x[m], pts.y[m], s=4, c=color, linewidths=0, zorder=3,
                   label=f"{name} ({m.sum():,})")
    ax.set_xlim(pts.x.min() - 1, pts.x.max() + 1)
    ax.set_ylim(pts.y.min() - 0.7, pts.y.max() + 0.7)
    ax.legend(title="Field survey", loc="lower right")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Irrigated and rainfed reference parcels", fontweight="bold")
    save_fig(fig, "parcels_map.png")


def plot_balance_and_size(lab):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    counts = lab["class_name"].value_counts().reindex(CLASS_COLORS)
    axes[0].bar(counts.index, counts.values, color=list(CLASS_COLORS.values()))
    for i, v in enumerate(counts.values):
        axes[0].text(i, v * 1.02, f"{v:,} ({100 * v / counts.sum():.1f}%)", ha="center")
    axes[0].set_ylabel("Parcels")
    axes[0].set_title("Parcels per class", fontweight="bold")
    bins = np.logspace(np.log10(lab["area_ha"].min()), np.log10(lab["area_ha"].max()), 50)
    for name, color in CLASS_COLORS.items():
        axes[1].hist(lab.loc[lab["class_name"] == name, "area_ha"], bins=bins, alpha=0.6, color=color, label=name)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Parcel area (ha, log scale)")
    axes[1].set_ylabel("Parcels")
    axes[1].set_title("Parcel size", fontweight="bold")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "class_balance_and_size.png")


def plot_timing(lab):
    month = lab["date"].dt.to_period("M")
    months = pd.period_range(month.min(), month.max(), freq="M")      # keep empty months visible
    table = pd.crosstab(month, lab["class_name"]).reindex(index=months, columns=list(CLASS_COLORS), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    x = np.arange(len(table))
    ax.bar(x, table["Irrigated"], color=CLASS_COLORS["Irrigated"], label="Irrigated")
    ax.bar(x, table["Rainfed"], bottom=table["Irrigated"], color=CLASS_COLORS["Rainfed"], label="Rainfed")
    ax.set_xticks(x, [str(m) for m in months], rotation=30, ha="right")
    ax.set_ylabel("Parcels")
    ax.set_title("Labelled parcels by survey month", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "survey_timing.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Survey.
    raw = gpd.read_file(FIELDS_FILE).to_crs(4326)
    print(f"Survey: {len(raw):,} parcels, {raw['date'].min():%Y-%m-%d} to {raw['date'].max():%Y-%m-%d}")

    # 2. Geometry cleaning.
    df = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    df["geometry"] = df.geometry.apply(repair)
    df = df[~df.geometry.to_wkb().duplicated()]
    print(f"With valid, unique geometry: {len(df):,}")

    # 3. Labels from the irrigation field.
    text = df["irrigation"].astype(str).str.strip().str.lower()
    df["label"] = np.select([text.isin(IRRIGATED_TEXT), text.isin(RAINFED_TEXT)], [1, 2], default=0)

    # 4. Cropland parcels with a label.
    lab = df[df["crop"].isin(CROP_KEEP) & (df["label"] > 0)].copy()
    lab["class_name"] = lab["label"].map(CLASS_NAMES)
    print(f"Labelled cropland parcels: {len(lab):,} {lab['class_name'].value_counts().to_dict()}")

    # 5. Attributes.
    utm = lab.to_crs(UTM)
    lab["area_ha"] = utm.area / 10_000
    xy = np.column_stack([utm.centroid.x, utm.centroid.y])
    dist, _ = cKDTree(xy).query(xy, k=2)
    lab["nn_dist_m"] = dist[:, 1]                                  # nearest OTHER parcel
    lab["ag_season"] = lab["date"].apply(agricultural_season)
    if CHECK_WORLDCOVER and WORLDCOVER_FILE.exists():
        print("WorldCover agreement (this takes a few minutes)...")
        lab["wc_cropland_frac"], lab["wc_major"] = worldcover_agreement(lab)
    else:
        lab["wc_cropland_frac"], lab["wc_major"] = np.nan, None

    # 6. Save.
    summary = lab.groupby(["ag_season", "class_name"]).agg(parcels=("label", "size"), area_ha=("area_ha", "sum")).round(1)
    summary.to_csv(OUT_DIR / "summary.csv")
    print(f"\nParcels per agricultural year and class:\n{summary.to_string()}")

    country = gpd.read_file(COUNTRY_FILE).to_crs(4326)
    plot_map(lab, country)
    plot_balance_and_size(lab)
    plot_timing(lab)

    cols = ["parcel_id", "date", "ag_season", "label", "class_name", "crop", "zone", "area_ha",
            "nn_dist_m", "wc_cropland_frac", "wc_major", "geometry"]
    out = lab[cols].copy()
    out["parcel_id"] = out["parcel_id"].astype(str)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    (OUT_DIR / "parcels_labelled.geojson").write_text(out.to_json(drop_id=True))
    print(f"\nSaved {OUT_DIR / 'parcels_labelled.geojson'} ({len(out):,} parcels)")


if __name__ == "__main__":
    main()
