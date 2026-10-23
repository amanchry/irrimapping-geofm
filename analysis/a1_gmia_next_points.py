"""
a1_gmia_next_points.py  —  inspect the GMIA-NEXT ground-truth points (global, then one country)
===============================================================================================

What it does
------------
GMIA-NEXT (Kebede 2026, Research Square preprint; maps: zenodo.org/records/17627111) is a global
30 m map of irrigated cropland for the 2023/24 season. Its ground-truth points
(zenodo.org/records/20732681; field `irrigated`: 1 = irrigated, 0 = rainfed / non-irrigated;
field `Source`: the dataset each point comes from) are checked here, and the points of one country
are saved as a possible INDEPENDENT test set (evaluation only, never training).

Steps
-----
Part A, global
  A1 read the points and the World Bank country boundaries;
  A2 data quality: counts vs the README, coordinate columns vs geometry, duplicates, conflicting labels;
     duplicates are removed, locations with both labels dropped;
  A3 classes per source;  A4 points per country and region;  A5 global density maps.
Part B, one country (COUNTRY_ISO, default Morocco)
  B1 points and their sources, map;
  B2 spacing between points;
  B3 Morocco only: distance to the survey parcels, label agreement for points inside a parcel,
     and the spatial fold (20 km block) each point falls in; points in blocks without survey
     parcels test the models in areas they have never seen;
  B4 ESA WorldCover class at each point (label 0 also covers non-cropland);
  B5 save the country points.

Input
-----
data/raw/gmia_next/GMIA-NEXT_GTPs_2024_v0_1/GMIA-NEXT_GTPs_2024_v0_1.shp   (move your GMIA-NEXT_GT folder here)
data/raw/gmia_next/WB_world_countries.geojson
outputs/02_folds/parcels_folds.geojson (for B3)  and  data/raw/worldcover/ESA_WorldCover_10m_2021.tif (for B4)

Output (outputs/analysis/gmia_next/)
------------------------------------
gmia_points_<ISO>.geojson, global_points_by_source.csv, global_points_by_country.csv,
quality_checks.csv, <ISO>_*.csv and figures

Run (from the project folder)
-----------------------------
    python analysis/a1_gmia_next_points.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
GMIA_DIR = ROOT / "data" / "raw" / "gmia_next"
POINTS_FILE = GMIA_DIR / "GMIA-NEXT_GTPs_2024_v0_1" / "GMIA-NEXT_GTPs_2024_v0_1.shp"
COUNTRIES_FILE = GMIA_DIR / "WB_world_countries.geojson"
SURVEY_FILE = ROOT / "outputs" / "02_folds" / "parcels_folds.geojson"
WORLDCOVER_FILE = ROOT / "data" / "raw" / "worldcover" / "ESA_WorldCover_10m_2021.tif"
OUT_DIR = ROOT / "outputs" / "analysis" / "gmia_next"

COUNTRY_ISO = "MAR"           # Part B country (ISO 3166 alpha-3, e.g. "MAR", "TUN")
UTM = 32629                   # metric CRS for Part B (UTM 29N for Morocco; 32632 for Tunisia)
BLOCK_KM = 20                 # same block size as 02_make_spatial_folds.py
README_COUNTS = {"points": 380_696, "irrigated": 230_683, "rainfed": 153_194}   # as stated in the dataset README
# ==========================================================================

CLASS_NAMES = {1: "Irrigated", 0: "Rainfed / non-irrigated"}
CLASS_COLORS = {1: "#1f77b4", 0: "#d62728"}
REGION_NAMES = {"AFR": "Sub-Saharan Africa", "MEN": "Middle East & North Africa", "EAP": "East Asia & Pacific",
                "ECA": "Europe & Central Asia", "LAC": "Latin America & Caribbean", "SOA": "South Asia",
                "Other": "Other (incl. USA, Canada)"}
WC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
            60: "Bare / sparse", 70: "Snow / ice", 80: "Water", 90: "Herbaceous wetland", 95: "Mangroves",
            100: "Moss / lichen"}


def save_fig(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def show(title, table):
    print(f"\n{title}\n{table.to_string()}")


# ------------------------------------------------------------ Part A
def quality_and_clean(pts):
    """A2: quality checks; then drop duplicate locations and locations carrying both labels."""
    same_xy = pts.duplicated(["x", "y"], keep=False)
    labels_per_xy = pts[same_xy].groupby(["x", "y"])["irrigated"].nunique()
    # Some sources have wrong long/lat columns; coordinates are always taken from the geometry.
    bad_cols = (pts["x"] - pts["long"]).abs().gt(1e-6) | (pts["y"] - pts["lat"]).abs().gt(1e-6)
    checks = pd.Series({
        "points in file": len(pts), "points stated in README": README_COUNTS["points"],
        "irrigated in file": int((pts["irrigated"] == 1).sum()), "irrigated in README": README_COUNTS["irrigated"],
        "rainfed in file": int((pts["irrigated"] == 0).sum()), "rainfed in README": README_COUNTS["rainfed"],
        "missing geometry": int(pts.geometry.isna().sum()),
        "long / lat columns differ from geometry": int(bad_cols.sum()),
        "points sharing coordinates": int(same_xy.sum()),
        "extra copies": int(pts.duplicated(["x", "y"]).sum()),
        "locations with conflicting labels": int((labels_per_xy > 1).sum()),
    }, name="value")
    checks.to_csv(OUT_DIR / "quality_checks.csv")
    show("A2. Quality checks", checks.to_frame())
    show("Sources where long / lat differ from the geometry", pts.loc[bad_cols, "Source"].value_counts().to_frame())

    conflict = pts.set_index(["x", "y"]).index.isin(labels_per_xy[labels_per_xy > 1].index)
    clean = pts[~conflict].drop_duplicates(["x", "y"]).copy()
    clean["class_name"] = clean["irrigated"].map(CLASS_NAMES)
    print(f"Kept {len(clean):,} of {len(pts):,} points (dropped {conflict.sum():,} conflicting, "
          f"{len(pts) - conflict.sum() - len(clean):,} duplicate copies)")
    return clean


def per_source(clean):
    """A3: classes per source (table + bar chart)."""
    table = pd.crosstab(clean["Source"], clean["class_name"]).reindex(columns=list(CLASS_NAMES.values()), fill_value=0)
    table["Total"] = table.sum(axis=1)
    table["Irrigated %"] = (100 * table[CLASS_NAMES[1]] / table["Total"]).round(1)
    table = table.sort_values("Total", ascending=False)
    table.to_csv(OUT_DIR / "global_points_by_source.csv")
    show("A3. Points per source", table)
    bs = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    y = np.arange(len(bs))
    ax.barh(y, bs[CLASS_NAMES[1]], color=CLASS_COLORS[1], label=CLASS_NAMES[1])
    ax.barh(y, bs[CLASS_NAMES[0]], left=bs[CLASS_NAMES[1]], color=CLASS_COLORS[0], label=CLASS_NAMES[0])
    ax.set_yticks(y, bs.index)
    ax.set_xlabel("Points")
    ax.set_title("GMIA-NEXT ground-truth points by source", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "global_points_by_source.png")


def per_country(clean, countries):
    """A4: country and World Bank region of each point (spatial join)."""
    joined = gpd.sjoin(clean, countries[["ISO_A3", "NAME", "WB_REGION", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]            # a point on a shared border can match two countries
    clean["ISO_A3"], clean["country"], clean["region"] = joined["ISO_A3"], joined["NAME"], joined["WB_REGION"]
    by_region = pd.crosstab(clean["region"].map(REGION_NAMES).fillna("Outside country polygons"), clean["class_name"])
    show("A4. Points per region", by_region.assign(Total=by_region.sum(axis=1)).sort_values("Total", ascending=False))
    by_country = pd.crosstab(clean["country"], clean["class_name"]).reindex(columns=list(CLASS_NAMES.values()), fill_value=0)
    by_country["Total"] = by_country.sum(axis=1)
    by_country = by_country.sort_values("Total", ascending=False)
    by_country.to_csv(OUT_DIR / "global_points_by_country.csv")
    mena = clean.loc[clean["region"] == "MEN", "country"].dropna().unique()
    show("Middle East & North Africa", by_country.loc[by_country.index.isin(mena)])
    return clean


def global_maps(clean, countries):
    """A5: density of each class on a world map."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 13), dpi=150)
    for ax, cls in zip(axes, [1, 0]):
        sub = clean[clean["irrigated"] == cls]
        countries.plot(ax=ax, facecolor="#f2f2f2", edgecolor="#bbbbbb", linewidth=0.3)
        hb = ax.hexbin(sub.geometry.x, sub.geometry.y, gridsize=(180, 60), extent=(-180, 180, -60, 75),
                       mincnt=1, norm=LogNorm(), cmap="Blues" if cls == 1 else "Reds")
        fig.colorbar(hb, ax=ax, fraction=0.02, pad=0.01, label="Points per cell")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 75)
        ax.set_title(f"{CLASS_NAMES[cls]} points ({len(sub):,})", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "global_points_density.png")


# ------------------------------------------------------------ Part B
def country_part(clean, countries):
    poly = countries[countries["ISO_A3"] == COUNTRY_ISO]
    assert len(poly), f"{COUNTRY_ISO} not found in the country boundaries"
    name = poly["NAME"].iloc[0]
    cp = clean[clean["ISO_A3"] == COUNTRY_ISO].copy()
    utm = cp.to_crs(UTM)
    cp["x_utm"], cp["y_utm"] = utm.geometry.x, utm.geometry.y
    show(f"B1. {name}: points per source", pd.crosstab(cp["Source"], cp["class_name"], margins=True, margins_name="Total"))

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    poly.plot(ax=ax, facecolor="#f0ebe0", edgecolor="#3a3a3a", linewidth=0.8)
    for cls in [0, 1]:
        sub = cp[cp["irrigated"] == cls]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=30, color=CLASS_COLORS[cls], edgecolors="k", linewidths=0.4,
                   zorder=3, label=f"{CLASS_NAMES[cls]} ({len(sub)})")
    ax.legend(loc="lower right", title="GMIA-NEXT label")
    ax.set_title(f"GMIA-NEXT points - {name}", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, f"{COUNTRY_ISO}_points_by_label.png")

    # B2. Spacing: close points add little independent information.
    xy = cp[["x_utm", "y_utm"]].to_numpy()
    d, _ = cKDTree(xy).query(xy, k=2)
    cp["nn_dist_m"] = d[:, 1]
    show("B2. Distance to the nearest other point (km)",
         cp.assign(nn_km=cp["nn_dist_m"] / 1000).groupby("class_name")["nn_km"].describe(percentiles=[0.1, 0.5, 0.9]).round(2))

    # B3. Overlap with the field survey and its spatial folds (Morocco only).
    if COUNTRY_ISO == "MAR" and SURVEY_FILE.exists():
        survey = gpd.read_file(SURVEY_FILE)
        s_utm = survey.to_crs(UTM)
        dist, _ = cKDTree(np.c_[s_utm.centroid.x, s_utm.centroid.y]).query(xy)
        cp["dist_to_survey_km"] = dist / 1000
        cp["dist_band"] = pd.cut(cp["dist_to_survey_km"], [0, 1, 5, 20, 50, np.inf],
                                 labels=["< 1 km", "1-5 km", "5-20 km", "20-50 km", "> 50 km"])
        show("B3. Distance to the nearest survey parcel", pd.crosstab(cp["dist_band"], cp["class_name"], margins=True))
        inside = gpd.sjoin(cp, survey[["label", "geometry"]], predicate="within")
        if len(inside):
            agree = pd.crosstab(inside["class_name"], inside["label"].map({1: "Irrigated", 2: "Rainfed"}))
            show(f"Points inside a survey parcel ({len(inside)}): GMIA label (rows) vs survey label", agree)
        size = BLOCK_KM * 1000
        cp["block_id"] = (np.floor(cp["x_utm"] / size).astype(int).astype(str) + "_" +
                          np.floor(cp["y_utm"] / size).astype(int).astype(str))
        cp["survey_fold"] = cp["block_id"].map(survey.drop_duplicates("block_id").set_index("block_id")["fold"])
        fold_label = cp["survey_fold"].map(lambda f: "no survey parcels in block" if pd.isna(f) else f"fold {int(f)}")
        show("Spatial fold of each point", pd.crosstab(fold_label, cp["class_name"], margins=True))

    # B4. WorldCover class at each point.
    if WORLDCOVER_FILE.exists():
        import rasterio
        with rasterio.open(WORLDCOVER_FILE) as src:
            b = src.bounds
            inside_r = (cp.geometry.x.between(b.left, b.right) & cp.geometry.y.between(b.bottom, b.top)).to_numpy()
            vals = np.zeros(len(cp), dtype=int)
            if inside_r.any():
                vals[inside_r] = [v[0] for v in src.sample(list(zip(cp.geometry.x[inside_r], cp.geometry.y[inside_r])))]
        cp["worldcover"] = pd.Series(vals, index=cp.index).map(WC_NAMES).fillna("outside raster")
        show("B4. WorldCover class at the points", pd.crosstab(cp["worldcover"], cp["class_name"], margins=True))

    # B5. Save.
    cols = [c for c in ["x", "y", "irrigated", "class_name", "Source", "nn_dist_m", "dist_to_survey_km",
                        "survey_fold", "worldcover", "geometry"] if c in cp.columns]
    out = cp[cols].rename(columns={"x": "lon", "y": "lat"})
    (OUT_DIR / f"gmia_points_{COUNTRY_ISO}.geojson").write_text(out.to_json(drop_id=True))
    print(f"\nSaved gmia_points_{COUNTRY_ISO}.geojson ({len(out)} points)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pts = gpd.read_file(POINTS_FILE).to_crs(4326)
    pts["x"], pts["y"] = pts.geometry.x, pts.geometry.y
    countries = gpd.read_file(COUNTRIES_FILE).to_crs(4326)
    print(f"A1. {len(pts):,} points")
    clean = quality_and_clean(pts)
    per_source(clean)
    clean = per_country(clean, countries)
    global_maps(clean, countries)
    country_part(clean, countries)


if __name__ == "__main__":
    main()
