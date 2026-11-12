"""
01_prepare_labels_Utah.py  —  irrigated / rainfed reference fields from the Utah Water-Related Land Use data
============================================================================================================

One of two label scripts (01_prepare_labels_Morocco.py, 01_prepare_labels_Utah.py). Both write the same
file layout, so scripts 02-04 run unchanged for either region (REGION at the top of each script).

Source
------
Water-Related Land Use (WRLU), Utah Division of Water Resources: one polygon layer per year (2017-2025) of
all agricultural fields in Utah and the parts of the river basins that extend into Idaho, Wyoming,
Colorado, Nevada and Arizona. Each field has a crop (Descriptio, CropGroup) and an irrigation method
(IRR_Method). The metadata defines "Dry Crop" as a field with no irrigation method applied, watered by
natural processes (rain) -> the rainfed class.

Steps
-----
1. Inspection of the source: fields and area per year, crop group and irrigation method for ALL years in
   FILES (source_counts.csv, source_years.png), so the choice of years and classes is documented.
2. Read the YEARS used for the dataset; agricultural fields only; geometry cleaned as for Morocco
   (repair invalid polygons, drop empty and duplicate geometries).
3. Label: IRRIGATED_METHODS (flood, sprinkler, drip; from 2025 also pivot and wheel line, which WRLU
   split off sprinkler) = irrigated (1); "Dry Crop" = rainfed (2).
   "Sub-irrigated" (natural high water table) is neither and is dropped.
4. Keep cropped fields only (CROP_GROUPS): fallow/idle land, pasture and the small groups (garden,
   orchard, small fruit) are dropped. Orchards are almost all irrigated in Utah (19 "Dry Crop"
   orchards in 2023), so they would only add an easy irrigated class.
4b. Export ALL labelled crop fields of YEARS before sampling (EXPORT_ALL_FIELDS): GeoPackage for QGIS (with
   a column `sampled` = field lies in a sampled cell), summary tables and the same figures as for the
   sample, so the sample can be compared with the whole population.
5. Sample whole tile cells (CELL_M = one 128 px tile of 03_build_tiles.py). Utah has >100,000 crop fields
   per year; downloading all of them is neither needed nor affordable. Sampling cells instead of single
   fields keeps every field of a sampled cell (dense label rasters, as in Morocco, and needed for the
   segmentation models of RQ2) and costs about one home tile per cell. Cells are drawn at random from
   two strata, cells with at least one rainfed field and cells with irrigated fields only, at most
   MAX_CELLS_PER_BLOCK per spatial block of 02_make_spatial_folds.py so the sample is spread over the
   region. The same cells are used in every year (temporal comparisons see the same places).
6. Attributes (same names as Morocco): parcel_id, ag_season, label, class_name, crop, zone (= river basin),
   area_ha, nn_dist_m (nearest other field of the same year), wc_cropland_frac / wc_major (optional:
   needs a WorldCover raster covering Utah in WORLDCOVER_FILE). Extra: year, state, county, crop_group,
   irr_method, cell_id.
7. Save the fields, summary tables and figures.

Agricultural year
-----------------
WRLU year Y = crops of the growing season of Y. The dataset uses the water year 1 October (Y-1) -
30 September (Y) (start_month in regions.py), written like Morocco as "Y-1/Y", e.g. WRLU 2024 -> "2023/24"
(1 Oct 2023 - 30 Sep 2024).

Years
-----
Utah is an independent second study area (no Morocco -> Utah transfer). WRLU 2024 is the training year;
WRLU 2021 is the temporal test year: 3 years apart, 36% of the crops differ from 2024 but only 0.07% of the
fields change class, so the test asks whether a model recognises irrigation in another year with other crops
and weather. Both years are always prepared together (01, 02), so cells, parcel numbers and folds are fixed once;
03 and 04 then process one year at a time (SEASONS). WRLU 2025 is not used: 94% of its crops repeat 2024.

Output (outputs/utah/dataset/01_labels/)
----------------------------------------
parcels_labelled.geojson   one polygon per field, same columns as Morocco (+ the extras above)
source_counts.csv          source inspection: fields and area per year, crop group, irrigation method
summary.csv                sampled fields and area per agricultural year and class
summary_by_zone.csv        sampled fields per river basin, year and class
sampled_cells.csv          the sampled cells with their stratum and field counts
all_crop_fields.gpkg       ALL labelled crop fields of YEARS (before sampling), column `sampled`
all_fields_summary.csv     all labelled crop fields per agricultural year and class (+ sample share)
all_fields_by_zone.csv     all labelled crop fields per river basin, year and class
all_fields_*.png           map (one panel per year), class balance and size, main crops - before sampling
*.png                      source_years, sampling_map, parcels_map, class_balance_and_size, crops_by_class

Run (from the project folder)
-----------------------------
    python ml_dataset_prep/01_prepare_labels_Utah.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")                      # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree
from shapely.ops import unary_union

from regions import admin_boundaries, dataset_dir, settings

# ================================ SETTINGS ================================
REGION = "utah"                            # projection, agricultural year, map boundaries: regions.py
ROOT = Path(__file__).resolve().parents[1]                 # project folder
SOURCE_DIR = ROOT / "data" / "Utah_field_data"
FILES = {                                  # WRLU layer per year (as downloaded)
    2017: "Water_Related_Land_Use_Statewide_2017/WaterRelatedLandUse_2017.shp",
    2018: "Water_Related_Land_Use_Statewide_2018/WaterRelatedLandUse_2018.shp",
    2019: "WaterRelatedLandUse_2019/WaterRelatedLandUse_2019.shp",
    2020: "Water_Related_Land_Use_(2020)/WaterRelatedLandUse_2020.shp",
    2021: "Water_Related_Land_Use_(2021)/WaterRelatedLandUse_2021.shp",
    2022: "WaterRelatedLandUse_2022/WaterRelatedLandUse_2022.shp",
    2023: "WaterRelatedLandUse_2023/WaterRelatedLandUse_2023.shp",
    2024: "WRLU_2024/Water_Related_Land_Use.shp",
    2025: "WaterRelatedLandUse_2025/WaterRelatedLandUse_2025.shp",
}
YEARS = [2024, 2021]                       # 2024 = training year ("2023/24"); 2021 = temporal test year ("2020/21")
INSPECT_ALL_YEARS = True                   # step 1 for every year in FILES (~10 s per year)
WORLDCOVER_FILE = ROOT / "data" / "worldcover" / "ESA_WorldCover_10m_2021_Utah.tif"   # optional; skipped if missing
CHECK_WORLDCOVER = True
EXPORT_ALL_FIELDS = True                  # step 4b: GeoPackage + figures of all labelled crop fields (~1 min)

CROP_GROUPS = ["Grain/Seeds", "Hay/Turf", "Field Crops"]   # cropped fields (see step 4)
EXCLUDE_CROPS = []                         # Descriptio values to drop, e.g. ["Alfalfa"] for a sensitivity run
IRRIGATED_METHODS = ["Flood", "Sprinkler", "Drip", "Pivot", "Wheel"]   # Pivot and Wheel (wheel line) split off Sprinkler from 2025
RAINFED_METHODS = ["Dry Crop"]
STATES = None                              # None = all fields of the WRLU basins; e.g. ["Utah"] for Utah only

SAMPLE = True                              # False = keep every field (very large tile store)
CELL_M = 1280                              # sampling cell = one tile of 03_build_tiles.py (128 px x 10 m)
N_RAINFED_CELLS = 400                      # cells with >= 1 rainfed field (in any of YEARS)
N_IRRIGATED_CELLS = 150                    # cells with irrigated fields only
MAX_CELLS_PER_BLOCK = 6                    # per spatial block (block_km in regions.py) and stratum
SEED = 42
# ==========================================================================

UTM = settings(REGION)["crs_epsg"]
START_MONTH = settings(REGION)["start_month"]
BLOCK_M = settings(REGION)["block_km"] * 1000
OUT_DIR = dataset_dir(REGION) / "01_labels"
CLASS_NAMES = {1: "Irrigated", 2: "Rainfed"}
CLASS_COLORS = {"Irrigated": "#1f77b4", "Rainfed": "#d62728"}
WC_NAMES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
            60: "Bare / sparse", 70: "Snow / ice", 80: "Water", 90: "Herbaceous wetland",
            95: "Mangroves", 100: "Moss / lichen"}
READ_COLS = ["Landuse", "CropGroup", "Descriptio", "IRR_Method", "State", "County", "Basin"]


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


def agricultural_season(year):
    """WRLU year -> agricultural year ending in it, e.g. 2024 -> '2023/24' (1 Oct 2023 - 30 Sep 2024)."""
    first = year - 1 if START_MONTH > 1 else year
    return f"{first}/{str(first + 1)[-2:]}"


def read_year(year, geometry=True):
    """Agricultural fields of one WRLU year (index = feature id in the source file)."""
    return pyogrio.read_dataframe(SOURCE_DIR / FILES[year], columns=READ_COLS, read_geometry=geometry,
                                  where="Landuse = 'Agricultural'", fid_as_index=True)


def label_of(method):
    return np.select([method.isin(IRRIGATED_METHODS), method.isin(RAINFED_METHODS)], [1, 2], default=0)


def worldcover_agreement(parcels):
    """Share of each field's 10 m pixels that WorldCover 2021 calls cropland (class 40), and the main class."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    frac, major = [], []
    with rasterio.open(WORLDCOVER_FILE) as src:
        for geom in parcels.to_crs(src.crs).geometry:
            try:
                arr, _ = rio_mask(src, [geom], crop=True, filled=False)
                values = arr.compressed()
                values = values[values > 0]
            except ValueError:                                     # field outside the raster
                values = np.array([], dtype=np.uint8)
            if values.size == 0:
                frac.append(np.nan)
                major.append(None)
                continue
            frac.append((values == 40).mean())
            classes, counts = np.unique(values, return_counts=True)
            major.append(WC_NAMES.get(int(classes[counts.argmax()]), str(classes[counts.argmax()])))
    return frac, major


# ------------------------------------------------------------ step 1: source inspection
def inspect_source(years):
    rows = []
    for y in years:
        d = pyogrio.read_dataframe(SOURCE_DIR / FILES[y], columns=["Landuse", "CropGroup", "IRR_Method", "Acres"],
                                   read_geometry=False, where="Landuse = 'Agricultural'")
        g = d.groupby(["CropGroup", "IRR_Method"]).agg(fields=("Acres", "size"), area_ha=("Acres", "sum")).reset_index()
        g["area_ha"] = (g["area_ha"] * 0.404686).round(0)
        g.insert(0, "year", y)
        rows.append(g)
        print(f"  {y}: {len(d):,} agricultural fields")
    return pd.concat(rows, ignore_index=True)


def plot_source_years(counts):
    used = counts[counts["CropGroup"].isin(CROP_GROUPS)]
    lab = pd.Series(label_of(used["IRR_Method"]), index=used.index).map(CLASS_NAMES)
    table = used.groupby([used["year"], lab])["fields"].sum().unstack().reindex(columns=list(CLASS_COLORS)).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    x = np.arange(len(table))
    for k, (name, color) in enumerate(CLASS_COLORS.items()):
        ax.bar(x + (k - 0.5) * 0.4, table[name], 0.4, color=color, label=name)
        for i, v in enumerate(table[name]):
            ax.text(i + (k - 0.5) * 0.4, v * 1.01, f"{v / 1000:.0f}k", ha="center", fontsize=7)
    ax.set_xticks(x, [str(y) for y in table.index])
    ax.set_ylabel("Fields")
    ax.set_title(f"WRLU crop fields per year ({', '.join(CROP_GROUPS)}) - all of the source, before sampling",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "source_years.png")


# ------------------------------------------------------------ step 5: cell sampling
def sample_cells(fields):
    """Random cells from two strata (with rainfed / irrigated only), at most MAX_CELLS_PER_BLOCK per block."""
    rng = np.random.default_rng(SEED)
    cells = pd.crosstab(fields["cell_id"], fields["label"]).reindex(columns=[1, 2], fill_value=0)
    cells.columns = ["n_irrigated", "n_rainfed"]
    first = fields.groupby("cell_id")[["cx", "cy"]].first()
    cells["block"] = (np.floor(first["cx"] / BLOCK_M).astype(int).astype(str) + "_"
                      + np.floor(first["cy"] / BLOCK_M).astype(int).astype(str))
    cells["stratum"] = np.where(cells["n_rainfed"] > 0, "rainfed", "irrigated_only")
    picked = []
    for stratum, n in (("rainfed", N_RAINFED_CELLS), ("irrigated_only", N_IRRIGATED_CELLS)):
        pool = cells[cells["stratum"] == stratum]
        pool = pool.loc[rng.permutation(pool.index)]
        pool = pool[pool.groupby("block").cumcount() < MAX_CELLS_PER_BLOCK]       # spread over blocks
        if len(pool) < n:
            print(f"  only {len(pool)} {stratum} cells available (asked {n})")
        picked.append(pool.head(n))
    sample = pd.concat(picked)
    cells["sampled"] = cells.index.isin(sample.index)
    return cells


# ------------------------------------------------------------ figures
def plot_sampling(cells, fields, admin):
    xy = fields.groupby("cell_id")[["cx", "cy"]].first()
    pts = gpd.GeoSeries(gpd.points_from_xy(xy["cx"], xy["cy"]), index=xy.index, crs=UTM).to_crs(4326)
    fig, ax = plt.subplots(figsize=(7, 8), dpi=150)
    admin.plot(ax=ax, facecolor="#f0ebe0", edgecolor="#3a3a3a", linewidth=0.6, zorder=1)
    styles = [(~cells["sampled"], "#bbbbbb", 1, "not sampled"),
              (cells["sampled"] & (cells["stratum"] == "irrigated_only"), CLASS_COLORS["Irrigated"], 5, "sampled: irrigated only"),
              (cells["sampled"] & (cells["stratum"] == "rainfed"), CLASS_COLORS["Rainfed"], 5, "sampled: with rainfed")]
    for mask, color, size, name in styles:
        p = pts[mask.reindex(pts.index).to_numpy()]
        ax.scatter(p.x, p.y, s=size, c=color, linewidths=0, zorder=3, label=f"{name} ({len(p):,})")
    b = pts.total_bounds
    ax.set_xlim(b[0] - 0.4, b[2] + 0.4)
    ax.set_ylim(b[1] - 0.3, b[3] + 0.3)
    ax.legend(title=f"Cells of {CELL_M / 1000:.2f} km with crop fields in any of these years", loc="lower left", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Sampled cells (same cells for WRLU {', '.join(map(str, YEARS))})", fontweight="bold")
    save_fig(fig, "sampling_map.png")


def plot_map(lab, admin):
    fig, ax = plt.subplots(figsize=(7, 8), dpi=150)
    admin.plot(ax=ax, facecolor="#f0ebe0", edgecolor="#3a3a3a", linewidth=0.6, zorder=1)
    pts = lab.geometry.representative_point()
    for name, color in CLASS_COLORS.items():
        m = (lab["class_name"] == name).to_numpy()
        ax.scatter(pts.x[m], pts.y[m], s=3, c=color, linewidths=0, zorder=3, label=f"{name} ({m.sum():,})")
    ax.set_xlim(pts.x.min() - 0.4, pts.x.max() + 0.4)
    ax.set_ylim(pts.y.min() - 0.3, pts.y.max() + 0.3)
    ax.legend(title=f"WRLU fields, {', '.join(map(str, YEARS))}", loc="lower left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Irrigated and rainfed reference fields", fontweight="bold")
    save_fig(fig, "parcels_map.png")


def plot_balance_and_size(lab, fname="class_balance_and_size.png", note=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    counts = lab["class_name"].value_counts().reindex(CLASS_COLORS)
    axes[0].bar(counts.index, counts.values, color=list(CLASS_COLORS.values()))
    for i, v in enumerate(counts.values):
        axes[0].text(i, v * 1.02, f"{v:,} ({100 * v / counts.sum():.1f}%)", ha="center")
    axes[0].set_ylabel("Fields")
    axes[0].set_title(f"Fields per class{note}", fontweight="bold")
    bins = np.logspace(np.log10(max(lab["area_ha"].min(), 1e-3)), np.log10(lab["area_ha"].max()), 50)
    for name, color in CLASS_COLORS.items():
        axes[1].hist(lab.loc[lab["class_name"] == name, "area_ha"], bins=bins, alpha=0.6, color=color, label=name)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Field area (ha, log scale)")
    axes[1].set_ylabel("Fields")
    axes[1].set_title("Field size", fontweight="bold")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, fname)


def plot_crops(lab, fname="crops_by_class.png", note=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    for ax, (name, color) in zip(axes, CLASS_COLORS.items()):
        top = lab.loc[lab["class_name"] == name, "crop"].value_counts().head(10)[::-1]
        ax.barh(top.index, top.values, color=color)
        for i, v in enumerate(top.values):
            ax.text(v * 1.01, i, f"{v:,}", va="center", fontsize=8)
        ax.set_title(f"{name}: main crops{note}", fontweight="bold")
        ax.set_xlabel("Fields")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, fname)


def plot_all_fields_map(allf, admin):
    """All labelled crop fields, one panel per year; fields in sampled cells outlined in black."""
    years = sorted(allf["year"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(6.5 * len(years), 8), dpi=150, squeeze=False)
    pts = allf.geometry.representative_point()
    b = pts.total_bounds
    for ax, y in zip(axes[0], years):
        admin.plot(ax=ax, facecolor="#f0ebe0", edgecolor="#3a3a3a", linewidth=0.6, zorder=1)
        in_year = (allf["year"] == y).to_numpy()
        for name, color in CLASS_COLORS.items():
            m = in_year & (allf["class_name"] == name).to_numpy()
            ax.scatter(pts.x[m], pts.y[m], s=0.4, c=color, linewidths=0, zorder=3 if name == "Rainfed" else 2,
                       label=f"{name} ({m.sum():,})")
        ax.set_xlim(b[0] - 0.4, b[2] + 0.4)
        ax.set_ylim(b[1] - 0.3, b[3] + 0.3)
        ax.legend(title=f"WRLU {y}: all labelled crop fields", loc="lower left", markerscale=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"All labelled crop fields, WRLU {y} (before sampling)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "all_fields_map.png")


def export_all_fields(fields, cells, admin):
    """Step 4b: every labelled crop field of YEARS, before sampling."""
    allf = fields.copy()
    allf["sampled"] = allf["cell_id"].isin(cells.index[cells["sampled"]]) if cells is not None else True
    allf["class_name"] = allf["label"].map(CLASS_NAMES)
    allf["ag_season"] = allf["year"].apply(agricultural_season)
    allf["area_ha"] = allf.geometry.area / 10_000
    allf = allf.rename(columns={"Descriptio": "crop", "Basin": "zone", "State": "state", "County": "county",
                                "CropGroup": "crop_group", "IRR_Method": "irr_method"})
    cols = ["parcel_id", "year", "ag_season", "label", "class_name", "crop", "crop_group", "irr_method", "zone",
            "state", "county", "area_ha", "cell_id", "sampled", "geometry"]
    allf = allf[cols].to_crs(4326)
    allf.to_file(OUT_DIR / "all_crop_fields.gpkg", layer="all_crop_fields", driver="GPKG")
    summary = allf.groupby(["year", "ag_season", "class_name"]).agg(
        fields=("label", "size"), area_ha=("area_ha", "sum"), median_area_ha=("area_ha", "median"),
        fields_sampled=("sampled", "sum")).round(2)
    summary["sampled_pct"] = (100 * summary["fields_sampled"] / summary["fields"]).round(1)
    summary.to_csv(OUT_DIR / "all_fields_summary.csv")
    pd.crosstab([allf["zone"], allf["year"]], allf["class_name"]).to_csv(OUT_DIR / "all_fields_by_zone.csv")
    print(f"All labelled crop fields (before sampling) -> all_crop_fields.gpkg:\n{summary.to_string()}")
    plot_all_fields_map(allf, admin)
    plot_balance_and_size(allf, "all_fields_balance_and_size.png", " - all crop fields, before sampling")
    plot_crops(allf, "all_fields_crops_by_class.png", " (all fields)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Source inspection.
    years = sorted(FILES) if INSPECT_ALL_YEARS else YEARS
    print(f"Source inspection ({len(years)} years)")
    source = inspect_source(years)
    source.to_csv(OUT_DIR / "source_counts.csv", index=False)
    plot_source_years(source)

    # 2-4. Fields of the dataset years, cleaned and labelled.
    parts = []
    for y in YEARS:
        d = read_year(y).to_crs(UTM)
        n0 = len(d)
        d = d[d.geometry.notna() & ~d.geometry.is_empty].copy()
        d["geometry"] = d.geometry.apply(repair)
        d = d[~d.geometry.to_wkb().duplicated()]
        d["label"] = label_of(d["IRR_Method"])
        keep = d["CropGroup"].isin(CROP_GROUPS) & (d["label"] > 0) & ~d["Descriptio"].isin(EXCLUDE_CROPS)
        if STATES:
            keep &= d["State"].isin(STATES)
        d = d[keep].copy()
        d["year"] = y
        d["parcel_id"] = [f"UT{y}_{i}" for i in d.index]
        print(f"WRLU {y}: {n0:,} agricultural fields -> {len(d):,} labelled crop fields "
              f"{d['label'].map(CLASS_NAMES).value_counts().to_dict()}")
        parts.append(d)
    fields = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=UTM)
    cent = fields.geometry.centroid
    fields["cx"], fields["cy"] = cent.x, cent.y
    fields["cell_id"] = (np.floor(fields["cx"] / CELL_M).astype(int).astype(str) + "_"
                         + np.floor(fields["cy"] / CELL_M).astype(int).astype(str))
    admin = admin_boundaries(REGION)

    # 5. Cell sampling.
    if SAMPLE:
        cells = sample_cells(fields)
        cells.to_csv(OUT_DIR / "sampled_cells.csv")
        plot_sampling(cells, fields, admin)
        lab = fields[fields["cell_id"].isin(cells.index[cells["sampled"]])].copy()
        s = cells[cells["sampled"]]
        print(f"Sampled {len(s)} cells ({(s['stratum'] == 'rainfed').sum()} with rainfed, "
              f"{(s['stratum'] == 'irrigated_only').sum()} irrigated only) of {len(cells):,} "
              f"in {s['block'].nunique()} blocks -> {len(lab):,} fields")
    else:
        cells = None
        lab = fields.copy()
    if EXPORT_ALL_FIELDS:
        export_all_fields(fields, cells, admin)

    # 6. Attributes.
    lab["class_name"] = lab["label"].map(CLASS_NAMES)
    lab["ag_season"] = lab["year"].apply(agricultural_season)
    lab["area_ha"] = lab.geometry.area / 10_000
    lab["nn_dist_m"] = np.nan
    for y, grp in lab.groupby("year"):                              # nearest OTHER field of the same year
        xy = grp[["cx", "cy"]].to_numpy()
        lab.loc[grp.index, "nn_dist_m"] = cKDTree(xy).query(xy, k=2)[0][:, 1] if len(grp) > 1 else np.nan
    lab = lab.rename(columns={"Descriptio": "crop", "Basin": "zone", "State": "state", "County": "county",
                              "CropGroup": "crop_group", "IRR_Method": "irr_method"})
    lab = lab.to_crs(4326)
    if CHECK_WORLDCOVER and WORLDCOVER_FILE.exists():
        print("WorldCover agreement (this takes a few minutes)...")
        lab["wc_cropland_frac"], lab["wc_major"] = worldcover_agreement(lab)
    else:
        if CHECK_WORLDCOVER:
            print(f"WorldCover check skipped: {WORLDCOVER_FILE.name} not found")
        lab["wc_cropland_frac"], lab["wc_major"] = np.nan, None

    # 7. Save.
    summary = lab.groupby(["ag_season", "class_name"]).agg(fields=("label", "size"), area_ha=("area_ha", "sum"),
                                                           median_area_ha=("area_ha", "median")).round(2)
    summary.to_csv(OUT_DIR / "summary.csv")
    print(f"\nFields per agricultural year and class:\n{summary.to_string()}")
    by_zone = pd.crosstab([lab["zone"], lab["ag_season"]], lab["class_name"])
    by_zone.to_csv(OUT_DIR / "summary_by_zone.csv")

    plot_map(lab, admin)
    plot_balance_and_size(lab)
    plot_crops(lab)

    cols = ["parcel_id", "date", "ag_season", "label", "class_name", "crop", "zone", "area_ha", "nn_dist_m",
            "wc_cropland_frac", "wc_major", "year", "state", "county", "crop_group", "irr_method", "cell_id", "geometry"]
    lab["date"] = None                                              # WRLU has no survey date per field
    out = lab[cols].reset_index(drop=True)
    (OUT_DIR / "parcels_labelled.geojson").write_text(out.to_json(drop_id=True))
    print(f"\nSaved {OUT_DIR / 'parcels_labelled.geojson'} ({len(out):,} fields)")


if __name__ == "__main__":
    main()
