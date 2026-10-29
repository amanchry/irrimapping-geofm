"""
01_inspect_dataset.py  —  check the ML-ready dataset before any model is trained
================================================================================

What it does
------------
First step of the irrigated-area study. Reads the dataset built by dataset/03_build_tiles.py and
dataset/04_make_views.py and checks that it is complete and sensible, so that problems in the data
are found before they show up as strange model results. Nothing is changed in the dataset.

Steps
-----
1. Inventory: parcels, tiles, and how many parcels reach each view (feature table, crops), by class,
   season and fold.
2. Label QA: overlapping parcels, parcels smaller than one pixel, parcels without inner (non-edge) pixels.
3. Observation QA: clear Sentinel-2 observations and Sentinel-1 scenes per month, per class; share of
   parcel-months without any clear Sentinel-2 view.
4. Patch purity of the foundation-model crops: share of the pooled patches that lies inside the parcel,
   for the patch-aligned crops and for the old centred crops (shows what the alignment gains).
5. Seasonal profiles per class from the parcel features (NDVI, NDWI, Sentinel-1 VH).
6. Example crops: true colour and NDVI for all 12 months, for a few random parcels per class.
7. Agreement with the legacy Earth Engine features (if outputs/legacy/03_s1s2_features exists):
   correlation of the parcel NDVI per month between the two pipelines.

Output (outputs/study/01_inspection/)
-------------------------------------
inventory.csv, label_qa.csv, observations_per_month.csv, purity_summary.csv, legacy_agreement.csv,
observations_per_month.png, purity.png, seasonal_profiles.png, examples_irrigated.png, examples_rainfed.png
For a closer look in QGIS: analysis/a3_export_chips_geotiff.py writes single crops as GeoTIFFs.

Run (from the project folder, after dataset/04_make_views.py)
-------------------------------------------------------------
    python study/01_inspect_dataset.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                         # project folder
WINDOW = "agyear"
TILE_ROOT = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW       # dataset/03_build_tiles.py
VIEWS = TILE_ROOT / "views"                                         # dataset/04_make_views.py
LEGACY_FEATURES = ROOT / "outputs" / "legacy" / "03_s1s2_features" / f"s1s2_features_{WINDOW}_30d.csv"
OUT_DIR = ROOT / "outputs" / "study" / "01_inspection"

EXAMPLE_PATCH = 16             # which crops to draw (8 or 16)
N_EXAMPLES = 3                 # random parcels per class in the example figures
SEED = 42
# ==========================================================================

CLASS_NAMES = {1: "irrigated", 2: "rainfed"}
CLASS_COLORS = {1: "#1F5F8B", 2: "#B4541A"}
NDVI_CMAP = LinearSegmentedColormap.from_list("ndvi", ["#8C510A", "#D8B365", "#F6E8C3", "#C7EAE5",
                                                       "#5AB4AC", "#01665E", "#003C30"])


def month_names(start_month, n):
    first = pd.Timestamp(year=2023, month=start_month, day=1)
    return [(first + pd.DateOffset(months=k)).strftime("%b") for k in range(n)]


def save_fig(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figure: {name}")


# ------------------------------------------------------------ 1. inventory
def inventory(parcels, tiles, feats, crop_index):
    rows = []
    groups = [("all", parcels)] + [(f"{k}={v}", g) for k in ("class_name", "ag_season", "fold")
                                   for v, g in parcels.groupby(k)]
    for name, g in groups:
        ids = set(g["parcel_id"])
        row = {"group": name, "parcels": len(g)}
        row["in_features_inner"] = int(feats["parcel_id"].isin(ids).sum()) if feats is not None else None
        for ps, idx in crop_index.items():
            row[f"in_crops_p{ps}"] = int(idx["parcel_id"].isin(ids).sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "inventory.csv", index=False)
    print(f"\n1. Inventory\n{table.to_string(index=False)}")
    if tiles is not None:
        print(f"   tiles: {len(tiles):,} | downloaded {int(tiles['downloaded'].sum()):,} | roles "
              f"{tiles['role'].value_counts().to_dict()} | tiles holding parcels of two folds: "
              f"{int(tiles['mixed_folds'].sum())}")


# ------------------------------------------------------------ 2. labels
def label_qa(parcels):
    rows = [("parcels", len(parcels)),
            ("smaller than one pixel centre (all_touched)", int(parcels["all_touched"].sum())),
            ("no pixel left after the inner buffer", int((parcels["n_px_inner"] == 0).sum())),
            ("median pixels per parcel", float(parcels["n_px"].median()))]
    ov = TILE_ROOT / "parcel_overlaps.csv"
    if ov.exists():
        o = pd.read_csv(ov)
        rows += [("overlapping parcel pairs", len(o)),
                 ("overlapping pairs with different labels", int((o["label_a"] != o["label_b"]).sum())),
                 ("overlaps covering > 50% of the smaller parcel", int((o["share_of_smaller"] > 0.5).sum()))]
    table = pd.DataFrame(rows, columns=["check", "value"])
    table.to_csv(OUT_DIR / "label_qa.csv", index=False)
    print(f"\n2. Label QA\n{table.to_string(index=False)}")


# ------------------------------------------------------------ 3. observations
def observation_qa(feats, months):
    n = len(months)
    rows = []
    for lab, g in feats.groupby("label"):
        for k in range(n):
            m = f"m{k + 1:02d}"
            row = {"class": CLASS_NAMES.get(lab, lab), "month": months[k],
                   "s2_clear_obs_mean": g[f"S2_NCLEAR_{m}"].mean(),
                   "pct_parcels_no_clear_s2": 100 * g[f"B4_{m}"].isna().mean()}
            for o in ("ASC", "DSC"):
                if f"S1_N_{o}_{m}" in g:
                    row[f"s1_{o.lower()}_scenes_mean"] = g[f"S1_N_{o}_{m}"].mean()
            rows.append(row)
    table = pd.DataFrame(rows).round(2)
    table.to_csv(OUT_DIR / "observations_per_month.csv", index=False)
    print(f"\n3. Observations per month (class means)\n{table.to_string(index=False)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for lab, g in table.groupby("class"):
        col = CLASS_COLORS[1 if lab == "irrigated" else 2]
        axes[0].plot(g["month"], g["s2_clear_obs_mean"], marker="o", color=col, label=lab)
        axes[1].plot(g["month"], g["pct_parcels_no_clear_s2"], marker="o", color=col, label=lab)
    axes[0].set_title("Clear Sentinel-2 observations per parcel and month", fontweight="bold")
    axes[1].set_title("Parcels without any clear Sentinel-2 view (%)", fontweight="bold")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "observations_per_month.png")


# ------------------------------------------------------------ 4. purity
def purity_qa(crop_index):
    rows, fig = [], None
    if not crop_index:
        return
    fig, axes = plt.subplots(1, len(crop_index), figsize=(6 * len(crop_index), 3.8), squeeze=False)
    for ax, (ps, idx) in zip(axes[0], sorted(crop_index.items())):
        for col, name, c in (("purity_centred", "centred crop (old)", "#8A919A"), ("purity", "patch-aligned crop", "#1F5F8B")):
            ax.hist(idx[col], bins=40, range=(0, 1), alpha=0.6, color=c, label=name)
            rows.append({"patch_px": ps, "crop": name, "median": idx[col].median(),
                         "pct_below_0.5": 100 * (idx[col] < 0.5).mean()})
        ax.set_title(f"Patch purity, {ps} x {ps}-pixel patches ({ps * 10} m)", fontweight="bold")
        ax.set_xlabel("share of pooled patches inside the parcel"); ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "purity.png")
    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT_DIR / "purity_summary.csv", index=False)
    print(f"\n4. Patch purity\n{table.to_string(index=False)}")


# ------------------------------------------------------------ 5. profiles
def seasonal_profiles(feats, months, orbit):
    n = len(months)
    variables = [("NDVI", "NDVI"), ("NDWI", "NDWI"), (f"VH_{orbit}", f"VH {orbit} (dB)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (var, title) in zip(axes, variables):
        cols = [f"{var}_m{k + 1:02d}" for k in range(n)]
        if not all(c in feats for c in cols):
            ax.set_visible(False)
            continue
        for lab, g in feats.groupby("label"):
            q = g[cols].quantile([0.25, 0.5, 0.75])
            ax.plot(months, q.loc[0.5], color=CLASS_COLORS[lab], label=CLASS_NAMES[lab], lw=2)
            ax.fill_between(months, q.loc[0.25], q.loc[0.75], color=CLASS_COLORS[lab], alpha=0.15)
        ax.set_title(title, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Parcel median per month (inner pixels): median and interquartile range per class", y=1.02)
    save_fig(fig, "seasonal_profiles.png")


# ------------------------------------------------------------ 6. example crops
def load_crops(folder, ids):
    """chips [n, months, bands, H, W] and masks for the requested parcel IDs (read part by part)."""
    index = pd.read_csv(folder / "index.csv", dtype={"parcel_id": str}).drop_duplicates("parcel_id").set_index("parcel_id")
    out = {}
    for part, sub in index.loc[ids].groupby("part"):
        d = np.load(folder / part)
        pos = {p: i for i, p in enumerate(d["parcel_id"].astype(str))}
        for pid in sub.index:
            out[pid] = (d["chips"][pos[pid]], d["mask"][pos[pid]])
    return out


def example_crops(crop_index, months):
    if EXAMPLE_PATCH not in crop_index:
        return
    folder = VIEWS / f"crops_p{EXAMPLE_PATCH}"
    info = json.loads((folder / "chips_info.json").read_text())
    bands, nodata = info["bands"], info["nodata"]
    b = {x: bands.index(x) for x in ("B2", "B3", "B4", "B8")}
    idx = crop_index[EXAMPLE_PATCH]
    rng = np.random.default_rng(SEED)
    for lab, name in CLASS_NAMES.items():
        pool = idx.loc[(idx["label"] == lab) & (idx["n_px_in_crop"] >= 50), "parcel_id"].to_numpy()
        if len(pool) == 0:
            continue
        ids = list(rng.choice(pool, size=min(N_EXAMPLES, len(pool)), replace=False))
        crops = load_crops(folder, ids)
        fig, axes = plt.subplots(2 * len(ids), len(months), figsize=(1.5 * len(months), 3.1 * len(ids)),
                                 gridspec_kw=dict(hspace=0.06, wspace=0.03), squeeze=False)
        for r, pid in enumerate(ids):
            chip, mask = crops[pid]
            for k in range(len(months)):
                raw = chip[k].astype(np.float32)
                raw[chip[k] == nodata] = np.nan
                rgb = np.dstack([raw[b["B4"]], raw[b["B3"]], raw[b["B2"]]]) / 2400
                rgb = np.nan_to_num(np.clip(rgb, 0, 1) ** 0.8, nan=0.85)
                ndvi = (raw[b["B8"]] - raw[b["B4"]]) / (raw[b["B8"]] + raw[b["B4"]] + 1e-6)
                for row, img, kw in ((2 * r, rgb, {}), (2 * r + 1, ndvi, dict(cmap=NDVI_CMAP, vmin=0, vmax=0.8))):
                    ax = axes[row, k]
                    ax.imshow(img, **kw)
                    ax.contour(mask.astype(float), levels=[0.5], colors=["#FFD23F" if row == 2 * r else "#B4541A"], linewidths=1)
                    ax.set_xticks([]); ax.set_yticks([])
                    if row == 0:
                        ax.set_title(months[k], fontsize=10, fontweight="bold")
            axes[2 * r, 0].set_ylabel(f"{pid[-12:]}\ntrue colour", fontsize=8)
            axes[2 * r + 1, 0].set_ylabel("NDVI", fontsize=8)
        fig.suptitle(f"Example {name} parcels: {EXAMPLE_PATCH}-px-aligned crops, 12 months (outline = parcel)",
                     fontweight="bold", y=1.0)
        save_fig(fig, f"examples_{name}.png")


# ------------------------------------------------------------ 7. legacy agreement
def legacy_agreement(feats, months):
    if not LEGACY_FEATURES.exists():
        print("\n7. Legacy Earth Engine features not found: agreement check skipped")
        return
    old = pd.read_csv(LEGACY_FEATURES, dtype={"parcel_id": str}).set_index("parcel_id")
    new = feats.set_index("parcel_id")
    common = old.index.intersection(new.index)
    rows = []
    for k in range(len(months)):
        a, b_ = old.loc[common, f"NDVI_p{k + 1:02d}"], new.loc[common, f"NDVI_m{k + 1:02d}"]
        ok = a.notna() & b_.notna()
        rows.append({"month_new": months[k], "period_old": f"p{k + 1:02d}", "parcels": int(ok.sum()),
                     "pearson_r": round(float(np.corrcoef(a[ok], b_[ok])[0, 1]), 3) if ok.sum() > 2 else None,
                     "mean_difference": round(float((b_[ok] - a[ok]).mean()), 4)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "legacy_agreement.csv", index=False)
    print(f"\n7. NDVI agreement, new tile features vs legacy Earth Engine table "
          f"(30-day periods vs calendar months, so not identical)\n{table.to_string(index=False)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = json.loads((TILE_ROOT / "tiles_info.json").read_text())
    months = month_names(info["start_month"], info["n_months"])
    parcels = pd.read_csv(TILE_ROOT / "parcels.csv", dtype={"parcel_id": str})
    tiles = pd.read_parquet(TILE_ROOT / "tiles.parquet") if (TILE_ROOT / "tiles.parquet").exists() else None
    feats = pd.read_csv(VIEWS / "parcel_features_inner.csv", dtype={"parcel_id": str}) \
        if (VIEWS / "parcel_features_inner.csv").exists() else None
    feats_all = pd.read_csv(VIEWS / "parcel_features_all.csv", dtype={"parcel_id": str}) \
        if (VIEWS / "parcel_features_all.csv").exists() else feats
    crop_index = {}
    for folder in sorted(VIEWS.glob("crops_p*")):
        if (folder / "index.csv").exists():
            crop_index[int(folder.name[len("crops_p"):])] = pd.read_csv(folder / "index.csv", dtype={"parcel_id": str})
    views_info = json.loads((VIEWS / "views_info.json").read_text()) if (VIEWS / "views_info.json").exists() else {}
    orbit = views_info.get("s1_orbit_for_crops_and_pixelsets", "ASC")

    inventory(parcels, tiles, feats, crop_index)
    label_qa(parcels)
    if feats_all is not None:
        observation_qa(feats_all, months)
    purity_qa(crop_index)
    if feats is not None:
        seasonal_profiles(feats, months, orbit)
        legacy_agreement(feats, months)
    example_crops(crop_index, months)
    print(f"\nSaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
