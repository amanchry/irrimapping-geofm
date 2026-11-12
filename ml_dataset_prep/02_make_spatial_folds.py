"""
02_make_spatial_folds.py  —  spatial cross-validation folds for the labelled parcels
====================================================================================

What it does
------------
Assigns every parcel to one of 5 folds so that a model can be trained on some areas and tested
on OTHER areas. Parcels are grouped into square blocks (BLOCK_KM x BLOCK_KM) and whole blocks go
to a fold, so a test parcel never has its next-door neighbour in the training data.

Why not a random split
----------------------
Surveyed parcels lie a few hundred metres apart and neighbours usually share the same class.
With a random split ~79% of test parcels have a training parcel within 250 m, so the score
measures memorising nearby fields, not mapping new areas (Roberts et al. 2017, Ecography;
Ploton et al. 2020, Nature Communications; Kattenborn et al. 2022, ISPRS Open J.).

Steps
-----
1. Read the labelled parcels and compute their centroids in metres (projection of the region, regions.py).
2. Class similarity by distance: for pairs of parcels of the same agricultural year at a given
   distance, the share with the same class. Where this curve reaches the random-placement level shows how far spatial
   dependence reaches, and so how large blocks must be.
3. Compare block sizes: for each size, build 5 balanced folds and report (a) rainfed parcels in
   the smallest fold, (b) class share per fold, (c) leakage = test parcels with a training parcel
   within 250 m / 1 km. A random split is included as reference.
4. Assign BLOCK_KM blocks to 5 folds with a greedy balance: blocks with most rainfed parcels are
   placed first, each into the fold furthest below its share of both classes.
5. Also save a stratified random 5-fold assignment (`random_fold`) to measure how much random
   validation overstates accuracy.
6. Save the folds, tables and figures.

How the folds are used
----------------------
models_run/03_train_evaluate.py rotates the 5 folds as in PASTIS (Garnot & Landrieu 2021): in each of
5 rounds one fold is the test fold, the next fold is the validation fold (chooses the hyperparameters)
and the other 3 folds train. Every parcel is tested once. The `split` column (fold 0 test, fold 1
validation, folds 2-4 train) is round 1 of that rotation, kept for quick checks.

The same place in two agricultural years falls in the same block, so it is always in the same fold.

Output (outputs/<REGION>/dataset/02_folds/)
-------------------------------------------
parcels_folds.geojson          all parcels + block_id, fold, random_fold, split
fold_counts.csv                parcels per class and fold
block_size_comparison.csv      the block-size table (step 3)
class_similarity_by_distance.csv / .png
leakage_check.csv              share of test parcels with a training parcel within 100 m ... 5 km
spatial_folds.png              map of folds + parcels per fold
spatial_folds_blocks.png       the blocks themselves, coloured by fold, with zooms on the N_ZOOMS densest areas
train_test_distance.png        distance from test parcels to the nearest training parcel

Run (from the project folder)
-----------------------------
    python ml_dataset_prep/02_make_spatial_folds.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")                       # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.spatial import cKDTree

from regions import admin_boundaries, dataset_dir, settings

# ================================ SETTINGS ================================
REGION = "morocco"           # "morocco" | "utah" (regions.py)
INPUT = dataset_dir(REGION) / "01_labels" / "parcels_labelled.geojson"   # from 01_prepare_labels_<Region>.py
OUT_DIR = dataset_dir(REGION) / "02_folds"

UTM = settings(REGION)["crs_epsg"]          # metric CRS for distances and blocks
N_FOLDS = 5
BLOCK_KM = settings(REGION)["block_km"]     # block size; justified by steps 2-3 (see block_size_comparison.csv)
CANDIDATE_KM = [2, 5, 10, 20, 25, 30, 50]
TEST_FOLD, VAL_FOLD = 0, 1   # only for the quick fixed split in column `split`
SEED = 42
N_ZOOMS, ZOOM_BLOCKS = 2, 5 # block map: zoom panels on the densest ZOOM_BLOCKS x ZOOM_BLOCKS windows
# ==========================================================================

CLASS_NAMES = {1: "Irrigated", 2: "Rainfed"}
CLASS_COLORS = {1: "#1f77b4", 2: "#d62728"}
FOLD_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def save_fig(fig, name):
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# ------------------------------------------------------------ step 2
def class_similarity(xy, labels, seasons):
    """Share of parcel pairs (same agricultural year) with the same class, per distance band, and the
    random-placement level. Pairs across years are left out: the same field in two years is not two samples."""
    p_irr = (labels == 1).mean()
    baseline = p_irr ** 2 + (1 - p_irr) ** 2              # chance two random parcels share a class
    edges = np.array([0, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000])
    pairs = cKDTree(xy).query_pairs(r=edges[-1], output_type="ndarray")   # all pairs closer than 50 km
    pairs = pairs[seasons[pairs[:, 0]] == seasons[pairs[:, 1]]]
    dist = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    band = np.digitize(dist, edges) - 1                    # distance band of each pair
    table = pd.DataFrame({
        "distance_to_km": edges[1:] / 1000,
        "pairs": np.bincount(band, minlength=len(edges))[: len(edges) - 1],
        "same_class_pct": [100 * same[band == k].mean() for k in range(len(edges) - 1)],
    }).round(1)
    return table, 100 * baseline, edges


# ------------------------------------------------------------ steps 3-5
def block_ids(xy, km):
    """Name of the km x km square each centroid falls in, e.g. '-12_171'."""
    size = km * 1000
    bx = np.floor(xy[:, 0] / size).astype(int)
    by = np.floor(xy[:, 1] / size).astype(int)
    return np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))


def assign_folds(blocks, labels, n_folds=N_FOLDS, seed=SEED):
    """Greedy, class-balanced assignment of whole blocks to folds.
    Blocks with the most rainfed parcels go first (they are the hard ones to balance); each goes to
    the fold whose larger class share would stay lowest. Ties are broken at random (fixed seed)."""
    table = pd.crosstab(blocks, labels).reindex(columns=[1, 2], fill_value=0)   # parcels per block and class
    rng = np.random.default_rng(seed)
    table = table.loc[rng.permutation(table.index)]                              # random order for ties
    table = table.sort_values([2, 1], ascending=False, kind="stable")
    totals = table.sum().to_numpy()
    load = np.zeros((n_folds, 2))                                                # parcels already in each fold
    block_fold = {}
    for block, row in table.iterrows():
        after = (load + row.to_numpy()) / totals                                 # class shares if added
        fold = int(np.argmin(after.max(axis=1) + 1e-6 * rng.random(n_folds)))
        load[fold] += row.to_numpy()
        block_fold[block] = fold
    return pd.Series(blocks).map(block_fold).to_numpy()


def random_folds(labels, n_folds=N_FOLDS, seed=SEED):
    """Stratified random folds: each class dealt round-robin after shuffling."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for c in (1, 2):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def leakage(xy, folds, radius_m):
    """Average over folds of the % of test parcels with a training parcel within radius_m."""
    shares = []
    for f in range(N_FOLDS):
        test = folds == f
        d, _ = cKDTree(xy[~test]).query(xy[test])          # distance to nearest training parcel
        shares.append((d <= radius_m).mean())
    return 100 * np.mean(shares)


def summarise(name, xy, labels, folds):
    per_fold = pd.crosstab(folds, labels).reindex(columns=[1, 2], fill_value=0)
    share = per_fold / per_fold.sum()
    return {
        "split": name,
        "rainfed_in_smallest_fold": int(per_fold[2].min()),
        "irrigated_share_range_pct": f"{100 * share[1].min():.0f}-{100 * share[1].max():.0f}",
        "rainfed_share_range_pct": f"{100 * share[2].min():.0f}-{100 * share[2].max():.0f}",
        "test_with_train_within_250m_pct": round(leakage(xy, folds, 250), 1),
        "test_with_train_within_1km_pct": round(leakage(xy, folds, 1000), 1),
    }


# ------------------------------------------------------------ figures
def plot_similarity(table, baseline, edges):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    mid = (edges[:-1] + edges[1:]) / 2 / 1000
    ax.plot(mid, table["same_class_pct"], marker="o", color="#333333")
    ax.axhline(baseline, color="#999999", linestyle="--", label=f"random placement ({baseline:.0f}%)")
    ax.axvline(BLOCK_KM, color=FOLD_COLORS[1], linestyle=":", label=f"block size {BLOCK_KM} km")
    ax.set_xscale("log")
    ax.set_xlabel("Distance between parcels (km, log scale)")
    ax.set_ylabel("Pairs with the same class (%)")
    ax.set_title("Class similarity between parcels by distance", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "class_similarity_by_distance.png")


def plot_folds(parcels, country, fold_table):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.3, 1]}, dpi=150)
    ax = axes[0]
    pts = gpd.GeoSeries(gpd.points_from_xy(parcels["x_utm"], parcels["y_utm"]), crs=UTM).to_crs(4326)
    country.plot(ax=ax, facecolor="#f0ebe0", edgecolor="#3a3a3a", linewidth=0.8, zorder=1)
    for f in range(N_FOLDS):
        m = parcels["fold"].to_numpy() == f
        ax.scatter(pts.x[m], pts.y[m], s=4, color=FOLD_COLORS[f], linewidths=0, zorder=3)
    ax.set_xlim(pts.x.min() - 0.8, pts.x.max() + 0.8)
    ax.set_ylim(pts.y.min() - 0.5, pts.y.max() + 0.5)
    ax.legend(handles=[Patch(color=FOLD_COLORS[f], label=f"Fold {f}") for f in range(N_FOLDS)], loc="lower right")
    ax.set_title(f"Spatial folds ({BLOCK_KM} km blocks)", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax = axes[1]
    x = np.arange(N_FOLDS)
    for k, (c, off) in enumerate([(1, -0.2), (2, 0.2)]):
        vals = fold_table[CLASS_NAMES[c]].to_numpy()
        ax.bar(x + off, vals, 0.4, color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        for i, v in enumerate(vals):
            ax.text(i + off, v + 20, f"{v:,}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"Fold {f}" for f in x])
    ax.set_ylabel("Parcels")
    ax.set_title("Parcels per fold", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "spatial_folds.png")


def densest_windows(bx, by, n_parcels, k=ZOOM_BLOCKS, n=N_ZOOMS):
    """Lower-left block of the n non-overlapping k x k block windows holding the most parcels."""
    counts = {}
    for x0 in range(bx.min() - k + 1, bx.max() + 1):
        for y0 in range(by.min() - k + 1, by.max() + 1):
            inside = (bx >= x0) & (bx < x0 + k) & (by >= y0) & (by < y0 + k)
            if inside.any():
                counts[(x0, y0)] = n_parcels[inside].sum()
    picked = []
    for (x0, y0), _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        if all(abs(x0 - a) >= k or abs(y0 - b) >= k for a, b in picked):
            picked.append((x0, y0))
        if len(picked) == n:
            break
    return picked


def plot_blocks(parcels, country):
    """The BLOCK_KM blocks coloured by fold (fold number in each block), parcels on top, zooms on dense areas.
    Drawn in the metric projection, so blocks are true squares."""
    size = BLOCK_KM * 1000
    blocks = parcels.groupby("block_id").agg(fold=("fold", "first"), n=("fold", "size"))
    xy = blocks.index.to_series().str.split("_", expand=True).astype(int)
    blocks["bx"], blocks["by"] = xy[0], xy[1]
    land = country.to_crs(UTM)
    windows = densest_windows(blocks["bx"].to_numpy(), blocks["by"].to_numpy(), blocks["n"].to_numpy())

    def draw(ax, numbers, point_size):
        land.plot(ax=ax, facecolor="#f0ebe0", edgecolor="#3a3a3a", linewidth=0.6, zorder=0)
        for b in blocks.itertuples():
            c = FOLD_COLORS[b.fold]
            ax.add_patch(Rectangle((b.bx * size, b.by * size), size, size, facecolor=c, alpha=0.3,
                                   edgecolor=c, linewidth=0.8, zorder=1))
            if numbers:
                ax.text((b.bx + 0.5) * size, (b.by + 0.5) * size, str(b.fold), ha="center", va="center",
                        fontsize=9, fontweight="bold", color=c, zorder=4, clip_on=True,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        for f in range(N_FOLDS):
            m = parcels["fold"].to_numpy() == f
            ax.scatter(parcels["x_utm"][m], parcels["y_utm"][m], s=point_size, color=FOLD_COLORS[f],
                       linewidths=0, zorder=3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig = plt.figure(figsize=(15, 9), dpi=150)
    grid = fig.add_gridspec(max(1, len(windows)), 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[:, 0])
    draw(ax, numbers=False, point_size=1)
    ax.set_xlim((blocks["bx"].min() - 1) * size, (blocks["bx"].max() + 2) * size)
    ax.set_ylim((blocks["by"].min() - 1) * size, (blocks["by"].max() + 2) * size)
    ax.set_title(f"All {len(blocks)} blocks of {BLOCK_KM} km, coloured by fold", fontweight="bold")
    ax.legend(handles=[Patch(facecolor=FOLD_COLORS[f], alpha=0.5, edgecolor=FOLD_COLORS[f], label=f"Fold {f}")
                       for f in range(N_FOLDS)], title=f"Squares = {BLOCK_KM} km blocks", loc="upper center",
              bbox_to_anchor=(0.5, -0.01), ncol=N_FOLDS, frameon=False)
    for k, (x0, y0) in enumerate(windows):
        letter = "AB"[k] if k < 2 else str(k + 1)
        ax.add_patch(Rectangle((x0 * size, y0 * size), ZOOM_BLOCKS * size, ZOOM_BLOCKS * size, fill=False,
                               linestyle="--", linewidth=1.2, edgecolor="#222222", zorder=5))
        ax.text(x0 * size, (y0 + ZOOM_BLOCKS) * size, f" {letter}", va="bottom", fontweight="bold", fontsize=12)
        zax = fig.add_subplot(grid[k, 1])
        draw(zax, numbers=True, point_size=3)
        zax.set_xlim(x0 * size, (x0 + ZOOM_BLOCKS) * size)
        zax.set_ylim(y0 * size, (y0 + ZOOM_BLOCKS) * size)
        zax.set_title(f"{letter} - zoom ({ZOOM_BLOCKS} x {ZOOM_BLOCKS} blocks)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "spatial_folds_blocks.png")


def plot_distance(d_random, d_spatial):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    bins = np.logspace(1, np.log10(max(d_spatial.max(), d_random.max())), 50)
    ax.hist(d_random, bins=bins, alpha=0.6, color="#999999", label=f"random split (median {np.median(d_random):.0f} m)")
    ax.hist(d_spatial, bins=bins, alpha=0.6, color=FOLD_COLORS[1], label=f"spatial split (median {np.median(d_spatial):.0f} m)")
    ax.set_xscale("log")
    ax.set_xlabel("Distance from test parcel to nearest training parcel (m, log scale)")
    ax.set_ylabel("Test parcels")
    ax.set_title("Train-test distance: random vs spatial split", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "train_test_distance.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Parcels and centroids in metres.
    parcels = gpd.read_file(INPUT)
    country = admin_boundaries(REGION)
    cent = parcels.to_crs(UTM).centroid
    parcels["x_utm"], parcels["y_utm"] = cent.x, cent.y
    xy = parcels[["x_utm", "y_utm"]].to_numpy()
    labels = parcels["label"].to_numpy()
    print(f"{len(parcels):,} parcels: {parcels['label'].map(CLASS_NAMES).value_counts().to_dict()}")

    # 2. How far does class similarity reach?
    sim, baseline, edges = class_similarity(xy, labels, parcels["ag_season"].to_numpy())
    sim.to_csv(OUT_DIR / "class_similarity_by_distance.csv", index=False)
    print(f"Random-placement level {baseline:.1f}% | same-class share by distance:\n{sim.to_string(index=False)}")
    plot_similarity(sim, baseline, edges)

    # 3. Compare block sizes (random split as reference).
    rows = [summarise("random", xy, labels, random_folds(labels))]
    for km in CANDIDATE_KM:
        rows.append(summarise(f"blocks {km} km", xy, labels, assign_folds(block_ids(xy, km), labels)))
    compare = pd.DataFrame(rows).set_index("split")
    compare.to_csv(OUT_DIR / "block_size_comparison.csv")
    print(f"\nBlock-size comparison:\n{compare.to_string()}")

    # 4-5. Final folds: spatial (BLOCK_KM blocks) and random.
    parcels["block_id"] = block_ids(xy, BLOCK_KM)
    parcels["fold"] = assign_folds(parcels["block_id"].to_numpy(), labels)
    parcels["random_fold"] = random_folds(labels)
    parcels["split"] = np.select([parcels["fold"] == TEST_FOLD, parcels["fold"] == VAL_FOLD],
                                 ["test", "val"], default="train")

    fold_table = pd.crosstab(parcels["fold"], parcels["label"].map(CLASS_NAMES))
    fold_table["blocks"] = parcels.groupby("fold")["block_id"].nunique()
    fold_table.to_csv(OUT_DIR / "fold_counts.csv")
    print(f"\n{parcels['block_id'].nunique()} blocks of {BLOCK_KM} km -> parcels per fold:\n{fold_table.to_string()}")
    plot_folds(parcels, country, fold_table)
    plot_blocks(parcels, country)

    # Leakage check for the fixed split (test fold vs folds 2-4), spatial vs random.
    fold, rfold = parcels["fold"].to_numpy(), parcels["random_fold"].to_numpy()
    d_spatial, _ = cKDTree(xy[~np.isin(fold, [TEST_FOLD, VAL_FOLD])]).query(xy[fold == TEST_FOLD])
    d_random, _ = cKDTree(xy[~np.isin(rfold, [TEST_FOLD, VAL_FOLD])]).query(xy[rfold == TEST_FOLD])
    radii = [100, 250, 500, 1000, 5000]
    check = pd.DataFrame({"random_split_pct": [100 * (d_random <= r).mean() for r in radii],
                          "spatial_split_pct": [100 * (d_spatial <= r).mean() for r in radii]},
                         index=[f"train parcel within {r} m" for r in radii]).round(1)
    check.to_csv(OUT_DIR / "leakage_check.csv")
    print(f"\nTest parcels with a training parcel nearby (%):\n{check.to_string()}")
    plot_distance(d_random, d_spatial)

    # 6. Save the parcels with their folds (written as text so an existing file is overwritten in place).
    cols = ["parcel_id", "date", "ag_season", "label", "class_name", "zone", "area_ha",
            "block_id", "fold", "random_fold", "split", "geometry"]
    out = parcels[[c for c in cols if c in parcels.columns]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    (OUT_DIR / "parcels_folds.geojson").write_text(out.to_json(drop_id=True))
    print(f"\nSaved parcels_folds.geojson ({len(out):,} parcels) in {OUT_DIR}")


if __name__ == "__main__":
    main()
