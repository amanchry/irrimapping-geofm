"""
05_breakdown_oof.py  —  where do the classifiers succeed or fail? Out-of-fold results by zone, season and purity
================================================================================================================

What it does
------------
The headline metrics of 03_train_evaluate.py / 04_compare_models.py pool all parcels. In this dataset the class
is strongly tied to the region: 94% of the rainfed parcels lie in Doukkala, and Gharb, Tadla and El Haouz hold
almost only irrigated parcels. A classifier can therefore score well by recognising the region instead of the
irrigation signal (shortcut learning). The foundation-model chips also mix the parcel with its neighbours
(patch purity). This script splits the existing out-of-fold predictions so both effects become visible.
No model is trained; it only reads oof_predictions.csv of every main run of one experiment in one region
(REGION). In Utah the same questions apply: 58% of the rainfed fields lie in the Bear River basin.

Zones per region (settings FOCUS_ZONE, IRRIGATED_ONLY): the focus zone is the zone where both classes are common
(Morocco: Doukkala; Utah: the zone with most rainfed parcels, found automatically); irrigated-only zones are
listed (Morocco) or found automatically (zones with >= 100 parcels and < 2% rainfed).

Steps
-----
1. Read the out-of-fold predictions of every run (as 04_compare_models.py) and keep the parcels predicted in all
   runs. Add zone, agricultural year, area and patch purity (p8 and p16 chips) from the dataset.
2. Per zone and agricultural year: parcels per class, rainfed recall and precision, false-rainfed rate
   (share of irrigated parcels predicted rainfed), and macro-F1 where both classes have >= MIN_PER_CLASS parcels.
3. Focus zone only (Morocco: Doukkala): macro-F1 with a 95% block-bootstrap CI (the zone's blocks resampled,
   the same draws for every run) and the paired difference against the baseline.
4. Irrigated-only zones (Morocco: Gharb, Tadla, El Haouz): false-rainfed rate with a 95% block-bootstrap CI.
5. By patch purity (16-px chips) and by parcel area: macro-F1 per bin.
6. Save tables and one figure.

Output: experiments/<EXPERIMENT>/results/<REGION>/breakdown/
------------------------------------------------------------
by_zone_season.csv           per run x classifier x zone x agricultural year
focus_zone.csv               macro-F1 in the focus zone with CI, and difference vs the baseline with CI
irrigated_only_zones.csv     false-rainfed rate in the irrigated-only zones with CI
by_purity.csv, by_area.csv   macro-F1 per bin
breakdown.png                focus-zone macro-F1 and false-rainfed rate per input

Run (from the project folder)
-----------------------------
    python models_run/05_breakdown_oof.py
"""
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_utils import env_override, quick_from_env, results_root, views_name  # noqa: E402

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
EXPERIMENT = "E02_frozen_pca"                              # folder in experiments/ whose runs are analysed
REGION = "morocco"                                         # "morocco" | "utah"
VIEWS_NAME = None                                          # None = the region's default views folder
EXCLUDE_RUNS = []                                          # e.g. ["agrifm_agyear_tiles_concat"] for E01 (random AgriFM weights)
FOCUS_ZONE = {"morocco": "Doukkala", "utah": None}         # None = the zone with most rainfed parcels
IRRIGATED_ONLY = {"morocco": ["Gharb", "Tadla", "El Haouz"], "utah": None}   # None = found automatically
CLASSIFIERS = ["rf", "logreg", "svm", "mlp", "ltae", "psetae"]   # heads to report (all = None)
WINDOW = "agyear"
MIN_PER_CLASS = 20                                         # macro-F1 only where both classes have this many parcels
PURITY_BINS = [0, 0.25, 0.5, 0.75, 1.0]
AREA_BINS_HA = [0, 0.5, 1, 2, 5, np.inf]
N_BOOT = 1000
SEED = 42
# ==========================================================================

EXPERIMENT = os.environ.get("EXPERIMENT", EXPERIMENT)
env_override(globals())
VIEWS_NAME = views_name(REGION, VIEWS_NAME)
RUNS_DIR = results_root(EXPERIMENT, quick_from_env()) / REGION
OUT_DIR = RUNS_DIR / "breakdown"
TILE_ROOT = ROOT / "outputs" / REGION / "dataset" / "03_tiles_128px" / WINDOW
ZONE_NAMES = {"el haouz": "El Haouz"}                      # spelling in the survey -> name used here
LE_RUN = re.compile(r"_f\d{3}_d\d+$")                     # label-efficiency runs are not broken down


# ------------------------------------------------------------ metrics (rainfed = 2 is the positive class)
def counts(y, p):
    t, q = y == 2, p == 2
    return (t & q).sum(), (~t & q).sum(), (t & ~q).sum(), (~t & ~q).sum()


def f1(a, b, c):
    return 2 * a / (2 * a + b + c) if (2 * a + b + c) else np.nan


def macro_f1(y, p):
    tp, fp, fn, tn = counts(y, p)
    return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2


def false_rainfed_rate(y, p):
    irr = y == 1
    return (p[irr] == 2).mean() if irr.any() else np.nan


def block_samples(blocks, rng):
    """N_BOOT index arrays, whole blocks drawn with replacement."""
    groups = {b: np.flatnonzero(blocks == b) for b in np.unique(blocks)}
    names = np.array(list(groups))
    return [np.concatenate([groups[b] for b in rng.choice(names, len(names))]) for _ in range(N_BOOT)]


# ------------------------------------------------------------ data
def load():
    tables = []
    for f in sorted(RUNS_DIR.glob("*/oof_predictions.csv")):
        if f.parent.name in EXCLUDE_RUNS or LE_RUN.search(f.parent.name):
            continue
        t = pd.read_csv(f, dtype={"parcel_id": str}).rename(columns={"model": "classifier"})
        tables.append(t.assign(run=f.parent.name))
    oof = pd.concat(tables, ignore_index=True)
    if CLASSIFIERS:
        oof = oof[oof["classifier"].isin(CLASSIFIERS)]
    common = set.intersection(*oof.groupby("run")["parcel_id"].apply(set))
    oof = oof[oof["parcel_id"].isin(common)]

    parcels = pd.read_csv(TILE_ROOT / "parcels.csv", dtype={"parcel_id": str})
    parcels["zone"] = parcels["zone"].replace(ZONE_NAMES)
    meta = parcels.set_index("parcel_id")[["zone", "ag_season", "area_ha"]]
    for ps in (8, 16):
        f = TILE_ROOT / VIEWS_NAME / f"chips_p{ps}" / "index.csv"
        if f.exists():                                     # chips stay on the cluster for Utah; purity then from qa_purity.csv
            idx = pd.read_csv(f, dtype={"parcel_id": str})
            meta[f"purity_p{ps}"] = idx.drop_duplicates("parcel_id").set_index("parcel_id")["purity"]
    qa = TILE_ROOT / VIEWS_NAME / "qa_purity.csv"
    if "purity_p16" not in meta and qa.exists():
        q = pd.read_csv(qa, dtype={"parcel_id": str})
        if "parcel_id" in q and "purity_p16" in q:
            meta["purity_p16"] = q.drop_duplicates("parcel_id").set_index("parcel_id")["purity_p16"]
    oof = oof.join(meta, on="parcel_id")
    baseline = next((r for r in oof["run"].unique() if r.startswith("s1s2")), None)   # E05 has none
    print(f"{len(common):,} parcels in all runs | runs: {sorted(oof['run'].unique())} | baseline {baseline}")
    return oof, baseline


# ------------------------------------------------------------ tables
def by_group(oof, keys):
    rows = []
    for k, g in oof.groupby(["run", "classifier"] + keys, observed=True):
        y, p = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        tp, fp, fn, tn = counts(y, p)
        n_irr, n_rf = int((y == 1).sum()), int((y == 2).sum())
        both = min(n_irr, n_rf) >= MIN_PER_CLASS
        rows.append(dict(zip(["run", "classifier"] + keys, k), n_irrigated=n_irr, n_rainfed=n_rf,
                         macro_f1=macro_f1(y, p) if both else np.nan,
                         recall_rainfed=tp / n_rf if n_rf else np.nan,
                         precision_rainfed=tp / (tp + fp) if (tp + fp) else np.nan,
                         false_rainfed_rate=false_rainfed_rate(y, p)))
    return pd.DataFrame(rows)


def with_ci(oof, mask, metric, baseline):
    """metric with block-bootstrap CI for the parcels in mask; paired difference vs the baseline (same classifier)."""
    sub = oof[mask]
    ref = sub.drop_duplicates("parcel_id").set_index("parcel_id")
    ids = ref.index.to_numpy()
    y = ref["y_true"].to_numpy()
    samples = block_samples(ref["block_id"].to_numpy(), np.random.default_rng(SEED))
    preds = {k: g.set_index("parcel_id").loc[ids, "y_pred"].to_numpy() for k, g in sub.groupby(["run", "classifier"])}
    draws = {k: np.array([metric(y[s], p[s]) for s in samples]) for k, p in preds.items()}
    rows = []
    for (run, clf), p in preds.items():
        row = {"run": run, "classifier": clf, "n_parcels": len(y), "n_rainfed": int((y == 2).sum()),
               "n_blocks": ref["block_id"].nunique(), "estimate": metric(y, p),
               "ci_low": np.nanquantile(draws[(run, clf)], 0.025), "ci_high": np.nanquantile(draws[(run, clf)], 0.975)}
        b = (baseline, clf) if (baseline, clf) in preds else (baseline, "rf")
        if baseline is not None and run != baseline and b in preds:
            d = draws[(run, clf)] - draws[b]
            row.update(delta_vs_baseline=metric(y, p) - metric(y, preds[b]),
                       delta_ci_low=np.nanquantile(d, 0.025), delta_ci_high=np.nanquantile(d, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def zones(oof):
    """Focus zone and irrigated-only zones of the region (settings, or found from the labels)."""
    per_zone = oof.drop_duplicates("parcel_id").groupby("zone")["y_true"].agg(n="size", rainfed=lambda s: (s == 2).sum())
    focus = FOCUS_ZONE.get(REGION) or per_zone["rainfed"].idxmax()
    irr = IRRIGATED_ONLY.get(REGION)
    if irr is None:
        irr = list(per_zone[(per_zone["n"] >= 100) & (per_zone["rainfed"] < 0.02 * per_zone["n"])].index)
    return focus, irr


def plot(douk, irr_only, path, focus, irr_zones):
    panels = [(douk, f"{focus} only: macro-F1")]
    if not irr_only.empty:
        panels.append((irr_only, f"{', '.join(irr_zones[:4])}{' …' if len(irr_zones) > 4 else ''}: "
                                 f"irrigated predicted rainfed"))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), dpi=130, squeeze=False)
    axes = axes[0]
    for ax, (t, title) in zip(axes, panels):
        runs = list(dict.fromkeys(t["run"]))
        clfs = list(dict.fromkeys(t["classifier"]))
        w = 0.8 / len(clfs)
        for j, c in enumerate(clfs):
            s = t[t["classifier"] == c].set_index("run").reindex(runs)
            x = np.arange(len(runs)) + (j - (len(clfs) - 1) / 2) * w
            ax.errorbar(x, s["estimate"], yerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]],
                        fmt="o", capsize=3, label=c)
        ax.set_xticks(range(len(runs)), [r.split("_")[0] for r in runs], rotation=20)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, title="classifier")
    fig.suptitle(f"{EXPERIMENT} · {REGION}: out-of-fold results by zone (95% block-bootstrap CI)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof, baseline = load()

    zs = by_group(oof, ["zone", "ag_season"])
    zs.round(3).to_csv(OUT_DIR / "by_zone_season.csv", index=False)

    focus, irr_zones = zones(oof)
    douk = with_ci(oof, oof["zone"] == focus, macro_f1, baseline)
    douk.round(4).to_csv(OUT_DIR / "focus_zone.csv", index=False)
    print(f"\n{focus} only ({douk['n_parcels'].iloc[0]:,} parcels, {douk['n_rainfed'].iloc[0]} rainfed, "
          f"{douk['n_blocks'].iloc[0]} blocks): macro-F1\n{douk.drop(columns=['n_parcels', 'n_rainfed', 'n_blocks']).round(3).to_string(index=False)}")

    if not irr_zones or not oof["zone"].isin(irr_zones).any():
        irr = pd.DataFrame()
        print("\nNo irrigated-only zone in this region (every zone with >= 100 parcels has >= 2% rainfed): skipped")
    else:
        irr = with_ci(oof, oof["zone"].isin(irr_zones), false_rainfed_rate, baseline)
        irr.round(4).to_csv(OUT_DIR / "irrigated_only_zones.csv", index=False)
    if not irr.empty:
        print(f"\nIrrigated-only zones {irr_zones} ({irr['n_parcels'].iloc[0]:,} parcels): share of irrigated predicted rainfed\n"
          f"{irr.drop(columns=['n_parcels', 'n_rainfed', 'n_blocks']).round(3).to_string(index=False)}")

    if "purity_p16" not in oof:
        oof["purity_p16"] = np.nan
    oof["purity_bin"] = pd.cut(oof["purity_p16"], PURITY_BINS, include_lowest=True)
    oof["area_bin"] = pd.cut(oof["area_ha"], AREA_BINS_HA)
    pur = by_group(oof, ["purity_bin"])
    area = by_group(oof, ["area_bin"])
    pur.round(3).to_csv(OUT_DIR / "by_purity.csv", index=False)
    area.round(3).to_csv(OUT_DIR / "by_area.csv", index=False)
    for name, t, key in [("16-px patch purity", pur, "purity_bin"), ("parcel area (ha)", area, "area_bin")]:
        if t.empty:
            print(f"\nMacro-F1 by {name}: no data (Utah: per-parcel purity needs the chip index, on the cluster)")
            continue
        wide = t.pivot_table(index=key, columns=["run", "classifier"], values="macro_f1", observed=True).round(3)
        wide.columns = [f"{r.split('_')[0]}/{c}" for r, c in wide.columns]
        n = t.drop_duplicates(key).set_index(key)[["n_irrigated", "n_rainfed"]]
        print(f"\nMacro-F1 by {name}:\n{n.join(wide).to_string()}")

    plot(douk, irr, OUT_DIR / "breakdown.png", focus, irr_zones)
    print(f"\nSaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
