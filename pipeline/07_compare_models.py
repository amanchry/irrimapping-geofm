"""
07_compare_models.py  —  compare all inputs (S1/S2 baseline and foundation models) on the same parcels
======================================================================================================

What it does
------------
Collects the out-of-fold predictions of every run of 06_train_evaluate.py and answers the main
research question: does a foundation-model input classify irrigated vs rainfed parcels better than
the Sentinel-1/2 baseline, under the same spatial validation?

Steps
-----
1. Find every run in outputs/06_models/ (each has oof_predictions.csv).
2. Keep the parcels predicted in ALL runs, so every input is scored on exactly the same parcels.
3. For every run x classifier: macro-F1, rainfed F1, balanced accuracy and overall accuracy with a
   95% block-bootstrap interval. The SAME bootstrap samples (whole 20 km blocks) are used for every
   run, which makes the differences paired.
4. Difference in macro-F1 against the baseline run (same classifier), with its 95% interval and a
   bootstrap p-value; p-values are adjusted for the number of comparisons (Holm). An interval that
   excludes 0 means a real difference (Foody 2009, Remote Sensing of Environment).
5. Save the tables and a figure.

Output (outputs/07_comparison/)
-------------------------------
metrics_all_runs.csv          run, classifier, metric, estimate, ci_low, ci_high
differences_vs_baseline.csv   run, classifier, delta macro-F1, ci_low, ci_high, p, p_holm, differs
comparison_macro_f1.png       macro-F1 with 95% CI per input and classifier
Run (from the project folder, after script 06 has run for each input)
---------------------------------------------------------------------
    python pipeline/07_compare_models.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
RUNS_DIR = ROOT / "outputs" / "06_models"
OUT_DIR = ROOT / "outputs" / "07_comparison"
BASELINE_RUN = "s1s2-all_agyear_30d"                       # run folder of the Sentinel-1/2 baseline
RUNS = None                                                # None = every run found; or a list of run names
N_BOOT = 1000
SEED = 42
# ==========================================================================

def _counts(y, p):
    """Confusion counts with rainfed (2) as positive: tp, fp, fn, tn."""
    pos_t, pos_p = y == 2, p == 2
    return (pos_t & pos_p).sum(), (~pos_t & pos_p).sum(), (pos_t & ~pos_p).sum(), (~pos_t & ~pos_p).sum()


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def macro_f1(y, p):
    tp, fp, fn, tn = _counts(y, p)
    return (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2          # rainfed F1 and irrigated F1


def f1_rainfed(y, p):
    tp, fp, fn, _ = _counts(y, p)
    return _f1(tp, fp, fn)


def balanced_accuracy(y, p):
    tp, fp, fn, tn = _counts(y, p)
    return (tp / (tp + fn) + tn / (tn + fp)) / 2


def overall_accuracy(y, p):
    return (y == p).mean()


# Plain numpy versions (fast enough for thousands of bootstrap samples); same definitions as scikit-learn.
METRICS = {"macro_f1": macro_f1, "f1_rainfed": f1_rainfed,
           "balanced_accuracy": balanced_accuracy, "overall_accuracy": overall_accuracy}


def load_runs():
    """All out-of-fold predictions: one table with run, classifier, parcel_id, y_true, y_pred, block_id."""
    tables = []
    for f in sorted(RUNS_DIR.glob("*/oof_predictions.csv")):
        run = f.parent.name
        if RUNS is not None and run not in RUNS:
            continue
        t = pd.read_csv(f, dtype={"parcel_id": str})
        tables.append(t.rename(columns={"model": "classifier"}).assign(run=run))
        print(f"  found {run}: {t['parcel_id'].nunique():,} parcels, classifiers {sorted(t['model'].unique())}")
    if not tables:
        raise SystemExit(f"No runs with oof_predictions.csv in {RUNS_DIR}")
    return pd.concat(tables, ignore_index=True)


def bootstrap_samples(blocks, n_boot, seed):
    """Index arrays of n_boot block-bootstrap samples (whole blocks drawn with replacement)."""
    groups = {b: np.flatnonzero(blocks == b) for b in np.unique(blocks)}
    names = np.array(list(groups))
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[b] for b in rng.choice(names, size=len(names), replace=True)])
            for _ in range(n_boot)]


def plot_comparison(table, path):
    m = table[table["metric"] == "macro_f1"]
    runs = list(dict.fromkeys(m["run"]))
    classifiers = list(dict.fromkeys(m["classifier"]))
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(runs)), 4.5), dpi=130)
    width = 0.8 / len(classifiers)
    colors = plt.cm.tab10.colors
    for j, clf in enumerate(classifiers):
        sub = m[m["classifier"] == clf].set_index("run").reindex(runs)
        x = np.arange(len(runs)) + (j - (len(classifiers) - 1) / 2) * width
        ax.errorbar(x, sub["estimate"], yerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
                    fmt="o", color=colors[j], capsize=3, label=clf)
    ax.set_xticks(range(len(runs)), runs, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1 (95% block-bootstrap CI)")
    ax.set_title("Irrigated vs rainfed: all inputs on the same parcels (nested spatial CV)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False, title="classifier")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def holm(p):
    """Holm-adjusted p-values (controls the chance of any false 'difference' across all comparisons)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Runs:")
    oof = load_runs()

    # 2. Parcels present in every run.
    per_run = oof.groupby("run")["parcel_id"].apply(set)
    common = sorted(set.intersection(*per_run))
    oof = oof[oof["parcel_id"].isin(common)]
    print(f"{len(common):,} parcels predicted in all {len(per_run)} runs")

    # Same parcel order and the same bootstrap samples for every run x classifier.
    ref = oof.drop_duplicates("parcel_id").set_index("parcel_id").loc[common]
    y = ref["y_true"].to_numpy()
    samples = bootstrap_samples(ref["block_id"].to_numpy(), N_BOOT, SEED)
    preds = {(r, c): g.set_index("parcel_id").loc[common, "y_pred"].to_numpy()
             for (r, c), g in oof.groupby(["run", "classifier"])}

    # 3. Metrics with CI.
    rows, boot_f1 = [], {}
    for (run, clf), p in preds.items():
        for name, fn in METRICS.items():
            draws = np.array([fn(y[s], p[s]) for s in samples])
            if name == "macro_f1":
                boot_f1[(run, clf)] = draws
            rows.append({"run": run, "classifier": clf, "metric": name, "estimate": fn(y, p),
                         "ci_low": np.quantile(draws, 0.025), "ci_high": np.quantile(draws, 0.975)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "metrics_all_runs.csv", index=False)
    wide = table[table["metric"] == "macro_f1"].pivot(index="run", columns="classifier", values="estimate").round(3)
    print(f"\nMacro-F1 (rows = input, columns = classifier):\n{wide.to_string()}")

    # 4. Differences against the baseline, same classifier.
    if BASELINE_RUN in per_run.index:
        diffs = []
        for (run, clf), draws in boot_f1.items():
            if run == BASELINE_RUN or (BASELINE_RUN, clf) not in boot_f1:
                continue
            d = draws - boot_f1[(BASELINE_RUN, clf)]
            point = METRICS["macro_f1"](y, preds[(run, clf)]) - METRICS["macro_f1"](y, preds[(BASELINE_RUN, clf)])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            diffs.append({"run": run, "classifier": clf, "delta_macro_f1": point,
                          "ci_low": np.quantile(d, 0.025), "ci_high": np.quantile(d, 0.975), "p": p})
        if diffs:
            diffs = pd.DataFrame(diffs)
            diffs["p_holm"] = holm(diffs["p"])
            diffs["differs"] = (diffs["ci_low"] > 0) | (diffs["ci_high"] < 0)
            diffs.round(4).to_csv(OUT_DIR / "differences_vs_baseline.csv", index=False)
            print(f"\nDifference in macro-F1 vs {BASELINE_RUN} (positive = better than the baseline):\n"
                  f"{diffs.round(3).to_string(index=False)}")
    else:
        print(f"\nBaseline run '{BASELINE_RUN}' not found: no differences computed.")

    plot_comparison(table, OUT_DIR / "comparison_macro_f1.png")
    print(f"\nSaved tables and figure in {OUT_DIR}")


if __name__ == "__main__":
    main()
