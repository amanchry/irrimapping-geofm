"""
04_compare_models.py  —  compare all inputs (S2 baseline and foundation models) on the same parcels
==================================================================================================

What it does
------------
Collects the out-of-fold predictions of every run of 03_train_evaluate.py and answers the main
research question: does a foundation-model input classify irrigated vs rainfed parcels better than
the Sentinel-2 baseline, under the same spatial validation? One region at a time (REGION).

Steps
-----
1. Find every main run of the experiment in the region (each has oof_predictions.csv), plus EXTRA_RUNS from
   other experiments (e.g. to compare PCA and L-TAE runs of the same input). Label-efficiency runs
   (names ending in _f010_d1 ...) are kept apart for step 6.
2. Keep the parcels predicted in ALL runs, so every input is scored on exactly the same parcels.
3. For every run x classifier: macro-F1, rainfed F1, balanced accuracy and overall accuracy with a
   95% block-bootstrap interval. The SAME bootstrap samples (whole 20 km blocks) are used for every
   run, which makes the differences paired.
4. Difference in macro-F1 against the baseline run (same classifier; runs with their own model, such as PSE-TAE, against the baseline's BASELINE_CLASSIFIER), with its 95% interval and a
   bootstrap p-value; p-values are adjusted for the number of comparisons (Holm). A difference counts
   ("differs") only if the 95% interval excludes 0 AND the Holm-adjusted p-value is below 0.05
   (Foody 2009, Remote Sensing of Environment; evaluation protocol §8).
5. Save the tables and a figure.
6. Label efficiency: for every run and classifier, macro-F1 and rainfed F1 at each training fraction (mean over
   the block draws, with a block-bootstrap CI on the same parcels and samples), the 100% point from the main run.

Output: experiments/<EXPERIMENT>/results/<REGION>/comparison/ (experiment mode)
        or outputs/<REGION>/study/04_comparison/ (EXPERIMENT = None)
-------------------------------
metrics_all_runs.csv          run, classifier, metric, estimate, ci_low, ci_high
differences_vs_baseline.csv   run, classifier, delta macro-F1, ci_low, ci_high, p, p_holm, ci_excludes_0, differs
comparison_macro_f1.png       macro-F1 with 95% CI per input and classifier
learning_curve.csv / .png     label efficiency (if the experiment has label-efficiency runs)
Run (from the project folder, after 03_train_evaluate.py has run for each input)
---------------------------------------------------------------------
    python experiments/run_experiment.py      # experiment mode: runs this script as the "compare" step
    python models_run/04_compare_models.py      # development mode (EXPERIMENT = None)
"""
import json
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
from experiment_utils import env_override, quick_from_env, results_root  # noqa: E402

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
EXPERIMENT = None              # e.g. "E02_frozen_pca": runs of experiments/<name>/results/ | None = settings below
REGION = "morocco"             # "morocco" | "utah": one region per comparison
EXTRA_RUNS = []                # runs of other experiments in the same region, as "<experiment>/<run>" (experiment mode)
RUNS_DIR = ROOT / "outputs" / REGION / "study" / "03_models"
OUT_DIR = ROOT / "outputs" / REGION / "study" / "04_comparison"
BASELINE_RUN = "s1s2-s2_agyear_tiles-inner"                # run folder of the baseline (Sentinel-2 only, D14)
RUNS = None                    # None = every run in RUNS_DIR | list of run folders
BASELINE_CLASSIFIER = "rf"     # runs with their own model (e.g. psetae) are compared with this baseline classifier
N_BOOT = 1000
SEED = 42

# ------------------------------------------------------------ experiment mode
# experiments/run_experiment.py sets EXPERIMENT; all runs in its results/ folder are compared, the
# feature run (s1s2-..., Sentinel-2 only by default) is the baseline, and the section "compare" of config.yaml can change the settings.
EXPERIMENT = os.environ.get("EXPERIMENT", EXPERIMENT)
env_override(globals())                                      # REGION from run_experiment.py
QUICK = quick_from_env()                                     # quick check: experiments/<E>/results_quick/
RUNS_DIR = ROOT / "outputs" / REGION / "study" / "03_models"  # development mode (after the region override)
OUT_DIR = ROOT / "outputs" / REGION / "study" / "04_comparison"
if EXPERIMENT:
    import yaml
    RUNS_DIR = results_root(EXPERIMENT, QUICK) / REGION
    OUT_DIR = RUNS_DIR / "comparison"
    RUNS = None
    _found = sorted(f.parent.name for f in RUNS_DIR.glob("*/oof_predictions.csv"))
    _main = [r for r in _found if not re.search(r"_f\d{3}_d\d+$", r)]
    BASELINE_RUN = next((r for r in _main if r.startswith("s1s2-s2")),      # this experiment's own baseline run
                        next((r for r in _main if r.startswith("s1s2-")), None))
    _cfg = yaml.safe_load((ROOT / "experiments" / EXPERIMENT / "config.yaml").read_text(encoding="utf-8"))
    for _key, _value in (_cfg.get("compare") or {}).items():
        if _key not in globals():
            raise SystemExit(f"config.yaml: unknown setting '{_key}' for 04_compare_models.py")
        globals()[_key] = _value
    if BASELINE_RUN is None:                                     # no baseline run of its own (E05): the E02 baseline
        BASELINE_RUN = next((r for r in EXTRA_RUNS if "/s1s2-s2" in r), "E02_frozen_pca/s1s2-s2_agyear_tiles-inner")
LE_RUN = re.compile(r"^(?P<base>.+)_f(?P<frac>\d{3})_d(?P<draw>\d+)$")   # label-efficiency run names
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


def load_runs(label_efficiency=False):
    """Out-of-fold predictions: one table with run, classifier, parcel_id, y_true, y_pred, block_id.
    label_efficiency=False: main runs (+ EXTRA_RUNS); True: only the label-efficiency runs of this experiment."""
    tables = []
    files = [(f.parent.name, f) for f in sorted(RUNS_DIR.glob("*/oof_predictions.csv"))
             if bool(LE_RUN.match(f.parent.name)) == label_efficiency]
    if not label_efficiency:
        for r in EXTRA_RUNS:                                             # "<experiment>/<run>" or "<experiment>/*"
            exp, name = r.split("/", 1)
            exp_dir = results_root(exp, QUICK) / REGION
            if name == "*":                                              # every main run of that experiment
                found = [(f"{exp}/{f.parent.name}", f) for f in sorted(exp_dir.glob("*/oof_predictions.csv"))
                         if not LE_RUN.match(f.parent.name)]
                if not found:
                    raise SystemExit(f"EXTRA_RUNS {r}: no finished run in {exp_dir} (run {exp} first)")
                files += found
            else:
                files.append((r, exp_dir / name / "oof_predictions.csv"))
    if label_efficiency and not files:
        return None
    for run, f in files:
        if RUNS is not None and run not in RUNS and not label_efficiency:
            continue
        if not f.exists():
            raise SystemExit(f"{f} not found (EXTRA_RUNS)")
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
    ax.set_title(f"Irrigated vs rainfed: all inputs on the same parcels{' (' + EXPERIMENT + ')' if EXPERIMENT else ''}",
                 fontweight="bold")
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
    global BASELINE_RUN
    if BASELINE_RUN not in per_run.index:                       # e.g. E05: the S2 baseline among the EXTRA_RUNS
        BASELINE_RUN = next((r for r in per_run.index if r.split("/")[-1].startswith("s1s2-s2")
                             and not LE_RUN.match(r.split("/")[-1])), BASELINE_RUN)
    if BASELINE_RUN in per_run.index:
        print(f"Baseline run: {BASELINE_RUN}")
        diffs = []
        for (run, clf), draws in boot_f1.items():
            if run == BASELINE_RUN or "/" in run:              # own runs only: the Holm family is this experiment's
                continue
            ref_clf = clf if (BASELINE_RUN, clf) in boot_f1 else BASELINE_CLASSIFIER   # e.g. psetae vs baseline rf
            if (BASELINE_RUN, ref_clf) not in boot_f1:
                continue
            d = draws - boot_f1[(BASELINE_RUN, ref_clf)]
            point = METRICS["macro_f1"](y, preds[(run, clf)]) - METRICS["macro_f1"](y, preds[(BASELINE_RUN, ref_clf)])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            diffs.append({"run": run, "classifier": clf, "baseline_classifier": ref_clf, "delta_macro_f1": point,
                          "ci_low": np.quantile(d, 0.025), "ci_high": np.quantile(d, 0.975), "p": p})
        if diffs:
            diffs = pd.DataFrame(diffs)
            diffs["p_holm"] = holm(diffs["p"])
            diffs["ci_excludes_0"] = (diffs["ci_low"] > 0) | (diffs["ci_high"] < 0)
            diffs["differs"] = diffs["ci_excludes_0"] & (diffs["p_holm"] < 0.05)     # protocol §8: CI and Holm
            diffs.round(4).to_csv(OUT_DIR / "differences_vs_baseline.csv", index=False)
            print(f"\nDifference in macro-F1 vs {BASELINE_RUN} (positive = better than the baseline):\n"
                  f"{diffs.round(3).to_string(index=False)}")
    else:
        raise SystemExit(f"Baseline run '{BASELINE_RUN}' not found among the runs: no comparison possible")

    # 5. This experiment's runs against the earlier experiments' runs of the SAME input (EXTRA_RUNS):
    #    E03 MLP vs E02 LogReg / RF / SVM, E04 L-TAE vs E03 MLP and E02 classifiers. Own Holm family.
    def input_key(run):
        return re.sub(r"_(concat|mean|mlp|ltae)$", "", run.split("/")[-1])
    own = [(r, c) for (r, c) in boot_f1 if "/" not in r]
    other = [(r, c) for (r, c) in boot_f1 if "/" in r]
    pairs = []
    for (run, clf) in own:
        for (orun, oclf) in other:
            if input_key(run) != input_key(orun):
                continue
            d = boot_f1[(run, clf)] - boot_f1[(orun, oclf)]
            point = METRICS["macro_f1"](y, preds[(run, clf)]) - METRICS["macro_f1"](y, preds[(orun, oclf)])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            pairs.append({"run": run, "classifier": clf, "other_run": orun, "other_classifier": oclf,
                          "delta_macro_f1": point, "ci_low": np.quantile(d, 0.025), "ci_high": np.quantile(d, 0.975),
                          "p": p})
    if pairs:
        pairs = pd.DataFrame(pairs)
        pairs["p_holm"] = holm(pairs["p"])
        pairs["ci_excludes_0"] = (pairs["ci_low"] > 0) | (pairs["ci_high"] < 0)
        pairs["differs"] = pairs["ci_excludes_0"] & (pairs["p_holm"] < 0.05)
        pairs.round(4).to_csv(OUT_DIR / "differences_vs_earlier_experiments.csv", index=False)
        print(f"\nSame input, this experiment's head minus the earlier experiments' heads:\n"
              f"{pairs.round(3).to_string(index=False)}")

    plot_comparison(table, OUT_DIR / "comparison_macro_f1.png")
    learning_curve(preds, common, y, samples)
    print(f"\nSaved tables and figure in {OUT_DIR}")


def learning_curve(main_preds, common, y, samples):
    """Label efficiency: macro-F1 and rainfed F1 per run, classifier and training fraction, on the parcels of the
    main comparison and the same bootstrap samples; draws are averaged (the CI covers blocks, the spread over
    draws is given as sd_draws). The 100% point is the main run; for neural runs (oof_per_seed.csv) each seed is
    scored on its own and averaged, so every point of the curve is a single-seed result."""
    le = load_runs(label_efficiency=True)
    if le is None:
        return
    le = le[le["parcel_id"].isin(common)]
    rows = []
    for (run, clf), g in le.groupby(["run", "classifier"]):
        m = LE_RUN.match(run)
        g = g.set_index("parcel_id").reindex(common)
        if g["y_pred"].isna().any():
            print(f"  {run} {clf}: not all common parcels predicted, skipped")
            continue
        rows.append({"base": m["base"], "classifier": clf, "fraction": int(m["frac"]) / 100, "draw": int(m["draw"]),
                     "pred": g["y_pred"].to_numpy().astype(int)})
    for (run, clf), p in main_preds.items():
        seed_file = (RUNS_DIR / run / "oof_per_seed.csv") if "/" not in run else None
        if seed_file is not None and seed_file.exists():           # neural run: one point per seed, like the
            s = pd.read_csv(seed_file, dtype={"parcel_id": str})   # single-seed label-efficiency runs
            for k, (seed, g) in enumerate(s.groupby("seed")):
                g = g.set_index("parcel_id").reindex(common)
                rows.append({"base": run, "classifier": clf, "fraction": 1.0, "draw": k + 1,
                             "pred": np.where(g["score"].to_numpy() >= 0.5, 2, 1)})
        else:
            rows.append({"base": run, "classifier": clf, "fraction": 1.0, "draw": 0, "pred": p})
    out = []
    groups = pd.DataFrame(rows).groupby(["base", "classifier", "fraction"])
    for (base, clf, frac), g in groups:
        for name in ("macro_f1", "f1_rainfed"):
            fn = METRICS[name]
            per_draw = np.array([fn(y, p) for p in g["pred"]])
            boot = np.array([np.mean([fn(y[s], p[s]) for p in g["pred"]]) for s in samples])
            out.append({"run": base, "classifier": clf, "fraction": frac, "metric": name, "n_draws": len(g),
                        "estimate": per_draw.mean(), "sd_draws": per_draw.std(ddof=1) if len(g) > 1 else 0.0,
                        "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975)})
    curve = pd.DataFrame(out)
    curve = curve[curve.groupby(["run", "classifier"])["fraction"].transform("nunique") > 1]   # runs with LE runs
    parcels = {}                                                   # mean training parcels per round, per fraction
    for f in RUNS_DIR.glob("*/run_config.json"):
        m = LE_RUN.match(f.parent.name)
        cfg = json.loads(f.read_text())
        if m and cfg.get("block_draws"):
            parcels.setdefault((m["base"], int(m["frac"]) / 100), []).append(
                np.mean([d["parcels"] for d in cfg["block_draws"]]))
    curve["train_parcels_per_round"] = [round(float(np.mean(parcels[(r, f)]))) if (r, f) in parcels else np.nan
                                        for r, f in zip(curve["run"], curve["fraction"])]
    if curve.empty:
        return
    curve.round(4).to_csv(OUT_DIR / "learning_curve.csv", index=False)
    classifiers = list(dict.fromkeys(curve["classifier"]))
    fig, axes = plt.subplots(2, len(classifiers), figsize=(4.6 * len(classifiers), 7.5), dpi=130, squeeze=False)
    for j, clf in enumerate(classifiers):
        for i, metric in enumerate(["macro_f1", "f1_rainfed"]):
            ax = axes[i][j]
            for k, (run, g) in enumerate(curve.query("classifier == @clf and metric == @metric").groupby("run")):
                g = g.sort_values("fraction")
                ax.plot(g["fraction"] * 100, g["estimate"], marker="o", color=plt.cm.tab10.colors[k % 10], label=run)
                ax.fill_between(g["fraction"] * 100, g["ci_low"], g["ci_high"], color=plt.cm.tab10.colors[k % 10],
                                alpha=0.12)
            ax.set_xscale("log")
            ax.set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
            ax.set_xlabel("Training blocks used (%)")
            ax.set_title(f"{clf}: {metric.replace('_', ' ')}", fontweight="bold")
            ax.grid(linestyle="--", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Label efficiency ({REGION}{', ' + EXPERIMENT if EXPERIMENT else ''}): mean over block draws, "
                 f"95% block-bootstrap CI", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Label efficiency: {curve['run'].nunique()} runs x {len(classifiers)} classifiers -> learning_curve.csv")


if __name__ == "__main__":
    main()
