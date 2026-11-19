"""
E01_s2_pixel_baseline/run.py  —  E01: Sentinel-2 pixel time series with random forest and gradient boosting
============================================================================================================

Question: how well do the raw Sentinel-2 time series of single pixels separate irrigated, rainfed and
not-cropland pixels? E01 is the reference for every later experiment.

What it does
------------
1. Reads the labelled pixels of pixelwise_run/04_make_pixel_samples.py (pixelwise_run/pixel_common.py): Morocco 2023/24 + 2024/25, Utah 2023/24.
2. Features per pixel: 12 months x (12 Sentinel-2 bands + NDVI + NDWI) = 168 values. Months without a clear
   observation are filled by linear interpolation in time; a band never observed is filled with the training median.
3. PASTIS rotation (5 rounds): every hyperparameter set in the grid is trained on the 3 training folds and scored
   on the validation fold (3-class macro-F1); the best one predicts the test fold.
4. Out-of-fold metrics with 95% block-bootstrap CI; grouped permutation importance per month and per band
   (random forest; test folds; drop in 3-class macro-F1).

Output: experiments/E01_s2_pixel_baseline/results/<region>/<model>/
    oof_predictions.parquet, metrics.csv, metrics_by_group.csv, confusion.png, run_config.json, run.log,
    chosen_params.csv, importance_month.csv, importance_band.csv (rf)

Run (from the project folder; CPU)
----------------------------------
    python experiments/E01_s2_pixel_baseline/run.py
    # cluster: sbatch --export=ALL,SCRIPT=experiments/E01_s2_pixel_baseline/run.py,EXPERIMENT_REGION=utah jobs/step_cpu.sbatch
QUICK_TEST = True runs on 20,000 pixels with a small grid (fast check of the chain).
"""
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pixelwise_run"))
from pixel_common import (CLASSES, S2_BANDS, add_indices, env_setting, feature_names, fill_time,  # noqa: E402
                          get_logger, load_samples, results_dir, rounds, s2_series, save_run)

# ================================ SETTINGS ================================
EXP_DIR = Path(__file__).resolve().parent
REGION = env_setting("REGION", "morocco")          # "morocco" | "utah"
MODELS = ["rf", "hgb"]                             # random forest, histogram gradient boosting
RF_GRID = {"max_features": ["sqrt", 0.33], "min_samples_leaf": [1, 5]}
RF_TREES = 500
HGB_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]}
HGB_MAX_ITER = 1000                                # early stopping on an internal 10% split (n_iter_no_change 20)
IMPORTANCE = True                                  # permutation importance per month and band (rf)
IMPORTANCE_PIXELS = 20000                          # test pixels per round used for the importance
N_JOBS = -1
SEED = 42
QUICK_TEST = env_setting("QUICK_TEST", False)
# ==========================================================================

if QUICK_TEST:
    RF_GRID = {"max_features": ["sqrt"], "min_samples_leaf": [5]}
    HGB_GRID = {"learning_rate": [0.1], "max_leaf_nodes": [31]}
    RF_TREES, HGB_MAX_ITER, IMPORTANCE_PIXELS = 100, 200, 2000


def make_model(name, params):
    if name == "rf":
        return RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced", n_jobs=N_JOBS,
                                      random_state=SEED, **params)
    return HistGradientBoostingClassifier(max_iter=HGB_MAX_ITER, class_weight="balanced", early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, random_state=SEED, **params)


def grid(name):
    g = RF_GRID if name == "rf" else HGB_GRID
    return [dict(zip(g, v)) for v in product(*g.values())]


def macro_f1(y, p):
    return f1_score(y, np.array(CLASSES)[p.argmax(1)], labels=CLASSES, average="macro", zero_division=0)


def permutation_importance(model, x, y, rng):
    """Drop in 3-class macro-F1 when all features of one month (or one band over all months) are shuffled."""
    base = macro_f1(y, model.predict_proba(x))
    names = feature_names()
    groups = {f"m{m + 1:02d}": [i for i, n in enumerate(names) if n.endswith(f"_m{m + 1:02d}")] for m in range(12)}
    groups.update({b: [i for i, n in enumerate(names) if n.startswith(f"{b}_m")] for b in S2_BANDS + ["NDVI", "NDWI"]})
    out = {}
    for g, cols in groups.items():
        xp = x.copy()
        perm = rng.permutation(len(x))
        xp[:, cols] = x[perm][:, cols]
        out[g] = base - macro_f1(y, model.predict_proba(xp))
    return out


def main():
    df, X, info = load_samples(REGION, quick=QUICK_TEST)
    feats = add_indices(fill_time(s2_series(X, info))).reshape(len(df), -1)        # [N, 168]
    y, folds = df["label"].to_numpy(), df["fold"].to_numpy()
    for name in MODELS:
        out_dir = results_dir(EXP_DIR, REGION, name, QUICK_TEST)
        log = get_logger(out_dir)
        log.info(f"E01 {REGION} {name}: {len(df):,} pixels, {feats.shape[1]} features")
        proba = np.zeros((len(df), 3), dtype=np.float32)
        chosen, imp_rows = [], []
        for k, train_f, val_f, test_f in rounds():
            tr, va, te = np.isin(folds, train_f), folds == val_f, folds == test_f
            med = np.nanmedian(feats[tr], axis=0)
            x = np.where(np.isnan(feats), med, feats)
            best = (-1, None, None)
            for params in grid(name):
                t0 = time.time()
                m = make_model(name, params).fit(x[tr], y[tr])
                f1 = macro_f1(y[va], m.predict_proba(x[va]))
                log.info(f"  round {k + 1} {params}: validation macro-F1 {f1:.4f} ({time.time() - t0:.0f}s)")
                if f1 > best[0]:
                    best = (f1, params, m)
            proba[te] = best[2].predict_proba(x[te])
            chosen.append({"round": k + 1, "test_fold": test_f, "val_fold": val_f, "val_macro_f1": best[0], **best[1]})
            if IMPORTANCE and name == "rf":
                rng = np.random.default_rng(SEED + k)
                idx = np.flatnonzero(te)
                idx = rng.choice(idx, min(IMPORTANCE_PIXELS, len(idx)), replace=False)
                imp_rows.append({"round": k + 1, **permutation_importance(best[2], x[idx], y[idx], rng)})
        tables = {"chosen_params.csv": pd.DataFrame(chosen)}
        if imp_rows:
            imp = pd.DataFrame(imp_rows).drop(columns="round").agg(["mean", "std"]).T.reset_index(names="group")
            tables["importance_month.csv"] = imp[imp["group"].str.match(r"m\d\d")]
            tables["importance_band.csv"] = imp[~imp["group"].str.match(r"m\d\d")]
        settings = {"experiment": "E01", "region": REGION, "model": name, "features": "12 months x (12 S2 bands + NDVI + NDWI)",
                    "grid": RF_GRID if name == "rf" else HGB_GRID, "rf_trees": RF_TREES, "hgb_max_iter": HGB_MAX_ITER,
                    "seed": SEED}
        save_run(out_dir, df, proba, settings, log, tables, QUICK_TEST)


if __name__ == "__main__":
    main()
