"""
03_train_evaluate.py  —  train and evaluate irrigated vs rainfed classifiers for ANY input
==========================================================================================

What it does
------------
One script for every input, so all of them are evaluated in exactly the same way:
  INPUT = "s1s2"       Sentinel-1/2 parcel features per month (ml_dataset_prep/04_make_views.py, view A)  -> the baseline
  INPUT = "clay"       Clay embeddings              (02a_embed_clay.py)
  INPUT = "prithvi"    Prithvi-EO-2.0 embeddings    (02b_embed_prithvi.py)
  INPUT = "terramind"  TerraMind embeddings         (02c_embed_terramind.py)
  INPUT = "agrifm"     AgriFM embeddings            (02d_embed_agrifm.py)
For each input, 4 classifiers are trained on top of the (frozen) features:
  logreg  logistic regression (linear; on embeddings this is the standard "linear probe")
  rf      random forest        hgb  histogram gradient boosting        svm  RBF support vector machine

Steps
-----
1. Read the input features and the spatial folds (02_make_spatial_folds.py); keep parcels in both.
   s1s2:       fill fully cloudy periods by linear interpolation in time, per parcel and variable
               (uses only that parcel's values: no leakage). S1S2_VARIABLES = "s2" / "s1" keeps
               only optical / radar variables (ablation).
   embeddings: months with no clear pixel are left empty (filled with the training median).
               FEATURE_MODE "concat": all steps side by side, then standardised and reduced to
               PCA_DIM components INSIDE each model (PCA fitted on training folds only);
               "mean": the average over steps.
2. Nested spatial cross-validation (main result):
     outer loop  each of the 5 spatial folds (20 km blocks) is the test fold once;
     inner loop  hyperparameters tuned by spatial CV on the other 4 folds only;
     the model is refit on those 4 folds and predicts the test fold.
   Every parcel is predicted once, by a model that never saw its block.
3. Metrics on all out-of-fold predictions with 95% CI from a block bootstrap (whole blocks
   resampled): macro-F1 (main), F1 / precision / recall per class, balanced and overall accuracy,
   PR-AUC for rainfed, Brier score and expected calibration error.
4. Paired comparison of the 4 classifiers on the same parcels (difference in macro-F1, 95% CI).
5. Spatial vs random 5-fold CV with the same hyperparameters: how much random CV overstates accuracy.
6. Grouped permutation importance with the random forest of each outer fold:
   s1s2 by variable (NDVI, VH, ...) and by period; embeddings by step (month).
7. Final models refit on all parcels (for mapping later); all tables and figures saved.
Compare inputs with 04_compare_models.py.

Output (outputs/study/03_models/<RUN_NAME>/)
--------------------------------------
metrics_summary.csv            headline metrics with 95% CI, one column per classifier
metrics_with_ci.csv            all metrics: estimate, ci_low, ci_high
metrics_per_fold.csv           metrics of each outer fold
paired_model_differences.csv   macro-F1 differences between classifiers with CI
cv_random_vs_spatial.csv/.png  optimism of random CV
oof_predictions.csv            every parcel's out-of-fold prediction and rainfed score (used by 04_compare_models.py)
selected_hyperparameters.csv, inner_tuning_scores.csv, importance.csv/.png, confusion_matrices.png,
seasonal_profiles.png (s1s2), gap_filling.csv (s1s2), final_model_<name>.joblib, run_config.json, run.log

Run (from the project folder)
-----------------------------
    python models_run/03_train_evaluate.py
Change INPUT below and run again for each input. QUICK_TEST = True gives a fast check.
"""
import ast
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib
matplotlib.use("Agg")                     # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
INPUT = "terramind"                        # "s1s2" | "clay" | "prithvi" | "terramind" | "agrifm"
WINDOW = "agyear"                     # time window of the dataset (ml_dataset_prep/03_build_tiles.py)

# S1/S2 baseline only
INTERVAL_DAYS = 30                    # legacy Earth Engine table only (S1S2_SOURCE = 'gee'): 30 or 15
START_MONTH = 9                       # first month of the window (labels of the periods)
S1S2_VARIABLES = "all"                # "all" | "s2" (optical only) | "s1" (radar only)
S1S2_SOURCE = "tiles"                 # "tiles" = parcel features cut from the tile store (04_make_views.py, view A;
                                      #           same pixels as the foundation-model chips) | "gee" = old script-03 table
TILE_FEATURES = "inner"               # tiles only: "inner" (edge pixels removed) | "all" (all parcel pixels)
S1_ORBIT = "chips"                    # tiles only: "chips" = same Sentinel-1 orbit as the foundation-model chips
                                      # (views_info.json) | "ASC" | "DSC" | "both" (both orbits as separate variables)

# Foundation-model embeddings only
EMB_SOURCE = "tiles"                  # "tiles" = embeddings of the ml_dataset_prep/04_make_views.py chips (<model>_agyear_tiles.npz) | "legacy" = older separately downloaded chips
FEATURE_MODE = "concat"               # "concat" (all months + PCA) | "mean" (average over months)
PCA_DIM = 128                         # PCA components in concat mode (fitted on training folds only)

SPLIT_FILE = ROOT / "outputs" / "dataset" / "02_folds" / "parcels_folds.geojson"
VIEWS_DIR = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW / "views"
if S1S2_SOURCE == "tiles":
    S1S2_FILE = VIEWS_DIR / f"parcel_features_{TILE_FEATURES}.csv"
    if S1_ORBIT == "chips" and INPUT == "s1s2":                   # orbit chosen by ml_dataset_prep/04_make_views.py for the chips
        if not (VIEWS_DIR / "views_info.json").exists():
            raise SystemExit(f"{VIEWS_DIR / 'views_info.json'} not found: run 03_build_tiles.py and 04_make_views.py "
                             f"first, or set S1S2_SOURCE = 'gee'")
        S1_ORBIT = json.loads((VIEWS_DIR / "views_info.json").read_text())["s1_orbit_for_chips_and_pixelsets"]
    S1S2_TAG = f"tiles-{TILE_FEATURES}-{S1_ORBIT.lower()}"
else:
    S1S2_FILE = ROOT / "outputs" / "legacy" / "03_s1s2_features" / f"s1s2_features_{WINDOW}_{INTERVAL_DAYS}d.csv"
    S1S2_TAG = f"{INTERVAL_DAYS}d"
EMB_TAG = f"{WINDOW}_tiles" if EMB_SOURCE == "tiles" else WINDOW
EMB_FILE = ROOT / "outputs" / "study" / "02_embeddings" / f"{INPUT}_{EMB_TAG}.npz"
RUN_NAME = (f"s1s2-{S1S2_VARIABLES}_{WINDOW}_{S1S2_TAG}" if INPUT == "s1s2"
            else f"{INPUT}_{EMB_TAG}_{FEATURE_MODE}")
OUT_DIR = ROOT / "outputs" / "study" / "03_models" / RUN_NAME

SEED = 42
RF_TREES = 500
N_BOOT = 1000                         # block-bootstrap draws for the confidence intervals
N_JOBS = -1                           # CPU cores for the random forest (-1 = all)
QUICK_TEST = False                    # True = fast check (100 trees, 100 bootstraps, one setting per model)
# ==========================================================================

N_FEATURES = 0                        # set in main(); used by preprocess() to size the PCA


def preprocess():
    """Steps every classifier gets first. s1s2: fill any remaining gap with the training median.
    Embeddings: fill empty months, standardise, and (concat) reduce with PCA - all fitted on training folds only."""
    if INPUT == "s1s2":
        return [SimpleImputer(strategy="median")]
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if FEATURE_MODE == "concat" and N_FEATURES > PCA_DIM:
        steps.append(PCA(n_components=PCA_DIM, random_state=SEED))
    return steps


FOLD_HOOK = None                      # set in main(): importance for each outer test fold

# ============================== SHARED CODE ===============================
# Metrics, classifiers, nested spatial cross-validation and plots - the same for every input.

CLASSES = [1, 2]                                   # 1 = irrigated, 2 = rainfed
CLASS_NAMES = {1: "Irrigated", 2: "Rainfed"}
CLASS_COLORS = {1: "#1f77b4", 2: "#d62728"}
HEADLINE = ["macro_f1", "f1_rainfed", "f1_irrigated", "recall_rainfed", "precision_rainfed",
            "balanced_accuracy", "overall_accuracy", "pr_auc_rainfed", "brier", "ece"]

if QUICK_TEST:                                     # small settings to check that everything runs
    RF_TREES, N_BOOT = 100, 100


def setup_logging(log_file):
    """Log to the screen and to run.log at the same time."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(log_file, mode="w"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def load_split(split_file):
    """Folds from 3_split_train_test.py: one row per parcel with label, block_id, fold, random_fold."""
    split = gpd.read_file(split_file)[["parcel_id", "label", "block_id", "fold", "random_fold"]]
    split["parcel_id"] = split["parcel_id"].astype(str)
    return pd.DataFrame(split).set_index("parcel_id")


# ---------------------------------------------------------------- metrics
def expected_calibration_error(is_rainfed, p_rainfed, n_bins=10):
    """Average gap between predicted probability and observed frequency, over 10 probability bins."""
    bins = np.minimum((p_rainfed * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(is_rainfed[in_bin].mean() - p_rainfed[in_bin].mean())
    return ece


def binary_metrics(y_true, y_pred, score_rainfed=None, is_probability=True):
    """All scores for one set of predictions. Rainfed is the positive class for PR-AUC and calibration.
    score_rainfed: probability (or SVM decision score) that the parcel is rainfed."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = {
        "n": len(y_true),
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for c in CLASSES:
        name = CLASS_NAMES[c].lower()
        m[f"precision_{name}"] = precision_score(y_true, y_pred, pos_label=c, zero_division=0)
        m[f"recall_{name}"] = recall_score(y_true, y_pred, pos_label=c, zero_division=0)
        m[f"f1_{name}"] = f1_score(y_true, y_pred, pos_label=c, zero_division=0)
    if score_rainfed is not None and len(np.unique(y_true)) == 2:
        is_rainfed = (y_true == 2).astype(int)
        score = np.asarray(score_rainfed, dtype=float)
        m["roc_auc"] = roc_auc_score(is_rainfed, score)
        m["pr_auc_rainfed"] = average_precision_score(is_rainfed, score)   # threshold-free, minority class
        if is_probability:                                                 # only for real probabilities
            m["brier"] = brier_score_loss(is_rainfed, score)
            m["ece"] = expected_calibration_error(is_rainfed, score)
    return m


def block_bootstrap(pred, n_boot, seed):
    """Estimate and 95% interval of every metric, resampling whole spatial blocks with replacement.
    Parcels in one block are correlated, so blocks (not parcels) are the independent units."""
    blocks = pred["block_id"].to_numpy()
    groups = {b: np.flatnonzero(blocks == b) for b in np.unique(blocks)}
    names = np.array(list(groups))
    y, yhat, s = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy(), pred["score"].to_numpy()
    is_prob = bool(pred["is_probability"].iloc[0])
    point = binary_metrics(y, yhat, s, is_prob)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sel = np.concatenate([groups[b] for b in rng.choice(names, size=len(names), replace=True)])
        if len(np.unique(y[sel])) == 2:
            draws.append(binary_metrics(y[sel], yhat[sel], s[sel], is_prob))
    draws = pd.DataFrame(draws)
    rows = [{"metric": k, "estimate": v, "ci_low": np.nanquantile(draws[k], 0.025),
             "ci_high": np.nanquantile(draws[k], 0.975)} for k, v in point.items() if k != "n"]
    return pd.DataFrame(rows).set_index("metric")


def paired_differences(oof, n_boot, seed):
    """Macro-F1 difference for every pair of models on the SAME parcels, with a block-bootstrap 95% CI.
    An interval that excludes 0 means the two models really differ (Foody 2009, RSE)."""
    models = list(oof)
    blocks = oof[models[0]]["block_id"].to_numpy()
    groups = {b: np.flatnonzero(blocks == b) for b in np.unique(blocks)}
    names = np.array(list(groups))
    y = oof[models[0]]["y_true"].to_numpy()
    preds = {m: oof[m]["y_pred"].to_numpy() for m in models}
    f1 = lambda m, sel: f1_score(y[sel], preds[m][sel], average="macro", labels=CLASSES, zero_division=0)
    rng = np.random.default_rng(seed)
    samples = [np.concatenate([groups[b] for b in rng.choice(names, size=len(names), replace=True)])
               for _ in range(n_boot)]
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            everything = np.arange(len(y))
            diffs = [f1(a, sel) - f1(b, sel) for sel in samples]
            rows.append({"model_a": a, "model_b": b, "macro_f1_a_minus_b": f1(a, everything) - f1(b, everything),
                         "ci_low": np.quantile(diffs, 0.025), "ci_high": np.quantile(diffs, 0.975)})
    table = pd.DataFrame(rows)
    table["differs"] = (table["ci_low"] > 0) | (table["ci_high"] < 0)
    return table


# ---------------------------------------------------------------- models
# Each builder returns a fresh, untrained scikit-learn pipeline. preprocess() (defined in the
# input-specific settings) adds the steps every model needs, e.g. filling empty values.
def build_logreg(C=1.0):
    """Logistic regression: a linear model. On embeddings this is the standard 'linear probe'."""
    return make_pipeline(*preprocess(), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def build_rf(max_features="sqrt", min_samples_leaf=1, class_weight=None):
    """Random forest: many decision trees on random subsets of parcels and features, majority vote."""
    return make_pipeline(*preprocess(),
                         RandomForestClassifier(n_estimators=RF_TREES, max_features=max_features,
                                                min_samples_leaf=min_samples_leaf, class_weight=class_weight,
                                                n_jobs=N_JOBS, random_state=SEED))


def build_hgb(learning_rate=0.1, max_leaf_nodes=31, class_weight=None):
    """Histogram gradient boosting: trees added one after another, each fixing the previous errors.
    Early stopping is off because it would hold out a RANDOM (not spatial) validation split."""
    return make_pipeline(*preprocess(),
                         HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                                        max_iter=300, early_stopping=False,
                                                        class_weight=class_weight, random_state=SEED))


def build_svm(C=1.0, gamma="scale"):
    """Support vector machine with an RBF kernel (used in Gao et al. 2018; Elwan et al. 2022).
    No probability calibration (it would need a random internal split); its decision score is used for PR-AUC."""
    return make_pipeline(*preprocess(), StandardScaler(),
                         SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced", random_state=SEED))


# name -> (builder, hyperparameter grid searched in the inner spatial CV)
MODELS = {
    "logreg": (build_logreg, {"C": [0.01, 0.1, 1.0, 10.0]}),
    "rf": (build_rf, {"max_features": ["sqrt", 0.3], "min_samples_leaf": [1, 5],
                      "class_weight": [None, "balanced_subsample"]}),
    "hgb": (build_hgb, {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31],
                        "class_weight": [None, "balanced"]}),
    "svm": (build_svm, {"C": [1.0, 10.0], "gamma": ["scale", 0.01]}),
}
if QUICK_TEST:                                     # one setting per model
    MODELS = {name: (build, {k: v[:1] for k, v in grid.items()}) for name, (build, grid) in MODELS.items()}


def predict(model, X):
    """Class predictions plus a 'rainfed' score: probability if the model has one, else SVM decision value."""
    y_hat = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return y_hat, proba[:, list(model.classes_).index(2)], True
    return y_hat, model.decision_function(X), False   # SVC: positive = class 2 (rainfed)


# ---------------------------------------------------------------- cross-validation
def tune_inner(build, grid, X, y, folds):
    """Inner loop: pick the hyperparameters with the best macro-F1 over spatial CV on the training folds only.
    Predictions of all inner folds are pooled before scoring."""
    best, best_score, scores = None, -1.0, []
    for params in ParameterGrid(grid):
        y_hat = np.empty(len(y), dtype=int)
        for f in np.unique(folds):
            inner_test = folds == f
            model = build(**params).fit(X[~inner_test], y[~inner_test])
            y_hat[inner_test] = model.predict(X[inner_test])
        score = f1_score(y, y_hat, average="macro", labels=CLASSES, zero_division=0)
        scores.append({"params": str(params), "inner_macro_f1": score})
        if score > best_score:
            best, best_score = params, score
    return best, best_score, scores


def nested_spatial_cv(data, features, log):
    """Outer loop: each spatial fold is the test fold once. Hyperparameters are tuned on the other
    folds only (inner loop), the model is refit on all of them and predicts the held-out fold.
    Every parcel is therefore predicted exactly once by a model that never saw its area."""
    X, y = data[features].to_numpy(dtype=np.float32), data["label"].to_numpy()
    folds = data["fold"].to_numpy()
    oof = {name: [] for name in MODELS}          # out-of-fold predictions per model
    selected, inner_log, fold_scores = [], [], []
    for outer in sorted(np.unique(folds)):
        test = folds == outer
        train = ~test
        log.info(f"Outer fold {outer}: train {train.sum():,} parcels, test {test.sum():,} parcels")
        for name, (build, grid) in MODELS.items():
            t0 = time.time()
            params, inner_score, scores = tune_inner(build, grid, X[train], y[train], folds[train])
            model = build(**params).fit(X[train], y[train])
            y_hat, score, is_prob = predict(model, X[test])
            oof[name].append(pd.DataFrame({
                "parcel_id": data.index[test], "outer_fold": outer, "y_true": y[test], "y_pred": y_hat,
                "score": score, "is_probability": is_prob, "block_id": data["block_id"].to_numpy()[test]}))
            m = binary_metrics(y[test], y_hat, score, is_prob)
            fold_scores.append({"model": name, "outer_fold": outer, "params": str(params), **m})
            selected.append({"model": name, "outer_fold": outer, "params": str(params), "inner_macro_f1": inner_score})
            inner_log += [{"model": name, "outer_fold": outer, **s} for s in scores]
            log.info(f"  {name:6s} {str(params):75s} inner {inner_score:.3f} | test macro-F1 {m['macro_f1']:.3f} "
                     f"({time.time() - t0:.0f}s)")
            if name == "rf" and FOLD_HOOK is not None:      # input-specific extra (e.g. importance)
                FOLD_HOOK(model, data.iloc[np.flatnonzero(test)], features, outer)
    oof = {name: pd.concat(parts, ignore_index=True) for name, parts in oof.items()}
    return oof, pd.DataFrame(selected), pd.DataFrame(inner_log), pd.DataFrame(fold_scores)


def fixed_params_cv(build, params, X, y, folds):
    """Plain k-fold CV with fixed hyperparameters (for the spatial-vs-random comparison)."""
    rows = []
    for f in np.unique(folds):
        test = folds == f
        model = build(**params).fit(X[~test], y[~test])
        y_hat = model.predict(X[test])
        rows.append({"fold": f, "macro_f1": f1_score(y[test], y_hat, average="macro", labels=CLASSES),
                     "f1_rainfed": f1_score(y[test], y_hat, pos_label=2, zero_division=0)})
    return pd.DataFrame(rows)


def most_common_params(selected, name):
    """Hyperparameters chosen most often across the outer folds (used for the final model)."""
    chosen = selected.loc[selected["model"] == name, "params"].value_counts().index[0]
    return ast.literal_eval(chosen)   # back from its string form, e.g. "{'C': 0.1}"


# ---------------------------------------------------------------- plots
def plot_confusions(oof, summary, path, title):
    fig, axes = plt.subplots(1, len(oof), figsize=(4.4 * len(oof), 4.2), dpi=120)
    for ax, (name, pred) in zip(np.atleast_1d(axes), oof.items()):
        cm = confusion_matrix(pred["y_true"], pred["y_pred"], labels=CLASSES)
        pct = cm / cm.sum(axis=1, keepdims=True)              # row % = recall of each class
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        names = [CLASS_NAMES[c] for c in CLASSES]
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.1%}", ha="center", va="center",
                        color="white" if pct[i, j] > 0.6 else "black")
        ax.set_title(f"{name}: macro-F1 {summary[name].loc['macro_f1', 'estimate']:.3f}", fontweight="bold")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv(cv, path, title):
    """Per-fold macro-F1 and rainfed F1, random vs spatial CV, same hyperparameters."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    colors = {"random": "#9e9e9e", "spatial": "#1f77b4"}
    models = list(dict.fromkeys(cv["model"]))
    for ax, metric in zip(axes, ["macro_f1", "f1_rainfed"]):
        for i, m in enumerate(models):
            for j, scheme in enumerate(["random", "spatial"]):
                v = cv.query("model == @m and cv == @scheme")[metric].to_numpy()
                x = i + (j - 0.5) * 0.3
                ax.scatter(np.full(len(v), x), v, color=colors[scheme], s=25, zorder=3)
                ax.hlines(v.mean(), x - 0.1, x + 0.1, color=colors[scheme], linewidth=2.5, zorder=4)
        ax.set_xticks(range(len(models)), models)
        ax.set_title(metric.replace("_", " "), fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=c, label=f"{s} CV")
                            for s, c in colors.items()], frameon=False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt_ci(table):
    return table.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}-{r.ci_high:.3f}]", axis=1)


# ---------------------------------------------------------------- whole protocol
def run_protocol(data, features, out_dir, run_name, log):
    """1 nested spatial CV -> 2 pooled metrics with 95% CI -> 3 paired model comparison ->
    4 spatial vs random CV -> 5 final models on all parcels -> save everything."""
    X, y = data[features].to_numpy(dtype=np.float32), data["label"].to_numpy()
    log.info(f"{len(data):,} parcels, {len(features)} features, folds "
             f"{data['fold'].value_counts().sort_index().to_dict()}")

    log.info("1/5 Nested spatial cross-validation")
    oof, selected, inner_log, fold_scores = nested_spatial_cv(data, features, log)

    log.info("2/5 Out-of-fold metrics with block-bootstrap 95% CI")
    summary = {name: block_bootstrap(pred, N_BOOT, SEED) for name, pred in oof.items()}
    table = pd.DataFrame({m: fmt_ci(t) for m, t in summary.items()}).reindex(HEADLINE).dropna(how="all")
    log.info("\n" + table.to_string())
    per_fold = fold_scores.groupby("model")[["macro_f1", "f1_rainfed"]].agg(["mean", "std"]).round(3)
    log.info("Per outer fold (mean, SD):\n" + per_fold.to_string())
    plot_confusions(oof, summary, out_dir / "confusion_matrices.png",
                    f"{run_name}: all parcels, out-of-fold (nested spatial CV)")

    log.info("3/5 Paired model differences (macro-F1, same parcels)")
    pairs = paired_differences(oof, N_BOOT, SEED)
    log.info("\n" + pairs.round(3).to_string(index=False))

    log.info("4/5 Spatial vs random 5-fold CV (same hyperparameters)")
    rows = []
    for name, (build, _) in MODELS.items():
        params = most_common_params(selected, name)
        for scheme, col in [("spatial", "fold"), ("random", "random_fold")]:
            res = fixed_params_cv(build, params, X, y, data[col].to_numpy())
            rows.append(res.assign(model=name, cv=scheme))
            log.info(f"  {name:6s} {scheme:7s} macro-F1 {res['macro_f1'].mean():.3f} ± {res['macro_f1'].std():.3f}")
    cv = pd.concat(rows, ignore_index=True)
    plot_cv(cv, out_dir / "cv_random_vs_spatial.png", f"{run_name}: random vs spatial 5-fold CV")

    log.info("5/5 Final models on all parcels (most often selected hyperparameters)")
    final_params = {}
    for name, (build, _) in MODELS.items():
        final_params[name] = most_common_params(selected, name)
        joblib.dump(build(**final_params[name]).fit(X, y), out_dir / f"final_model_{name}.joblib")

    # Save everything.
    pd.concat(oof, names=["model"]).to_csv(out_dir / "oof_predictions.csv")
    pd.concat(summary, names=["model"]).to_csv(out_dir / "metrics_with_ci.csv")
    table.to_csv(out_dir / "metrics_summary.csv")
    fold_scores.to_csv(out_dir / "metrics_per_fold.csv", index=False)
    selected.to_csv(out_dir / "selected_hyperparameters.csv", index=False)
    inner_log.to_csv(out_dir / "inner_tuning_scores.csv", index=False)
    pairs.to_csv(out_dir / "paired_model_differences.csv", index=False)
    cv.to_csv(out_dir / "cv_random_vs_spatial.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps({
        "run": run_name, "input": INPUT, "window": WINDOW,
        "settings": ({"source": S1S2_SOURCE, "feature_file": S1S2_FILE.name, "variables": S1S2_VARIABLES,
                      **({"tile_features": TILE_FEATURES, "s1_orbit": S1_ORBIT} if S1S2_SOURCE == "tiles"
                         else {"interval_days": INTERVAL_DAYS})} if INPUT == "s1s2"
                     else {"feature_mode": FEATURE_MODE, "pca_dim": PCA_DIM, "embedding_file": EMB_FILE.name}),
        "n_parcels": int(len(data)), "n_features": len(features),
        "validation": "nested spatial CV (outer 5 folds, inner spatial CV on the training folds)",
        "seed": SEED, "rf_trees": RF_TREES, "n_boot": N_BOOT, "quick_test": QUICK_TEST,
        "grids": {k: v[1] for k, v in MODELS.items()}, "final_params": final_params}, indent=2, default=str))
    return oof, summary
# ============================ END SHARED CODE =============================


# ============================ INPUT-SPECIFIC CODE ===========================
S1_VARIABLES = {"VV", "VH", "VV_ASC", "VH_ASC", "VV_DSC", "VH_DSC"}
TILE_META = ["n_px_used", "used_inner", "fold", "label", "ag_season"]      # not features
TILE_COUNTS = ("S2_NCLEAR", "S1_N_ASC", "S1_N_DSC")                          # observation counts: QA only, not features


def split_name(col):
    """'NDVI_p03' -> ('NDVI', 'p03');  'm03_d0012' -> ('m03', 'd0012')."""
    a, b = col.rsplit("_", 1)
    return a, b


# ---------------------------------------------------------------- S1/S2 features
def load_s1s2(split):
    """S1/S2 feature table joined to the folds; keeps the variables chosen in S1S2_VARIABLES."""
    feats = pd.read_csv(S1S2_FILE, dtype={"parcel_id": str}).drop_duplicates("parcel_id").set_index("parcel_id")
    if S1S2_SOURCE == "tiles":
        feats = tile_feature_table(feats)
    features = [c for c in feats.columns if c not in ("label", "ag_season")]
    if S1S2_VARIABLES == "s2":
        features = [c for c in features if split_name(c)[0] not in S1_VARIABLES]
    elif S1S2_VARIABLES == "s1":
        features = [c for c in features if split_name(c)[0] in S1_VARIABLES]
    return split.join(feats[features], how="inner"), features


def tile_feature_table(feats):
    """Parcel features of 04_make_views.py -> the same layout as the script-03 table.
    Drops bookkeeping and observation-count columns, keeps the chosen Sentinel-1 orbit and renames it
    to VV_mXX / VH_mXX (with S1_ORBIT = "both", VV_ASC_mXX ... are kept as four radar variables)."""
    def var(c):                                                   # 'VV_ASC_m03' -> 'VV_ASC'; 'label' -> 'label'
        return c.rsplit("_", 1)[0] if "_" in c else c
    feats = feats.drop(columns=[c for c in feats.columns if c in TILE_META[:3] or var(c) in TILE_COUNTS])
    if S1_ORBIT in ("ASC", "DSC"):
        other = "DSC" if S1_ORBIT == "ASC" else "ASC"
        feats = feats.drop(columns=[c for c in feats.columns if var(c) in (f"VV_{other}", f"VH_{other}")])
        feats = feats.rename(columns={c: c.replace(f"_{S1_ORBIT}_", "_") for c in feats.columns
                                      if var(c) in (f"VV_{S1_ORBIT}", f"VH_{S1_ORBIT}")})
    return feats


def fill_gaps(data, features):
    """Linear interpolation in time, per parcel and variable; gaps at the start/end take the nearest period.
    Returns the filled table and the share of filled values per parcel."""
    data = data.copy()
    was_empty = data[features].isna()
    for var in dict.fromkeys(split_name(c)[0] for c in features):
        cols = sorted([c for c in features if split_name(c)[0] == var], key=lambda c: split_name(c)[1])
        data[cols] = data[cols].interpolate(axis=1, limit_direction="both")
    return data, was_empty.mean(axis=1)


def period_label(period):
    """'p03' -> start date of that 30-day period, e.g. 'Nov 01'; 'm03' (tile features, calendar months) -> 'Nov'."""
    first = pd.Timestamp(year=2023, month=START_MONTH, day=1)
    if period.startswith("m"):
        return (first + pd.DateOffset(months=int(period[1:]) - 1)).strftime("%b")
    return (first + pd.Timedelta(days=(int(period[1:]) - 1) * INTERVAL_DAYS)).strftime("%b %d")


def plot_profiles(data, features, path):
    """Median and inter-quartile range per class over the year (irrigated parcels stay green in summer)."""
    periods = sorted({split_name(c)[1] for c in features})
    variables = [v for v in ["NDVI", "NDWI", "VH"] if f"{v}_{periods[0]}" in features]
    if not variables:
        return
    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 4), dpi=120, squeeze=False)
    for ax, var in zip(axes[0], variables):
        cols = [f"{var}_{p}" for p in periods]
        for lab in CLASSES:
            q = data.loc[data["label"] == lab, cols].quantile([0.25, 0.5, 0.75])
            x = [period_label(p) for p in periods]
            ax.plot(x, q.loc[0.5], color=CLASS_COLORS[lab], label=CLASS_NAMES[lab], linewidth=2)
            ax.fill_between(x, q.loc[0.25], q.loc[0.75], color=CLASS_COLORS[lab], alpha=0.15)
        ax.set_title(var + (" (dB)" if var == "VH" else ""), fontweight="bold")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False)
    fig.suptitle(f"{INTERVAL_DAYS}-day profiles ({WINDOW}): median and IQR per class", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- embeddings
STEP_NAMES = []


def load_embeddings(split):
    """Embedding file -> one row per parcel joined to the folds.
    concat: columns m01_d0000 ... (step, dimension); mean: d0000 ... (average over observed steps)."""
    d = np.load(EMB_FILE, allow_pickle=False)
    emb = d["embeddings"].astype(np.float32)                  # [parcels, steps, dims]
    emb[~d["valid"]] = np.nan                                 # unobserved step -> empty
    n, n_steps, dims = emb.shape
    if FEATURE_MODE == "concat":
        cols = [f"m{s + 1:02d}_d{k:04d}" for s in range(n_steps) for k in range(dims)]
        table = pd.DataFrame(emb.reshape(n, n_steps * dims), columns=cols)
    else:
        cols = [f"d{k:04d}" for k in range(dims)]
        table = pd.DataFrame(np.nanmean(emb, axis=1), columns=cols)
    table.index = pd.Index(d["parcel_id"].astype(str), name="parcel_id")
    return split.join(table, how="inner"), cols, [str(s) for s in d["steps"]]


# ---------------------------------------------------------------- importance
IMPORTANCE_ROWS = []


def shuffle_drop(model, X, y, cols, rng, repeats):
    """Drop in macro-F1 when the given columns are shuffled together across parcels."""
    base = f1_score(y, model.predict(X), average="macro")
    drops = []
    for _ in range(repeats):
        Xp = X.copy()
        Xp[:, cols] = X[np.ix_(rng.permutation(len(X)), cols)]
        drops.append(base - f1_score(y, model.predict(Xp), average="macro"))
    return float(np.mean(drops))


def fold_importance(model, test_data, features, outer):
    """Random forest of one outer fold, on its test fold: importance by variable and period (s1s2) or by step."""
    X, y = test_data[features].to_numpy(dtype=np.float32), test_data["label"].to_numpy()
    rng = np.random.default_rng(SEED)
    groups = {}
    if INPUT == "s1s2":
        for i, c in enumerate(features):
            var, period = split_name(c)
            groups.setdefault(("variable", var), []).append(i)
            groups.setdefault(("period", period_label(period)), []).append(i)
    elif FEATURE_MODE == "concat" and len(STEP_NAMES) > 1:
        for i, c in enumerate(features):
            step = int(c[1:3]) - 1
            groups.setdefault(("step", STEP_NAMES[step]), []).append(i)
    for (kind, name), cols in groups.items():
        IMPORTANCE_ROWS.append({"kind": kind, "group": name, "outer_fold": outer,
                                "drop_macro_f1": shuffle_drop(model, X, y, cols, rng, repeats=5)})


def plot_importance(table, path):
    kinds = list(dict.fromkeys(table["kind"]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(7 * len(kinds), 4.5), dpi=120, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        g = table[table["kind"] == kind].groupby("group", sort=False)["drop_macro_f1"].agg(["mean", "std"])
        if kind == "variable":
            g = g.sort_values("mean")
            ax.barh(g.index, g["mean"], xerr=g["std"], color="#5b8c5a")
            ax.set_xlabel("Drop in macro-F1 (mean ± SD over outer folds)")
        else:
            ax.bar(g.index, g["mean"], yerr=g["std"], color="#4c78a8")
            ax.set_ylabel("Drop in macro-F1 (mean ± SD over outer folds)")
            ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.set_title(f"Importance by {kind}", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{RUN_NAME}: grouped permutation importance (random forest)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    global FOLD_HOOK, N_FEATURES, STEP_NAMES
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(OUT_DIR / "run.log")
    t0 = time.time()
    split = load_split(SPLIT_FILE)

    # 1. Input features joined to the folds.
    if INPUT == "s1s2":
        data, features = load_s1s2(split)
        data = data[~data[features].isna().all(axis=1)]              # parcels with no satellite data at all
        data, filled = fill_gaps(data, features)
        filled.rename("filled_fraction").to_frame().join(data[["label"]]).to_csv(OUT_DIR / "gap_filling.csv")
        log.info(f"{RUN_NAME}: filled by interpolation {filled.mean():.1%} of values, {(filled > 0).mean():.1%} of parcels")
        plot_profiles(data, features, OUT_DIR / "seasonal_profiles.png")
    else:
        data, features, STEP_NAMES = load_embeddings(split)
        log.info(f"{RUN_NAME}: {len(STEP_NAMES)} steps, empty (unobserved) {data[features].isna().mean().mean():.1%}")
    N_FEATURES = len(features)
    log.info(f"{RUN_NAME}: {len(data):,} parcels {data['label'].map(CLASS_NAMES).value_counts().to_dict()} | "
             f"{N_FEATURES} features" + (f" -> PCA {PCA_DIM}" if INPUT != "s1s2" and FEATURE_MODE == "concat"
                                         and N_FEATURES > PCA_DIM else ""))

    # 2-5, 7. Nested spatial CV, metrics, paired comparison, spatial vs random, final models.
    FOLD_HOOK = fold_importance                                       # 6. importance inside the outer loop
    run_protocol(data, features, OUT_DIR, RUN_NAME, log)

    if IMPORTANCE_ROWS:
        importance = pd.DataFrame(IMPORTANCE_ROWS)
        importance.to_csv(OUT_DIR / "importance.csv", index=False)
        plot_importance(importance, OUT_DIR / "importance.png")
    log.info(f"Done in {(time.time() - t0) / 60:.1f} min -> {OUT_DIR}")


if __name__ == "__main__":
    main()
