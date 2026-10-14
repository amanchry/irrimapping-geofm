"""Evaluation protocol shared by all irrigated / rainfed models.

Labels follow the split notebook: 1 = irrigated, 2 = rainfed. Rainfed (the minority class) is the
positive class for ROC-AUC and PR-AUC; all other scores are reported for both classes.
"""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import ParameterGrid

CLASSES = [1, 2]
CLASS_NAMES = {1: "Irrigated", 2: "Rainfed"}
HEADLINE = ["macro_f1", "f1_rainfed", "f1_irrigated", "balanced_accuracy", "overall_accuracy", "kappa", "mcc", "pr_auc_rainfed"]


# ---------------------------------------------------------------- metrics
def binary_metrics(y_true, y_pred, p_rainfed=None) -> dict:
    """All scores for one set of predictions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = {
        "n": len(y_true),
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=CLASSES),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for c in CLASSES:
        name = CLASS_NAMES[c].lower()
        m[f"precision_{name}"] = precision_score(y_true, y_pred, pos_label=c, zero_division=0)
        m[f"recall_{name}"] = recall_score(y_true, y_pred, pos_label=c, zero_division=0)
        m[f"f1_{name}"] = f1_score(y_true, y_pred, pos_label=c, zero_division=0)
    if p_rainfed is not None and len(np.unique(y_true)) == 2:
        is_rainfed = (y_true == 2).astype(int)
        m["roc_auc"] = roc_auc_score(is_rainfed, p_rainfed)
        m["pr_auc_rainfed"] = average_precision_score(is_rainfed, p_rainfed)
    return m


def block_bootstrap(y_true, y_pred, p_rainfed, groups, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """Point estimate and percentile CI of every metric, resampling whole spatial blocks.

    Parcels in one block are correlated, so resampling parcels would give intervals that are too narrow.
    """
    y_true, y_pred, p_rainfed, groups = map(np.asarray, (y_true, y_pred, p_rainfed, groups))
    point = binary_metrics(y_true, y_pred, p_rainfed)
    uniq = np.unique(groups)
    idx_by_block = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_block[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y_true[idx])) < 2:
            continue
        draws.append(binary_metrics(y_true[idx], y_pred[idx], p_rainfed[idx]))
    draws = pd.DataFrame(draws)
    rows = []
    for k, v in point.items():
        if k == "n":
            continue
        lo, hi = np.nanquantile(draws[k], [alpha / 2, 1 - alpha / 2]) if k in draws else (np.nan, np.nan)
        rows.append({"metric": k, "estimate": v, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).set_index("metric")


# ---------------------------------------------------------------- model selection
def predict(model, X):
    """Class labels and rainfed probability."""
    proba = model.predict_proba(X)
    p_rainfed = proba[:, list(model.classes_).index(2)]
    return model.predict(X), p_rainfed


def tune_on_validation(models: dict, X: pd.DataFrame, y: pd.Series, train_mask, val_mask, score: str = "macro_f1") -> pd.DataFrame:
    """Fit every parameter combination on the training folds and score it on the validation fold.

    `models` maps a name to (build_fn, param_grid); build_fn(**params) returns an unfitted estimator.
    """
    rows = []
    for name, (build, grid) in models.items():
        for params in ParameterGrid(grid):
            model = build(**params).fit(X[train_mask], y[train_mask])
            y_hat, p = predict(model, X[val_mask])
            rows.append({"model": name, "params": params, **binary_metrics(y[val_mask], y_hat, p)})
    return pd.DataFrame(rows).sort_values(["model", score], ascending=[True, False])


def best_params(tuning: pd.DataFrame, score: str = "macro_f1") -> dict:
    """Best parameter set per model."""
    return {m: g.sort_values(score, ascending=False).iloc[0]["params"] for m, g in tuning.groupby("model")}


def cross_validate(build: Callable, X: pd.DataFrame, y: pd.Series, folds: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-fold-out evaluation. Returns per-fold metrics and out-of-fold predictions."""
    rows, oof = [], []
    for f in sorted(folds.unique()):
        test = (folds == f).to_numpy()
        model = build().fit(X[~test], y[~test])
        y_hat, p = predict(model, X[test])
        rows.append({"fold": f, **binary_metrics(y[test], y_hat, p)})
        oof.append(pd.DataFrame({"fold": f, "y_true": y[test], "y_pred": y_hat, "p_rainfed": p}, index=X.index[test]))
    return pd.DataFrame(rows).set_index("fold"), pd.concat(oof)


def summarise_cv(per_fold: pd.DataFrame, metrics=HEADLINE) -> pd.Series:
    """'mean ± std' strings for the headline metrics."""
    cols = [m for m in metrics if m in per_fold]
    return pd.Series({m: f"{per_fold[m].mean():.3f} ± {per_fold[m].std():.3f}" for m in cols})


# ---------------------------------------------------------------- plots
def plot_confusion(y_true, y_pred, ax, title=""):
    """Confusion matrix with counts and row-normalised percentages (recall per class)."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    names = [CLASS_NAMES[c] for c in CLASSES]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.1%}", ha="center", va="center",
                    color="white" if pct[i, j] > 0.6 else "black", fontsize=10)
    ax.set_title(title, fontweight="bold")


def plot_cv_comparison(results: pd.DataFrame, metric: str = "macro_f1", ax=None):
    """Per-fold scores for each model under random and spatial cross-validation.

    `results` has columns model, cv, fold and the metric.
    """
    ax = ax or plt.gca()
    models = sorted(results["model"].unique())
    schemes = ["random", "spatial"]
    colors = {"random": "#9e9e9e", "spatial": "#1f77b4"}
    for i, m in enumerate(models):
        for j, s in enumerate(schemes):
            v = results.query("model == @m and cv == @s")[metric].to_numpy()
            x = i + (j - 0.5) * 0.3
            ax.scatter(np.full(len(v), x), v, color=colors[s], s=25, zorder=3)
            ax.hlines(v.mean(), x - 0.1, x + 0.1, color=colors[s], linewidth=2.5, zorder=4)
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=colors[s], label=f"{s} CV") for s in schemes],
              frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
