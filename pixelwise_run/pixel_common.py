"""
pixel_common.py  —  shared code of the pixel-wise experiments E01–E05 (not run on its own)
==========================================================================================

Every experiment script (experiments/E0*/run.py) uses this module, so all experiments read the same pixels, use
the same folds and rotation, and are scored with the same metrics.

1. Samples: the labelled pixels of the surveyed parcels (pixelwise_run/04_make_pixel_samples.py) for a region and
   its training years (Morocco 2023/24 + 2024/25 pooled, Utah 2023/24). Two classes: 1 irrigated, 2 rainfed (no
   cropland mask, no not-cropland class). Pixels with fold -1 are dropped. Each pixel keeps `sample_idx`, its row in
   that year's pixels.parquet.
2. Sentinel-2 series: 12 months x 12 bands (reflectance x 10,000); months without a clear observation
   (S2_NCLEAR = 0 or NODATA) are NaN, optionally filled by linear interpolation in time; NDVI and NDWI per month.
3. Validation: PASTIS rotation over the 5 spatial folds. Round k: test fold k, validation fold k+1 (mod 5),
   training = the other 3 folds.
4. Metrics: headline = macro-F1 (mean F1 of irrigated and rainfed); per-class F1 / precision / recall; overall
   accuracy. 95% CI by block bootstrap over the 20 km blocks, computed
   from per-block confusion matrices (fast, and identical draws for every experiment: paired comparisons).
5. Saving: oof_predictions.parquet, metrics.csv, metrics_by_group.csv, confusion.png, run_config.json, run.log in
   experiments/<ID>/results[_quick]/<region>/<run>/.
"""
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [1, 2]                     # 1 irrigated, 2 rainfed (the surveyed cropland parcels)
CLASS_NAMES = {1: "irrigated", 2: "rainfed"}
N_CLASSES = len(CLASSES)
SEASONS = {"morocco": ["2023/24", "2024/25"], "utah": ["2023/24"]}      # training years per region
N_FOLDS = 5
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
N_BOOT = 1000
BOOT_SEED = 2024                     # same bootstrap draws in every experiment (paired comparisons)


# ---------------------------------------------------------------- settings from the environment
def env_setting(name, default):
    """Value of EXPERIMENT_<NAME> if set (Slurm jobs), else the script's setting."""
    v = os.environ.get(f"EXPERIMENT_{name}")
    if v is None or v == "":
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes")
    return type(default)(v) if default is not None else v


# ---------------------------------------------------------------- samples
def samples_dir(region, season):
    return ROOT / "outputs" / region / "data_training_prep" / "04_pixel_samples" / season.replace("/", "-")


def load_samples(region, seasons=None, quick=False, seed=42):
    """Pixel table (one row per pixel) and the pixel series X [N, 12, 19] int16 of all training years.
    quick = True keeps a random 20,000 pixels (for a fast check of the whole chain)."""
    dfs, xs, info = [], [], None
    for season in seasons or SEASONS[region]:
        d = samples_dir(region, season)
        if not (d / "pixels.parquet").exists():
            raise SystemExit(f"{d}/pixels.parquet not found: run pixelwise_run/04_make_pixel_samples.py "
                             f"for {region} {season} first")
        df = pd.read_parquet(d / "pixels.parquet")
        x = np.load(d / "pixels_X.npy", mmap_mode="r")
        info = json.loads((d / "samples_info.json").read_text())
        df["season"] = season
        df["sample_idx"] = np.arange(len(df))
        keep = (df["fold"].to_numpy() >= 0) & df["label"].isin(CLASSES).to_numpy()
        dfs.append(df[keep])
        xs.append(np.asarray(x[keep]))
    df = pd.concat(dfs, ignore_index=True)
    X = np.concatenate(xs)
    if quick and len(df) > 20000:
        pick = np.sort(np.random.default_rng(seed).choice(len(df), 20000, replace=False))
        df, X = df.iloc[pick].reset_index(drop=True), X[pick]
    df["parcel_key"] = df["season"] + "|" + df["parcel_id"].astype(str)
    return df, X, info


def month_positions(n_months=12):
    """Day of the agricultural year at the middle of each month (for the temporal encodings)."""
    return [int(15 + 30.44 * m) for m in range(n_months)]


def month_dates(season, start_month, n_months=12):
    """15th of each month of an agricultural year, e.g. '2023/24' with start 9 -> 2023-09-15 ... 2024-08-15."""
    y0 = int(season[:4])
    return [pd.Timestamp(y0 + (start_month - 1 + m) // 12, (start_month - 1 + m) % 12 + 1, 15) for m in range(n_months)]


# ---------------------------------------------------------------- Sentinel-2 features
def s2_series(X, info):
    """[N, 12, 12] float32 reflectance x 10,000; NaN where the month has no clear observation."""
    mb = info["month_bands"]
    s = X[:, :, [mb.index(b) for b in S2_BANDS]].astype(np.float32)
    bad = (s == info["nodata"]).any(axis=2)
    if "S2_NCLEAR" in mb:
        bad |= X[:, :, mb.index("S2_NCLEAR")] <= 0
    s[bad] = np.nan
    return s


def clear_months(series):
    """[N, T] True where the month has a clear observation."""
    return ~np.isnan(series).any(axis=2)


def fill_time(a):
    """Linear interpolation over the month axis of a [N, T, C] array with NaN; first / last months take the
    nearest observed month; a pixel without any observation in a band stays NaN."""
    n, t, c = a.shape
    steps = np.arange(t, dtype=np.float32)[None, :, None]
    valid = ~np.isnan(a)
    prev = np.maximum.accumulate(np.where(valid, steps, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(valid, steps, t)[:, ::-1], axis=1)[:, ::-1]
    pv = np.take_along_axis(a, np.clip(prev, 0, t - 1).astype(int), axis=1)
    nv = np.take_along_axis(a, np.clip(nxt, 0, t - 1).astype(int), axis=1)
    hp, hn = prev >= 0, nxt < t
    w = (steps - prev) / np.maximum(nxt - prev, 1)
    filled = np.where(hp & hn, pv + (nv - pv) * w, np.where(hp, pv, np.where(hn, nv, np.nan)))
    return np.where(valid, a, filled).astype(np.float32)


def add_indices(series):
    """Append NDVI (B8, B4) and NDWI (B3, B8; McFeeters 1996) per month: [N, T, 12] -> [N, T, 14]."""
    b = {name: series[:, :, i] for i, name in enumerate(S2_BANDS)}
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b["B8"] - b["B4"]) / (b["B8"] + b["B4"])
        ndwi = (b["B3"] - b["B8"]) / (b["B3"] + b["B8"])
    return np.concatenate([series, ndvi[..., None], ndwi[..., None]], axis=2).astype(np.float32)


def feature_names(n_months=12):
    return [f"{v}_m{m + 1:02d}" for m in range(n_months) for v in S2_BANDS + ["NDVI", "NDWI"]]


# ---------------------------------------------------------------- validation
def rounds():
    """PASTIS rotation: (round, training folds, validation fold, test fold)."""
    for k in range(N_FOLDS):
        val = (k + 1) % N_FOLDS
        yield k, [f for f in range(N_FOLDS) if f not in (k, val)], val, k


def class_weights(y):
    """Weights inversely proportional to the class counts (CLASSES), mean 1."""
    counts = np.array([(y == c).sum() for c in CLASSES], dtype=float)
    w = np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
    return w


def standardiser(train_values):
    """Per-channel mean / std over the training pixels (last axis = channel), ignoring NaN."""
    flat = train_values.reshape(-1, train_values.shape[-1])
    mean, std = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------- metrics
def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)


def metrics_from_confusion(cm_pix):
    """cm_pix: N_CLASSES x N_CLASSES pixel confusion (rows true class, columns predicted class, order of CLASSES).
    macro_f1 (headline) = mean F1 of the classes (irrigated, rainfed)."""
    out = {}
    tp = np.diag(cm_pix)
    fp, fn = cm_pix.sum(0) - tp, cm_pix.sum(1) - tp
    f1 = _f1(tp, fp, fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    out["macro_f1"] = float(f1.mean())
    out["overall_accuracy"] = float(tp.sum() / max(cm_pix.sum(), 1))
    for i, c in enumerate(CLASSES):
        out[f"f1_{CLASS_NAMES[c]}"] = float(f1[i])
        out[f"precision_{CLASS_NAMES[c]}"] = float(prec[i])
        out[f"recall_{CLASS_NAMES[c]}"] = float(rec[i])
    return out


def block_confusions(oof, blocks):
    """Per-block pixel confusion matrices [B, N_CLASSES, N_CLASSES], in the order of `blocks`."""
    pos = {b: i for i, b in enumerate(blocks)}
    cm_pix = np.zeros((len(blocks), N_CLASSES, N_CLASSES))
    bi = oof["block_id"].map(pos).to_numpy()
    np.add.at(cm_pix, (bi, oof["label"].to_numpy() - 1, oof["pred"].to_numpy() - 1), 1)
    return cm_pix


def bootstrap_draws(n_blocks, n_boot=N_BOOT, seed=BOOT_SEED):
    """[n_boot, n_blocks] counts of each block in each draw (blocks sorted by name)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks) for _ in range(n_boot)])


def score(oof, n_boot=N_BOOT):
    """Point estimates and 95% block-bootstrap CI: DataFrame (metric, value, ci_low, ci_high)."""
    blocks = sorted(oof["block_id"].unique(), key=str)
    cm_pix = block_confusions(oof, blocks)
    point = metrics_from_confusion(cm_pix.sum(0))
    draws = bootstrap_draws(len(blocks), n_boot)
    boot = pd.DataFrame([metrics_from_confusion(np.tensordot(w, cm_pix, 1)) for w in draws])
    rows = [{"metric": k, "value": v, "ci_low": boot[k].quantile(0.025), "ci_high": boot[k].quantile(0.975)}
            for k, v in point.items()]
    return pd.DataFrame(rows), boot


def score_groups(oof):
    """Point estimates per year and per zone."""
    rows = []
    for col in ["season", "zone"]:
        for value, g in oof.groupby(col):
            if col == "zone" and value == "":
                continue
            blocks = sorted(g["block_id"].unique(), key=str)
            m = metrics_from_confusion(block_confusions(g, blocks).sum(0))
            rows.append({"group": col, "value": value, "pixels": len(g), **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- output
def results_dir(exp_dir, region, run, quick=False):
    d = Path(exp_dir) / ("results_quick" if quick else "results") / region / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logger(out_dir):
    log = logging.getLogger(str(out_dir))
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(Path(out_dir) / "run.log", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def git_commit():
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "experiments"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return c + (" (uncommitted changes)" if dirty else "")
    except OSError:
        return ""


def save_run(out_dir, df, proba, settings, log, extra_tables=None, quick=False):
    """Write the out-of-fold predictions, metrics with CI, metrics per group, confusion figure and the run config.
    proba: [N, N_CLASSES] out-of-fold class probabilities in the order of df."""
    oof = df[["season", "sample_idx", "label", "parcel_key", "parcel_id", "fold", "block_id", "zone", "tile_id",
              "x", "y"]].copy()
    for i, c in enumerate(CLASSES):
        oof[f"p_{CLASS_NAMES[c]}"] = proba[:, i].astype(np.float32)
    oof["pred"] = np.array(CLASSES)[proba.argmax(1)]
    oof.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    metrics, _ = score(oof, 100 if quick else N_BOOT)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    score_groups(oof).to_csv(out_dir / "metrics_by_group.csv", index=False)
    for name, table in (extra_tables or {}).items():
        table.to_csv(out_dir / name, index=False)
    plot_confusion(oof, out_dir / "confusion.png")
    (out_dir / "run_config.json").write_text(json.dumps({
        **settings, "git_commit": git_commit(), "finished": time.strftime("%Y-%m-%d %H:%M"),
        "machine": platform.node(), "pixels": len(oof), "quick": quick}, indent=2, default=str))
    m = metrics.set_index("metric")
    for k in ["macro_f1", "f1_rainfed"]:
        log.info(f"{k}: {m.loc[k, 'value']:.3f} [{m.loc[k, 'ci_low']:.3f}, {m.loc[k, 'ci_high']:.3f}]")
    log.info(f"Saved {out_dir}")
    return oof, metrics


def plot_confusion(oof, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = pd.crosstab(oof["label"], oof["pred"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_numpy()
    share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    names = [CLASS_NAMES[c].capitalize() for c in CLASSES]
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{share[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks(range(N_CLASSES), names, fontsize=8)
    ax.set_yticks(range(N_CLASSES), names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
