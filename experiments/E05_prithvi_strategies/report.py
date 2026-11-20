"""
E05_prithvi_strategies/report.py  —  tables, paired tests and figures of E05 (frozen / LoRA / fine-tuned / scratch)
==================================================================================================================

Reads the finished runs results/<region>/<strategy>_p<percent>/ and writes
  results/<region>/summary.csv   one row per run: scores with 95% block-bootstrap CI, trainable parameters, epochs,
                                 time per epoch, GPU hours, labelled parcels used
  results/<region>/paired.csv    paired differences at every training share: fine-tuned - frozen (effect of training
                                 the encoder), fine-tuned - scratch (effect of the pretrained weights), frozen -
                                 scratch, LoRA - frozen (effect of the adapters), fine-tuned - LoRA (full vs
                                 parameter-efficient), LoRA - scratch; same pixels and block-bootstrap draws as
                                 05_compare_experiments.py, Holm correction per metric over all comparisons of this
                                 table (only the finished runs enter)
  docs/experiments/figures/e05_curves.png          training loss, validation loss, validation macro-F1 per epoch
                                                   (100% of the labels; mean over rounds and seeds, band = range)
  docs/experiments/figures/e05_label_fraction.png  out-of-fold macro-F1 against the share of training parcels, with
                                                   E04 U-TAE and E01 random forest (100%) as reference lines
  docs/experiments/figures/e05_confusion.png       confusion matrices at 100%

Run (Mac, after the results are copied from the cluster)
--------------------------------------------------------
    python experiments/E05_prithvi_strategies/report.py
"""
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pixelwise_run"))
from pixel_common import block_confusions, bootstrap_draws, metrics_from_confusion, score  # noqa: E402

# ================================ SETTINGS ================================
REGION = "utah"
STRATEGIES = {"scratch": ("From scratch", "#969696"), "frozen": ("Frozen", "#3182bd"),
              "lora": ("LoRA", "#31a354"), "finetune": ("Full fine-tuning", "#e6550d")}
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
COMPARISONS = [("finetune", "frozen"), ("finetune", "scratch"), ("frozen", "scratch"),
               ("lora", "frozen"), ("finetune", "lora"), ("lora", "scratch")]
METRICS = ["cropland_macro_f1", "macro_f1_3class"]
REFERENCE_RUNS = {"E04 U-TAE": "E04_utae/results/{region}/utae", "E01 random forest": "E01_s2_pixel_baseline/results/{region}/rf"}
CURVE_FRACTION = 1.0                               # training share shown in the learning curves
# ==========================================================================

EXP = Path(__file__).resolve().parent
RES = EXP / "results" / REGION
FIG = ROOT / "docs" / "experiments" / "figures"
CLASSES = ["irrigated", "rainfed", "not cropland"]

spec = importlib.util.spec_from_file_location("compare", ROOT / "pixelwise_run" / "05_compare_experiments.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def run_name(strategy, fraction):
    return f"{strategy}_p{round(fraction * 100):03d}"


def finished():
    return {(s, f): RES / run_name(s, f) for s in STRATEGIES for f in FRACTIONS
            if (RES / run_name(s, f) / "oof_predictions.parquet").exists()}


def trainable_millions(d):
    m = re.search(r"trainable parameters: ([\d.]+) M", (d / "run.log").read_text())
    return float(m.group(1)) if m else np.nan


def summary(runs):
    rows = []
    for (s, f), d in runs.items():
        m = pd.read_csv(d / "metrics.csv").set_index("metric")
        h = pd.read_csv(d / "training_history.csv")
        per_run = h.groupby(["round", "seed"])
        best_epoch = h.loc[per_run["val_macro_f1"].idxmax(), "epoch"]
        row = {"run": d.name, "strategy": s, "fraction": f}
        for x in METRICS + ["f1_irrigated", "f1_rainfed", "f1_not_cropland"]:
            if x in m.index:
                row[x] = m.loc[x, "value"]
                if x in METRICS:
                    row[f"{x}_low"], row[f"{x}_high"] = m.loc[x, "ci_low"], m.loc[x, "ci_high"]
        row.update({"trainable_M": trainable_millions(d), "epochs_mean": per_run["epoch"].max().mean(),
                    "best_epoch_mean": best_epoch.mean(), "seconds_per_epoch": h["seconds"].median(),
                    "gpu_hours": h["seconds"].sum() / 3600})
        sub = d / "label_subset.csv"
        if sub.exists():
            t = pd.read_csv(sub)
            row["parcels_per_round"] = (t["parcels_irrigated"] + t["parcels_rainfed"]).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fraction", "strategy"])


def paired(runs):
    oofs = {k: pd.read_parquet(d / "oof_predictions.parquet").sort_values(["season", "sample_idx"]).reset_index(drop=True)
            for k, d in runs.items()}
    first = next(iter(oofs.values()))
    for k, o in oofs.items():
        if not np.array_equal(o[["season", "sample_idx"]].to_numpy(), first[["season", "sample_idx"]].to_numpy()):
            raise SystemExit(f"{run_name(*k)} was scored on other pixels than the other runs")
    blocks = sorted(first["block_id"].unique(), key=str)
    draws = bootstrap_draws(len(blocks))
    boot = {k: compare.boot_metrics(o, blocks, draws) for k, o in oofs.items()}
    point = {k: metrics_from_confusion(block_confusions(o, blocks).sum(0)) for k, o in oofs.items()}
    rows = []
    for f in FRACTIONS:
        for a, b in COMPARISONS:
            if (a, f) not in oofs or (b, f) not in oofs:
                continue
            for x in METRICS:
                d = boot[(a, f)][x] - boot[(b, f)][x]
                rows.append({"fraction": f, "comparison": f"{a} - {b}", "metric": x,
                             "difference": point[(a, f)][x] - point[(b, f)][x],
                             "ci_low": d.quantile(0.025), "ci_high": d.quantile(0.975),
                             "p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = t.groupby("metric")["p"].transform(lambda p: compare.holm(p.to_numpy()))
        t["counts"] = ((t["ci_low"] > 0) | (t["ci_high"] < 0)) & (t["p_holm"] < 0.05)
    return t


def fig_curves(runs):
    panels = [("loss", "training loss"), ("val_loss", "validation loss"),
              ("val_macro_f1", "validation 3-class macro-F1")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for s, (label, color) in STRATEGIES.items():
        d = runs.get((s, CURVE_FRACTION))
        if d is None:
            continue
        h = pd.read_csv(d / "training_history.csv")
        for a, (col, title) in zip(ax, panels):
            g = h.groupby("epoch")[col]
            a.plot(g.mean().index, g.mean(), color=color, label=label)
            a.fill_between(g.min().index, g.min(), g.max(), color=color, alpha=0.15, lw=0)
            a.set_title(title, fontsize=10)
            a.set_xlabel("epoch")
            a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{REGION.title()}, {CURVE_FRACTION:.0%} of the training parcels: mean over rounds and seeds "
                 "(band = range; runs stop early at different epochs)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "e05_curves.png", dpi=130)
    plt.close(fig)


def fig_fraction(table):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    refs = {}
    for name, path in REFERENCE_RUNS.items():
        p = ROOT / "experiments" / path.format(region=REGION) / "metrics.csv"
        if p.exists():
            refs[name] = pd.read_csv(p).set_index("metric")["value"]
    for a, x in zip(ax, METRICS):
        for s, (label, color) in STRATEGIES.items():
            t = table[table["strategy"] == s].sort_values("fraction")
            if t.empty:
                continue
            pct = t["fraction"] * 100
            a.errorbar(pct, t[x], yerr=[t[x] - t[f"{x}_low"], t[f"{x}_high"] - t[x]], color=color, marker="o",
                       capsize=3, label=label)
        for (name, r), ls in zip(refs.items(), ["--", ":"]):
            a.axhline(r[x], color="k", lw=0.8, ls=ls, label=f"{name} (100%)")
        a.set_xscale("log")
        a.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
        a.minorticks_off()
        a.set_xlabel("share of the training parcels")
        a.set_title({"cropland_macro_f1": "cropland macro-F1 (irrigated, rainfed)",
                     "macro_f1_3class": "3-class macro-F1"}[x], fontsize=10)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("out-of-fold score (95% block-bootstrap CI)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e05_label_fraction.png", dpi=130)
    plt.close(fig)


def fig_confusion(runs):
    keys = [(s, 1.0) for s in STRATEGIES if (s, 1.0) in runs]
    if not keys:
        return
    fig, ax = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4), squeeze=False)
    for a, k in zip(ax[0], keys):
        o = pd.read_parquet(runs[k] / "oof_predictions.parquet", columns=["label", "pred"])
        cm = pd.crosstab(o["label"], o["pred"]).reindex(index=[1, 2, 3], columns=[1, 2, 3], fill_value=0).to_numpy()
        share = cm / cm.sum(1, keepdims=True)
        a.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{share[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                       color="white" if share[i, j] > 0.5 else "black")
        a.set_xticks(range(3), CLASSES, fontsize=8)
        a.set_yticks(range(3), CLASSES, fontsize=8)
        a.set_xlabel("predicted")
        a.set_ylabel("reference")
        a.set_title(f"{REGION.title()} · {STRATEGIES[k[0]][0]} (100%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e05_confusion.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    runs = finished()
    if not runs:
        raise SystemExit(f"no finished E05 runs in {RES}")
    FIG.mkdir(parents=True, exist_ok=True)
    table = summary(runs)
    table.to_csv(RES / "summary.csv", index=False)
    print(table[["run"] + METRICS + ["trainable_M", "best_epoch_mean", "seconds_per_epoch", "gpu_hours"]]
          .round(3).to_string(index=False))
    t = paired(runs)
    if len(t):
        t.to_csv(RES / "paired.csv", index=False)
        print(t.round(3).to_string(index=False))
    fig_curves(runs)
    fig_fraction(table)
    fig_confusion(runs)
    print(f"Figures in {FIG}")
