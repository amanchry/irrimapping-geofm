"""
report_figures.py  —  figures for the experiment pages (docs/experiments/figures/)
==================================================================================

Reads the finished runs in experiments/E0*/results/<region>/<run>/ and writes, for every <tag> in FIGURES:
  scores_<tag>.png        cropland and 3-class macro-F1 with 95% block-bootstrap CI, per region and run
  confusion_<tag>.png     row-normalised confusion matrices of the runs in "confusion"
and once:
  e01_importance.png      E01 random forest: permutation importance per month and per band

Run (Mac, after the results are copied from the cluster)
--------------------------------------------------------
    python experiments/report_figures.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixelwise_run"))
from regions import settings  # noqa: E402

# ================================ SETTINGS ================================
FIGURES = {
    "e01_e02": {"scores": {"E01 random forest": "E01_s2_pixel_baseline/rf",
                           "E01 gradient boosting": "E01_s2_pixel_baseline/hgb",
                           "E02 TempCNN": "E02_temporal_nn/tempcnn"},
                "confusion": {"E01 random forest": "E01_s2_pixel_baseline/rf", "E02 TempCNN": "E02_temporal_nn/tempcnn"}},
    "e03_e04": {"scores": {"E01 random forest": "E01_s2_pixel_baseline/rf", "E02 TempCNN": "E02_temporal_nn/tempcnn",
                           "E03 U-TAE": "E03_utae/utae", "E04 Prithvi": "E04_frozen_geofm/prithvi",
                           "E04 Clay": "E04_frozen_geofm/clay", "E04 TerraMind": "E04_frozen_geofm/terramind",
                           "E04 AgriFM": "E04_frozen_geofm/agrifm"},
                "confusion": {"E01 random forest": "E01_s2_pixel_baseline/rf", "E03 U-TAE": "E03_utae/utae",
                              "E04 TerraMind": "E04_frozen_geofm/terramind"}},
    "e04": {"confusion": {"Prithvi": "E04_frozen_geofm/prithvi", "Clay": "E04_frozen_geofm/clay",
                          "TerraMind": "E04_frozen_geofm/terramind", "AgriFM": "E04_frozen_geofm/agrifm"}},
}
REGIONS = {"morocco": "Morocco", "utah": "Utah"}
# ==========================================================================

EXP = Path(__file__).resolve().parent
OUT = EXP.parent / "docs" / "experiments" / "figures"
CLASSES = ["irrigated", "rainfed", "not cropland"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def res(run, region):
    exp, model = run.split("/")
    return EXP / exp / "results" / region / model


def month_labels(region):
    s = settings(region)["start_month"]
    return [MONTHS[(s - 1 + k) % 12] for k in range(12)]


COLORS = {"E01": ["#9ecae1", "#3182bd"], "E02": ["#fdae6b", "#e6550d"], "E03": ["#74c476"],
          "E04": ["#bcbddc", "#9e9ac8", "#756bb1", "#54278f"]}


def run_colors(runs):
    used = {k: 0 for k in COLORS}
    out = []
    for label in runs:
        e = label[:3]
        out.append(COLORS[e][used[e] % len(COLORS[e])])
        used[e] += 1
    return out


def scores(tag, RUNS):
    fig, ax = plt.subplots(1, 2, figsize=(max(12, 1.7 * len(RUNS) + 5), 4.2), sharey=False)
    colors = run_colors(RUNS)
    for a, (region, name) in zip(ax, REGIONS.items()):
        for j, metric in enumerate(["cropland_macro_f1", "macro_f1_3class"]):
            for i, (label, run) in enumerate(RUNS.items()):
                m = pd.read_csv(res(run, region) / "metrics.csv").set_index("metric").loc[metric]
                x = j * (len(RUNS) + 1) + i
                a.bar(x, m["value"], color=colors[i], label=label if j == 0 else None)
                a.errorbar(x, m["value"], yerr=[[m["value"] - m["ci_low"]], [m["ci_high"] - m["value"]]],
                           color="k", capsize=3, lw=1)
                a.text(x, m["ci_low"] - 0.012, f"{m['value']:.3f}", ha="center", va="top", fontsize=7, rotation=90,
                       color="white")
        a.set_xticks([(len(RUNS) - 1) / 2, len(RUNS) + 1 + (len(RUNS) - 1) / 2])
        a.set_xticklabels(["cropland macro-F1\n(irrigated, rainfed)", "3-class macro-F1"])
        a.set_ylim(0.5, 1.0)
        a.set_title(name)
        a.grid(axis="y", alpha=0.3)
    ax[0].set_ylabel("out-of-fold score (95% block-bootstrap CI)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=len(RUNS), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / f"scores_{tag}.png", dpi=130)
    plt.close(fig)


def confusion(tag, RUNS_CONFUSION):
    fig, ax = plt.subplots(len(REGIONS), len(RUNS_CONFUSION), figsize=(4.4 * len(RUNS_CONFUSION), 8))
    for r, (region, name) in enumerate(REGIONS.items()):
        for c, (label, run) in enumerate(RUNS_CONFUSION.items()):
            o = pd.read_parquet(res(run, region) / "oof_predictions.parquet", columns=["label", "pred"])
            cm = pd.crosstab(o["label"], o["pred"]).reindex(index=[1, 2, 3], columns=[1, 2, 3], fill_value=0).to_numpy()
            share = cm / cm.sum(1, keepdims=True)
            a = ax[r, c]
            a.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for k in range(3):
                    a.text(k, i, f"{share[i, k]:.2f}\n({cm[i, k]:,})", ha="center", va="center", fontsize=8,
                           color="white" if share[i, k] > 0.5 else "black")
            a.set_xticks(range(3), CLASSES, fontsize=8)
            a.set_yticks(range(3), CLASSES, fontsize=8)
            a.set_xlabel("predicted")
            a.set_ylabel("reference")
            a.set_title(f"{name} · {label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{tag}.png", dpi=130)
    plt.close(fig)


def importance():
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    for r, (region, name) in enumerate(REGIONS.items()):
        d = res("E01_s2_pixel_baseline/rf", region)
        im = pd.read_csv(d / "importance_month.csv")
        ib = pd.read_csv(d / "importance_band.csv").sort_values("mean", ascending=False)
        ax[r, 0].bar(month_labels(region), im["mean"], yerr=im["std"], color="#3182bd", capsize=2)
        ax[r, 0].set_title(f"{name}: importance per month (agricultural year)", fontsize=10)
        ax[r, 1].bar(ib["group"], ib["mean"], yerr=ib["std"], color="#e6550d", capsize=2)
        ax[r, 1].set_title(f"{name}: importance per band (all months)", fontsize=10)
        for a in ax[r]:
            a.set_ylabel("drop in 3-class macro-F1")
            a.axhline(0, color="k", lw=0.5)
            a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "e01_importance.png", dpi=130)
    plt.close(fig)


def attention():
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    colors = {1: "#3182bd", 2: "#e6550d", 3: "#969696"}
    for a, (region, name) in zip(ax, REGIONS.items()):
        att = pd.read_csv(res("E02_temporal_nn/ltae", region) / "attention_by_month.csv")
        mean = att.drop(columns=["round", "seed"]).groupby("class").mean()
        for c, row in mean.iterrows():
            a.plot(month_labels(region), row.to_numpy(), marker="o", color=colors[c], label=CLASSES[c - 1])
        a.axhline(1 / 12, color="k", lw=0.6, ls=":")
        a.set_title(f"{name}: L-TAE attention per month", fontsize=10)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("mean attention (uniform = 1/12)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "e02_attention.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, f in FIGURES.items():
        if "scores" in f:
            scores(tag, f["scores"])
        confusion(tag, f["confusion"])
    importance()                                   # (the E02 L-TAE attention figure is on hold with the pixel L-TAE)
    print(f"Figures in {OUT}")
