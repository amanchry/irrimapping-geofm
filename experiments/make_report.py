"""
make_report.py  —  write the result tables and figures of an experiment into its report page
==========================================================================================

What it does
------------
1. Reads the experiment's config (experiments/<EXPERIMENT>/config.yaml) and its results for every region
   (results/<region>/comparison/ from models_run/04_compare_models.py, results/<region>/<run>/ from the training
   scripts). E01 (before the two regions) has its results directly in results/ and is read as Morocco.
2. Writes the figures into the documentation, docs/experiments/<EXPERIMENT>/figures/ (file names start with the
   region): macro-F1 per input and classifier (95% CI), macro-F1 per test fold, the learning curves (label
   efficiency) and the confusion matrices of every run.
3. Fills the part of docs/experiments/<EXPERIMENT>/report.md between <!-- AUTO:RESULTS:START --> and
   <!-- AUTO:RESULTS:END --> with the setup, the result tables and the figures. Everything outside these markers
   (answer, findings, interpretation) is written by hand and never changed. A missing report.md is created from
   experiments/_template/REPORT.md.
4. Refreshes the experiment list (docs/experiments/index.md), the results overview (docs/results.md) and the
   Experiments part of the site navigation (docs/SUMMARY.md: the entries under "All experiments").
Experiments with "listed: false" in their config are left out of the overview pages and the navigation.

Run (from the project folder)
-----------------------------
    python experiments/make_report.py          # or as the "report" step of run_experiment.py
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
import yaml

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]               # project folder
EXPERIMENT = "E02_frozen_pca"                             # folder name in experiments/ (None = only the overview pages)
# ==========================================================================

EXPERIMENT = os.environ.get("EXPERIMENT", EXPERIMENT)
EXP_ROOT = ROOT / "experiments"                 # configs, results, scripts
DOCS = ROOT / "docs"                            # the MkDocs documentation
REPO = "https://github.com/amanchry/irrimapping-geofm/tree/main"   # links to result tables on GitHub
INPUT_NAMES = {"clay": "Clay v1.5", "prithvi": "Prithvi-EO-2.0", "terramind": "TerraMind", "agrifm": "AgriFM",
               "psetae": "PSE-TAE"}
CLF_NAMES = {"logreg": "LogReg", "rf": "RF", "hgb": "HGB", "svm": "SVM", "mlp": "MLP", "ltae": "L-TAE",
             "psetae": "PSE-TAE"}
CLF_FULL = {"logreg": "Logistic regression (LogReg)", "rf": "Random forest (RF)",
            "hgb": "Gradient boosting (HGB)", "svm": "Support vector machine (SVM)",
            "mlp": "Multilayer perceptron (MLP)", "ltae": "L-TAE + MLP (temporal attention)",
            "psetae": "PSE-TAE (pixel-set encoder + temporal attention)"}     # names in the report tables
COLORS = {"S2 baseline": "#6B7480", "S1/S2 baseline": "#6B7480", "Clay v1.5": "#1F5F8B", "Prithvi-EO-2.0": "#B4541A",
          "TerraMind": "#2E8B57", "AgriFM": "#7B4FA0", "PSE-TAE": "#C08A00"}
REGION_NAMES = {"morocco": "Morocco", "utah": "Utah 2023/24"}
LE_NAME = re.compile(r"_f\d{3}_d\d+$")                       # label-efficiency run folders
STATUS = {"done": "✅ done", "ready": "🟦 ready to run", "running": "⏳ running", "planned": "📝 planned",
          "parked": "⏸ parked"}


def input_name(run):
    """Run folder -> readable input name ('E02_frozen_pca/clay_...' from another experiment keeps its ID)."""
    prefix, _, base = run.rpartition("/")
    if base.startswith("s1s2"):
        name = "S2 baseline" if base.startswith("s1s2-s2") else "S1/S2 baseline"
    else:
        name = INPUT_NAMES.get(base.split("_")[0].replace("-s1s2", ""), base)
    if base.endswith("_mlp"):
        name += " (MLP)"
    elif base.endswith("_ltae"):
        name += " (L-TAE)"
    return f"{name} [{prefix.split('_')[0]}]" if prefix else name


ORDER = ["S2 baseline", "S1/S2 baseline", "Clay v1.5", "Prithvi-EO-2.0", "TerraMind", "AgriFM", "PSE-TAE"]


def base_name(name):
    return name.split(" [")[0].replace(" (L-TAE)", "").replace(" (MLP)", "")


def run_order(runs):
    """Runs sorted as: baseline, Clay, Prithvi, TerraMind, AgriFM, others; other experiments' runs after their own."""
    def key(run):
        name = input_name(run)
        base = base_name(name)
        return (ORDER.index(base) if base in ORDER else len(ORDER), "[" in name, name)
    return sorted(dict.fromkeys(runs), key=key)


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fold_descriptions(region):
    """{fold: 'Doukkala 180, Souss 59'}: where the rainfed parcels of each fold are (zones with >= 10 of them),
    counting only the agricultural years of the region's views (Utah's fold file also holds 2021)."""
    import pyogrio
    sys.path.insert(0, str(ROOT / "models_run"))
    from experiment_utils import region_paths, views_name
    paths = region_paths(region, "agyear", views_name(region))
    if not paths["split_file"].exists():
        return {}
    props = pyogrio.read_dataframe(paths["split_file"], columns=["ag_season", "label", "zone", "fold"],
                                   read_geometry=False)
    info = paths["views_dir"] / "views_info.json"
    seasons = json.loads(info.read_text()).get("seasons") if info.exists() else None
    if seasons:
        props = props[props["ag_season"].isin(seasons)]
    props["zone"] = props["zone"].astype(str).str.strip().str.title()
    out = {}
    for fold, g in props.groupby("fold"):
        rain = g[g["label"] == 2]["zone"].value_counts()
        rain = rain[rain >= 10]
        out[int(fold)] = ", ".join(f"{z} {n}" for z, n in rain.items())
    return out


# ------------------------------------------------------------ figures
def figure_macro_f1(metrics, path):
    m = metrics[metrics["metric"] == "macro_f1"].copy()
    m["input"] = m["run"].map(input_name)
    inputs = [input_name(r) for r in run_order(m["run"])]
    clfs = [c for c in CLF_NAMES if c in set(m["classifier"])]
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(clfs) * 1.2 + 2), 4.2), dpi=150)
    width = 0.8 / len(inputs)
    for j, inp in enumerate(inputs):
        sub = m[m["input"] == inp].set_index("classifier").reindex(clfs)
        x = np.arange(len(clfs)) + (j - (len(inputs) - 1) / 2) * width
        ax.bar(x, sub["estimate"], width * 0.92, color=COLORS.get(base_name(inp), "#999"),
               label=inp, alpha=0.95 if j == 0 else 0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, sub["estimate"], yerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
                    fmt="none", ecolor="#1C2733", elinewidth=0.8, capsize=2)
    lo = max(0.0, np.floor((m["ci_low"].min() - 0.03) * 20) / 20)
    ax.set_ylim(lo, 1.0)
    ax.set_xticks(range(len(clfs)), [CLF_NAMES[c] for c in clfs])
    ax.set_ylabel("Macro-F1 (95% block-bootstrap CI)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=min(3, len(inputs)), fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_per_fold(per_fold, clfs, labels, path):
    fig, axes = plt.subplots(1, len(clfs), figsize=(5.2 * len(clfs), 3.8), dpi=150, sharey=True, squeeze=False)
    for ax, clf in zip(axes[0], clfs):
        sub = per_fold[per_fold["classifier"] == clf]
        for inp, g in sorted(sub.groupby("input", sort=False),
                             key=lambda t: ORDER.index(base_name(t[0])) if base_name(t[0]) in ORDER else len(ORDER)):
            g = g.sort_values("outer_fold")
            base = base_name(inp)
            is_base = base.endswith("baseline")
            ax.plot(g["outer_fold"], g["macro_f1"], marker="o", color=COLORS.get(base, "#999"), label=inp,
                    linewidth=2.2 if is_base else 1.5, linestyle="--" if is_base else "-")
        folds = sorted(per_fold["outer_fold"].unique())
        ax.set_xticks(folds, [f"fold {f}" for f in folds], fontsize=8)
        ax.set_title(CLF_NAMES.get(clf, clf), fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Macro-F1 on the test fold")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ tables
def table_macro_f1(metrics, clfs):
    m = metrics[metrics["metric"] == "macro_f1"]
    best = m.groupby("classifier")["estimate"].max()
    lines = ["| Input | " + " | ".join(CLF_FULL[c] for c in clfs) + " |",
             "| --- | " + " | ".join("---:" for _ in clfs) + " |"]
    for run in run_order(m["run"]):
        cells = []
        for c in clfs:
            r = m[(m["run"] == run) & (m["classifier"] == c)]
            if r.empty:
                cells.append("–")
                continue
            v = r["estimate"].iloc[0]
            cells.append(f"**{v:.3f}**" if np.isclose(v, best[c]) else f"{v:.3f}")
        lines.append(f"| {input_name(run)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_with_ci(metrics, clfs, metric_names):
    lines = ["| Input | Classifier | " + " | ".join(metric_names.values()) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in metric_names) + " |"]
    for run in run_order(metrics["run"]):
        for c in clfs:
            cells = []
            for key in metric_names:
                r = metrics[(metrics["run"] == run) & (metrics["classifier"] == c) & (metrics["metric"] == key)]
                cells.append("–" if r.empty else
                             f"{r['estimate'].iloc[0]:.3f} [{r['ci_low'].iloc[0]:.2f}–{r['ci_high'].iloc[0]:.2f}]")
            if any(x != "–" for x in cells):
                lines.append(f"| {input_name(run)} | {CLF_FULL.get(c, c)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict(row):
    if row["ci_low"] > 0:
        return "**better**" if row["p_holm"] < 0.05 else "higher, not significant after Holm"
    if row["ci_high"] < 0:
        return "**worse**" if row["p_holm"] < 0.05 else "lower, not significant after Holm"
    return "no clear difference"


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table_differences(diffs, clfs):
    d = diffs[diffs["classifier"].isin(clfs)].copy()
    d["_o"] = d["run"].map({r: i for i, r in enumerate(run_order(d["run"]))})
    d = d.sort_values(["_o", "classifier"], key=lambda c: c.map(lambda v: clfs.index(v) if v in clfs else v)
                      if c.name == "classifier" else c)
    d["verdict"] = d.apply(verdict, axis=1)
    lines = ["| Input | Classifier | Difference in macro-F1 (Δ) | 95% confidence interval (CI) | p-value (Holm-corrected) | Verdict |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for _, r in d.iterrows():
        vs = "" if r["classifier"] == r["baseline_classifier"] else f" (vs baseline {CLF_NAMES[r['baseline_classifier']]})"
        lines.append(f"| {input_name(r['run'])} | {CLF_FULL.get(r['classifier'], r['classifier'])}{vs} | "
                     f"{r['delta_macro_f1']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                     f"{fmt_p(r['p_holm'])} | {r['verdict']} |")
    return "\n".join(lines), d


def table_per_fold(per_fold, clf):
    sub = per_fold[per_fold["classifier"] == clf]
    wide = sub.pivot_table(index="input", columns="outer_fold", values="macro_f1", sort=False)
    wide = wide.reindex(sorted(wide.index, key=lambda n: ORDER.index(base_name(n)) if base_name(n) in ORDER else len(ORDER)))
    folds = list(wide.columns)
    lines = ["| Input | " + " | ".join(f"Fold {f}" for f in folds) + " | Mean ± standard deviation (SD) |",
             "| --- | " + " | ".join("---:" for _ in folds) + " | ---: |"]
    for inp, row in wide.iterrows():
        lines.append(f"| {inp} | " + " | ".join(fmt(v) for v in row) +
                     f" | {row.mean():.3f} ± {row.std(ddof=1):.3f} |")
    return "\n".join(lines)


def describe_setup(cfg, run_configs):
    te = cfg.get("train_evaluate") or {}
    lt = cfg.get("ltae") or {}
    steps = cfg.get("steps", [])
    validation = te.get("VALIDATION", lt.get("VALIDATION", "pastis"))
    val_text = {"pastis": "5 spatial folds (20 km blocks); per round 3 folds train, the next fold chooses the "
                          "hyperparameters or stops training, 1 fold tests (PASTIS rotation)",
                "nested": "5 spatial folds (20 km blocks); hyperparameters by inner spatial CV on the 4 training "
                          "folds, model refit on all 4 (nested spatial CV)"}.get(validation, validation)
    if "ltae" in steps and lt.get("HEAD", "ltae") == "ltae":
        agg_text = "L-TAE attention over the 12 monthly embeddings, trained with the head (months without data masked)"
        heads = "L-TAE + MLP head (16 heads, d_k 8, d_model 256, MLP 256-128), 3 seeds"
    elif "ltae" in steps:
        agg_text = f"12 monthly embeddings side by side → standardise → PCA {lt.get('PCA_DIM', 128)} (training folds only)"
        heads = "MLP head 128 → 256 → 128 → 2 (batch normalisation, dropout 0.2), 3 seeds"
    elif "psetae" in steps:
        agg_text = "PSE-TAE temporal attention on the pixel sets (no foundation model)"
        heads = "PSE-TAE trained from scratch, 3 seeds"
    else:
        agg_text = {"concat": f"12 monthly embeddings side by side → standardise → PCA {te.get('PCA_DIM', 128)} "
                              f"(fitted on the training folds only)",
                    "mean": "average of the 12 monthly embeddings"}.get(te.get("FEATURE_MODE", "concat"))
        heads = ("logistic regression (linear probe), random forest, gradient boosting (HGB), RBF SVM"
                 if cfg.get("id") == "E01" else "logistic regression (linear probe), random forest, RBF SVM")
    le = cfg.get("label_efficiency")
    n = sorted({rc.get("n_parcels") for rc in run_configs.values() if rc.get("n_parcels")})
    rows = [("Question", cfg.get("question", "")),
            ("Research question", cfg.get("research_question", "")),
            ("Regions", ", ".join(REGION_NAMES.get(r, r) for r in cfg.get("regions", ["morocco"]))),
            ("Inputs", ", ".join({"s1s2": "S2 baseline"}.get(i, INPUT_NAMES.get(i, i)) for i in cfg.get("inputs", []))
             or "pixel sets (Sentinel-2)"),
            ("Temporal aggregation", agg_text),
            ("Head", heads),
            ("Validation", val_text),
            ("Label efficiency", f"also trained on {', '.join(f'{f:.0%}' for f in le['fractions'])} of the training "
                                 f"blocks, {le['draws']} block draw{'s' if le['draws'] > 1 else ''} each (same draws in every experiment)" if le else ""),
            ("Parcels", " / ".join(f"{x:,}" for x in n) + " (inputs scored on the parcels common to all runs)" if n else ""),
            ("Uncertainty", "95% confidence interval (CI) from a block bootstrap (1,000 draws of whole 20 km blocks), the same draws for "
                            "every run → paired differences; p-values Holm-adjusted"),
            ("Changed from", cfg.get("changes_from", "–"))]
    return "| | |\n| --- | --- |\n" + "\n".join(f"| {k} | {v} |" for k, v in rows if v)


def table_learning_curve(curve, clfs):
    """Macro-F1 (mean over draws) per run and training fraction, for the headline classifiers."""
    c = curve[(curve["metric"] == "macro_f1") & curve["classifier"].isin(clfs)]
    fracs = sorted(c["fraction"].unique())
    lines = ["| Input | Head | " + " | ".join(f"{f:.0%}" for f in fracs) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in fracs) + " |"]
    if "train_parcels_per_round" in c:
        tp = c.groupby("fraction")["train_parcels_per_round"].mean()
        lines.append("| *training parcels per round* | | " + " | ".join(
            "all" if f >= 1 else ("–" if pd.isna(tp.get(f)) else f"{tp[f]:,.0f}") for f in fracs) + " |")
    for run in run_order(c["run"]):
        for clf in clfs:
            g = c[(c["run"] == run) & (c["classifier"] == clf)].set_index("fraction")
            if g.empty:
                continue
            cells = [("–" if f not in g.index else f"{g.loc[f, 'estimate']:.3f}"
                      + (f" ± {g.loc[f, 'sd_draws']:.3f}" if f < 1 and g.loc[f, 'n_draws'] > 1 else "")) for f in fracs]
            lines.append(f"| {input_name(run)} | {CLF_FULL.get(clf, clf)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------ REPORT.md
def replace_block(text, name, content):
    start, end = f"<!-- AUTO:{name}:START -->", f"<!-- AUTO:{name}:END -->"
    if start not in text:
        raise SystemExit(f"Markers {start} / {end} not found")
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    note = "<!-- generated by experiments/make_report.py: do not edit between these markers -->"
    return f"{pre}{start}\n{note}\n\n{content.strip()}\n\n{end}{post}"


def result_dirs(exp_dir, cfg):
    """[(region, results folder)]: results/<region>/ for every region, or results/ itself (E01, Morocco only)."""
    res = exp_dir / "results"
    if (res / "comparison" / "metrics_all_runs.csv").exists():
        return [("morocco", res)]
    return [(r, res / r) for r in cfg.get("regions", ["morocco"]) if (res / r / "comparison" / "metrics_all_runs.csv").exists()]


def table_vs_earlier(pairs):
    """This experiment's head minus the earlier experiments' heads on the same input (own Holm family)."""
    lines = ["| Input | This experiment | Earlier run | Difference in macro-F1 (Δ) | 95% confidence interval (CI) | p-value (Holm-corrected) | Verdict |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    pairs = pairs.copy()
    pairs["_o"] = pairs["run"].map({r: i for i, r in enumerate(run_order(pairs["run"]))})
    for _, r in pairs.sort_values(["_o", "other_run", "other_classifier"]).iterrows():
        lines.append(f"| {base_name(input_name(r['run']))} | {CLF_FULL.get(r['classifier'], r['classifier'])} | "
                     f"{input_name(r['other_run']).split(' [')[1].rstrip(']')} {CLF_NAMES.get(r['other_classifier'], r['other_classifier'])} | "
                     f"{r['delta_macro_f1']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {fmt_p(r['p_holm'])} | "
                     f"{verdict(r)} |")
    return "\n".join(lines)


def seed_table(res, runs):
    """Macro-F1 of each seed of the neural runs (metrics_per_seed.csv)."""
    lines = ["| Input | Head | Macro-F1 per seed | Mean ± standard deviation (SD) |", "| --- | --- | --- | ---: |"]
    for run in run_order(runs):
        f = res / run / "metrics_per_seed.csv"
        if "/" in run or not f.exists():
            continue
        s = pd.read_csv(f)
        head = run.rsplit("_", 1)[-1] if run.endswith(("_mlp", "_ltae")) else "psetae"
        lines.append(f"| {base_name(input_name(run))} | {CLF_FULL.get(head, head)} | "
                     + ", ".join(f"{v:.3f}" for v in s["macro_f1"]) +
                     f" | {s['macro_f1'].mean():.3f} ± {s['macro_f1'].std(ddof=1) if len(s) > 1 else 0:.3f} |")
    return "\n".join(lines) if len(lines) > 2 else ""


def figure_training_curves(res, runs, path):
    """Validation macro-F1 per epoch (first seed, every round) of the neural runs."""
    hist = [(run, pd.read_csv(res / run / "training_history.csv")) for run in run_order(runs)
            if "/" not in run and (res / run / "training_history.csv").exists()]
    if not hist:
        return False
    fig, axes = plt.subplots(1, len(hist), figsize=(4.2 * len(hist), 3.4), dpi=130, squeeze=False, sharey=True)
    for ax, (run, h) in zip(axes[0], hist):
        if "seed" in h:
            h = h[h["seed"] == h["seed"].min()]
        for fold, g in h.groupby("outer_fold"):
            ax.plot(g["epoch"], g["val_macro_f1"], label=f"test fold {fold}", linewidth=1.3)
        ax.set_title(input_name(run), fontsize=9, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("validation macro-F1")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_attention(res, runs, path):
    """L-TAE attention per month, mean over the test folds, per class (attention_by_month.csv)."""
    att = [(run, pd.read_csv(res / run / "attention_by_month.csv")) for run in run_order(runs)
           if "/" not in run and (res / run / "attention_by_month.csv").exists()]
    if not att:
        return False
    fig, axes = plt.subplots(1, len(att), figsize=(4.2 * len(att), 3.4), dpi=130, squeeze=False, sharey=True)
    for ax, (run, a) in zip(axes[0], att):
        steps = list(dict.fromkeys(a["step"]))
        for c, name, color in [(1, "irrigated", "#1f77b4"), (2, "rainfed", "#d62728")]:
            m = a[a["class"] == c].groupby("step", sort=False)["attention"].mean().reindex(steps)
            ax.plot(steps, m.to_numpy(), marker="o", color=color, label=name)
        ax.set_title(input_name(run), fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("attention weight (mean over heads)")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def breakdown_parts(res, figs, tag, region):
    """Results by zone (script 05): focus zone with CI, irrigated-only zones, parcel area."""
    b = res / "breakdown"
    if not (b / "focus_zone.csv").exists():
        return []
    fz = pd.read_csv(b / "focus_zone.csv")
    lines = ["| Input | Head | Macro-F1 [95% confidence interval] | Difference from the baseline [95% confidence interval] |", "| --- | --- | ---: | ---: |"]
    for run in run_order(fz["run"]):
        for _, r in fz[fz["run"] == run].iterrows():
            delta = ("–" if pd.isna(r.get("delta_vs_baseline", np.nan)) else
                     f"{r['delta_vs_baseline']:+.3f} [{r['delta_ci_low']:+.3f}, {r['delta_ci_high']:+.3f}]")
            lines.append(f"| {input_name(run)} | {CLF_FULL.get(r['classifier'], r['classifier'])} | "
                         f"{r['estimate']:.3f} [{r['ci_low']:.2f}–{r['ci_high']:.2f}] | {delta} |")
    first = fz.iloc[0]
    parts = ["### Results by zone",
             f"Only the focus zone, where both classes are common ({int(first['n_parcels']):,} parcels, "
             f"{int(first['n_rainfed'])} rainfed, {int(first['n_blocks'])} blocks): does the input separate the "
             "classes inside one region, or does it recognise the region? Script `05_breakdown_oof.py`.",
             how_to_read("zone"),
             "\n".join(lines)]
    if (b / "breakdown.png").exists():
        (figs / f"{tag}breakdown.png").write_bytes((b / "breakdown.png").read_bytes())
        parts.append(f"![Results by zone, {region}](figures/{tag}breakdown.png)")
    parts.append(f"Tables per zone and season, patch purity and parcel area: `{res.relative_to(ROOT).as_posix()}/breakdown/`.")
    return parts


# Short "how to read this" notes for readers new to the tables; collapsed until clicked (pymdownx.details).
HOW_TO_READ = {
    "macro_f1": """\
**What:** one score per input (row) and classifier (column), between 0 and 1. Higher is better.

**Macro-F1:** F1 is computed for each class separately (irrigated, rainfed) and the two are averaged. Rainfed
parcels are the minority, so a model that labels everything "irrigated" still gets a high overall accuracy but a
low macro-F1. That is why macro-F1 is the main metric.

**How:** every parcel is predicted once, by the model that did *not* see its 20 km block in training (out-of-fold).
All inputs are scored on the same parcels, so the numbers can be compared directly.

**Read it as:** about 0.90 = very good; 0.75 = clear errors on the rainfed class; 0.50 = no better than guessing one class.""",
    "differences": """\
**What:** each row compares one input with the baseline (the Sentinel feature classifier), using the same classifier.

- **Difference in macro-F1 (Δ):** the input's score minus the baseline's. +0.013 means 0.013 higher than the baseline; −0.158
  means 0.158 lower.
- **95% confidence interval (CI), e.g. [−0.021, +0.025]:** the range in which the true difference probably lies. It comes from
  resampling whole 20 km blocks 1,000 times (block bootstrap) and recomputing Δ each time.
    - The range **includes 0** (one end negative, the other positive): the data cannot tell which input is
      better.
    - The range is **entirely above 0**, e.g. [+0.011, +0.049]: the input is better.
    - The range is **entirely below 0**, e.g. [−0.215, −0.085]: the input is worse.
- **p-value (Holm-corrected):** how likely a difference this large would be if the two inputs were really equal. Many pairs are
  tested at once, so each p-value is raised (Holm correction) to avoid false "wins" by chance. 1.000 = no evidence
  of a difference.
- **Verdict:** "better" or "worse" only if the CI excludes 0 **and** the Holm-corrected p-value < 0.05. "Higher/lower, not
  significant after Holm" = the CI excludes 0, but the evidence does not survive the correction.

**Why paired:** both inputs are scored on the same parcels and the same bootstrap draws, so a difference reflects
the inputs, not which parcels happened to be drawn.""",
    "vs_earlier": """\
**What:** the same input with two different classifiers, e.g. Clay + MLP (this experiment) minus Clay + SVM (E02).
It shows whether the classifier, not the input, makes the difference.

Δ, the 95% confidence interval and the verdict are read as in the table above: a CI that includes 0 means no clear difference.""",
    "per_class": """\
**What:** the main metric and three others for the headline classifiers, each with its 95% confidence interval in brackets
(the range the true value probably lies in).

- **Macro-F1:** mean F1 of the two classes (main metric).
- **F1 of the rainfed class:** how well the minority class is found. Precision and recall of "rainfed" are combined, so it is
  usually the lowest number and the most informative one.
- **Balanced accuracy:** mean recall of the two classes. 0.5 = guessing.
- **Overall accuracy:** share of all parcels labelled correctly. It looks high even for weak models, because most
  parcels are irrigated, so do not compare inputs on it alone.

**Wide CIs** (e.g. 0.63–0.88) mean the score changes a lot from block to block, so small differences between
rows are not meaningful.""",
    "label_efficiency": """\
**What:** how the score changes when the model is trained with fewer labelled parcels.

- **Columns:** share of the training blocks used (10%, 25%, 50%; 100% = the main result). The row *training
  parcels per round* gives the actual number of parcels. Validation and test folds are always complete, so every
  column is tested on the same parcels.
- **0.807 ± 0.091:** the mean over 3 different random draws of blocks ± their standard deviation. A large ± means
  the result depends on *which* blocks were drawn.
- **Read it as:** an input that stays high at 10% needs fewer labels (it is "label-efficient"). If its curve rises
  steeply, it needs many labels.

**Why whole blocks:** dropping single parcels would leave neighbours of every parcel in training. Dropping whole
20 km blocks simulates having field data from fewer places.""",
    "seeds": """\
**What:** neural networks start from random weights, so each run (seed) gives a slightly different result. The
table shows each seed on its own. Small differences between seeds (≈ 0.01) mean the result is stable, and a
difference between inputs smaller than this spread is not meaningful.""",
    "zone": """\
**What:** the score inside the one zone where both classes are common (the focus zone), not over the whole region.

**Why:** across a whole region, a model can score well just by learning *where* parcels are (one zone is almost all
irrigated, another has the rainfed parcels). Inside one zone, location does not help, so this shows whether the
input separates irrigated from rainfed fields themselves.

**Difference from the baseline [95% confidence interval]:** as in the difference table. A CI that includes 0 means no clear difference.""",
    "per_fold": """\
**What:** the score on each test fold separately. Each fold is a different set of 20 km blocks, so the five numbers
show how much the result changes from one part of the region to another.

- **Mean ± standard deviation (SD):** average of the five folds and their spread. This can differ slightly from the main table, which
  scores all parcels at once.
- **Read it as:** a low fold points to a part of the region where the model struggles. The fold description above
  lists where its rainfed parcels are. An input with a smaller SD is more consistent across space.""",
    "importance": """\
**What:** one month (or one variable) at a time is shuffled on the test fold, and the drop in macro-F1 is measured.
A large drop = the model relies on that month. Close to 0 = the model does not need it. Bars below 0 are noise.

For foundation models after PCA, importance is usually spread thinly over all months, so it says little about which
months the embedding encodes.""",
}


def how_to_read(key, what="table"):
    """Collapsed admonition with the reading note of a table (opens on click)."""
    body = "\n".join(("    " + line) if line else "" for line in HOW_TO_READ[key].splitlines())
    return f'??? info "How to read this {what}"\n\n{body}'


def region_section(exp, cfg, region, res, figs, headline):
    """Tables and figures of one region."""
    comp = res / "comparison"
    metrics = pd.read_csv(comp / "metrics_all_runs.csv")
    diffs = (pd.read_csv(comp / "differences_vs_baseline.csv") if (comp / "differences_vs_baseline.csv").exists()
             else pd.DataFrame())
    all_clfs = [c for c in CLF_NAMES if c in set(metrics["classifier"])]
    head = [c for c in headline if c in all_clfs] or all_clfs[:2]
    per_fold, run_configs = [], {}
    for run in dict.fromkeys(metrics["run"]):
        if "/" in run:                                              # run of another experiment (EXTRA_RUNS)
            other = EXP_ROOT / run.split("/")[0] / "results"
            rdir = (other / region if (other / region).exists() else other) / run.split("/", 1)[1]
        else:
            rdir = res / run
        if (rdir / "metrics_per_fold.csv").exists():
            pf = pd.read_csv(rdir / "metrics_per_fold.csv")
            if "model" not in pf:
                pf["model"] = run.split("/")[-1].split("_")[0]
            per_fold.append(pf.rename(columns={"model": "classifier"}).assign(input=input_name(run)))
        if (rdir / "run_config.json").exists():
            run_configs[run] = json.loads((rdir / "run_config.json").read_text())
    per_fold = pd.concat(per_fold, ignore_index=True) if per_fold else pd.DataFrame()
    labels = fold_descriptions(region)
    tag = f"{region}_"
    figure_macro_f1(metrics, figs / f"{tag}macro_f1_by_input.png")
    fold_clfs = [c for c in head + ["svm"] if c in all_clfs][:3]
    if not per_fold.empty:
        figure_per_fold(per_fold, fold_clfs, labels, figs / f"{tag}macro_f1_per_fold.png")
    base = "S2 baseline" if any(r.split("/")[-1].startswith("s1s2-s2") for r in metrics["run"]) else "S1/S2 baseline"
    parts = [f"## {REGION_NAMES.get(region, region)}",
             "### Macro-F1 of every input and head",
             "Out-of-fold predictions of all 5 spatial folds; **bold** = best input for that head. Runs marked "
             "[E0x] come from an earlier experiment, on the same parcels and bootstrap draws.",
             how_to_read("macro_f1"),
             table_macro_f1(metrics, all_clfs),
             f"![Macro-F1 per input and head, {region}](figures/{tag}macro_f1_by_input.png)"]
    if not diffs.empty:
        dtable, _ = table_differences(diffs, head)
        parts += [f"### Difference from the {base} (headline heads)",
                  "Δ = input minus baseline with the same head (runs with their own model against the baseline's "
                  "random forest), on the same parcels and the same bootstrap draws. **Better / worse** = the 95% confidence "
                  "interval (CI) excludes 0 **and** the Holm-corrected p-value < 0.05.",
                  how_to_read("differences"),
                  dtable]
        other = diffs[~diffs["classifier"].isin(head)]
        if not other.empty:
            parts += ["<details markdown=\"1\"><summary>Other heads (robustness check)</summary>\n",
                      table_differences(diffs, [c for c in all_clfs if c not in head])[0], "\n</details>"]
    if (comp / "differences_vs_earlier_experiments.csv").exists():
        parts += ["### Same input, earlier experiments' heads",
                  "Δ = this experiment's head minus the head of an earlier experiment on the same input, same "
                  "parcels and bootstrap draws (own Holm family). E03: MLP vs the E02 classifiers; E04: L-TAE vs the "
                  "E03 MLP and the E02 classifiers.",
                  how_to_read("vs_earlier"),
                  table_vs_earlier(pd.read_csv(comp / "differences_vs_earlier_experiments.csv"))]
    parts += ["### Per-class metrics (headline heads, 95% confidence interval)",
              how_to_read("per_class"),
              table_with_ci(metrics, head, {"macro_f1": "Macro-F1", "f1_rainfed": "F1 of the rainfed class",
                                            "balanced_accuracy": "Balanced accuracy", "overall_accuracy": "Overall accuracy"})]
    if (comp / "learning_curve.csv").exists():
        curve = pd.read_csv(comp / "learning_curve.csv")
        if (comp / "learning_curve.png").exists():
            (figs / f"{tag}learning_curve.png").write_bytes((comp / "learning_curve.png").read_bytes())
        parts += ["### Label efficiency",
                  "Macro-F1 when trained on a share of the training blocks (validation and test folds complete); mean "
                  "± SD over the 3 block draws. Draws are nested (the 10% blocks are part of the 25% and 50% blocks) "
                  "and identical in every experiment. Neural heads: every point is a single-seed result (the 100% "
                  "point is the mean over its 3 seeds scored one by one).",
                  how_to_read("label_efficiency"),
                  table_learning_curve(curve, head + [c for c in all_clfs if c not in head]),
                  f"![Learning curves, {region}](figures/{tag}learning_curve.png)"]
    st = seed_table(res, metrics["run"].unique())
    if st:
        parts += ["### Spread over seeds (neural heads)",
                  "Macro-F1 of each seed on its own; the result above is the mean probability of the seeds.",
                  how_to_read("seeds"), st]
    if figure_attention(res, metrics["run"].unique(), figs / f"{tag}attention_by_month.png"):
        parts += ["### Which months the L-TAE attends to",
                  "Attention weight per month (mean over heads and test folds), per class (first seed).",
                  f"![Attention per month, {region}](figures/{tag}attention_by_month.png)"]
    imp = []
    for p in sorted(res.glob("*/importance.png")):
        if LE_NAME.search(p.parent.name):
            continue
        name = f"{tag}importance_{p.parent.name}.png"
        (figs / name).write_bytes(p.read_bytes())
        imp.append(f"[{input_name(p.parent.name)}](figures/{name})")
    if imp:
        parts += ["### Which months matter (permutation importance, random forest)",
                  "Drop in macro-F1 when a month (or a baseline variable) is shuffled on the test fold: " + " · ".join(imp),
                  how_to_read("importance", "figure")]
    if figure_training_curves(res, metrics["run"].unique(), figs / f"{tag}training_curves.png"):
        parts += ["### Training curves",
                  "Validation macro-F1 per epoch in each round (first seed); the best epoch is kept (early stopping).",
                  f"![Training curves, {region}](figures/{tag}training_curves.png)"]
    parts += breakdown_parts(res, figs, tag, region)
    if not per_fold.empty:
        parts += ["### Macro-F1 per test fold",
                  "Each fold is a set of 20 km blocks, used once as the test fold. Rainfed test parcels per fold: "
                  + "; ".join(f"fold {f}: {d}" for f, d in labels.items()) + ".",
                  how_to_read("per_fold"),
                  *[f"**{CLF_NAMES.get(c, c)}**\n\n" + table_per_fold(per_fold, c) for c in fold_clfs],
                  f"![Macro-F1 per test fold, {region}](figures/{tag}macro_f1_per_fold.png)"]
    files = []
    for p in sorted(res.glob("*/confusion_matrices.png")):
        if LE_NAME.search(p.parent.name):
            continue
        name = f"{tag}confusion_{p.parent.name}.png"
        (figs / name).write_bytes(p.read_bytes())
        files.append((p.parent.name, f"figures/{name}"))
    rel = res.relative_to(ROOT).as_posix()
    parts += ["### Files\n\n" + "\n".join([
              f"- Tables: [`{rel}/comparison/`]({REPO}/{rel}/comparison) (all metrics with CI, differences vs "
              "baseline, learning curves)",
              f"- Per run ([`{rel}/<run>/`]({REPO}/{rel})): out-of-fold predictions, metrics per fold (and per seed), "
              "run settings and log",
              "- Confusion matrices: " + (" · ".join(f"[{input_name(run)}]({path})" for run, path in files) or "–")])]
    return parts, run_configs, len(metrics["run"].unique())


def build_report(exp):
    exp_dir = EXP_ROOT / exp
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8"))
    dirs = result_dirs(exp_dir, cfg)
    if not dirs:
        print(f"{exp}: no results/<region>/comparison yet (run the compare step first): report not written")
        return
    headline = (cfg.get("report") or {}).get("HEADLINE_CLASSIFIERS", ["rf", "logreg"])
    page_dir = DOCS / "experiments" / exp
    figs = page_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    sections, run_configs, n_runs = [], {}, 0
    for region, res in dirs:
        parts, rc, n = region_section(exp, cfg, region, res, figs, headline)
        sections += parts
        run_configs.update(rc)
        n_runs += n
    info_file = exp_dir / "results" / "run_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    last = info.get("runs", [{}])[-1] if info.get("runs") else {}
    parts = ["## Setup", describe_setup(cfg, run_configs), *sections,
             "## Code and settings\n\n" + "\n".join([
                 f"- Settings: [`experiments/{exp}/config.yaml`]({REPO.replace('/tree/', '/blob/')}/experiments/{exp}/config.yaml)",
                 f"- Code version: `{last.get('git_commit', 'see run_config.json of each run')}`"
                 + (f", run {last.get('started')} – {last.get('finished')}" if last.get("finished") else "")])]
    report = page_dir / "report.md"
    text = report.read_text() if report.exists() else (EXP_ROOT / "_template" / "REPORT.md").read_text()
    text = text.replace("{{TITLE}}", cfg["title"]).replace("{{ID}}", cfg["id"])
    report.write_text(replace_block(text, "RESULTS", "\n\n".join(parts)))
    print(f"{exp}: docs/experiments/{exp}/report.md updated ({n_runs} runs, regions {[r for r, _ in dirs]})")


# ------------------------------------------------------------ overview pages
def headline_of(exp_dir, cfg):
    """One-line result of a finished experiment for the overview pages (per region)."""
    texts = []
    for region, res in result_dirs(exp_dir, cfg):
        comp = res / "comparison"
        m = pd.read_csv(comp / "metrics_all_runs.csv")
        m = m[(m["metric"] == "macro_f1") & ~m["run"].str.contains("/")]
        if m.empty:
            continue
        best = m.loc[m["estimate"].idxmax()]
        base = m[m["run"].str.startswith("s1s2")]
        text = f"best {input_name(best['run'])} + {CLF_NAMES.get(best['classifier'])} {best['estimate']:.3f}"
        if not base.empty:
            b = base.loc[base["estimate"].idxmax()]
            text += f"; best baseline {CLF_NAMES.get(b['classifier'])} {b['estimate']:.3f}"
        if (comp / "differences_vs_baseline.csv").exists():
            d = pd.read_csv(comp / "differences_vs_baseline.csv")
            d = d[~d["run"].str.contains("/")]
            better = d[(d["p_holm"] < 0.05) & (d["ci_low"] > 0)]
            worse = d[(d["p_holm"] < 0.05) & (d["ci_high"] < 0)]
            text += f"; {len(better)} of {len(d)} pairs better, {len(worse)} worse than the baseline (Holm)"
        texts.append((f"{REGION_NAMES.get(region, region)}: " if len(cfg.get("regions", [])) > 1 else "") + text)
    return " · ".join(texts) or "–"


def update_overviews():
    rows, results = [], []
    for cfg_file in sorted(EXP_ROOT.glob("E*/config.yaml")):
        cfg = yaml.safe_load(cfg_file.read_text())
        if cfg.get("listed", True) is False:
            continue
        exp_dir = cfg_file.parent
        name = exp_dir.name
        report = f"[report]({name}/report.md)" if (DOCS / "experiments" / name / "report.md").exists() else "–"
        result = headline_of(exp_dir, cfg) if cfg.get("status") == "done" else "–"
        rows.append(f"| **{cfg['id']}** | [{cfg['title']}]({name}/index.md) | {cfg.get('research_question', '')} | "
                    f"{STATUS.get(cfg.get('status'), cfg.get('status'))} | {result} | {report} |")
        if cfg.get("status") == "done":
            results.append((name, cfg, result))

    index = DOCS / "experiments" / "index.md"
    table = ("| ID | Experiment | Research question (RQ) | Status | Headline result (macro-F1) | Report |\n"
             "| --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows))
    index.write_text(replace_block(index.read_text(), "INDEX", table))

    overview = DOCS / "results.md"
    if overview.exists():
        blocks = []
        for name, cfg, result in results:
            figs_all = sorted((DOCS / "experiments" / name / "figures").glob("*macro_f1_by_input.png"))
            figs_region = [f for f in figs_all if f.name != "macro_f1_by_input.png"]   # per-region figures win over
            blocks.append(f"#### {cfg['id']} · {cfg['title']}\n\n{cfg.get('question', '')}\n\n"  # the old single one
                          f"- **Result:** {result}.\n- **Answer:** {cfg.get('answer', 'see the report')}\n"
                          f"- Full report: [{cfg['id']} report](experiments/{name}/report.md)\n\n"
                          + "\n\n".join(f"![{cfg['id']}](experiments/{name}/figures/{f.name})"
                                        for f in (figs_region or figs_all)))
        overview.write_text(replace_block(overview.read_text(), "RESULTS",
                                          "\n\n".join(blocks) or "_No finished experiment yet._"))
    summary = DOCS / "SUMMARY.md"                                   # site navigation: the Experiments section
    if summary.exists():
        nav = []
        for cfg_file in sorted(EXP_ROOT.glob("E*/config.yaml")):
            cfg = yaml.safe_load(cfg_file.read_text())
            name = cfg_file.parent.name
            if cfg.get("listed", True) is False or not (DOCS / "experiments" / name / "index.md").exists():
                continue
            nav.append(f"    * [{cfg['id']} · {cfg.get('short_title', cfg['title'])}](experiments/{name}/index.md)")
            if (DOCS / "experiments" / name / "report.md").exists():
                nav.append(f"        * [Report](experiments/{name}/report.md)")
        # the lines after "* [All experiments](...)" up to the next tab ("* ...") are the experiment entries
        lines = summary.read_text().splitlines()
        i = next(k for k, l in enumerate(lines) if "](experiments/index.md)" in l) + 1
        j = next((k for k in range(i, len(lines)) if lines[k].startswith("* ")), len(lines))
        summary.write_text("\n".join(lines[:i] + nav + lines[j:]) + "\n")
    print(f"Updated {index.relative_to(ROOT)}, {overview.relative_to(ROOT)} and the site navigation")


if __name__ == "__main__":
    if EXPERIMENT:
        build_report(EXPERIMENT)
    update_overviews()
