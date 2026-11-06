"""
make_report.py  —  write the result tables and figures of an experiment into its REPORT.md
==========================================================================================

What it does
------------
1. Reads the experiment's config.yaml and its results (results/comparison/ from
   models_run/04_compare_models.py, and results/<run>/ from the training scripts).
2. Writes two figures into results/figures/: macro-F1 per input and classifier (95% CI), and
   macro-F1 per test fold (which region is hard).
3. Fills the part of REPORT.md between <!-- AUTO:START --> and <!-- AUTO:END --> with the setup,
   the result tables and the figures. Everything outside these markers (the answer, the
   interpretation) is written by hand and never changed. A missing REPORT.md is created from
   experiments/_template/REPORT.md.
4. Refreshes the experiment list in experiments/README.md and the results overview in
   docs/05_results.md (between their AUTO markers).
Experiments with "listed: false" in their config are left out of the two overview pages.

Run (from the project folder)
-----------------------------
    python experiments/make_report.py          # or as the "report" step of run_experiment.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]               # project folder
EXPERIMENT = "E01_frozen_pca_nested"                      # folder name in experiments/ (None = only the overview pages)
FOLDS_FILE = ROOT / "outputs" / "dataset" / "02_folds" / "parcels_folds.geojson"   # to name the regions of each fold
# ==========================================================================

EXPERIMENT = os.environ.get("EXPERIMENT", EXPERIMENT)
EXP_ROOT = ROOT / "experiments"
INPUT_NAMES = {"s1s2": "S1/S2 baseline", "clay": "Clay v1.5", "prithvi": "Prithvi-EO-2.0",
               "terramind": "TerraMind", "agrifm": "AgriFM", "psetae": "PSE-TAE"}
CLF_NAMES = {"logreg": "LogReg", "rf": "RF", "hgb": "HGB", "svm": "SVM", "psetae": "PSE-TAE", "ltae": "L-TAE"}
COLORS = {"S1/S2 baseline": "#6B7480", "Clay v1.5": "#1F5F8B", "Prithvi-EO-2.0": "#B4541A",
          "TerraMind": "#2E8B57", "AgriFM": "#7B4FA0", "PSE-TAE": "#C08A00"}
STATUS = {"done": "✅ done", "ready": "🟦 ready to run", "running": "⏳ running", "planned": "📝 planned",
          "parked": "⏸ parked"}


def input_name(run):
    """Run folder -> readable input name ('E01/clay_...' from another experiment keeps its prefix)."""
    prefix, _, base = run.rpartition("/")
    key = "s1s2" if base.startswith("s1s2") else base.split("_")[0]
    name = INPUT_NAMES.get(key, base)
    if "_ltae" in base:
        name += " (L-TAE)"
    return f"{name} [{prefix.split('_')[0]}]" if prefix else name


ORDER = ["S1/S2 baseline", "Clay v1.5", "Prithvi-EO-2.0", "TerraMind", "AgriFM", "PSE-TAE"]


def run_order(runs):
    """Runs sorted as: baseline, Clay, Prithvi, TerraMind, AgriFM, others; other experiments' runs after their own."""
    def key(run):
        name = input_name(run)
        base = name.split(" [")[0].replace(" (L-TAE)", "")
        return (ORDER.index(base) if base in ORDER else len(ORDER), "[" in name, name)
    return sorted(dict.fromkeys(runs), key=key)


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fold_descriptions():
    """{fold: 'Doukkala 180, Souss 59'}: where the rainfed parcels of each fold are (zones with >= 10 of them)."""
    if not FOLDS_FILE.exists():
        return {}
    props = pd.DataFrame([f["properties"] for f in json.loads(FOLDS_FILE.read_text())["features"]])
    props["zone"] = props["zone"].str.strip().str.title()
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
        ax.bar(x, sub["estimate"], width * 0.92, color=COLORS.get(inp.split(" [")[0].replace(" (L-TAE)", ""), "#999"),
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
                             key=lambda t: ORDER.index(t[0]) if t[0] in ORDER else len(ORDER)):
            g = g.sort_values("outer_fold")
            base = inp.split(" [")[0].replace(" (L-TAE)", "")
            ax.plot(g["outer_fold"], g["macro_f1"], marker="o", color=COLORS.get(base, "#999"), label=inp,
                    linewidth=2.2 if base == "S1/S2 baseline" else 1.5,
                    linestyle="--" if base == "S1/S2 baseline" else "-")
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
    lines = ["| Input | " + " | ".join(CLF_NAMES[c] for c in clfs) + " |",
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
                lines.append(f"| {input_name(run)} | {CLF_NAMES.get(c, c)} | " + " | ".join(cells) + " |")
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
    lines = ["| Input | Classifier | Δ macro-F1 | 95% CI | p (Holm) | Verdict |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for _, r in d.iterrows():
        vs = "" if r["classifier"] == r["baseline_classifier"] else f" (vs baseline {CLF_NAMES[r['baseline_classifier']]})"
        lines.append(f"| {input_name(r['run'])} | {CLF_NAMES.get(r['classifier'], r['classifier'])}{vs} | "
                     f"{r['delta_macro_f1']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                     f"{fmt_p(r['p_holm'])} | {r['verdict']} |")
    return "\n".join(lines), d


def table_per_fold(per_fold, clf):
    sub = per_fold[per_fold["classifier"] == clf]
    wide = sub.pivot_table(index="input", columns="outer_fold", values="macro_f1", sort=False)
    wide = wide.reindex(sorted(wide.index, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER)))
    folds = list(wide.columns)
    lines = ["| Input | " + " | ".join(f"Fold {f}" for f in folds) + " | Mean ± SD |",
             "| --- | " + " | ".join("---:" for _ in folds) + " | ---: |"]
    for inp, row in wide.iterrows():
        lines.append(f"| {inp} | " + " | ".join(fmt(v) for v in row) +
                     f" | {row.mean():.3f} ± {row.std(ddof=1):.3f} |")
    return "\n".join(lines)


def describe_setup(cfg, run_configs):
    te = cfg.get("train_evaluate") or {}
    validation = te.get("VALIDATION", "pastis")
    val_text = {"pastis": "5 spatial folds (20 km blocks); per round 3 folds train, the next fold chooses the "
                          "hyperparameters, 1 fold tests (PASTIS rotation)",
                "nested": "5 spatial folds (20 km blocks); hyperparameters by inner spatial CV on the 4 training "
                          "folds, model refit on all 4 (nested spatial CV)"}.get(validation, validation)
    agg = cfg.get("temporal_aggregation", te.get("FEATURE_MODE", "concat"))
    agg_text = {"concat": f"12 monthly embeddings side by side → standardise → PCA {te.get('PCA_DIM', 128)} "
                          f"(fitted on the training folds only)",
                "mean": "average of the 12 monthly embeddings",
                "ltae": "L-TAE over the 12 monthly embeddings, trained on the training folds"}.get(agg, agg)
    n = sorted({rc.get("n_parcels") for rc in run_configs.values() if rc.get("n_parcels")})
    rows = [("Question", cfg.get("question", "")),
            ("Research question", cfg.get("research_question", "")),
            ("Inputs", ", ".join(INPUT_NAMES.get(i, i) for i in cfg.get("inputs", []))),
            ("Temporal aggregation (GeoFMs)", agg_text),
            ("Classifiers", "logistic regression (linear probe), random forest, gradient boosting (HGB), RBF SVM"),
            ("Validation", val_text),
            ("Parcels", " / ".join(f"{x:,}" for x in n) + " (inputs scored on the parcels common to all runs)"),
            ("Uncertainty", "95% CI from a block bootstrap (1,000 draws of whole 20 km blocks), the same draws for "
                            "every run → paired differences; p-values Holm-adjusted"),
            ("Changed from", cfg.get("changes_from", "–"))]
    return "| | |\n| --- | --- |\n" + "\n".join(f"| {k} | {v} |" for k, v in rows if v)


# ------------------------------------------------------------ REPORT.md
def replace_block(text, name, content):
    start, end = f"<!-- AUTO:{name}:START -->", f"<!-- AUTO:{name}:END -->"
    if start not in text:
        raise SystemExit(f"Markers {start} / {end} not found")
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    note = "<!-- generated by experiments/make_report.py: do not edit between these markers -->"
    return f"{pre}{start}\n{note}\n\n{content.strip()}\n\n{end}{post}"


def build_report(exp):
    exp_dir = EXP_ROOT / exp
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
    res = exp_dir / "results"
    comp = res / "comparison"
    if not (comp / "metrics_all_runs.csv").exists():
        print(f"{exp}: no results/comparison yet (run the compare step first): report not written")
        return
    metrics = pd.read_csv(comp / "metrics_all_runs.csv")
    diffs = (pd.read_csv(comp / "differences_vs_baseline.csv") if (comp / "differences_vs_baseline.csv").exists()
             else pd.DataFrame())
    headline = (cfg.get("report") or {}).get("HEADLINE_CLASSIFIERS", ["rf", "logreg"])
    all_clfs = [c for c in CLF_NAMES if c in set(metrics["classifier"])]

    # per-fold metrics and run settings of every run
    per_fold, run_configs = [], {}
    for run in dict.fromkeys(metrics["run"]):
        rdir = (EXP_ROOT / run.split("/")[0] / "results" / run.split("/", 1)[1]) if "/" in run else res / run
        if (rdir / "metrics_per_fold.csv").exists():
            pf = pd.read_csv(rdir / "metrics_per_fold.csv")
            if "model" not in pf:
                pf["model"] = run.split("_")[0].split("/")[-1]
            per_fold.append(pf.rename(columns={"model": "classifier"}).assign(input=input_name(run)))
        if (rdir / "run_config.json").exists():
            run_configs[run] = json.loads((rdir / "run_config.json").read_text())
    per_fold = pd.concat(per_fold, ignore_index=True) if per_fold else pd.DataFrame()
    labels = fold_descriptions()

    figs = res / "figures"
    figs.mkdir(exist_ok=True)
    figure_macro_f1(metrics, figs / "macro_f1_by_input.png")
    fold_clfs = [c for c in headline + ["svm"] if c in all_clfs][:3]
    if not per_fold.empty:
        figure_per_fold(per_fold, fold_clfs, labels, figs / "macro_f1_per_fold.png")

    parts = ["## Setup", describe_setup(cfg, run_configs),
             "## Results", "### Macro-F1 of every input and classifier",
             "Out-of-fold predictions of all 5 spatial folds; **bold** = best input for that classifier.",
             table_macro_f1(metrics, all_clfs),
             "![Macro-F1 per input and classifier](results/figures/macro_f1_by_input.png)"]
    if not diffs.empty:
        dtable, d = table_differences(diffs, headline)
        parts += ["### Difference from the S1/S2 baseline (headline classifiers)",
                  "Δ = input minus baseline with the same classifier, on the same parcels and the same bootstrap "
                  "draws. **Better / worse** = the 95% CI excludes 0 **and** the Holm-adjusted p < 0.05 "
                  "(Holm corrects for testing all inputs × classifiers at once).",
                  dtable]
        other = diffs[~diffs["classifier"].isin(headline)]
        if not other.empty:
            parts += ["<details markdown=\"1\"><summary>Other classifiers (robustness check)</summary>\n",
                      table_differences(diffs, [c for c in all_clfs if c not in headline])[0], "\n</details>"]
    parts += ["### Per-class metrics (headline classifiers, 95% CI)",
              table_with_ci(metrics, headline, {"macro_f1": "Macro-F1", "f1_rainfed": "F1 rainfed",
                                                "balanced_accuracy": "Balanced acc.",
                                                "overall_accuracy": "Overall acc."})]
    if not per_fold.empty:
        parts += ["### Macro-F1 per test fold",
                  "Each fold is a set of 20 km blocks, used once as the test fold. Rainfed test parcels per fold: "
                  + "; ".join(f"fold {f}: {d}" for f, d in labels.items()) + ".",
                  *[f"**{CLF_NAMES.get(c, c)}**\n\n" + table_per_fold(per_fold, c) for c in fold_clfs],
                  "![Macro-F1 per test fold](results/figures/macro_f1_per_fold.png)"]
    files = sorted(p.relative_to(exp_dir).as_posix() for p in res.glob("*/confusion_matrices.png"))
    info = json.loads((res / "run_info.json").read_text()) if (res / "run_info.json").exists() else {}
    last = info.get("runs", [{}])[-1] if info.get("runs") else {}
    parts += ["### Files\n\n" + "\n".join([
              "- Tables: [`results/comparison/`](results/comparison/) (all metrics with CI, differences vs baseline)",
              "- Per run (`results/<run>/`): out-of-fold predictions, metrics per fold, selected hyperparameters, "
              "confusion matrices, run log",
              *[f"- Confusion matrices: [{Path(f).parent.name}]({f})" for f in files],
              f"- Code version: `{last.get('git_commit', 'see run_config.json of each run')}`"
              + (f", run {last.get('started')} – {last.get('finished')}" if last else "")])]

    report = exp_dir / "REPORT.md"
    text = report.read_text() if report.exists() else (EXP_ROOT / "_template" / "REPORT.md").read_text()
    text = text.replace("{{TITLE}}", cfg["title"]).replace("{{ID}}", cfg["id"])
    report.write_text(replace_block(text, "RESULTS", "\n\n".join(parts)))
    print(f"{exp}: REPORT.md updated ({len(metrics['run'].unique())} runs)")


# ------------------------------------------------------------ overview pages
def headline_of(exp_dir, cfg):
    """One-line result of a finished experiment for the overview pages."""
    comp = exp_dir / "results" / "comparison"
    if not (comp / "metrics_all_runs.csv").exists():
        return "–"
    m = pd.read_csv(comp / "metrics_all_runs.csv")
    m = m[m["metric"] == "macro_f1"]
    best = m.loc[m["estimate"].idxmax()]
    base = m[m["run"].str.split("/").str[-1].str.startswith("s1s2")]
    text = f"best {input_name(best['run'])} + {CLF_NAMES.get(best['classifier'])} {best['estimate']:.3f}"
    if not base.empty:
        b = base.loc[base["estimate"].idxmax()]
        text += f"; best baseline {CLF_NAMES.get(b['classifier'])} {b['estimate']:.3f}"
    if (comp / "differences_vs_baseline.csv").exists():
        d = pd.read_csv(comp / "differences_vs_baseline.csv")
        better = d[(d["p_holm"] < 0.05) & (d["ci_low"] > 0)]
        worse = d[(d["p_holm"] < 0.05) & (d["ci_high"] < 0)]
        text += (f"; {len(better)} of {len(d)} input-classifier pairs better, {len(worse)} worse than the baseline "
                 f"(Holm)")
    return text


def update_overviews():
    rows, results = [], []
    for cfg_file in sorted(EXP_ROOT.glob("E*/config.yaml")):
        cfg = yaml.safe_load(cfg_file.read_text())
        if cfg.get("listed", True) is False:
            continue
        exp_dir = cfg_file.parent
        name = exp_dir.name
        report = f"[report]({name}/REPORT.md)" if (exp_dir / "REPORT.md").exists() else "–"
        result = headline_of(exp_dir, cfg) if cfg.get("status") == "done" else "–"
        rows.append(f"| **{cfg['id']}** | [{cfg['title']}]({name}/README.md) | {cfg.get('research_question', '')} | "
                    f"{STATUS.get(cfg.get('status'), cfg.get('status'))} | {result} | {report} |")
        if cfg.get("status") == "done":
            results.append((name, cfg, result))

    index = EXP_ROOT / "README.md"
    table = ("| ID | Experiment | RQ | Status | Headline result (macro-F1) | Report |\n"
             "| --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows))
    index.write_text(replace_block(index.read_text(), "INDEX", table))

    overview = ROOT / "docs" / "05_results.md"
    if overview.exists():
        blocks = []
        for name, cfg, result in results:
            blocks.append(f"#### {cfg['id']} · {cfg['title']}\n\n{cfg.get('question', '')}\n\n"
                          f"- **Result:** {result}.\n- **Answer:** {cfg.get('answer', 'see the report')}\n"
                          f"- Full report: [experiments/{name}/REPORT.md](../experiments/{name}/REPORT.md)\n\n"
                          f"![{cfg['id']}](../experiments/{name}/results/figures/macro_f1_by_input.png)")
        overview.write_text(replace_block(overview.read_text(), "RESULTS",
                                          "\n\n".join(blocks) or "_No finished experiment yet._"))
    print(f"Updated {index.relative_to(ROOT)} and {overview.relative_to(ROOT)}")


if __name__ == "__main__":
    if EXPERIMENT:
        build_report(EXPERIMENT)
    update_overviews()
