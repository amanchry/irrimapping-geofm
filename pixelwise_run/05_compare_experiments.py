"""
05_compare_experiments.py  —  one table of all finished pixel-wise runs, and paired tests against a reference run
=============================================================================================================

What it does
------------
Collects every run of E01-E04 with results in experiments/E0*/results/<region>/<run>/ (oof_predictions.parquet;
E05 is compared within itself by experiments/E05_prithvi_strategies/report.py), checks that
all runs of a region were scored on the same pixels, and writes for each region:
  comparison.csv        one row per run: cropland macro-F1 and 3-class macro-F1 with 95% CI
  paired_vs_<ref>.csv   paired difference of every run to each reference run (same pixels, same block-bootstrap
                        draws): difference, 95% CI, two-sided bootstrap p-value and Holm-adjusted p-value; a
                        difference counts when its CI excludes 0 and the Holm-adjusted p < 0.05
The tables go to experiments/comparison/<region>/ (copied to the documentation when the results are reported).

Run (on the Mac, after the results are copied from the cluster)
---------------------------------------------------------------
    python pixelwise_run/05_compare_experiments.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pixel_common import block_confusions, bootstrap_draws, metrics_from_confusion, score  # noqa: E402

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1] / "experiments"   # experiment folders E0*/results/
EXPERIMENTS = "E0[1-4]_*"                          # E05 has its own tables and tests (E05_prithvi_strategies/report.py)
REGIONS = ["morocco", "utah"]
REFERENCES = ["E01_s2_pixel_baseline/rf",          # runs every other run is compared with (one table each)
              "E03_utae/utae"]
METRICS = ["cropland_macro_f1", "macro_f1_3class"]
EXCLUDE = ["E02_temporal_nn/ltae"]                 # runs left out of the table and the Holm family (pixel L-TAE:
                                                   # on hold, under review)
OUT_DIR = ROOT / "comparison"
# ==========================================================================


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def boot_metrics(oof, blocks, draws):
    cm_pix = block_confusions(oof, blocks)
    return pd.DataFrame([metrics_from_confusion(np.tensordot(w, cm_pix, 1)) for w in draws])


def main():
    for region in REGIONS:
        runs = {f"{p.parents[3].name}/{p.parent.name}": p for p in sorted(ROOT.glob(f"{EXPERIMENTS}/results/{region}/*/oof_predictions.parquet"))}
        runs = {k: p for k, p in runs.items() if k not in EXCLUDE}
        if not runs:
            print(f"{region}: no results yet")
            continue
        oofs = {k: pd.read_parquet(p).sort_values(["season", "sample_idx"]).reset_index(drop=True) for k, p in runs.items()}
        keys = {k: tuple(map(tuple, o[["season", "sample_idx"]].to_numpy())) for k, o in oofs.items()}
        base = next(iter(keys.values()))
        for k, v in keys.items():
            if v != base:
                raise SystemExit(f"{region}: {k} was scored on other pixels than {next(iter(keys))}")
        out = OUT_DIR / region
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for k, o in oofs.items():
            m = score(o)[0].set_index("metric")
            rows.append({"run": k, **{f"{x}{s}": m.loc[x, c] for x in METRICS
                                      for s, c in (("", "value"), ("_low", "ci_low"), ("_high", "ci_high"))}})
        table = pd.DataFrame(rows).sort_values("cropland_macro_f1", ascending=False)
        table.to_csv(out / "comparison.csv", index=False)
        print(f"\n== {region}\n" + table[["run"] + METRICS].round(3).to_string(index=False))
        for REFERENCE in REFERENCES:
            if REFERENCE not in oofs:
                print(f"  reference {REFERENCE} not finished: no paired tests")
                continue
            blocks = sorted(oofs[REFERENCE]["block_id"].unique(), key=str)
            draws = bootstrap_draws(len(blocks))
            ref = boot_metrics(oofs[REFERENCE], blocks, draws)
            ref_point = score(oofs[REFERENCE], n_boot=1)[0].set_index("metric")["value"]
            paired = []
            for k, o in oofs.items():
                if k == REFERENCE:
                    continue
                b = boot_metrics(o, blocks, draws)
                point = score(o, n_boot=1)[0].set_index("metric")["value"]
                for x in METRICS:
                    d = b[x] - ref[x]
                    paired.append({"run": k, "metric": x, "difference": point[x] - ref_point[x],
                                   "ci_low": d.quantile(0.025), "ci_high": d.quantile(0.975),
                                   "p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
            paired = pd.DataFrame(paired)
            if len(paired):
                paired["p_holm"] = paired.groupby("metric")["p"].transform(lambda p: holm(p.to_numpy()))
                paired["counts"] = ((paired["ci_low"] > 0) | (paired["ci_high"] < 0)) & (paired["p_holm"] < 0.05)
                name = REFERENCE.replace("/", "_")
                paired.to_csv(out / f"paired_vs_{name}.csv", index=False)
                print(paired[paired["metric"] == "cropland_macro_f1"].round(3).to_string(index=False))
    

if __name__ == "__main__":
    main()
