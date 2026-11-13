"""
run_experiment.py  —  run one experiment from its config file, step by step, for every region
==============================================================================================

What it does
------------
Reads experiments/<EXPERIMENT>/config.yaml and runs its steps in order, for every region in "regions"
(Morocco, Utah). Every step is an ordinary script of models_run/; this file only starts it with environment
variables (experiment, region, input, training fraction, draw, seeds), so the script takes its settings from the
config file and writes into experiments/<EXPERIMENT>/results/<region>/.

  step             script                              runs (per region)
  train_evaluate   models_run/03_train_evaluate.py     once per input in "inputs"
  ltae             models_run/03c_train_ltae.py        once per input in "inputs" (section "ltae": HEAD mlp / ltae)
  psetae           models_run/03b_train_psetae.py      once per file in "psetae_configs" whose REGION is the region
  compare          models_run/04_compare_models.py     once per region: all runs on the same parcels, learning curves
  breakdown        models_run/05_breakdown_oof.py      once per region (optional step)
  report           experiments/make_report.py          once: tables + figures into docs/experiments/<name>/report.md

Label efficiency: if the config has a "label_efficiency" block, every training run is repeated with each fraction
and draw (EXPERIMENT_FRACTION, EXPERIMENT_DRAW; neural runs with "seeds_per_draw" seeds), after the main runs.
plan_runs() lists every training run; jobs/submit_experiment.py uses it to send the same runs to Slurm.

Runs that already have results are skipped (RERUN = False), so an interrupted experiment continues where it
stopped. The config used, the code version (git commit) and the start / end times are saved in
results/run_info.json.

Run (from the project folder)
-----------------------------
    python experiments/run_experiment.py
QUICK = True runs every script in its QUICK_TEST mode (a fast check of the whole experiment); its results go to
experiments/<EXPERIMENT>/results_quick/ (never mixed with the real results) and no report is written.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models_run"))
from experiment_utils import clean_run_env, results_root  # noqa: E402

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]               # project folder
EXPERIMENT = "E02_frozen_pca"                      # folder name in experiments/
ONLY_STEPS = None                                         # None = all steps of the config | e.g. ["compare", "report"]
REGIONS = None                                            # None = "regions" of the config | e.g. ["utah"]
LABEL_EFFICIENCY = True                                   # False = only the main runs (no 10 / 25 / 50% runs)
RERUN = False                                             # True = run again even if a run already has results
QUICK = False                                             # True = QUICK_TEST mode in every script (fast check)
# ==========================================================================

SCRIPTS = {                                               # step -> (script, once per input?)
    "train_evaluate": (ROOT / "models_run" / "03_train_evaluate.py", True),
    "ltae": (ROOT / "models_run" / "03c_train_ltae.py", True),
    "psetae": (ROOT / "models_run" / "03b_train_psetae.py", False),
    "compare": (ROOT / "models_run" / "04_compare_models.py", False),
    "breakdown": (ROOT / "models_run" / "05_breakdown_oof.py", False),
    "report": (ROOT / "experiments" / "make_report.py", False),
}


def git_commit():
    """Code version the experiment ran with (empty if git is not available)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "models_run", "experiments"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return commit + (" (with uncommitted changes)" if dirty else "")
    except OSError:
        return ""


def psetae_configs_for(cfg, exp_dir, region):
    """PSE-TAE config files of the experiment whose REGION is the region."""
    out = []
    for name in cfg.get("psetae_configs") or []:
        c = yaml.safe_load((exp_dir / "configs" / name).read_text(encoding="utf-8")) or {}
        if c.get("REGION", "morocco") == region:
            out.append(name)
    return out


def plan_runs(cfg, exp_dir, steps=None, regions=None, label_efficiency=True):
    """Every training run of the experiment: dicts with step, region, item (input or PSE-TAE config file),
    fraction, draw, seeds (None = the script's N_SEEDS) and env (the environment variables of the run)."""
    steps = [s for s in cfg["steps"] if steps is None or s in steps]
    regions = regions or cfg.get("regions") or ["morocco"]
    le = cfg.get("label_efficiency") if label_efficiency else None
    fractions = [(1.0, 1, None)]
    if le:
        fractions += [(f, d, le.get("seeds_per_draw")) for f in le["fractions"] for d in range(1, le["draws"] + 1)]
    runs = []
    for frac, draw, seeds in fractions:                          # main runs first, then the label-efficiency runs
        for region in regions:
            for step in steps:
                if step not in ("train_evaluate", "ltae", "psetae"):
                    continue
                items = psetae_configs_for(cfg, exp_dir, region) if step == "psetae" else cfg["inputs"]
                for item in items:
                    env = {"EXPERIMENT_REGION": region, "EXPERIMENT_FRACTION": str(frac), "EXPERIMENT_DRAW": str(draw)}
                    env["PSETAE_CONFIG" if step == "psetae" else "EXPERIMENT_INPUT"] = item
                    if seeds and step in ("ltae", "psetae"):
                        env["EXPERIMENT_N_SEEDS"] = str(seeds)
                    runs.append({"step": step, "region": region, "item": item, "fraction": frac, "draw": draw,
                                 "seeds": seeds, "env": env})
    return runs


def run_script(step, env, label):
    script, _ = SCRIPTS[step]
    if not script.exists():
        raise SystemExit(f"Step '{step}' needs {script.relative_to(ROOT)}, which does not exist yet.")
    print(f"\n=== {label} {'=' * max(3, 70 - len(label))}", flush=True)
    t0 = time.time()
    done = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env)
    if done.returncode != 0:
        raise SystemExit(f"Step '{label}' failed (exit code {done.returncode}); fix it and run again "
                         f"(finished runs are kept).")
    print(f"--- {label}: {(time.time() - t0) / 60:.1f} min")


def main():
    exp_dir = ROOT / "experiments" / EXPERIMENT
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8"))
    steps = [s for s in cfg["steps"] if ONLY_STEPS is None or s in ONLY_STEPS]
    regions = REGIONS or cfg.get("regions") or ["morocco"]
    results = results_root(EXPERIMENT, QUICK)
    results.mkdir(exist_ok=True)

    info_file = results / "run_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {"runs": []}
    run = {"started": time.strftime("%Y-%m-%d %H:%M"), "steps": steps, "regions": regions, "git_commit": git_commit(),
           "python": platform.python_version(), "machine": platform.node(), "quick": QUICK, "config": cfg}
    runs = plan_runs(cfg, exp_dir, steps, regions, LABEL_EFFICIENCY)
    print(f"Experiment {EXPERIMENT}: {cfg['title']}\nSteps: {', '.join(steps)} | regions: {regions} | "
          f"inputs: {cfg.get('inputs')} | training runs: {len(runs)}")

    env = dict(clean_run_env(os.environ), EXPERIMENT=EXPERIMENT)   # no stray EXPERIMENT_* value from the shell
    if not RERUN:
        env["EXPERIMENT_SKIP_DONE"] = "1"
    if QUICK:
        env["EXPERIMENT_QUICK_TEST"] = "1"
    for r in runs:
        tag = "" if r["fraction"] >= 1 else f" · {r['fraction']:.0%} draw {r['draw']}"
        run_script(r["step"], dict(env, **r["env"]), f"{r['step']} · {r['region']} · {r['item']}{tag}")
    for step in [s for s in steps if s in ("compare", "breakdown")]:
        for region in regions:
            run_script(step, dict(env, EXPERIMENT_REGION=region), f"{step} · {region}")
    if "report" in steps and not QUICK:                                # quick results never go into the docs
        run_script("report", env, "report")

    run["finished"] = time.strftime("%Y-%m-%d %H:%M")
    info["runs"].append(run)
    info_file.write_text(json.dumps(info, indent=2, default=str))
    print(f"\nDone. Results in {results.relative_to(ROOT)}/<region>/; report: docs/experiments/{EXPERIMENT}/report.md")


if __name__ == "__main__":
    main()
