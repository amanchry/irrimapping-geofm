"""
run_experiment.py  —  run one experiment from its config file, step by step
===========================================================================

What it does
------------
Reads experiments/<EXPERIMENT>/config.yaml and runs its steps in order. Every step is an
ordinary script of models_run/; this file only starts it with the experiment's name, so the script
takes its settings from the config file and writes into experiments/<EXPERIMENT>/results/.

  step             script                              runs
  train_evaluate   models_run/03_train_evaluate.py     once per input in "inputs"
  ltae             models_run/03c_train_ltae.py        once per input in "inputs"   (not written yet)
  psetae           models_run/03b_train_psetae.py      once, or once per file in "psetae_configs" (configs/<file>)
  compare          models_run/04_compare_models.py     once: all runs of the experiment on the same parcels
  report           experiments/make_report.py          once: tables + figures into docs/experiments/<name>/report.md

Runs that already have results are skipped (RERUN = False), so an interrupted experiment can be
started again and continues where it stopped. The config used, the code version (git commit) and the
start / end times are saved in results/run_info.json.

Run (from the project folder)
-----------------------------
    python experiments/run_experiment.py
To repeat an experiment later: run it again with RERUN = True, or delete its results/ folder.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]               # project folder
EXPERIMENT = "E02_frozen_pca_pastis"                      # folder name in experiments/
ONLY_STEPS = None                                         # None = all steps of the config | e.g. ["compare", "report"]
RERUN = False                                             # True = run again even if a run already has results
# ==========================================================================

SCRIPTS = {                                               # step -> (script, once per input?)
    "train_evaluate": (ROOT / "models_run" / "03_train_evaluate.py", True),
    "ltae": (ROOT / "models_run" / "03c_train_ltae.py", True),
    "psetae": (ROOT / "models_run" / "03b_train_psetae.py", False),
    "compare": (ROOT / "models_run" / "04_compare_models.py", False),
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


def main():
    exp_dir = ROOT / "experiments" / EXPERIMENT
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
    steps = [s for s in cfg["steps"] if ONLY_STEPS is None or s in ONLY_STEPS]
    results = exp_dir / "results"
    results.mkdir(exist_ok=True)

    info_file = results / "run_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {"runs": []}
    run = {"started": time.strftime("%Y-%m-%d %H:%M"), "steps": steps, "git_commit": git_commit(),
           "python": platform.python_version(), "machine": platform.node(), "config": cfg}
    print(f"Experiment {EXPERIMENT}: {cfg['title']}\nSteps: {', '.join(steps)} | inputs: {cfg.get('inputs')}")

    env = dict(os.environ, EXPERIMENT=EXPERIMENT)
    if not RERUN:
        env["EXPERIMENT_SKIP_DONE"] = "1"
    for step in steps:
        script, per_input = SCRIPTS[step]
        if not script.exists():
            raise SystemExit(f"Step '{step}' needs {script.relative_to(ROOT)}, which does not exist yet.")
        configs = cfg.get("psetae_configs") if step == "psetae" else None     # one run per config file
        for inp in (cfg["inputs"] if per_input else configs or [None]):
            label = f"{step}" + (f" · {inp}" if inp else "")
            print(f"\n=== {label} {'=' * max(3, 70 - len(label))}", flush=True)
            t0 = time.time()
            var = "PSETAE_CONFIG" if configs else "EXPERIMENT_INPUT"
            step_env = dict(env, **{var: inp}) if inp else env
            done = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=step_env)
            if done.returncode != 0:
                raise SystemExit(f"Step '{label}' failed (exit code {done.returncode}); fix it and run again "
                                 f"(finished runs are kept).")
            print(f"--- {label}: {(time.time() - t0) / 60:.1f} min")

    run["finished"] = time.strftime("%Y-%m-%d %H:%M")
    info["runs"].append(run)
    info_file.write_text(json.dumps(info, indent=2, default=str))
    print(f"\nDone. Results in {results.relative_to(ROOT)}; report: docs/experiments/{EXPERIMENT}/report.md")


if __name__ == "__main__":
    main()
