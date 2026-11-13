"""
submit_slurm.py  —  send every run of an experiment to the cluster as Slurm jobs (run on the server)
===================================================================================================

What it does
------------
Lists the training runs of an experiment exactly as experiments/run_experiment.py does (plan_runs(): every region,
input or PSE-TAE config, and the label-efficiency fractions and draws) and submits one Slurm job per run:
  train_evaluate (03)        -> jobs/step_cpu.sbatch
  ltae (03c), psetae (03b)   -> jobs/step_gpu.sbatch
Then one CPU job per region that waits for all of them (--dependency=afterok) and runs the compare step
(04_compare_models.py) and, if the experiment has it, the breakdown step (05_breakdown_oof.py). The report is
written afterwards on the Mac (copy results/ back, then run_experiment.py with ONLY_STEPS = ["report"]).

Runs that already have results are skipped by the scripts themselves (EXPERIMENT_SKIP_DONE), so submitting
again after a failure only repeats the missing runs. DRY_RUN = True prints the sbatch commands without sending them.
QUICK = True writes to experiments/<EXPERIMENT>/results_quick/, never to results/.

Experiments compared with earlier ones (EXTRA_RUNS in the "compare" section, e.g. E03 with "E02_frozen_pca/*"):
the compare job of a region is only submitted if every main run of those experiments is finished in that region,
or if WAIT_FOR lists the job ids it must wait for (e.g. the E02 compare jobs). Otherwise submit the compare step
later (ONLY_STEPS = ["compare"]). A compare job also stays pending if one of its runs failed (afterok): fix the run,
submit again (only the missing runs are repeated) and cancel the old compare job (scancel).

Run (on the server, from the project folder)
--------------------------------------------
    python experiments/submit_slurm.py
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models_run"))
from experiment_utils import results_root  # noqa: E402

# Every variable a run may read; each job gets all of them (empty = not set), so nothing leaks from the shell.
RUN_VARS = ["EXPERIMENT_REGION", "EXPERIMENT_INPUT", "PSETAE_CONFIG", "EXPERIMENT_FRACTION", "EXPERIMENT_DRAW",
            "EXPERIMENT_N_SEEDS", "EXPERIMENT_VIEWS", "EXPERIMENT_QUICK_TEST", "EXPERIMENT_SKIP_DONE"]

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "E02_frozen_pca"          # folder in experiments/
REGIONS = None                         # None = "regions" of the config | e.g. ["utah"]
LABEL_EFFICIENCY = True                # False = only the main runs
ONLY_STEPS = None                      # None = all training steps of the config | e.g. ["ltae"]
QUICK = False                          # True = QUICK_TEST mode in every script (check the whole chain first)
WAIT_FOR = []                          # extra job ids the compare jobs wait for (e.g. the E02 compare jobs)
DRY_RUN = True                         # True = only print the commands
# ==========================================================================

SCRIPTS = {"train_evaluate": ("models_run/03_train_evaluate.py", "jobs/step_cpu.sbatch"),
           "ltae": ("models_run/03c_train_ltae.py", "jobs/step_gpu.sbatch"),
           "psetae": ("models_run/03b_train_psetae.py", "jobs/step_gpu.sbatch"),
           "compare": ("models_run/04_compare_models.py", "jobs/step_cpu.sbatch"),
           "breakdown": ("models_run/05_breakdown_oof.py", "jobs/step_cpu.sbatch")}


def load_plan_runs():
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "experiments" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan_runs


def sbatch(script, job, env, name, dependency=None):
    values = {k: "" for k in RUN_VARS}
    values.update({"EXPERIMENT_SKIP_DONE": "1", "EXPERIMENT_QUICK_TEST": "1" if QUICK else ""}, **env)
    exports = ",".join(["ALL", f"SCRIPT={script}", f"EXPERIMENT={EXPERIMENT}"] + [f"{k}={v}" for k, v in values.items()])
    cmd = ["sbatch", "--parsable", f"--job-name={name}", f"--export={exports}"]
    if dependency:
        cmd.append(f"--dependency=afterok:{':'.join(dependency)}")
    cmd.append(job)
    if DRY_RUN:
        print(" ".join(cmd))
        return f"<{name}>"
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]


def borrowed_runs_complete(cfg, region, plan_runs):
    """True if every main run of the experiments in EXTRA_RUNS ("<experiment>/*" or "<experiment>/<run>") is
    finished in this region."""
    missing = []
    for item in (cfg.get("compare") or {}).get("EXTRA_RUNS") or []:
        exp, name = item.split("/", 1)
        other_dir = ROOT / "experiments" / exp
        other = yaml.safe_load((other_dir / "config.yaml").read_text(encoding="utf-8"))
        res = results_root(exp, QUICK) / region
        if name == "*":
            expected = [r for r in plan_runs(other, other_dir, None, [region], False)]
            done = len(list(res.glob("*/oof_predictions.csv")))
            if done < len(expected):
                missing.append(f"{exp}: {done} of {len(expected)} main runs")
        elif not (res / name / "oof_predictions.csv").exists():
            missing.append(f"{exp}/{name}")
    if missing:
        print(f"  {region}: borrowed runs not finished yet ({'; '.join(missing)})")
    return not missing


def main():
    exp_dir = ROOT / "experiments" / EXPERIMENT
    cfg = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8"))
    (ROOT / "jobs" / "logs").mkdir(parents=True, exist_ok=True)
    runs = load_plan_runs()(cfg, exp_dir, ONLY_STEPS, REGIONS, LABEL_EFFICIENCY)
    ids = {}
    if ONLY_STEPS and not runs:                                  # e.g. ONLY_STEPS = ["compare"]: compare jobs only
        ids = {r: [] for r in (REGIONS or cfg.get("regions") or ["morocco"])}
    for r in runs:
        script, job = SCRIPTS[r["step"]]
        frac = "" if r["fraction"] >= 1 else f"_f{round(r['fraction'] * 100):03d}d{r['draw']}"
        name = f"{EXPERIMENT[:3]}_{r['region'][:2]}_{r['item'].replace('.yaml', '')}{frac}"
        ids.setdefault(r["region"], []).append(sbatch(script, job, r["env"], name))
    plan_runs = load_plan_runs()
    for region, deps in ids.items():
        if not WAIT_FOR and not borrowed_runs_complete(cfg, region, plan_runs):
            print(f"  {region}: compare job NOT submitted; submit it later (ONLY_STEPS = ['compare']) or set WAIT_FOR")
            continue
        for step in [s for s in cfg["steps"] if s in ("compare", "breakdown") and (ONLY_STEPS is None or s in ONLY_STEPS)]:
            script, job = SCRIPTS[step]
            sbatch(script, job, {"EXPERIMENT_REGION": region}, f"{EXPERIMENT[:3]}_{region[:2]}_{step}",
                   (deps + [str(j) for j in WAIT_FOR]) or None)
    print(f"{EXPERIMENT}: {len(runs)} training jobs" + (" (dry run, nothing sent)" if DRY_RUN else " submitted")
          + f"; compare per region after all of them: {list(ids)}")


if __name__ == "__main__":
    main()
