"""
experiment_utils.py  —  shared helpers of the training scripts 03, 03b and 03c (not run on its own)
====================================================================================================

What it holds
-------------
1. Region and paths: the folds, views and embeddings of a region (Morocco or Utah), so every training script
   reads the same files.
2. Experiment-mode overrides: run_experiment.py (and the Slurm jobs) pass the region, input, training fraction,
   draw and number of seeds as environment variables; the scripts read them with env_override().
3. Label efficiency: draw_training_blocks() picks the training blocks of one round for a training fraction
   (10 / 25 / 50%). The draw depends only on the region, the round, the fraction and the draw number, never on
   the input or the script, so draw k uses the same blocks in E02, E03, E04 and E05 (paired comparisons).

Block draw (label efficiency)
-----------------------------
Only parcels of the agricultural years of the views folder count (draw_table()). Draws are NESTED: draw k orders
the rainfed blocks and the irrigated-only blocks once (seed from region, round and draw), and a fraction takes the
first blocks of each list, so the 10% blocks are part of the 25% blocks, which are part of the 50% blocks. The
ordering is accepted only if its 10% blocks already hold MIN_RAINFED rainfed parcels (otherwise the next sub-seed),
so every fraction keeps enough rainfed parcels and a learning curve shows the effect of adding blocks, not of
drawing different ones.
The training blocks of the round are split into blocks with at least one rainfed parcel and irrigated-only
blocks, and the same fraction is taken from each group (at least one block from each), so every subset keeps
rainfed parcels. In Morocco most rainfed parcels lie in a few Doukkala blocks. The blocks come from the fold file
(all parcels of the region), not from one input's table, so the parcels left out because an input has no data for
them do not change the draw.
"""
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml_dataset_prep"))
from regions import settings as region_settings  # noqa: E402  (ml_dataset_prep/regions.py)

DEFAULT_VIEWS = {"morocco": "views", "utah": "views_2023-24"}      # views folder of each study area
MIN_RAINFED = 20                                                     # rainfed training parcels a draw must keep
MIN_FRACTION = 0.10                                                  # smallest fraction; it must keep MIN_RAINFED


# ---------------------------------------------------------------- region and paths
def views_name(region, views=None):
    """The views folder of a region: the one given, or the region's default."""
    return views or DEFAULT_VIEWS[region]


def views_suffix(views):
    """'views' -> '' ; 'views_2023-24' -> '_2023-24' (the suffix of the embedding file names)."""
    return views[len("views"):]


def region_paths(region, window, views):
    """Files of a region used by the training scripts."""
    region_settings(region)                                          # stops on an unknown region
    dataset = ROOT / "outputs" / region / "dataset"
    views_dir = dataset / "03_tiles_128px" / window / views
    return {
        "split_file": dataset / "02_folds" / "parcels_folds.geojson",
        "views_dir": views_dir,
        "embeddings_dir": ROOT / "outputs" / region / "study" / "02_embeddings",
        "study_dir": ROOT / "outputs" / region / "study",
        "embedding_tag": f"{window}_tiles{views_suffix(views)}",
    }


def start_month(region):
    return region_settings(region)["start_month"]


# ---------------------------------------------------------------- experiment-mode overrides
ENV = {                                                              # setting -> environment variable
    "REGION": "EXPERIMENT_REGION",
    "VIEWS_NAME": "EXPERIMENT_VIEWS",
    "TRAIN_FRACTION": "EXPERIMENT_FRACTION",
    "FRACTION_DRAW": "EXPERIMENT_DRAW",
    "N_SEEDS": "EXPERIMENT_N_SEEDS",
    "QUICK_TEST": "EXPERIMENT_QUICK_TEST",                          # "1" = fast check of a whole experiment
}


def env_override(namespace):
    """Replace REGION, VIEWS_NAME, TRAIN_FRACTION, FRACTION_DRAW and N_SEEDS in a script's settings by the
    environment variables set by run_experiment.py or a Slurm job (only those the script has)."""
    for name, var in ENV.items():
        if name in namespace and os.environ.get(var):
            value = os.environ[var]
            namespace[name] = (float(value) if name == "TRAIN_FRACTION" else
                               int(value) if name in ("FRACTION_DRAW", "N_SEEDS") else
                               value.lower() in ("1", "true", "yes") if name == "QUICK_TEST" else value)


def fraction_tag(fraction, draw):
    """Run-name suffix of a label-efficiency run: '' for all training blocks, '_f010_d1' for 10%, draw 1."""
    if fraction >= 1.0:
        return ""
    return f"_f{round(fraction * 100):03d}_d{draw}"


def results_root(experiment, quick=False):
    """experiments/<experiment>/results/, or results_quick/ for QUICK_TEST runs, so a quick check of the chain
    never leaves files that a real run would take as "already done"."""
    return ROOT / "experiments" / experiment / ("results_quick" if quick else "results")


def results_dir(experiment, region, run_name, fallback, quick=False):
    """experiments/<experiment>/results[_quick]/<region>/<run>/ in experiment mode, else <fallback>/<run>/."""
    if experiment:
        return results_root(experiment, quick) / region / run_name
    return Path(fallback) / (run_name + ("_quick" if quick else ""))


def quick_from_env():
    return os.environ.get("EXPERIMENT_QUICK_TEST", "").lower() in ("1", "true", "yes")


def clean_run_env(env):
    """A copy of env without any EXPERIMENT_* / PSETAE_CONFIG variable, so a value left in the shell can never
    reach a run that did not set it (run_experiment.py and submit_slurm.py set every variable a run needs)."""
    return {k: v for k, v in env.items() if not (k.startswith("EXPERIMENT") or k == "PSETAE_CONFIG")}


# ---------------------------------------------------------------- label efficiency
def draw_table(split_file, views_dir):
    """Parcels the block draws are made from: the fold file (all parcels of the region) restricted to the
    agricultural years of the views folder (views_info.json "seasons"; Utah's fold file holds 2024 AND 2021).
    Columns parcel_id, block_id, fold, label. The same for every input and script that uses these views."""
    import pyogrio
    table = pyogrio.read_dataframe(split_file, columns=["parcel_id", "ag_season", "block_id", "fold", "label"],
                                   read_geometry=False)
    info_file = Path(views_dir) / "views_info.json"
    seasons = json.loads(info_file.read_text()).get("seasons") if info_file.exists() else None
    if seasons:
        table = table[table["ag_season"].isin(seasons)]
    table["parcel_id"] = table["parcel_id"].astype(str)
    return table[["parcel_id", "block_id", "fold", "label"]].reset_index(drop=True)


def draw_training_blocks(split, train_folds, fraction, draw, region, round_id, min_rainfed=MIN_RAINFED):
    """Blocks kept for training in one round. split: fold table of ALL parcels of the region with columns
    block_id, fold, label (1 irrigated, 2 rainfed). Returns the set of block ids (all blocks if fraction >= 1).
    Nested: the blocks of a smaller fraction are always part of a larger fraction's blocks (same draw)."""
    train = split[split["fold"].isin(train_folds)]
    blocks = train.groupby("block_id")["label"].agg(lambda s: (s == 2).sum())     # rainfed parcels per block
    if fraction >= 1.0:
        return set(blocks.index)
    if fraction < MIN_FRACTION:
        raise SystemExit(f"TRAIN_FRACTION {fraction} is below MIN_FRACTION {MIN_FRACTION}")
    rainfed_blocks = np.array(sorted(blocks.index[blocks > 0], key=str))
    irrigated_blocks = np.array(sorted(blocks.index[blocks == 0], key=str))
    n_take = lambda f, n: max(1, round(f * n)) if n else 0
    base = zlib.crc32(f"{region}|{int(round_id)}|{int(draw)}".encode())          # no fraction: nested draws
    for attempt in range(500):
        rng = np.random.default_rng(base + attempt)
        r_order, i_order = rng.permutation(rainfed_blocks), rng.permutation(irrigated_blocks)
        smallest = set(r_order[:n_take(MIN_FRACTION, len(r_order))]) | set(i_order[:n_take(MIN_FRACTION, len(i_order))])
        if blocks.loc[list(smallest)].sum() >= min_rainfed:
            return (set(r_order[:n_take(fraction, len(r_order))]) |
                    set(i_order[:n_take(fraction, len(i_order))]))
    raise SystemExit(f"No ordering of the training blocks keeps {min_rainfed} rainfed parcels in its first "
                     f"{MIN_FRACTION:.0%} ({region}, round {round_id})")


def describe_draw(split, chosen, train_folds):
    """Numbers of a draw for the log and run_config.json."""
    train = split[split["fold"].isin(train_folds)]
    kept = train[train["block_id"].isin(chosen)]
    return {"blocks": int(len(chosen)), "blocks_available": int(train["block_id"].nunique()),
            "parcels": int(len(kept)), "rainfed": int((kept["label"] == 2).sum())}


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))
