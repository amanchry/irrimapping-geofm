"""
E02_temporal_nn/run.py  —  E02: temporal neural networks on the Sentinel-2 pixel time series
===========================================================================================

Question: does a temporal neural network beat the classical baseline (E01) on the same pixels and input?

Models
------
tempcnn   TempCNN (Pelletier et al. 2019): 3 x Conv1d(64, k=5) + BN + ReLU + dropout over the 12 months,
          dense 256, output 3 classes. Input: the gap-filled series (as E01).
ltae      L-TAE (Garnot & Landrieu 2020) on single pixels: 16 heads, d_k 8, d_model 256, MLP 256-128, output 3
          classes. Months without a clear observation are masked in the attention (the gap-filled values are
          not attended).
Input per month: 12 Sentinel-2 bands + NDVI + NDWI (14 channels), standardised per channel with the training
pixels of the round; values still missing after gap filling -> 0 (= the training mean).

Training: AdamW (lr 1e-3, weight decay 1e-4), class-weighted cross-entropy, batch 512, early stopping on the
validation fold's 3-class macro-F1 (patience 10, at most 100 epochs); N_SEEDS seeds per round, the out-of-fold
probability is the seed mean. Same PASTIS rotation and metrics as E01 (pixelwise_run/pixel_common.py).

Output: experiments/E02_temporal_nn/results/<region>/<model>/  (as E01, plus training_history.csv and, for ltae,
attention_by_month.csv: mean attention of each month per class on the test folds)

Run (GPU)
---------
    python experiments/E02_temporal_nn/run.py
    # cluster: sbatch --export=ALL,SCRIPT=experiments/E02_temporal_nn/run.py,EXPERIMENT_REGION=morocco jobs/step_gpu.sbatch
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pixelwise_run"))
from nn_common import DEVICE, LTAEClassifier, TempCNN, predict_proba, train_pixel_model  # noqa: E402
from pixel_common import (CLASSES, add_indices, class_weights, clear_months, env_setting, fill_time,  # noqa: E402
                          get_logger, load_samples, month_positions, results_dir, rounds, s2_series, save_run,
                          standardiser)

# ================================ SETTINGS ================================
EXP_DIR = Path(__file__).resolve().parent
REGION = env_setting("REGION", "morocco")          # "morocco" | "utah"
MODELS = ["tempcnn", "ltae"]
N_SEEDS = 3                                        # seeds SEED, SEED + 1000, SEED + 2000
SEED = 42
LR, WEIGHT_DECAY, BATCH_SIZE = 1e-3, 1e-4, 512
EPOCHS, PATIENCE = 100, 10
LTAE_ARGS = {"n_head": 16, "d_k": 8, "d_model": 256, "mlp": (256, 128), "dropout": 0.2}
QUICK_TEST = env_setting("QUICK_TEST", False)
# ==========================================================================

if QUICK_TEST:
    N_SEEDS, EPOCHS, PATIENCE = 1, 3, 2


def build(name, d_in):
    if name == "tempcnn":
        return TempCNN(d_in)
    return LTAEClassifier(d_in, month_positions(), **LTAE_ARGS)


def main():
    df, X, info = load_samples(REGION, quick=QUICK_TEST)
    raw = s2_series(X, info)
    observed = clear_months(raw)                                        # [N, 12] for the L-TAE mask
    series = add_indices(fill_time(raw))                                # [N, 12, 14]
    y, folds = df["label"].to_numpy(), df["fold"].to_numpy()
    for name in MODELS:
        out_dir = results_dir(EXP_DIR, REGION, name, QUICK_TEST)
        log = get_logger(out_dir)
        log.info(f"E02 {REGION} {name}: {len(df):,} pixels, device {DEVICE}")
        proba = np.zeros((len(df), 3), dtype=np.float32)
        history, att_rows = [], []
        for k, train_f, val_f, test_f in rounds():
            tr, va, te = (np.flatnonzero(np.isin(folds, train_f)), np.flatnonzero(folds == val_f),
                          np.flatnonzero(folds == test_f))
            mean, std = standardiser(series[tr])
            x = np.nan_to_num((series - mean) / std, nan=0.0).astype(np.float32)
            mask = observed.copy()
            mask[~mask.any(1)] = True                                     # never observed: attend to all months
            w = class_weights(y[tr])
            probs = []
            for s in range(N_SEEDS):
                seed = SEED + 1000 * s
                log.info(f"  round {k + 1} seed {seed}: train {len(tr):,} | val {len(va):,} | test {len(te):,}")
                model, hist = train_pixel_model(build(name, x.shape[2]), x, y, tr, va, w, seed, mask=mask, lr=LR,
                                                weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                                patience=PATIENCE, log=log)
                history += [dict(h, round=k + 1, seed=seed) for h in hist]
                probs.append(predict_proba(model, x[te], mask[te]))
                if name == "ltae":
                    att = []
                    for s0 in range(0, len(te), 4096):
                        predict_proba(model, x[te[s0:s0 + 4096]], mask[te[s0:s0 + 4096]])
                        att.append(model.last_attention.mean(1).cpu().numpy())
                    att = np.concatenate(att)
                    for c in CLASSES:
                        sel = y[te] == c
                        if sel.any():
                            att_rows.append({"round": k + 1, "seed": seed, "class": c,
                                             **{f"m{m + 1:02d}": float(v) for m, v in enumerate(att[sel].mean(0))}})
            proba[te] = np.mean(probs, axis=0)
        tables = {"training_history.csv": pd.DataFrame(history)}
        if att_rows:
            tables["attention_by_month.csv"] = pd.DataFrame(att_rows)
        settings = {"experiment": "E02", "region": REGION, "model": name, "n_seeds": N_SEEDS, "lr": LR,
                    "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "patience": PATIENCE,
                    "ltae": LTAE_ARGS if name == "ltae" else None, "input": "12 months x (12 S2 bands + NDVI + NDWI)"}
        save_run(out_dir, df, proba, settings, log, tables, QUICK_TEST)


if __name__ == "__main__":
    main()
