"""
E03_utae/run.py  —  E03: U-TAE trained from scratch on the 128 px tiles
======================================================================

Question: does a spatio-temporal network of the same input, trained from scratch, match the frozen foundation
models (E04)? It also tests the tile training pipeline before the large encoders.

Model (pixelwise_run/tile_common.py, UTAE; Garnot & Landrieu 2021)
----------------------------------------------------------------
Input: the 12 Sentinel-2 bands of the 12 months, standardised per band with the training pixels of the round;
missing months -> 0 and masked in the temporal attention. Encoder widths 64-64-64-128, decoder 32-32-64-128,
L-TAE at 1/8 resolution with 16 heads (d_k 4, d_model 256), attention masks aggregate the skip features of every
level (the PASTIS configuration).

Training: masked cross-entropy at the sampled pixels of the training folds (class weights), AdamW lr 1e-3,
random flips / rotations, early stopping on the validation pixels' 3-class macro-F1; N_SEEDS seeds per round.

Output: experiments/E03_utae/results/<region>/utae/ (as E01, plus training_history.csv)

Run (GPU)
---------
    python experiments/E03_utae/run.py
    # cluster: sbatch --export=ALL,SCRIPT=experiments/E03_utae/run.py,EXPERIMENT_REGION=morocco jobs/step_gpu.sbatch
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pixelwise_run"))
from pixel_common import env_setting  # noqa: E402
from tile_common import UTAE, run_tile_experiment, s2_stats  # noqa: E402

# ================================ SETTINGS ================================
EXP_DIR = Path(__file__).resolve().parent
REGION = env_setting("REGION", "morocco")
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 4                                          # tiles per step
EPOCHS, PATIENCE = 100, 10
N_SEEDS = 3
WORKERS = 8
QUICK_TEST = env_setting("QUICK_TEST", False)
# ==========================================================================

if QUICK_TEST:
    EPOCHS, PATIENCE, N_SEEDS, WORKERS, BATCH = 2, 1, 1, 0, 2


def build(train_rows, df, X, info):
    mean, std = s2_stats(df, X, info, train_rows)
    return UTAE(12, mean=mean, std=std)


if __name__ == "__main__":
    run_tile_experiment(EXP_DIR, "E03", REGION, "utae", build,
                        {"model": "U-TAE", "encoder_widths": [64, 64, 64, 128], "decoder_widths": [32, 32, 64, 128],
                         "n_head": 16, "d_k": 4, "d_model": 256},
                        lr=LR, weight_decay=WEIGHT_DECAY, batch=BATCH, epochs=EPOCHS, patience=PATIENCE,
                        n_seeds=N_SEEDS, workers=WORKERS, quick=QUICK_TEST)
