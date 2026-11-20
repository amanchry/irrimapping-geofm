"""
E04_frozen_geofm/run.py  —  E04: frozen geospatial foundation models + L-TAE + UPerNet decoder (RQ1)
=====================================================================================================

Question: do frozen geospatial foundation models with a light decoder classify the pixels better than the pixel
models (E01-E03) and a spatio-temporal network trained from scratch (E03)?

Model (pixelwise_run/tile_common.py, GeoFMSegmenter)
--------------------------------------------------
Encoder (frozen, pretrained weights, bands and normalisation from pixelwise_run/encoders.py):
  prithvi    Prithvi-EO-2.0-300M-TL, B2 B3 B4 B8A B11 B12, 4 months per pass with date and location, 16 px patches
  clay       Clay v1.5, 10 bands, one month per pass with week and location, 8 px patches
  terramind  TerraMind 1.0 base, 12 bands (S2L2A), one month per pass, 16 px patches
  agrifm     AgriFM, 10 bands, all 12 months in one pass (time merged inside the model), region band statistics (D15)
  presto     Presto, every pixel's 12-month series (10 bands + NDVI), 128-d per pixel (pixelwise_run/encoders.py)
  tessera    TESSERA v2 Medium, every pixel's clear months (10 bands, day of year), 128-d per pixel
Temporal head: L-TAE over the 12 months at every patch position (months without a clear pixel in the patch are
masked); AgriFM, Presto and TESSERA have none (time is handled inside the encoder). Decoder: 4-scale feature
pyramid + UPerNet head, at 10 m (pixel models: the 10 m feature map pooled to 4 scales).
Only the temporal head and the decoder are trained.

Training: masked cross-entropy at the sampled pixels of the training folds (class weights), AdamW, random flips /
rotations, early stopping on the validation pixels' macro-F1 (irrigated, rainfed); N_SEEDS seeds per round. Scored on the same
out-of-fold pixels as every other experiment.

Output: experiments/E04_frozen_geofm/results/<region>/<encoder>/ (as E01, plus training_history.csv)

Run (GPU; one encoder per job)
------------------------------
    python experiments/E04_frozen_geofm/run.py
    # cluster: sbatch --export=ALL,SCRIPT=experiments/E04_frozen_geofm/run.py,EXPERIMENT_REGION=morocco,EXPERIMENT_ENCODER=clay jobs/step_gpu.sbatch
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pixelwise_run"))
from pixel_common import env_setting  # noqa: E402
from tile_common import GeoFMSegmenter, run_tile_experiment  # noqa: E402

# ================================ SETTINGS ================================
EXP_DIR = Path(__file__).resolve().parent
REGION = env_setting("REGION", "morocco")          # "morocco" (first) | "utah" (later)
ENCODER = env_setting("ENCODER", "clay")           # "prithvi" | "clay" | "terramind" | "agrifm" | "presto" | "tessera"
LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH = 8                                          # tiles per step (x 12 months through the encoder)
EPOCHS, PATIENCE = 50, 8
N_SEEDS = 3
WORKERS = 8                                        # data-loading processes
QUICK_TEST = env_setting("QUICK_TEST", False)      # 60 tiles, 2 epochs, 1 seed
# ==========================================================================

if QUICK_TEST:
    EPOCHS, PATIENCE, N_SEEDS, WORKERS, BATCH = 2, 1, 1, 0, 2


def build(train_rows, df, X, info):
    return GeoFMSegmenter(ENCODER, REGION, frozen=True)


if __name__ == "__main__":
    run_tile_experiment(EXP_DIR, "E04", REGION, ENCODER, build,
                        {"encoder": ENCODER, "frozen": True, "temporal_head": "ltae" if ENCODER in ("prithvi", "clay", "terramind") else "none",
                         "decoder": "UPerNet"},
                        lr=LR, weight_decay=WEIGHT_DECAY, batch=BATCH, epochs=EPOCHS, patience=PATIENCE,
                        n_seeds=N_SEEDS, workers=WORKERS, quick=QUICK_TEST)
