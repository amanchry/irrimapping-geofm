"""
E05_lora/run.py  —  E05: fine-tuning the best E04 encoder with LoRA (RQ2)
========================================================================

Question: does adapting the encoder (low-rank adapters, LoRA; Hu et al. 2022) improve on the frozen encoder of
E04 with the same decoder, data and training?

Model: exactly the E04 model of ENCODER (pixelwise_run/tile_common.py, GeoFMSegmenter), with LoRA adapters of rank
LORA_RANK on the attention projections (qkv / to_qkv) of every encoder block; TESSERA: on the feed-forward
layers (linear1 / linear2), because its nn.MultiheadAttention packs q, k, v in one weight. The pretrained weights stay frozen;
the adapters, the temporal head and the decoder are trained. Set ENCODER to the best encoder of E04.

Output: experiments/E05_lora/results/<region>/<encoder>_lora<rank>/ (as E04)

Run (GPU)
---------
    python experiments/E05_lora/run.py
    # cluster: sbatch --export=ALL,SCRIPT=experiments/E05_lora/run.py,EXPERIMENT_REGION=morocco,EXPERIMENT_ENCODER=clay jobs/step_gpu.sbatch
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pixelwise_run"))
from pixel_common import env_setting  # noqa: E402
from tile_common import GeoFMSegmenter, run_tile_experiment  # noqa: E402

# ================================ SETTINGS ================================
EXP_DIR = Path(__file__).resolve().parent
REGION = env_setting("REGION", "morocco")
ENCODER = env_setting("ENCODER", "clay")           # the best encoder of E04 (any of the six)
LORA_RANK, LORA_ALPHA = 8, 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH = 4                                          # smaller than E04: gradients flow through the encoder
EPOCHS, PATIENCE = 50, 8
N_SEEDS = 3
WORKERS = 8
QUICK_TEST = env_setting("QUICK_TEST", False)
# ==========================================================================

if QUICK_TEST:
    EPOCHS, PATIENCE, N_SEEDS, WORKERS, BATCH = 2, 1, 1, 0, 2


def build(train_rows, df, X, info):
    return GeoFMSegmenter(ENCODER, REGION, frozen=True, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)


if __name__ == "__main__":
    run_tile_experiment(EXP_DIR, "E05", REGION, f"{ENCODER}_lora{LORA_RANK}", build,
                        {"encoder": ENCODER, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
                         "lora_targets": "feed-forward linear1/linear2" if ENCODER == "tessera" else "attention qkv", "decoder": "UPerNet"},
                        lr=LR, weight_decay=WEIGHT_DECAY, batch=BATCH, epochs=EPOCHS, patience=PATIENCE,
                        n_seeds=N_SEEDS, workers=WORKERS, quick=QUICK_TEST)
