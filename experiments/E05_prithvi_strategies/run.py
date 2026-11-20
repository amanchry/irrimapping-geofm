"""
E05_prithvi_strategies/run.py  —  E05: Prithvi frozen vs LoRA vs fully fine-tuned vs trained from scratch, with fewer labels
===========================================================================================================================

Questions
---------
1. How do the four training strategies behave (loss and F1 per epoch, overfitting)?
2. What does training the encoder (LoRA adapters or every weight) cost and add over the frozen encoder of E03?
3. Does fine-tuning beat the frozen encoder and the same network trained from scratch, does the parameter-efficient
   LoRA keep the gain of full fine-tuning, and do the answers change with the number of labelled parcels
   (10 / 25 / 50 / 100% of the training parcels)?

Model (pixelwise_run/tile_common.py, GeoFMSegmenter), the same network in every run
----------------------------------------------------------------------------------
Encoder  Prithvi-EO-2.0-300M-TL (ViT-Large, 24 blocks, 1024-d, 16 px patches; B2 B3 B4 B8A B11 B12; 4 months per
         pass with date and location), output of the last block: an 8 x 8 patch grid per month
Head     L-TAE over the 12 months at every patch position -> 4-scale pyramid (x4, x2, x1, x1/2; ViTDet / the
         Feature2Pyramid neck of PANGAEA) -> UPerNet -> class scores on 32 x 32, bilinear to 128 x 128 (as E03)

Strategies (STRATEGY)
---------------------
  frozen     pretrained encoder, locked (= E03); only the head is trained
  lora       pretrained encoder, locked; low-rank adapters (LoRA, Hu et al. 2021) on the attention (qkv) and MLP
             (fc1, fc2) layers of all 24 blocks, rank 16, alpha 32 (as Marti-Escofet et al. 2025 for Prithvi:
             about 5.5 M adapter weights); adapters and head trained, adapters at the head's learning rate
  finetune   pretrained encoder, every weight trained, with a 10x smaller learning rate than the head
  scratch    the same encoder with random initial weights, every weight trained
Training share (FRACTION): 0.1, 0.25, 0.5 or 1.0 of the training-fold parcels (per class, nested draws;
tile_common.label_subset). Validation and test folds are always complete, so every run is scored on the same
out-of-fold pixels as E01-E04.

Training (identical for the four strategies): masked class-weighted cross-entropy, AdamW, weight decay 1e-4, batch 8
tiles, bfloat16, flips / rotations, 1 epoch linear warm-up, gradient clipping at norm 1, early stopping on the
validation 3-class macro-F1 (patience 15, at most 60 epochs), 3 seeds per round.

Output: experiments/E05_prithvi_strategies/results/<region>/<strategy>_p<percent>/ (as E03, plus label_subset.csv
for fractions < 1). Figures and paired tests: experiments/E05_prithvi_strategies/report.py.

Run (GPU; one job per strategy and fraction, 16 jobs)
-----------------------------------------------------
    python experiments/E05_prithvi_strategies/run.py
    # cluster (frozen, finetune, scratch first; lora later):
    for s in frozen finetune scratch lora; do for f in 0.1 0.25 0.5 1.0; do
      sbatch --time=72:00:00 --export=ALL,SCRIPT=experiments/E05_prithvi_strategies/run.py,EXPERIMENT_REGION=utah,EXPERIMENT_STRATEGY=$s,EXPERIMENT_FRACTION=$f jobs/step_gpu.sbatch
    done; done
First a quick test per strategy (EXPERIMENT_QUICK_TEST=1: 60 tiles, 2 epochs, 1 seed) to read the epoch time and
the GPU memory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pixelwise_run"))
from pixel_common import env_setting  # noqa: E402
from tile_common import GeoFMSegmenter, run_tile_experiment  # noqa: E402

# ================================ SETTINGS ================================
EXP_DIR = Path(__file__).resolve().parent
REGION = env_setting("REGION", "utah")             # "utah" | "morocco"
STRATEGY = env_setting("STRATEGY", "frozen")       # "frozen" | "lora" | "finetune" | "scratch"
FRACTION = env_setting("FRACTION", 1.0)            # share of the training parcels: 0.1 | 0.25 | 0.5 | 1.0
ENCODER = "prithvi"
LR = 1e-4                                          # head (L-TAE + pyramid + UPerNet); scratch: also the encoder
ENCODER_LR = {"frozen": None, "lora": 1e-4, "finetune": 1e-5, "scratch": 1e-4}.get(STRATEGY)   # trained encoder weights
LORA_RANK, LORA_ALPHA = 16, 32                     # adapter rank and scaling (alpha / rank = 2)
LORA_TARGETS = ("qkv", "fc1", "fc2")               # attention and MLP linear layers of every Prithvi block
WEIGHT_DECAY = 1e-4
BATCH = 8                                          # tiles per step (x 12 months through the encoder)
EPOCHS, PATIENCE = 60, 15
WARMUP_EPOCHS = 1                                  # linear learning-rate warm-up
GRAD_CLIP = 1.0                                    # gradient norm
N_SEEDS = 3
SUBSET_SEED = 7                                    # parcel draw for FRACTION < 1 (the same for every strategy)
WORKERS = 8                                        # data-loading processes
QUICK_TEST = env_setting("QUICK_TEST", False)      # 60 tiles, 2 epochs, 1 seed
# ==========================================================================

if QUICK_TEST:
    EPOCHS, PATIENCE, N_SEEDS, WORKERS, BATCH = 2, 1, 1, 0, 2


def build(train_rows, df, X, info):
    if STRATEGY == "frozen":
        return GeoFMSegmenter(ENCODER, REGION, frozen=True)
    if STRATEGY == "lora":
        return GeoFMSegmenter(ENCODER, REGION, frozen=True, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
                              lora_targets=LORA_TARGETS)
    return GeoFMSegmenter(ENCODER, REGION, frozen=False, pretrained=STRATEGY == "finetune")


if __name__ == "__main__":
    if STRATEGY not in ("frozen", "lora", "finetune", "scratch"):
        raise SystemExit(f"unknown STRATEGY {STRATEGY}")
    run_name = f"{STRATEGY}_p{round(FRACTION * 100):03d}"
    run_tile_experiment(EXP_DIR, "E05", REGION, run_name, build,
                        {"encoder": ENCODER, "strategy": STRATEGY, "pretrained": STRATEGY != "scratch",
                         "encoder_trained": {"frozen": "no", "lora": "adapters"}.get(STRATEGY, "all"),
                         **({"lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_targets": list(LORA_TARGETS)}
                            if STRATEGY == "lora" else {}),
                         "temporal_head": "ltae", "decoder": "UPerNet"},
                        lr=LR, weight_decay=WEIGHT_DECAY, batch=BATCH, epochs=EPOCHS, patience=PATIENCE,
                        n_seeds=N_SEEDS, workers=WORKERS, quick=QUICK_TEST, label_fraction=FRACTION,
                        subset_seed=SUBSET_SEED, encoder_lr=ENCODER_LR, warmup_epochs=WARMUP_EPOCHS,
                        grad_clip=GRAD_CLIP)
