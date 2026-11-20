# Prithvi fine-tuning for smallholder crops · Tomotaki-Dawoud et al. (2025)

!!! quote "Citation"
    Tomotaki-Dawoud, K., Chaliganti, R., Chauhan, S. C., Wolffram, P., Türkes, O., & Bosse, S. (2025). Fine-tuning
    the Prithvi foundation model for crop-type and health mapping in smallholder farms. *Procedia Computer Science*
    270, 5055–5064 (KES 2025). [doi:10.1016/j.procs.2025.09.632](https://doi.org/10.1016/j.procs.2025.09.632).
    Code and data: [gitlab.hhi.fraunhofer.de/dawoud/finetune-prithvi-for-smallholders](https://gitlab.hhi.fraunhofer.de/dawoud/finetune-prithvi-for-smallholders.git)

**In one line:** on a small crop-type dataset, fully fine-tuning Prithvi-EO-2.0 beat the same network with a frozen
encoder and the same network trained from scratch, with the largest gain over training from scratch when few labels
were used.

## What they did

- **Data and region:** smallholder farms in Telangana, India (Telangana Crop Health Challenge data, ADeX platform).
  Of 10,608 farm records, 601 parcels remained after quality filtering (February 2023 – August 2024).
- **Tasks:** two segmentation tasks on chips around one parcel: crop type (seven crops: paddy, cotton, Bengal gram,
  groundnut, chillies, red gram, maize) and crop health (healthy, stressed, diseased, pests). Pixels in a buffer
  around the parcel are labelled "background" and ignored in the loss.
- **Input:** Harmonized Landsat Sentinel-2 (HLS, 30 m), the six Prithvi bands (blue, green, red, NIR, SWIR1, SWIR2),
  three dates per parcel (early, mid and late season between sowing and harvest; ≥ 97% valid pixels, ≤ 10% cloud).
  Each chip is resampled with nearest neighbour to 446 × 446 px at 0.5 m, so that a small field covers enough
  pixels.
- **Model:** Prithvi-EO-2.0-300M as encoder with a segmentation decoder, three strategies:
    1. **frozen encoder:** only decoder and head trained;
    2. **full fine-tuning:** all encoder and decoder weights trained;
    3. **from scratch:** the same architecture trained from random weights (reference).
- **Training:** AdamW, learning rate 10⁻⁴, 100 epochs; random flips and affine shifts; cross-entropy, focal loss and
  Dice loss compared; class-balanced sampling of the batches.
- **Evaluation:** one fixed train / validation split of the parcels (no spatial blocks); macro-F1, IoU and accuracy of
  the best model on the validation set; training-set fractions from 20% to 100% for the label-efficiency curve.

## Key results

Dice loss, all training data (Table 1 of the paper):

| Strategy | Crop type: macro-F1 | Crop type: IoU | Crop type: accuracy | Crop health: macro-F1 | Crop health: IoU |
| --- | --- | --- | --- | --- | --- |
| From scratch | 0.63 | 0.49 | 0.69 | 0.31 | 0.22 |
| Frozen encoder | 0.56 | 0.50 | 0.68 | 0.20 | 0.17 |
| **Full fine-tuning** | **0.79** | **0.67** | **0.82** | **0.32** | **0.25** |

- **Crop type:** full fine-tuning was best by a clear margin, converged faster and reached a lower loss. The frozen
  encoder was the weakest, below training from scratch.
- **Crop health** (≥ 80% "healthy"): all strategies were poor; full fine-tuning was only 0.01 above training from
  scratch, and its higher accuracy hid overfitting to the majority class.
- **Fewer labels:** the pretrained, fully fine-tuned model was well ahead of training from scratch at 20–40% of the
  training data for crop type (40–60% for crop health). With more data the pretrained model levelled off while the
  model trained from scratch kept improving.
- **Loss:** Dice loss gave the best macro-F1 and minority-class recall of the three losses.

## What this research takes from it

- **The design of [E05](../experiments/E05_prithvi_strategies.md):** the same three strategies (frozen, full
  fine-tuning, from scratch) with the same Prithvi-EO-2.0-300M encoder, loss and score per epoch, and score against the
  share of training labels. It is the closest published comparison of the three strategies for crops with Prithvi.
- **The expectation to test:** fine-tuning beats the frozen encoder, and pretraining matters most with few labels. E05
  checks whether this holds for irrigated / rainfed at 10 m.
- **A warning for the imbalanced class:** fine-tuning overfitted the majority class in the crop-health task; rainfed
  is the minority class here (about 1 rainfed to 6 irrigated parcels in Morocco, 1 to 5 in Utah), so E05 reports per-class F1 and compares
  validation-loss curves, not only accuracy.

## Differences and limitations

- **Very small dataset and a single split:** 601 parcels, one random train / validation split, no repeated runs, no
  confidence intervals or tests, and the best model is chosen and scored on the same validation set. The differences
  may partly reflect chance; E05 uses spatial folds, 3 seeds, a separate test fold and block-bootstrap tests.
- **Strong upsampling:** 30 m HLS resampled to 0.5 m (factor 60) means the encoder sees blocky images far from its
  pretraining scale. This may explain the weak frozen encoder: its features do not suit these inputs, and fine-tuning
  can correct that. E05 uses 10 m Sentinel-2 tiles close to Prithvi's pretraining data (30 m HLS), so the frozen encoder
  is less handicapped.
- **Three dates** instead of 12 monthly composites; one parcel per chip instead of 128 × 128 px tiles with many fields.
- **One learning rate for all weights** (10⁻⁴), while E05 uses a ten times smaller rate for the encoder (discriminative
  fine-tuning) and a warm-up, which is the usual guard against destroying the pretrained features.
- The paper does not give the decoder type, batch size or early-stopping rule, so the set-up cannot be reproduced
  exactly from the text.
