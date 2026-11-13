# E06 · Label efficiency: frozen GeoFMs vs Sentinel-2 features with fewer training labels

*Learning curves: every input trained on 10, 25, 50 and 100% of the training blocks, to see whether foundation models need fewer labels.*

| | |
| --- | --- |
| **Status** | 📝 planned — needs E02–E03 and the `TRAIN_FRACTION` setting |
| **Research question** | RQ1 — Q5 of the [experiment plan](../../phase2_models/experiment_plan.md#6-e06--label-efficiency) |
| **Builds on** | [E02](../E02_frozen_pca/index.md), [E04](../E04_frozen_ltae/index.md), [E05](../E05_psetae_scratch/index.md), [E03](../E03_frozen_pca_mlp/index.md) |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E06_label_efficiency/config.yaml) |
| **Report** | written when the results are in |

## What this experiment builds on

Every method choice of this experiment and the paper it comes from (full list: [Reference papers](../../references/index.md)).

| What this experiment takes | Reference |
| --- | --- |
| Label efficiency of foundation embeddings for cropland mapping | Mughees et al. (2026), [From foundation embeddings to cropland maps: label efficiency, temporal transferability …](https://arxiv.org/abs/2609.17138) |
| Embeddings built for mapping from sparse labels | Brown et al. (2025), [AlphaEarth Foundations … from sparse label data](https://arxiv.org/abs/2507.22291) |
| Subsampling whole spatial blocks, not parcels | Roberts et al. (2017), [Cross-validation strategies for data with … spatial … structure](https://doi.org/10.1111/ecog.02881) |
| Same heads and validation as E02–E03 | Garnot & Landrieu (2021), [PASTIS / U-TAE](https://arxiv.org/abs/2107.07933) · [summary](../../references/pastis-utae.md) |
| Confidence intervals by resampling whole blocks | Künsch (1989), [The jackknife and the bootstrap for general stationary observations](https://doi.org/10.1214/aos/1176347265) |
| Paired differences between inputs with CIs | Foody (2009), [Classification accuracy comparison …](https://doi.org/10.1016/j.rse.2009.03.014) |
| Correction for testing many input × head pairs | Holm (1979), A simple sequentially rejective multiple test procedure, *Scand. J. Stat.* 6 |
| No kappa coefficient | Pontius & Millones (2011), [Death to Kappa](https://doi.org/10.1080/01431161.2011.552923) |

## Question
How much training data does each input need? Do the foundation models gain most when labels are few?

## Why
The main argument for foundation models is that their pretrained representation needs fewer labels. With all
training data, a frozen GeoFM may only match Sentinel-2 features (as in E01); with few labels the difference can
be large. The learning curve shows where a foundation model starts to pay off.

## Setup
| Setting | Value |
| --- | --- |
| Training size | 10, 25, 50% of the training blocks of each round (100% = the E02–E03 runs) |
| Sampling | whole blocks, drawn separately from blocks with rainfed parcels and from irrigated-only blocks; a draw with fewer than 20 rainfed parcels is redrawn |
| Validation and test folds | unchanged, so only the amount of training data varies |
| Repeats | 3 draws per fraction; the same draws for every input (paired) |
| Inputs and heads | every input with the linear probe and SVM; the best neural setting per input (E03 or E04); PSE-TAE for the pixel sets |
| Regions | Morocco and Utah 2023/24 |

## Results to show
Learning curves of macro-F1 and rainfed F1 against the training fraction (mean over draws, block-bootstrap CI), one
panel per head; the fraction at which each GeoFM reaches the full-data score of the Sentinel-2 baseline.

## Outcome
To be filled in when the results are in.
