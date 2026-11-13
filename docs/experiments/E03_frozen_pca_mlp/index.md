# E03 · Frozen GeoFMs + PCA + MLP head

*Exactly the E02 input (12 months → PCA 128) with a trained MLP head instead of the classical classifiers: E02 → E03 isolates the effect of the head.*

| | |
| --- | --- |
| **Status** | ✅ done (13 Nov 2026), Morocco and Utah |
| **Research question** | RQ1 — Q2 of the [experiment plan](../../phase2_models/experiment_plan.md#4-design-what-is-varied) |
| **Builds on** | [E02](../E02_frozen_pca/index.md) |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E03_frozen_pca_mlp/config.yaml) |
| **Report** | **[REPORT.md](report.md)** |

## What this experiment builds on

Every method choice of this experiment and the paper it comes from (full list: [Reference papers](../../references/index.md)).

| What this experiment takes | Reference |
| --- | --- |
| Frozen encoder + trained neural head (the usual GeoFM benchmark set-up) | Marsocci et al. (2024), [PANGAEA](https://arxiv.org/abs/2412.04204) · [summary](../../references/pangaea.md) |
| Embeddings + MLP head | Feng et al. (2025), [TESSERA](https://arxiv.org/abs/2506.20380) |
| Simple heads on embeddings matter little (tested here) | Mughees et al. (2026), [From foundation embeddings to cropland maps: label efficiency, temporal transferability …](https://arxiv.org/abs/2609.17138) |
| Validation rotation; early stopping on the validation fold | Garnot & Landrieu (2021), [PASTIS / U-TAE](https://arxiv.org/abs/2107.07933) · [summary](../../references/pastis-utae.md) |
| Spatial blocks (20 km) instead of random splits | Roberts et al. (2017), [Cross-validation strategies for data with … spatial … structure](https://doi.org/10.1111/ecog.02881) |
| Why random CV overstates accuracy (shown in every run) | Ploton et al. (2020), [Spatial validation reveals poor predictive performance …](https://doi.org/10.1038/s41467-020-18321-y) |
| Confidence intervals by resampling whole blocks | Künsch (1989), [The jackknife and the bootstrap for general stationary observations](https://doi.org/10.1214/aos/1176347265) |
| Paired differences between inputs with CIs | Foody (2009), [Classification accuracy comparison …](https://doi.org/10.1016/j.rse.2009.03.014) |
| Correction for testing many input × head pairs | Holm (1979), A simple sequentially rejective multiple test procedure, *Scand. J. Stat.* 6 |
| No kappa coefficient | Pontius & Millones (2011), [Death to Kappa](https://doi.org/10.1080/01431161.2011.552923) |
| Model compared: Clay v1.5 (10 S2 bands, date and location inputs) | Clay Foundation (2024), [Clay v1.5](https://huggingface.co/made-with-clay/Clay) |
| Model compared: Prithvi-EO-2.0 (6 HLS bands, date and location inputs) | Szwarcman et al. (2024), [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732) |
| Model compared: TerraMind (12 S2 bands; S1 only in ablation A5) | Jakubik et al. (2025), [TerraMind](https://arxiv.org/abs/2504.11171) |
| Model compared: AgriFM (10 S2 bands, whole year in one pass) | Li et al. (2025), [AgriFM](https://arxiv.org/abs/2505.21357) |

## Question
Does a trainable neural head get more out of the same PCA input than logistic regression, random forest and SVM?

## Why
GeoFM benchmarks evaluate frozen encoders with a trained neural head. E03 gives every input the same small MLP on
exactly the E02 input, so a difference between E02 and E03 comes from the head alone. E03 is also the reference for
E04: E03 → E04 changes only how the months are combined (PCA → L-TAE), with the same MLP head.

## What changes
| | E02 | E03 |
| --- | --- | --- |
| Input | 12 months concatenated → PCA 128 (baseline: 180 features) | same |
| Head | LogReg, RF, SVM (grid on the validation fold) | MLP 128 → 256 → 128 → 2, ReLU, batch normalisation, dropout 0.2 |
| Training | – | AdamW 1e-3, weight decay 1e-4, batch 128, class-weighted cross-entropy, ≤ 100 epochs, early stopping on validation macro-F1 (patience 15) |
| Seeds | 1 | 3 (mean ± SD) |

Inputs (all Sentinel-2 only): S2 baseline, Clay, Prithvi, TerraMind, AgriFM. Regions: Morocco and Utah 2023/24.

## Label efficiency (part of this experiment)

Every input is also trained on **10, 25 and 50% of the training blocks**, so the report shows a learning curve next
to the main result (Q5 of the [experiment plan](../../phase2_models/experiment_plan.md#6-label-efficiency-inside-e02e05)).

| Setting | Value |
| --- | --- |
| Fractions | 10, 25, 50% of the training blocks of each round; 100% = the main run |
| Sampling | whole blocks: rainfed and irrigated-only blocks each put in a random order (per region, round and draw), and a fraction takes the first blocks of each list, so the draws are **nested** (10% ⊂ 25% ⊂ 50%); the order is accepted only if its first 10% hold ≥ 20 rainfed parcels. No fraction below 10%. |
| Draws | 3 per fraction; draw *k* is the same in E02–E05 and for every input (paired) |
| Seeds | 1 seed per draw (the main run keeps its 3 seeds) |
| Validation and test folds | unchanged |
| Runs | 5 inputs × 3 fractions × 3 draws = **45** per region, for the MLP head |
| Result | macro-F1 and rainfed F1 against the training fraction, one line per input, Morocco and Utah |

## How to run

```bash
# on the Mac (all regions, main and label-efficiency runs, compare, report):
#   experiments/run_experiment.py:  EXPERIMENT = "E03_frozen_pca_mlp"
python experiments/run_experiment.py
# on the cluster: one Slurm job per run, then one compare job per region (EXPERIMENT and DRY_RUN in the script)
python experiments/submit_slurm.py
```

- **Order:** after E02 (compares with its runs: `EXTRA_RUNS: "E02_frozen_pca/*"`).
- **Hardware:** GPU (CPU works). `QUICK = True` checks the whole chain in a few minutes first.
- **Results:** `experiments/E03_frozen_pca_mlp/results/<region>/`; the report page is written by the `report` step on the Mac.

## Outcome
The MLP head is as good as E02's SVM for every input (all differences within ±0.01, none significant) and clearly
better than LogReg and RF for most GeoFMs. A trained head therefore removes the weak-head problem, but it does not
change the ranking of the inputs. **Morocco:** the S2 baseline with an MLP is the best result so far (0.908). Clay,
Prithvi and TerraMind are 0.01–0.02 lower (not significant), and AgriFM is worse. **Utah:** Clay, Prithvi and TerraMind
with an MLP beat the S2 baseline with an MLP by 0.04–0.05 (significant; +0.08–0.09 in Bear River). The S2 baseline's
best head is still E02's random forest (0.764), so the best GeoFM (Clay + SVM, 0.791) leads by about 0.03. E03 also
corrects E02: with an MLP, the S2 features find most of the rainfed parcels in Souss (84%), so the GeoFMs' advantage
there was a head effect. Full tables, figures and interpretation are in the **[report](report.md)**.
