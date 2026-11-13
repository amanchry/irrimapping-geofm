# E04 · Frozen GeoFMs + L-TAE temporal aggregation + MLP head

*The 12 monthly embeddings are combined by a trained attention model (L-TAE) instead of PCA; the same MLP head as E03, so E03 → E04 isolates the temporal aggregation.*

| | |
| --- | --- |
| **Status** | 🟦 ready to run (after E02 and E03; code tested on both regions) |
| **Research question** | RQ1 — Q3 of the [experiment plan](../../phase2_models/experiment_plan.md#4-design-what-is-varied): how the monthly embeddings are combined |
| **Builds on** | [E03](../E03_frozen_pca_mlp/index.md) (same MLP head), [E02](../E02_frozen_pca/index.md) |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E04_frozen_ltae/config.yaml) |

## What this experiment builds on

Every method choice of this experiment and the paper it comes from (full list: [Reference papers](../../references/index.md)).

| What this experiment takes | Reference |
| --- | --- |
| Label efficiency of foundation embeddings for cropland mapping | Mughees et al. (2026), [From foundation embeddings to cropland maps: label efficiency, temporal transferability …](https://arxiv.org/abs/2609.17138) |
| Embeddings built for mapping from sparse labels | Brown et al. (2025), [AlphaEarth Foundations … from sparse label data](https://arxiv.org/abs/2507.22291) |
| L-TAE: the temporal attention model and its default size | Garnot & Landrieu (2020), [L-TAE](https://arxiv.org/abs/2007.00586) · [summary](../../references/ltae.md) |
| L-TAE to combine the per-date encodings of single-date GeoFMs; frozen encoder + trained head | Marsocci et al. (2024), [PANGAEA](https://arxiv.org/abs/2412.04204) · [summary](../../references/pangaea.md) |
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

## Question
Does a **learned** temporal aggregation of the 12 monthly embeddings (L-TAE) classify irrigated vs rainfed
parcels better than concatenating the months and reducing them with PCA?

## Why
Clay, Prithvi and TerraMind give one embedding per month. E01/E02 put the 12 months side by side and reduce them
with PCA (a fixed, unsupervised step). PANGAEA (Marsocci et al. 2024) combines the monthly encodings of
single-date models with a trained L-TAE and found it better than a linear mapping. E04 tests whether that holds
for irrigated vs rainfed mapping. The same L-TAE is trained on the monthly Sentinel-2 features, so the baseline gets
the same temporal model ([protocol §5](../../phase2_models/evaluation.md#5-temporal-aggregation-two-pipelines)).

## What changes
| | E03 (PCA + MLP) | E04 (L-TAE + MLP) |
| --- | --- | --- |
| Monthly embeddings → parcel vector | concat 12 months → standardise → PCA 128 | L-TAE attention over the 12 months (trained) |
| Head | MLP 256-128 | L-TAE output → MLP 256-128 (trained together) |
| Trained on | 3 training folds; early stopping on the validation fold | same |
| Inputs (Sentinel-2 only) | S2 baseline, Clay, Prithvi, TerraMind, AgriFM | S2 baseline, Clay, Prithvi, TerraMind (AgriFM has no months) |
| Seeds | 3 | 3 |

E04 is also compared with the classical classifiers of E02.

The comparison step also reads the E03 and E02 runs (`EXTRA_RUNS`), so PCA and L-TAE are compared on the same
parcels with the same bootstrap draws. Regions: Morocco and Utah 2023/24.

## Implementation (`models_run/03c_train_ltae.py`, `HEAD: ltae`)
1. Reads the same embedding files as `03_train_evaluate.py` (`<model>_agyear_tiles[_2023-24].npz`, 12 steps × D) or
   the monthly Sentinel-2 features (15 variables × 12 months; gaps filled by interpolation in time, as in script 03).
2. Standardises each channel with the training parcels' observed months; months without a clear pixel are masked in
   the attention, not filled.
3. L-TAE (Garnot & Landrieu 2020) with the paper's default size (16 heads with one learned master query each, d_k 8,
   d_model 256, sinusoidal encoding of the mid-month day, output MLP 256-128), then a linear layer with 2 outputs;
   class-weighted cross-entropy, AdamW.
4. PASTIS rotation: train on 3 folds, early stopping on the validation fold (macro-F1, patience 15), test on the test
   fold; 3 seeds per round, the out-of-fold probability is their mean.
5. Writes `oof_predictions.csv`, `metrics_per_fold.csv`, `metrics_with_ci.csv`, `metrics_per_seed.csv`,
   `training_history.csv`, `attention_by_month.csv` and `run_config.json` in the format of script 03
   (`model = "ltae"`), so `04_compare_models.py` and `make_report.py` treat it like any other run.

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
| Runs | 4 inputs × 3 fractions × 3 draws = **36** per region, for L-TAE + MLP |
| Result | macro-F1 and rainfed F1 against the training fraction, one line per input, Morocco and Utah |

## How to run

```bash
# on the Mac (all regions, main and label-efficiency runs, compare, report):
#   experiments/run_experiment.py:  EXPERIMENT = "E04_frozen_ltae"
python experiments/run_experiment.py
# on the cluster: one Slurm job per run, then one compare job per region (EXPERIMENT and DRY_RUN in the script)
python experiments/submit_slurm.py
```

- **Order:** after E02 and E03 (compares with their runs).
- **Hardware:** GPU. `QUICK = True` checks the whole chain in a few minutes first.
- **Results:** `experiments/E04_frozen_ltae/results/<region>/`; the report page is written by the `report` step on the Mac.

## Outcome
To be filled in when the results are in.
