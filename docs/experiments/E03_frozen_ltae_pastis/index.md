# E03 · Frozen GeoFMs + L-TAE temporal aggregation

| | |
| --- | --- |
| **Status** | 📝 planned — needs `models_run/03c_train_ltae.py` |
| **Research question** | RQ1 — how the monthly embeddings are combined |
| **Builds on** | [E02](../E02_frozen_pca_pastis/index.md) |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E03_frozen_ltae_pastis/config.yaml) |

## Question
Does a **learned** temporal aggregation of the 12 monthly embeddings (L-TAE) classify irrigated vs rainfed
parcels better than concatenating the months and reducing them with PCA?

## Why
Clay, Prithvi and TerraMind give one embedding per month. E01/E02 put the 12 months side by side and reduce them
with PCA (a fixed, unsupervised step). PANGAEA (Marsocci et al. 2024) combines the monthly encodings of
single-date models with a trained L-TAE and found it better than a linear mapping. E03 tests whether that holds
for irrigated vs rainfed mapping. The same L-TAE is trained on the monthly S1/S2 features, so the baseline gets
the same temporal model ([protocol §5](../../phase2_models/evaluation.md#5-temporal-aggregation-two-pipelines)).

## What changes
| | E02 (PCA pipeline) | E03 (L-TAE pipeline) |
| --- | --- | --- |
| Monthly embeddings → parcel vector | concat 12 months → standardise → PCA 128 | L-TAE attention over the 12 months (trained) |
| Classifier | LogReg, RF, HGB, SVM | L-TAE output → small MLP head (trained together) |
| Trained on | 3 training folds | 3 training folds; early stopping on the validation fold |
| Inputs | S1/S2, Clay, Prithvi, TerraMind, AgriFM | S1/S2, Clay, Prithvi, TerraMind (AgriFM has no months) |
| Seeds | 1 (deterministic classifiers) | 3 (neural network: mean ± SD over seeds) |

The comparison step also reads the E02 runs (`EXTRA_RUNS`), so PCA and L-TAE are compared on the same parcels with
the same bootstrap draws.

## Implementation plan (`models_run/03c_train_ltae.py`)
1. Read the same embedding files as `03_train_evaluate.py` (`<model>_agyear_tiles.npz`, 12 steps × D) or the
   monthly S1/S2 features (17 variables × 12 months).
2. Standardise per channel on the training folds; empty months are masked in the attention, not filled.
3. L-TAE (Garnot & Landrieu 2020) with the paper's default size (16 heads, d_k 8, d_model 256, MLP 256-128),
   then an MLP head with 2 outputs; class-weighted cross-entropy.
4. PASTIS rotation: train on 3 folds, early stopping on the validation fold (macro-F1), test on the test fold.
5. Write `oof_predictions.csv`, `metrics_per_fold.csv`, `metrics_with_ci.csv`, `run_config.json` in the same format
   as script 03, with `model = "ltae"`, so `04_compare_models.py` and `make_report.py` work unchanged.

## How to run (once 03c exists)
```bash
# experiments/run_experiment.py:  EXPERIMENT = "E03_frozen_ltae_pastis"
python experiments/run_experiment.py
```

## Outcome
To be filled in when the results are in.
