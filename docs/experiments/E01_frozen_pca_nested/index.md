# E01 · Frozen GeoFMs + PCA vs the S1/S2 baseline (nested spatial CV)

| | |
| --- | --- |
| **Status** | ✅ done (30 Oct 2026) — superseded by [E02](../E02_frozen_pca_pastis/index.md) as the main result; kept as the sensitivity check of the validation design |
| **Research question** | RQ1 — accuracy of each foundation model vs a Sentinel-based classifier |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E01_frozen_pca_nested/config.yaml) |
| **Report** | **[REPORT.md](report.md)** |

## Question
Do frozen foundation-model embeddings separate irrigated from rainfed parcels better than Sentinel-1/2 features,
with the same parcels, folds and classifiers?

## Why
First complete run on the shared 128 px tile store with patch-aligned chips: every input sees the same pixels,
the same monthly composites and the same parcels, so any difference comes from the representation.

## Setup
| | |
| --- | --- |
| Inputs | S1/S2 baseline (204 parcel features) · Clay v1.5 (chips p8) · Prithvi-EO-2.0 · TerraMind · AgriFM (chips p16) |
| Encoders | frozen; one embedding per parcel per month (AgriFM: per year), patch-share-weighted mean over the parcel |
| Temporal aggregation | concat 12 months → standardise → PCA 128 (fitted on the training folds only) |
| Classifiers | LogReg (linear probe), RF, HGB, SVM; grids in the [protocol](../../phase2_models/evaluation.md#4-classifiers) |
| Validation | nested spatial CV: 5 outer folds (20 km blocks); inner spatial CV on the 4 training folds chooses the hyperparameters; refit on 4 folds |

## How to run again
```bash
# experiments/run_experiment.py:  EXPERIMENT = "E01_frozen_pca_nested",  RERUN = True
python experiments/run_experiment.py
```
About 2 h per foundation-model input and 30 min for the baseline on the MacBook (CPU).

## Outcome
No foundation model beats the S1/S2 baseline: the best input-classifier pair is Clay + SVM (macro-F1 0.910) against
0.906 for the baseline with the same SVM, a difference well inside the noise. With a linear probe, Clay and
TerraMind equal the baseline; with a random forest every GeoFM is lower, and AgriFM is clearly worse. The GeoFMs do
better than the baseline in the hardest test fold. Full report: **[REPORT.md](report.md)**.
