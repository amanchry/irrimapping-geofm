[Home](../../README.md) › [Experiments](../README.md) › E02

# E02 · Frozen GeoFMs + PCA vs the S1/S2 baseline (PASTIS validation)

| | |
| --- | --- |
| **Status** | 🟦 ready to run |
| **Research question** | RQ1 — accuracy of each foundation model vs a Sentinel-based classifier |
| **Builds on** | [E01](../E01_frozen_pca_nested/README.md) |
| **Config** | [`config.yaml`](config.yaml) |
| **Report** | `REPORT.md`, written when the results are in |

## Question
Do frozen foundation-model embeddings separate irrigated from rainfed parcels better than Sentinel-1/2 features,
with the same parcels, folds and classifiers, under the **final validation design**?

## Why
E01 used nested spatial CV. The thesis now uses the PASTIS rotation for every model (classifiers and, later,
neural networks), so they are trained, tuned and tested in exactly the same way
([decision D3](../../docs/04_decisions.md#d3-validation-pastis-rotation-instead-of-nested-cv)). E02 is the
**main result of RQ1**; E01 becomes a sensitivity check of the validation design.

## What changes
| | E01 | E02 |
| --- | --- | --- |
| Validation | nested: inner spatial CV on the 4 training folds, refit on all 4 | PASTIS: 3 folds train, next fold validation, 1 fold test |
| Model trained on | 4 folds | 3 folds |

Inputs, embeddings, temporal aggregation (concat + PCA 128), classifiers, grids and metrics are unchanged; see the
[evaluation protocol](../../docs/03_protocol.md).

## How to run
```bash
# experiments/run_experiment.py:  EXPERIMENT = "E02_frozen_pca_pastis"
python experiments/run_experiment.py
```
Needs the embeddings of `models_run/02a–02d` (already in `outputs/study/02_embeddings/`). Run time on the
MacBook: about 2 h per foundation-model input, 30 min for the baseline (E01 timings; PASTIS tunes on one fold
instead of four, so it should be faster).

## Outcome
To be filled in when the results are in.
