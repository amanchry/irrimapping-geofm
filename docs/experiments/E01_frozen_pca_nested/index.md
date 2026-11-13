# E01 · Frozen GeoFMs + PCA + classical classifiers (nested spatial CV)

*First full comparison: frozen embeddings of four GeoFMs vs Sentinel-1/2 features, with nested spatial cross-validation. Done before D14; kept as the check of the validation design.*

| | |
| --- | --- |
| **Status** | ✅ done (30 Oct 2026) — superseded by [E02](../E02_frozen_pca/index.md) as the main result; kept as the sensitivity check of the validation design |
| **Research question** | RQ1 — accuracy of each foundation model vs a Sentinel-based classifier |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E01_frozen_pca_nested/config.yaml) |
| **Report** | **[REPORT.md](report.md)** |

## What this experiment builds on

Every method choice of this experiment and the paper it comes from (full list: [Reference papers](../../references/index.md)).

| What this experiment takes | Reference |
| --- | --- |
| Frozen encoder → parcel embedding → simple head | Tseng et al. (2023), [Presto](https://arxiv.org/abs/2304.14065) |
| Frozen-encoder evaluation of GeoFMs against supervised baselines | Marsocci et al. (2024), [PANGAEA](https://arxiv.org/abs/2412.04204) · [summary](../../references/pangaea.md) |
| Simple heads on embeddings are common and matter little | Mughees et al. (2026), [From foundation embeddings to cropland maps: label efficiency, temporal transferability …](https://arxiv.org/abs/2609.17138) |
| Random forest on frozen embeddings in agriculture | Nedungadi et al. (2026), [Foundation models meet agriculture: challenges beyond pretraining](https://arxiv.org/abs/2608.30392) |
| Closest study: GeoFM embeddings for irrigated vs rainfed mapping | Yang et al. (2026), [AlphaEarth embeddings for irrigated cropland mapping](https://doi.org/10.3390/rs18071065) · [summary](../../references/yang-2026-alphaearth-irrigated.md) |
| Logistic regression as the linear probe | Alain & Bengio (2016), [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644) |
| Random forest | Breiman (2001), [Random forests](https://doi.org/10.1023/A:1010933404324) |
| RBF SVM for irrigation mapping with Sentinel data | Gao et al. (2018), [Irrigation mapping using Sentinel-1 time series at field scale](https://doi.org/10.3390/rs10091495) |
| RBF SVM for irrigation mapping, Sentinel-1/2, incl. Morocco | Elwan et al. (2022), [Irrigation mapping on two contrasted climatic contexts using Sentinel-1 and Sentinel-2](https://doi.org/10.3390/w14050804) |
| Nested CV: tuning separated from testing | Varma & Simon (2006), [Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91) |
| Spatial folds for tuning as well as testing | Schratz et al. (2019), [Hyperparameter tuning … using spatial data](https://doi.org/10.1016/j.ecolmodel.2019.06.002) |
| Closest validation design: held-out regions, inner spatial CV, Sentinel-1/2 crops | Orynbaikyzy et al. (2022), [Spatial transferability of random forest models for crop type classification](https://doi.org/10.3390/rs14061493) |
| Spatial blocks (20 km) instead of random splits | Roberts et al. (2017), [Cross-validation strategies for data with … spatial … structure](https://doi.org/10.1111/ecog.02881) |
| Why random CV overstates accuracy (shown in every run) | Ploton et al. (2020), [Spatial validation reveals poor predictive performance …](https://doi.org/10.1038/s41467-020-18321-y) |
| Confidence intervals by resampling whole blocks | Künsch (1989), [The jackknife and the bootstrap for general stationary observations](https://doi.org/10.1214/aos/1176347265) |
| Paired differences between inputs with CIs | Foody (2009), [Classification accuracy comparison …](https://doi.org/10.1016/j.rse.2009.03.014) |
| Correction for testing many input × head pairs | Holm (1979), A simple sequentially rejective multiple test procedure, *Scand. J. Stat.* 6 |
| No kappa coefficient | Pontius & Millones (2011), [Death to Kappa](https://doi.org/10.1080/01431161.2011.552923) |
| Model compared: Clay v1.5 (10 S2 bands, date and location inputs) | Clay Foundation (2024), [Clay v1.5](https://huggingface.co/made-with-clay/Clay) |
| Model compared: Prithvi-EO-2.0 (6 HLS bands, date and location inputs) | Szwarcman et al. (2024), [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732) |
| Model compared: TerraMind (12 S2 bands + S1 VV, VH in E01, before D14) | Jakubik et al. (2025), [TerraMind](https://arxiv.org/abs/2504.11171) |
| Model compared: AgriFM (10 S2 bands, whole year in one pass; config normalisation in E01, before D15) | Li et al. (2025), [AgriFM](https://arxiv.org/abs/2505.21357) |
| Labels | Ouassanouan et al. (2026), [Crop and irrigation types ground-truth dataset for Moroccan agricultural regions](https://doi.org/10.1038/s41597-026-06993-y) |

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
