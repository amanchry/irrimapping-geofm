# E02 · Frozen GeoFMs + PCA + classical classifiers

*Main table of RQ1: frozen embeddings of four GeoFMs vs Sentinel-2 features, 12 months concatenated and reduced by PCA, classified by logistic regression, random forest and SVM, in Morocco and Utah.*

| | |
| --- | --- |
| **Status** | 📝 planned — embeddings ready for both regions; needs the region setting in scripts 03–05 |
| **Research question** | RQ1 — accuracy of each foundation model vs a Sentinel-based classifier |
| **Builds on** | [E01](../E01_frozen_pca_nested/index.md) |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E02_frozen_pca/config.yaml) |
| **Report** | written when the results are in |

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
| Validation rotation: 3 folds train, 1 validation, 1 test | Garnot & Landrieu (2021), [PASTIS / U-TAE](https://arxiv.org/abs/2107.07933) · [summary](../../references/pastis-utae.md) |
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
| Labels (Morocco) | Ouassanouan et al. (2026), [Crop and irrigation types ground-truth dataset for Moroccan agricultural regions](https://doi.org/10.1038/s41597-026-06993-y) |

## Question
Do frozen foundation-model embeddings separate irrigated from rainfed parcels better than Sentinel-2 features,
with the same parcels, folds and classifiers, under the **final validation design**? (Q1 of the
[experiment plan](../../phase2_models/experiment_plan.md#4-design-what-is-varied))

## Why
E01 used nested spatial CV. The study now uses the PASTIS rotation for every model (classifiers and, later,
neural networks), so they are trained, tuned and tested in exactly the same way
([decision D3](../../decisions.md#d3-validation-pastis-rotation-instead-of-nested-cv)). E02 is the
**main result of RQ1**; E01 becomes a sensitivity check of the validation design.

## What changes
| | E01 | E02 |
| --- | --- | --- |
| Validation | nested: inner spatial CV on the 4 training folds, refit on all 4 | PASTIS: 3 folds train, next fold validation, 1 fold test |
| Model trained on | 4 folds | 3 folds |
| Input data | baseline and TerraMind with Sentinel-1 | **Sentinel-2 only** for every input ([D14](../../decisions.md#d14-sentinel-2-only-for-every-input)) |
| Baseline | 204 S1/S2 features | 180 Sentinel-2 features (15 variables × 12 months) |
| AgriFM normalisation | config statistics | chip statistics of the region ([D15](../../decisions.md#d15-agrifm-normalised-with-the-statistics-of-the-regions-chips)) |
| Classifiers | LogReg, RF, HGB, SVM | LogReg (linear probe), RF, SVM ([D6](../../decisions.md#d6-four-classifiers-instead-of-one-random-forest)) |
| Regions | Morocco | Morocco and Utah 2023/24, analysed separately |

Temporal aggregation (concat + PCA 128), grids and metrics are unchanged; see the
[evaluation protocol](../../phase2_models/evaluation.md) and the [experiment plan](../../phase2_models/experiment_plan.md#e02--pca--classical-classifiers).

## How to run
```bash
# experiments/run_experiment.py:  EXPERIMENT = "E02_frozen_pca"
python experiments/run_experiment.py
```
Needs the embeddings of `models_run/02a–02d`, ready for both regions (`outputs/<region>/study/02_embeddings/`).
Runs on the cluster as one CPU job per region and input.

## Outcome
To be filled in when the results are in.
