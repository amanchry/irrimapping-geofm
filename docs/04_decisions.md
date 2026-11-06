[Home](../README.md) › [Documentation](README.md) › Decisions

# 4 · Decisions and departures from the proposal

Every point where the implementation differs from the [thesis proposal](../thesis_proposal/MScProposal_s3677133_AmanChaudhary.pdf),
with what the proposal said, what is done now, and why. Smaller method choices are in the
[progress log](../PROGRESS.md#method-decisions).

| # | Topic | Proposal | Now | Since |
| --- | --- | --- | --- | --- |
| [D1](#d1-spatial-folds-instead-of-a-7030-split) | Train / test split | 70 / 30 stratified + validation subset | 5 spatial folds of 20 km blocks | Oct 2026 |
| [D2](#d2-two-agricultural-years-per-parcel-season) | Imagery period | 2024 annual stack | each parcel's own agricultural year (2023/24 or 2024/25), Sep – Aug | Oct 2026 |
| [D3](#d3-validation-pastis-rotation-instead-of-nested-cv) | Validation design | validation subset from the training data | PASTIS rotation: 3 train / 1 validation / 1 test | 4 Nov 2026 |
| [D4](#d4-one-shared-tile-store-for-every-input) | Image preparation | annual stacks via STAC / xarray | one tile store from Google Earth Engine for every input | 27 Oct 2026 |
| [D5](#d5-frozen-encoder--classical-classifiers) | Frozen-encoder strategy | frozen encoder + decoder / head | frozen encoder → parcel embedding → four classifiers | Oct 2026 |
| [D6](#d6-four-classifiers-instead-of-one-random-forest) | Baseline classifier | Random Forest | four classifiers (RF headline) on every input | Oct 2026 |
| [D7](#d7-macro-f1-as-the-headline-metric) | Selection metric | mean F1 over folds ± SD | macro-F1 of the out-of-fold predictions, 95% block-bootstrap CI, per-fold mean ± SD | Oct 2026 |
| [D8](#d8-two-temporal-aggregation-pipelines) | Combining the months | not specified | PCA pipeline now; L-TAE pipeline next | 4 Nov 2026 |
| [D9](#d9-cloud-score-instead-of-qa60) | Cloud mask | cloud-masked composites | Cloud Score+ (cs_cdf ≥ 0.60) | Oct 2026 |
| [D10](#d10-transfer-region-under-review) | Transfer region | Tunisia, 54 GMIA-NEXT points | under review (Tunisia or US states with dryland labels); after Morocco | 4 Nov 2026 |
| [D11](#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model) | Classifier head and mapping | frozen encoder + head; maps 2021–2025 | parcel benchmark with classical heads (RQ1) + segmentation head and pixel models for the maps (RQ2–RQ4); neural baselines TempCNN and U-TAE | 6 Nov 2026 |

---

## D1. Spatial folds instead of a 70/30 split
- **Proposal (§4.1):** stratified 70 / 30 split, validation subset from the training data. (§4.2.4 already asks for
  spatial cross-validation.)
- **Now:** parcels grouped into 20 km blocks, blocks into 5 folds with equal class balance; every parcel tested
  once by a model that never saw its block.
- **Why:** with a random split, 79% of test parcels lie within 250 m of a training parcel (about 2% with 20 km
  blocks). Random splits overstate accuracy for mapping new areas (Roberts et al. 2017; Ploton et al. 2020).
- **Where:** [dataset §5](../ml_dataset_prep/README.md#5-script-02--make_spatial_folds), [protocol §2](03_protocol.md#2-spatial-splits).

## D2. Two agricultural years (per-parcel season)
- **Proposal (§4.1):** the 2024 annual image stack.
- **Now:** each parcel gets the 12 months (1 Sep – 31 Aug) of the season in which it was surveyed: 2023/24 or
  2024/25. Both seasons in one dataset, months aligned (month 1 = September).
- **Why:** the survey ran Dec 2023 – Jan 2025; using only 2024 would mix seasons, and dropping 2024/25 would lose 42%
  of the rainfed parcels. The agricultural year covers the winter rainfed cycle and the summer, when only irrigated
  fields stay green.
- **Consequence:** region and year are partly confounded (Gharb and Tadla only 2023/24, Souss only 2024/25).
- **Where:** [dataset §6.2](../ml_dataset_prep/README.md#62-two-agricultural-years).

## D3. Validation: PASTIS rotation instead of nested CV
- **Before (E01):** nested spatial CV, settings by inner CV on the 4 training folds, refit on 4.
- **Now:** per round 3 folds train, the next fold chooses the settings (or stops training), 1 fold tests, as in
  PASTIS (Garnot & Landrieu 2021).
- **Why:** one design for classical classifiers and neural networks (L-TAE, fine-tuning), which need a separate
  validation fold for early stopping. It is the standard design of the PASTIS crop-mapping benchmark.
- **Where:** [protocol §2](03_protocol.md#2-spatial-splits); E01 is kept as the sensitivity check.

## D4. One shared tile store for every input
- **Proposal:** annual Sentinel-1/2 composites via STAC and xarray.
- **Now:** Google Earth Engine builds non-overlapping 128 × 128 px tiles (1.28 km) with 12 monthly composites × 19
  bands; the baseline features and the foundation-model chips are all cut from these tiles.
- **Why:** STAC processing was too slow for the full set; with one store every input sees identical pixels,
  compositing and masks, and chips no longer overlap across folds.
- **Where:** [dataset §6–7](../ml_dataset_prep/README.md#6-script-03--build_tiles).

## D5. Frozen encoder + classical classifiers
- **Proposal (§4.2.1):** frozen encoder, train a decoder and classification head.
- **Now:** the task is per parcel, so the frozen encoder gives one embedding per parcel and the "head" is a
  classifier (linear probe and three others). Fine-tuning (LoRA, RQ2) follows as planned.
- **Why:** tests what the pretrained representation already contains, at equal cost for every model; the linear
  probe is the standard test of frozen features.

## D6. Four classifiers instead of one Random Forest
- **Proposal:** Random Forest baseline.
- **Now:** logistic regression, random forest, gradient boosting and SVM on every input (RF stays the headline).
- **Why:** a classifier that suits hand-built features (RF) may not suit dense embeddings (E01: RF is the weakest
  classifier on embeddings). Reporting all four keeps the comparison from depending on one choice.

## D7. Macro-F1 as the headline metric
- **Proposal (§4.2.4):** OA, precision, recall, F1; best model by mean F1 over the five folds, SD as consistency.
- **Now:** macro-F1 of all out-of-fold predictions with a 95% block-bootstrap CI as headline; per-fold mean ± SD is
  reported too, as the proposal asked.
- **Why:** with 86% irrigated parcels, overall accuracy and irrigated F1 are high for any model; macro-F1 weights the
  rainfed class equally. Paired bootstrap differences say whether a gap is real.

## D8. Two temporal aggregation pipelines
- **Proposal:** not specified.
- **Now:** PCA pipeline (concat 12 months → PCA 128) in E01 / E02; L-TAE pipeline in E03.
- **Why:** PANGAEA (Marsocci et al. 2024) found L-TAE better than linear mapping for single-date encoders; testing
  both shows whether the conclusion depends on how the months are combined.
- **Where:** [protocol §5](03_protocol.md#5-temporal-aggregation-two-pipelines).

## D9. Cloud Score+ instead of QA60
- **Now:** Sentinel-2 pixels kept when Cloud Score+ cs_cdf ≥ 0.60; scenes with up to 60% cloud admitted.
- **Why:** the QA60 band misses thin cloud and shadow; Cloud Score+ is the recommended mask in Earth Engine
  (Pasquarella et al. 2023).

## D10. Transfer region under review
- **Proposal (§3.2):** zero-shot transfer to Tunisia, validated on 54 GMIA-NEXT points (44 irrigated, 10 rainfed).
- **Now:** Morocco first. As an alternative, US state datasets with both irrigated and dryland fields (eastern
  Washington WSDA, Utah WRLU 2017–2025) would allow a spatial and a multi-year test with the same pipeline.
- **Why:** 10 rainfed points are too few to measure a rainfed F1; the US data needs a check of its labels
  (see the [progress log, 4–5 Nov 2026](../PROGRESS.md#change-log-newest-first)).
- **Decision due:** after RQ1 and RQ2, with the supervisors.

## D11. Two kinds of head: parcel benchmark and mapping model
- **Proposal (§4.2.1, §4.3):** frozen encoder with a decoder / head; annual irrigated maps 2021–2025 for Morocco.
- **Problem:** the RQ1 pipeline pools the embeddings over the **parcel polygon**. Annual maps of Morocco (or of a
  transfer region) have no polygons, and the GeoFM patches (80 m Clay, 160 m the others) are too coarse for parcels of
  0.83 ha (median, about 90 m across).
- **Now:**
  1. **RQ1 benchmark (unchanged):** frozen encoder → parcel embedding → LogReg (linear probe), RF, HGB, SVM; plus
     L-TAE + MLP head (E03). Classical heads on frozen embeddings are common practice (Presto; WorldCereal; AlphaEarth
     irrigated mapping, Yang et al. 2026) and the head matters little in recent comparisons (Mughees et al. 2026).
  2. **Mapping models (RQ2–RQ4):** frozen GeoFM + **UPerNet segmentation head** trained on the label pixels of the
     tiles (as in PANGAEA and Prithvi-EO-2.0), then LoRA with the same head (RQ2). Classical baseline: RF on the S1/S2
     features of each pixel.
  3. **Neural baselines from scratch:** **TempCNN** (per-pixel time series; Pelletier et al. 2019) and **U-TAE**
     (tiles; Garnot & Landrieu 2021). Both produce 10 m maps. PSE-TAE needs parcel polygons and stays a parcel-level
     extra (E04).
  4. **Same scoring:** pixel predictions are aggregated per parcel (mean probability) and scored with the same
     folds, parcels and bootstrap as RQ1, so every number stays comparable.
- **Why:** the thesis must end in wall-to-wall annual maps; a 1D-CNN beat RF for irrigated mapping
  (Bazzi et al. 2019) and TempCNN-type models transferred better across regions than GeoFM embeddings
  (Tong & Wang 2025), so a neural baseline is needed next to the classical one.
- **Where:** [literature notes](research/README.md#heads-and-mapping-what-geofm-crop-studies-use).

---
← [3 · Evaluation protocol](03_protocol.md) · Next: [5 · Results →](05_results.md)
