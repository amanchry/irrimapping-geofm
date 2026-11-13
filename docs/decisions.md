# Decisions and departures from the proposal

This page records every departure of the implementation from the [research proposal](assets/research_proposal.pdf),
stating the original plan, the current approach and the rationale. Smaller method choices are in the
[progress log](progress.md#method-decisions).

| # | Topic | Proposal | Now | Since |
| --- | --- | --- | --- | --- |
| [D1](#d1-spatial-folds-instead-of-a-7030-split) | Train / test split | 70 / 30 stratified + validation subset | 5 spatial folds of 20 km blocks | Oct 2026 |
| [D2](#d2-two-agricultural-years-per-parcel-season) | Imagery period | 2024 annual stack | each parcel's own agricultural year (2023/24 or 2024/25), Sep – Aug | Oct 2026 |
| [D3](#d3-validation-pastis-rotation-instead-of-nested-cv) | Validation design | validation subset from the training data | PASTIS rotation: 3 train / 1 validation / 1 test | 4 Nov 2026 |
| [D4](#d4-one-shared-tile-store-for-every-input) | Image preparation | annual stacks via STAC / xarray | one tile store from Google Earth Engine for every input | 27 Oct 2026 |
| [D5](#d5-frozen-encoder--classical-classifiers) | Frozen-encoder strategy | frozen encoder + decoder / head | frozen encoder → parcel embedding → four classifiers | Oct 2026 |
| [D6](#d6-four-classifiers-instead-of-one-random-forest) | Baseline classifier | Random Forest | three classifiers (LogReg, RF, SVM) and an MLP head on every input; gradient boosting dropped (12 Nov 2026) | Oct 2026 |
| [D7](#d7-macro-f1-as-the-headline-metric) | Selection metric | mean F1 over folds ± SD | macro-F1 of the out-of-fold predictions, 95% block-bootstrap CI, per-fold mean ± SD | Oct 2026 |
| [D8](#d8-two-temporal-aggregation-pipelines) | Combining the months | not specified | PCA pipeline now; L-TAE pipeline next | 4 Nov 2026 |
| [D9](#d9-cloud-score-instead-of-qa60) | Cloud mask | cloud-masked composites | Cloud Score+ (cs_cdf ≥ 0.60) | Oct 2026 |
| [D10](#d10-transfer-region-under-review) | Transfer region | Tunisia, 54 GMIA-NEXT points | replaced by D13: no transfer region; Utah as a second study area | 12 Nov 2026 |
| [D11](#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model) | Classifier head and mapping | frozen encoder + head; maps 2021–2025 | parcel benchmark with classical heads (RQ1) + segmentation head and pixel models for the maps (RQ2–RQ4); neural baselines TempCNN and U-TAE | 6 Nov 2026 |
| [D12](#d12-agrifm-with-its-pretrained-4-px-patch-embedding) | AgriFM encoder | not specified | pretrained 4 × 4 px patch embedding (output stride 32 px); all weights checked on loading | 6 Nov 2026 |
| [D13](#d13-utah-as-a-second-independent-study-area) | RQ3 / RQ4: second region and temporal test | zero-shot transfer of the Morocco model to Tunisia (RQ4); temporal test within Morocco (RQ3) | Utah (WRLU) as an independent second study area, no transfer; temporal test in Utah: train on WRLU 2024, test on WRLU 2021 on held-out folds | 12 Nov 2026 |
| [D14](#d14-sentinel-2-only-for-every-input) | Input data of the models | Sentinel-1 and Sentinel-2 | Sentinel-2 only for every GeoFM and the baselines; Sentinel-1 as an optional ablation (Morocco) | 12 Nov 2026 |
| [D15](#d15-agrifm-normalised-with-the-statistics-of-the-regions-chips) | AgriFM normalisation | not specified | per-band mean / std of the region's own chips; config statistics as sensitivity run | 12 Nov 2026 |

---

## D1. Spatial folds instead of a 70/30 split
- **Proposal (§4.1):** stratified 70 / 30 split, validation subset from the training data. (§4.2.4 already asks for
  spatial cross-validation.)
- **Now:** parcels grouped into 20 km blocks, blocks into 5 folds with equal class balance; every parcel tested
  once by a model that never saw its block.
- **Why:** with a random split, 79% of test parcels lie within 250 m of a training parcel (about 2% with 20 km
  blocks). Random splits overstate accuracy for mapping new areas (Roberts et al. 2017; Ploton et al. 2020).
- **Where:** [dataset §5](phase1_data/morocco.md#5-script-02--make_spatial_folds), [protocol §2](phase2_models/evaluation.md#2-spatial-splits).

## D2. Two agricultural years (per-parcel season)
- **Proposal (§4.1):** the 2024 annual image stack.
- **Now:** each parcel gets the 12 months (1 Sep – 31 Aug) of the season in which it was surveyed: 2023/24 or
  2024/25. Both seasons in one dataset, months aligned (month 1 = September).
- **Why:** the survey ran Dec 2023 – Jan 2025; using only 2024 would mix seasons, and dropping 2024/25 would lose 42%
  of the rainfed parcels. The agricultural year covers the winter rainfed cycle and the summer, when only irrigated
  fields stay green.
- **Consequence:** region and year are partly confounded (Gharb and Tadla only 2023/24, Souss only 2024/25).
- **Where:** [dataset §6.2](phase1_data/morocco.md#62-two-agricultural-years).

## D3. Validation: PASTIS rotation instead of nested CV
- **Before (E01):** nested spatial CV, settings by inner CV on the 4 training folds, refit on 4.
- **Now:** per round 3 folds train, the next fold chooses the settings (or stops training), 1 fold tests, as in
  PASTIS (Garnot & Landrieu 2021).
- **Why:** one design for classical classifiers and neural networks (L-TAE, fine-tuning), which need a separate
  validation fold for early stopping. It is the standard design of the PASTIS crop-mapping benchmark.
- **Where:** [protocol §2](phase2_models/evaluation.md#2-spatial-splits); E01 is kept as the sensitivity check.

## D4. One shared tile store for every input
- **Proposal:** annual Sentinel-1/2 composites via STAC and xarray.
- **Now:** Google Earth Engine builds non-overlapping 128 × 128 px tiles (1.28 km) with 12 monthly composites × 19
  bands; the baseline features and the foundation-model chips are all cut from these tiles.
- **Why:** STAC processing was too slow for the full set; with one store every input sees identical pixels,
  compositing and masks, and chips no longer overlap across folds.
- **Where:** [dataset §6–7](phase1_data/morocco.md#6-script-03--build_tiles).

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
- **Update (12 Nov 2026):** from E02 on, three classifiers: logistic regression (linear probe), random forest
  (standard remote-sensing classifier, the proposal's baseline) and SVM (strongest in E01), plus a trainable **MLP
  head** (E03, E04). Gradient boosting is dropped: in E01 it was never the best classifier and always within the
  confidence interval of SVM. Plan: [RQ1 · Experiment plan](phase2_models/experiment_plan.md).

## D7. Macro-F1 as the headline metric
- **Proposal (§4.2.4):** OA, precision, recall, F1; best model by mean F1 over the five folds, SD as consistency.
- **Now:** macro-F1 of all out-of-fold predictions with a 95% block-bootstrap CI as headline; per-fold mean ± SD is
  reported too, as the proposal asked.
- **Why:** with 86% irrigated parcels, overall accuracy and irrigated F1 are high for any model; macro-F1 weights the
  rainfed class equally. Paired bootstrap differences say whether a gap is real.

## D8. Two temporal aggregation pipelines
- **Proposal:** not specified.
- **Now:** PCA pipeline (concat 12 months → PCA 128) in E01 / E02; L-TAE pipeline in E04.
- **Why:** PANGAEA (Marsocci et al. 2024) found L-TAE better than linear mapping for single-date encoders; testing
  both shows whether the conclusion depends on how the months are combined.
- **Where:** [protocol §5](phase2_models/evaluation.md#5-temporal-aggregation-two-pipelines).

## D9. Cloud Score+ instead of QA60
- **Now:** Sentinel-2 pixels kept when Cloud Score+ cs_cdf ≥ 0.60; scenes with up to 60% cloud admitted.
- **Why:** the QA60 band misses thin cloud and shadow; Cloud Score+ is the recommended mask in Earth Engine
  (Pasquarella et al. 2023).

## D10. Transfer region under review
- **Proposal (§3.2):** zero-shot transfer to Tunisia, validated on 54 GMIA-NEXT points (44 irrigated, 10 rainfed).
- **Now:** Morocco first. As an alternative, US state datasets with both irrigated and dryland fields (eastern
  Washington WSDA, Utah WRLU 2017–2025) would allow a spatial and a multi-year test with the same pipeline.
- **Why:** 10 rainfed points are too few to measure a rainfed F1; the US data needs a check of its labels
  (see the [progress log, 4–5 Nov 2026](progress.md#change-log-newest-first)).
- **Decided (12 Nov 2026): Utah (WRLU) as the second region.** Explicit rainfed class ("Dry Crop"), 13,000–17,000
  rainfed crop fields per year, every year 2017–2025 with one method, one UTM zone. Eastern Washington was the
  fallback: most of its dryland-wheat fields come from the NASS Cropland Data Layer (irrigation not observed).
- **Superseded the same day by [D13](#d13-utah-as-a-second-independent-study-area):** Utah is not a transfer target
  but an independent second study area.

## D11. Two kinds of head: parcel benchmark and mapping model
- **Proposal (§4.2.1, §4.3):** frozen encoder with a decoder / head; annual irrigated maps 2021–2025 for Morocco.
- **Problem:** the RQ1 pipeline pools the embeddings over the **parcel polygon**. Annual maps of Morocco (or of a
  transfer region) have no polygons, and the GeoFM patches (80 m Clay, 160 m the others) are too coarse for parcels of
  0.83 ha (median, about 90 m across).
- **Now:**
  1. **RQ1 benchmark (unchanged):** frozen encoder → parcel embedding → LogReg (linear probe), RF, HGB, SVM; plus
     L-TAE + MLP head (E04). Classical heads on frozen embeddings are common practice (Presto; WorldCereal; AlphaEarth
     irrigated mapping, Yang et al. 2026) and the head matters little in recent comparisons (Mughees et al. 2026).
  2. **Mapping models (RQ2–RQ4):** frozen GeoFM + **UPerNet segmentation head** trained on the label pixels of the
     tiles (as in PANGAEA and Prithvi-EO-2.0), then LoRA with the same head (RQ2). Classical baseline: RF on the S1/S2
     features of each pixel.
  3. **Neural baselines from scratch:** **TempCNN** (per-pixel time series; Pelletier et al. 2019) and **U-TAE**
     (tiles; Garnot & Landrieu 2021). Both produce 10 m maps. PSE-TAE needs parcel polygons and stays a parcel-level
     extra (E05).
  4. **Same scoring:** pixel predictions are aggregated per parcel (mean probability) and scored with the same
     folds, parcels and bootstrap as RQ1, so every number stays comparable.
- **Why:** the research must end in wall-to-wall annual maps; a 1D-CNN beat RF for irrigated mapping
  (Bazzi et al. 2019) and TempCNN-type models transferred better across regions than GeoFM embeddings
  (Tong & Wang 2025), so a neural baseline is needed next to the classical one.
- **Where:** [literature notes](references/heads-and-mapping.md).

## D12. AgriFM with its pretrained 4-px patch embedding
- **Before:** `02d_embed_agrifm.py` built the Sentinel-2 encoder with the patch size of the repository's fine-tuning
  configuration (`configs/cropland_mapping.py`: 4 time steps × 2 × 2 px, output stride 16 px) and loaded
  `AgriFM.pth` through mmengine. No weight was loaded: the checkpoint names carry the prefix `encoder.`, and the
  pretrained patch embedding (`S2_patch_emd`, kernel 128 × 10 × 4 × 4 × 4) does not fit a 2 × 2 px patch. mmengine
  only warns about mismatched names and still reports success, so all AgriFM embeddings produced up to 6 Nov 2026
  (local and on the cluster, including E01) came from a randomly initialised encoder.
- **Now:** patch size 4 × 4 × 4 (the size of the pretrained patch embedding), output stride 32 px (320 m; 2 × 2
  cells per 64-px chip). The weights are mapped by name in the script and every encoder tensor must load; otherwise
  the script stops. On the cluster 349 of 349 encoder tensors load.
- **Why:** the frozen-encoder comparison (D5) must use each encoder as pretrained. In the fine-tuning set-up of the
  repository the 2 × 2 px patch embedding is trained anew; with a frozen encoder it would stay random. The alternative,
  resizing the 4 × 4 px kernel to 2 × 2 px, keeps a finer grid but replaces the pretrained first layer by an
  approximation.
- **Consequence:** AgriFM has the coarsest grid of the four models (320 m against 80–160 m), which matters for small
  parcels; the patch-share weighting is unchanged. Earlier AgriFM results are void and are re-run.
- **Where:** [`models_run/02d_embed_agrifm.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/models_run/02d_embed_agrifm.py),
  [Foundation models · inputs](foundation_models.md).

## D13. Utah as a second independent study area
- **Proposal (§3.2, RQ3, RQ4):** RQ4 applies the Morocco-trained model zero-shot to a transfer region (Tunisia,
  54 GMIA-NEXT points); RQ3 tests temporal consistency within Morocco.
- **Now:**
    - **Utah is an independent second study area.** It is trained and tested on its own data (WRLU labels), with
      the same pipeline, foundation models, classifiers and evaluation as Morocco; the two regions are reported side
      by side. There is no Morocco → Utah transfer.
    - **Temporal test in Utah:** train on WRLU 2024 (`2023/24`), test on WRLU 2021 (`2020/21`) on the folds held out
      in training (same fields in both years, paired comparison). See
      [Utah training data, section 10](phase1_data/utah.md#10-temporal-test-design).
    - **Years:** WRLU 2024 (training) and 2021 (test). Between them 36% of the crops change but only 0.07% of the
      fields change class; 2025 repeats 94% of the 2024 crops and is not an independent year; 2019 shows a sudden
      jump in class changes (1.35%) that may be a revision of the mapping.
- **Why:**
    - A Morocco → Utah transfer mixes a change of region, climate, crop calendar (Utah is 2–3 months later), label
      source and field size; a low score would not say which of these failed. Two independent studies show whether
      the conclusions on the foundation models hold in a second region.
    - Morocco cannot offer a clean temporal test (the surveyed regions differ between the two years); in Utah the
      same places are mapped every year.
    - Tunisia has 10 rainfed reference points, too few for a rainfed F1 (D10).
- **Consequence:** RQ4 changes from "spatial transferability of the Morocco model" to "do the results hold in a
  second region"; RQ3 is answered in Utah. To be agreed with the supervisors.
- **Where:** [Utah training data](phase1_data/utah.md), `ml_dataset_prep/01_prepare_labels_Utah.py`.

## D14. Sentinel-2 only for every input
- **Proposal:** Sentinel-1 and Sentinel-2 time series.
- **Before:** the baseline used Sentinel-2 + Sentinel-1 features (one orbit), TerraMind received Sentinel-2 +
  Sentinel-1, the other three GeoFMs Sentinel-2 only.
- **Now:** every input of the model comparison uses **Sentinel-2 only**: the four GeoFMs (TerraMind with
  `MODALITIES = ["S2L2A"]`), the feature baseline (`S1S2_VARIABLES = "s2"`) and PSE-TAE (`BANDS = "s2"`). No model is
  given an input it was not pretrained on, and no fusion of embeddings with radar features is designed for the
  optical-only models.
- **Why:**
    - Every model sees the same optical data, so a difference comes from the representation and the head, not from
      one model having radar.
    - Morocco and Utah are treated alike.
    - Sentinel-1 coverage of Utah changed between the years: after Sentinel-1B failed (December 2021) only
      Sentinel-1A covered the area in 2023/24 (only the descending orbit was usable), while the 2020/21 test year had
      both satellites. With radar, the 2024 → 2021 temporal test (D13) would partly measure a change in radar coverage.
- **Consequence:** the multimodal pretraining of TerraMind is not used in the main comparison. An optional ablation
  in Morocco (both orbits available) adds Sentinel-1 to TerraMind (`terramind-s1s2`) and to the baseline
  (`S1S2_VARIABLES = "all"`, PSE-TAE `morocco_s2s1`). E01 (done before this decision) used Sentinel-1 in the baseline
  and in TerraMind and is kept as it is.
- **Where:** [Foundation models · Configuration](foundation_models.md#configuration-in-this-study),
  [Models pipeline](phase2_models/frozen_encoders.md).

## D15. AgriFM normalised with the statistics of the region's chips
- **Before:** AgriFM inputs were normalised with the statistics of the repository's fine-tuning example
  (`configs/cropland_mapping.py`).
- **Problem:** neither the AgriFM paper (Li et al. 2025) nor the repository documents the normalisation of the
  pretraining data. The config statistics (blue 4,179 ± 4,042) describe raw, partly cloudy time series; the
  cloud-masked monthly composites of this study have blue 714 ± 344 (Morocco). With the config statistics the
  visible and NIR bands use only 8–30% of the expected spread and sit about 0.8 SD below zero, so the frozen encoder
  sees an almost constant input in these bands.
- **Now:** per-band mean and standard deviation of the region's own chips (`NORM = "data"` in `02d_embed_agrifm.py`):
  all valid pixels of 20 evenly spaced chip files, every 4th pixel; computed from imagery only (no labels), saved to
  `agrifm_norm_stats_<tag>.json` and reused. For the Utah temporal test the 2023/24 statistics are reused for 2020/21
  (`NORM_STATS_FILE`), so both years are scaled identically.
- **Why:** standardising each band to zero mean and unit variance is what a z-score normalisation with the pretraining
  statistics does on the pretraining data; when those statistics are unknown, the data's own statistics are the
  closest equivalent.
- **Sensitivity:** the config statistics are kept as a second embedding (`NORM = "config"`, `agrifm-cfgnorm_*.npz`;
  the Morocco file made on 6 Nov 2026 is this version).
- **Where:** [Foundation models · AgriFM](foundation_models.md#agrifm), `models_run/02d_embed_agrifm.py`.
