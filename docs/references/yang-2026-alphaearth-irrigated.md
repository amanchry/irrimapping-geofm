# AlphaEarth for irrigated mapping · Yang et al. (2026)

!!! quote "Citation"
    Yang, L., Gao, Y., Zhao, X., Liang, N., Ma, R., Xi, S., Zhang, X., & Wang, R. (2026). Evaluating the performance
    of AlphaEarth Foundation embeddings for irrigated cropland mapping across regions and years. *Remote Sensing*,
    18(7), 1065. [doi:10.3390/rs18071065](https://doi.org/10.3390/rs18071065)

**In one line:** the closest study to this research — foundation-model embeddings for irrigated vs non-irrigated
cropland, compared with Sentinel features and tested across years and regions.

## What they did
- **Regions:** Guanzhong Plain, China (main) and Kansas, USA (independent site); years 2022 and 2024, cross-year test
  on 2023.
- **Inputs:** [AlphaEarth](https://arxiv.org/abs/2507.22291) embeddings vs Sentinel-1/2 features.
- **Classifiers on the embeddings:** K-means, decision tree, random forest, gradient boosting (GBDT), linear SVM,
  RBF SVM, MLP.
- Feature separability by Jeffries–Matusita distance.

## Key results
- Embeddings separate the classes better than Sentinel features (best embedding dimension JM 1.58 vs NDWI 0.72).
- Overall accuracy about **0.95** in the Guanzhong Plain and **0.93** in Kansas; under GBDT, embeddings 0.95 vs
  Sentinel features 0.84. The paper's abstract, text and tables name different best classifiers (RF, GBDT, DT /
  RBF-SVM), so the "best head" is not clear-cut.
- Unsupervised K-means already reaches 0.85 / 0.81.
- **Across years** (train 2022, apply 2023 and 2024): stable, OA 0.87–0.88.
- **Across regions** (train China, apply Kansas, no retraining): collapses to OA about 0.2–0.47, mostly missed
  irrigated fields — attributed to different irrigation regimes, phenology and management.

## What this research takes from it
- Several classifier heads on frozen embeddings, as in our parcel benchmark ([E01](../experiments/E01_frozen_pca_nested/report.md)).
- A warning for RQ4: temporal transfer can be stable while spatial transfer fails; the transfer region must be close
  in irrigation practice ([decision D10](../decisions.md#d10-transfer-region-under-review)).

## Differences and limitations
- Two temperate regions and two to three years only; the contribution of each AlphaEarth input source cannot be separated.
- Pixel-level task; this research classifies parcels and compares four open GeoFMs against a Sentinel baseline.
