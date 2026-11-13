# PANGAEA · Marsocci et al. (2024)

!!! quote "Citation"
    Marsocci, V., et al. (2024). PANGAEA: a global and inclusive benchmark for geospatial foundation models. *arXiv*.
    [arXiv:2412.04204](https://arxiv.org/abs/2412.04204)

**In one line:** a common benchmark that evaluates many geospatial foundation models in the same way, and finds that
they do not consistently beat supervised models trained from scratch.

## What they did
- Many GeoFMs on a range of Earth-observation tasks (segmentation, change detection, regression), including the crop
  datasets PASTIS and a South Sudan crop-type set.
- **Frozen encoders** with the same trainable decoder for every model (**UPerNet**).
- **Time series with single-date models:** each date is encoded separately, then the dates are combined by a
  **linear mapping** or an **L-TAE** ([L-TAE](ltae.md)).
- Supervised baselines trained from scratch (UNet, ViT); label-fraction experiments.

## Key results
- GeoFMs **do not consistently outperform** the supervised baselines.
- They help most when **labels are few**.
- **L-TAE** aggregation generally beats the linear mapping for multi-temporal tasks.
- Adding SAR to optical input did not always help.

## What this research takes from it
- Frozen-encoder evaluation with an identical head for every model as the main comparison
  ([protocol](../phase2_models/evaluation.md)).
- The **L-TAE pipeline** next to the PCA pipeline ([E04](../experiments/E04_frozen_ltae/index.md),
  [decision D8](../decisions.md#d8-two-temporal-aggregation-pipelines)).
- A result where GeoFMs only equal a strong baseline is a plausible outcome, not a failure ([E01](../experiments/E01_frozen_pca_nested/report.md)).
- For maps: a segmentation head (UPerNet) on the frozen encoder ([decision D11](../decisions.md#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model)).

## Differences and limitations
- Pixel-level segmentation benchmarks; this research classifies parcels (one label per field), so the "head" is a
  classifier on a pooled parcel embedding.
