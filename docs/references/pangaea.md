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
- The set-up of [E04](../experiments/E04_frozen_geofm.md), as in PANGAEA: the frozen encoder encodes the months, an
  **L-TAE** combines them at every patch position (for the encoders that do not model time themselves), a 4-scale
  feature pyramid and a **UPerNet** decoder predict every pixel of the 128 px tile.
- A result where GeoFMs only equal a strong baseline is a plausible outcome, not a failure; E04 is compared with
  the Sentinel-2 baselines E01, E02 and the U-TAE trained from scratch (E03).
- Archived parcel-wise set-up: the L-TAE pipeline next to the PCA pipeline
  ([archived E04](../archive/parcel_wise_classification/E04_frozen_ltae/index.md),
  [decision D8](../decisions.md#d8-two-temporal-aggregation-pipelines)).

## Differences and limitations
- Dense segmentation benchmarks; here the labels are sparse (labelled pixels inside surveyed parcels and
  not-cropland pixels), so the loss is computed on the labelled pixels only.
