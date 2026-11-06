# Phase 5 · Annual irrigated-cropland maps

!!! note "Starter page"
    Planned for March 2027. This page will be extended when the mapping work begins.

**Aim:** produce annual 10 m maps of irrigated and rainfed cropland for Morocco (2021–2025) and for the transfer
region, using the best mapping model of Phases 2–4, and assess them against independent data.

## Planned workflow

| Step | Description |
| --- | --- |
| Imagery | monthly Sentinel-1/2 composites per agricultural year, with the Phase 1 tile pipeline ([component 1c](../phase1_data/overview.md#components)) |
| Cropland mask | restriction of the maps to cropland (WorldCereal, GLAD cropland or ESA WorldCover; see [Data sources](../data_source/index.md#auxiliary-and-reference-maps)) |
| Inference | the pixel-level mapping model of [RQ2](../phase2_models/fine_tuning.md), applied tile by tile |
| Accuracy assessment | independent reference points (GMIA-NEXT) and held-out parcels; area estimates with confidence intervals |
| Comparison | national statistics (FAO AQUASTAT) and existing products (LGRIP30, GMIE, WorldCereal) |


## Outputs

- annual irrigated-cropland maps, Morocco, 2021–2025, 10 m;
- maps of the transfer region (RQ4);
- accuracy and area tables per year and region.
