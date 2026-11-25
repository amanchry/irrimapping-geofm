# Phase 5 · Annual irrigated-cropland maps

!!! note "Starter page"
    Planned for March 2027. This page will be extended when the mapping work begins.

**Aim:** produce annual 10 m maps of irrigated and rainfed cropland for Morocco (2021–2025) and for Utah, the
second study area ([D13](../decisions.md#d13-utah-as-a-second-independent-study-area)), using the best model of
Phases 2–4, and assess them against independent data.

## Planned workflow

| Step | Description |
| --- | --- |
| Imagery | 12 monthly Sentinel-2 composites per agricultural year on the 10.24 km blocks of `data_download/` (Earth Search STAC, [D21](../decisions.md#d21-sentinel-2-and-sentinel-1-from-open-stac-catalogues-earth-engine-only-for-dynamic-world)): every block with at least 164 Dynamic World cropland pixels (1% of a 128 px tile; `02a_dynamic_world.py`, one cropland file per block and year) and every block holding training tiles ([D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them); [Phase 1](../phase1_data/overview.md#components), [Cropland extent §2](../phase1_data/cropland_extent.md#2-blocks-and-downloads)). The earlier STAC download (`s2_download/`, [D18](../decisions.md#d18-wall-to-wall-composites-from-stac-for-the-maps)) was retired. Sentinel-2 only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)) |
| Cropland extent | the **annual Dynamic World cropland** of each agricultural year (`data_download/02a_dynamic_world.py`: "crops" the most frequent Dynamic World class in at least 2 of the 12 months), the layer that also selects the downloaded blocks ([Cropland extent](../phase1_data/cropland_extent.md#1-annual-dynamic-world-cropland)). The experiments classify only the surveyed parcels (irrigated or rainfed) and use no mask ([D19](../decisions.md#d19-two-class-experiments-cropland-extent-in-the-mapping-phase)). Before the maps, the layer is checked on its **omission** of the surveyed irrigated and rainfed parcels and on its **commission** (land in the layer that is not cropland, against the reference maps; annual Dynamic World land cover of `data_download/04c_ref_dynamic_world_annual.py`, ESA WorldCover, LGRIP30). The union of four cropland products ([D17](../decisions.md#d17-cropland-mask-from-four-products-and-a-not-cropland-class)) is archived ([Union cropland mask](../archive/cropland_union_mask.md)) |
| Land that is not cropland | **to be decided in this phase:** handled by the Dynamic World cropland alone, or by an extra "not cropland" class added to the mapping model; predictions are kept inside the cropland layer |
| Inference | the best model of the experiments E01–E05 ([Experiments](../experiments/index.md)), applied to the downloaded blocks: each block is classified on its own, on its cropland pixels only (tile models on 128 px windows read from the block in memory, pixel models on each pixel's series), and the block maps are merged at the end; no map tile files are written |
| Accuracy assessment | a stratified random sample over the map (including land outside the cropland layer and not cropland), area-adjusted accuracy and area estimates with confidence intervals (Olofsson et al. 2014; [experiment plan §3](../experiments/plan.md#3-after-the-experiments)); the GMIA-NEXT reference points as an independent comparison |
| Comparison | national statistics (FAO AQUASTAT) and existing products (LGRIP30, GMIE, WorldCereal) |


## Outputs

- annual irrigated-cropland maps, Morocco, 2021–2025, 10 m;
- annual maps of Utah;
- accuracy and area tables per year and region.
