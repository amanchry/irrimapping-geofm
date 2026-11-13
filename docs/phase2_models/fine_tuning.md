# RQ2 · Fine-tuning and mapping model

!!! note "Starter page"
    Design as planned; extended when the work begins (planned November 2026 – January 2027).

**Research question RQ2:** to what extent does parameter-efficient fine-tuning improve on a frozen encoder?

## Planned design

The frozen-encoder benchmark of RQ1 pools embeddings over parcel polygons and cannot produce wall-to-wall maps. RQ2
therefore introduces a pixel-level model that serves both as the fine-tuning experiment and as the mapping model of
[Phase 5](../phase5_maps/index.md) ([decision D11](../decisions.md#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model)).

| Component | Description |
| --- | --- |
| Frozen encoder + segmentation head | each GeoFM with a UPerNet head trained on the labelled pixels of the tiles, as in PANGAEA and Prithvi-EO-2.0 |
| Parameter-efficient fine-tuning | low-rank adaptation (LoRA) of the encoder with the same head |
| Full fine-tuning (optional) | all encoder weights trained, for the best model only |
| Classical baseline | random forest on the per-pixel Sentinel-1/2 features |
| Neural baselines | TempCNN (per-pixel time series) and U-TAE (image time series), trained from scratch |

Training, tuning and testing follow the [evaluation protocol](evaluation.md): the same spatial folds and the PASTIS
rotation. Pixel predictions are additionally aggregated to parcels, so that the results are directly comparable with
RQ1.

## To be specified

- loss and class weighting for the 14% rainfed share;
- LoRA rank and target layers per model;
- pixel-level and parcel-level metrics;
- experiments (to be numbered from E03).
