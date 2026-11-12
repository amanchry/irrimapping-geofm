# PASTIS and U-TAE · Garnot & Landrieu (2021)

!!! quote "Citation"
    Garnot, V. S. F., & Landrieu, L. (2021). Panoptic segmentation of satellite image time series with convolutional
    temporal attention networks. *ICCV 2021*. [arXiv:2107.07933](https://arxiv.org/abs/2107.07933)

**In one line:** the reference benchmark for crop mapping from Sentinel-2 time series, with a spatio-temporal model
(U-TAE) that segments whole image tiles.

## What they did
- **PASTIS dataset:** Sentinel-2 time series of agricultural parcels in France on **128 × 128 px** patches, with
  crop-type labels per pixel and per parcel; the patches are split into **5 folds**.
- **Fold rotation:** in each round 3 folds train, 1 fold validates, 1 fold tests; all five rotations are reported.
- **U-TAE:** a U-Net whose temporal dimension is collapsed by temporal self-attention (L-TAE) at the lowest
  resolution; the attention masks are reused at higher resolutions. Semantic and panoptic segmentation.

## Key results
- U-TAE set the state of the art for semantic segmentation of crop types on PASTIS; the panoptic head separates
  individual parcels.

## What this research takes from it
- The **PASTIS rotation** (3 train / 1 validation / 1 test) for every model
  ([decision D3](../decisions.md#d3-validation-pastis-rotation-instead-of-nested-cv),
  [protocol §2](../phase2_models/evaluation.md#2-spatial-splits)).
- **Non-overlapping 128 px tiles** as the shared image store ([dataset pipeline](../phase1_data/morocco.md)).
- **U-TAE as a neural baseline** for wall-to-wall maps ([decision D11](../decisions.md#d11-two-kinds-of-head-parcel-benchmark-and-mapping-model)).

## Differences and limitations
- Dense labels in France, many crop classes; this research has sparse parcel labels and two classes (irrigated,
  rainfed). Our folds are 20 km spatial blocks.
