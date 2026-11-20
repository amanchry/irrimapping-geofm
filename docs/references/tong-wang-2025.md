# Invariant features for crop classification · Tong & Wang (2025)

!!! quote "Citation"
    Tong, X.-Y., & Wang, S. (2025). Invariant features for global crop type classification. *arXiv*.
    [arXiv:2509.03497](https://arxiv.org/abs/2509.03497)

**In one line:** GeoFM embeddings work for transfer between nearby countries but break down across continents, where
a simple temporal CNN on Sentinel-2 time series transfers better.

## What they did
- Crop-type classification with transfer across countries, continents and hemispheres.
- **Inputs and heads:** Presto (128-d) and AlphaEarth (64-d) per-pixel embeddings with a random forest (500 trees) or
  an 8-layer MLP; harmonic features with RF / MLP; deep models on 5-day median Sentinel-2 series (CropNet with 2D
  spectral-temporal convolutions, a 1D variant, TempCNN, ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny).
- Time-series augmentation (time shift, time scale, magnitude warping).

## Key results
- GeoFM embeddings are competitive when the shift is mild (between countries).
- They **degrade strongly across continents and hemispheres** (e.g. USA → Australia: overall accuracy 45% for
  Presto and 46% for AlphaEarth, against 87% for CropNet).
- Simple 5-day Sentinel-2 series with CropNet beat harmonic features and GeoFM embeddings in every transfer setting;
  augmentation adds further gains. The authors conclude that inductive bias matters more than model scale or
  pretraining.

## What this research takes from it
- A **temporal CNN (TempCNN) as neural baseline** next to the GeoFMs ([E02](../experiments/E02_temporal_nn.md)).
- An expectation for RQ4: in a second region with different practices (Utah,
  [D13](../decisions.md#d13-utah-as-a-second-independent-study-area)) simpler temporal models may compare better
  with the GeoFMs than in Morocco.

## Differences and limitations
- Crop types, not irrigation; pixel-level; different label sources per country.
