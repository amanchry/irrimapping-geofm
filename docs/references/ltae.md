# L-TAE · Garnot & Landrieu (2020)

!!! quote "Citation"
    Garnot, V. S. F., & Landrieu, L. (2020). Lightweight temporal self-attention for classifying satellite images
    time series. *AALTD workshop, ECML-PKDD 2020*. [arXiv:2007.00586](https://arxiv.org/abs/2007.00586)

**In one line:** a small temporal-attention module that learns which dates of a time series matter, and collapses
the series into one vector.

## What they did
- Starts from the temporal attention encoder (TAE) of PSE-TAE and makes it lighter: each attention head has a single
  learnt **master query** instead of a query computed from the sequence; channels are split between heads.
- A positional encoding of the acquisition date keeps the order of the observations.
- Evaluated on parcel-level crop-type classification from Sentinel-2 time series.

## Key results
- Matches or beats the heavier attention and recurrent models with far fewer parameters and less compute.

## What this research takes from it
- A **pixel L-TAE** (attention over the 12 monthly Sentinel-2 values of one pixel, trained from scratch) was run in
  [E02](../experiments/E02_temporal_nn.md); it is on hold and not reported.
- The **temporal head** of the tile models: the L-TAE combines the 12 monthly features at every patch position of
  the encoders that do not merge all 12 months themselves (Prithvi, Clay, TerraMind), before the UPerNet decoder ([E04](../experiments/E04_frozen_geofm.md),
  [E05](../experiments/E05_prithvi_strategies.md)); U-TAE ([E03](../experiments/E03_utae.md)) uses the same
  attention inside the network.
- Archived parcel-wise set-up: attention over the 12 monthly parcel embeddings
  ([archived E04](../archive/parcel_wise_classification/E04_frozen_ltae/index.md)).
- Default size from the paper (16 heads, d_k 8, d_model 256, MLP 256–128).
- PANGAEA uses the same module to aggregate dates for single-date GeoFMs ([PANGAEA](pangaea.md)).

## Differences and limitations
- Designed for raw reflectance time series of many dates; here it runs on 12 monthly composites or embeddings.
