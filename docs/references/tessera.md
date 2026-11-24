# TESSERA · Feng et al. (2025)

!!! quote "Citation"
    Feng, Z., Atzberger, C., Jaffer, S., Knezevic, J., Sormunen, S., Young, R., Lisaius, M. C., Immitzer, M.,
    Jackson, T., Ball, J., Coomes, D. A., Madhavapeddy, A., Blake, A., & Keshav, S. (2025). TESSERA: temporal
    embeddings of surface spectra for Earth representation and analysis. [arXiv:2506.20380](https://arxiv.org/abs/2506.20380)
    (summary based on v7, April 2026). Code, weights and embeddings: [github.com/ucam-eo/tessera](https://github.com/ucam-eo/tessera)

**In one line:** a small per-pixel foundation model that turns a year of Sentinel-1 and Sentinel-2 observations into
one 128-d embedding per 10 m pixel, trained so that the embedding does not depend on which cloud-free dates were
available; the embeddings are released for the whole globe.

## What they did

- **Model.** A per-pixel model that reads the valid (cloud-free) Sentinel-1 and Sentinel-2 observations of one year
  without compositing, each with its day of year, and returns one 128-d embedding per 10 m pixel. It is trained
  self-supervised (Barlow Twins: two random subsets of 40 dates of the same pixel must give the same embedding),
  with global shuffling and mix-up as regularisers, on about 800 M pixels from 3,012 MGRS tiles (2017–2024).
  Architecture, inputs, pretraining and the v2 teacher and students: [TESSERA model page](../foundation_models/tessera.md).
- **Product:** global annual 10 m embeddings, with open weights, code and the `geotessera` Python library to download
  them ("embeddings as data").
- **Evaluation:** six benchmarks with Sentinel-1/2 time series (tree species, PASTIS-R parcels, Austrian crop
  classification and segmentation, biomass, canopy height), at 1%, 30% and 100% of the labels. TESSERA, Presto and
  AlphaEarth embeddings stay frozen with a small head (2-layer MLP per pixel, or a small U-Net for patches); the
  patch-based foundation models (Prithvi, CROMA, DOFA and others) get a UPerNet head and are scored both frozen and
  fine-tuned.

## Key results

- **Crop classification (Austrian Crop, per pixel):** F1 66.15 with 1% of the labels, against 37.22 for AlphaEarth
  and 32.74 for Presto; about 0.5 F1 with only 4 labelled samples per class.
- **Crop segmentation (Austrian Crop):** mIoU 53.12, against 25.70 for AlphaEarth and 34.04 for Presto.
- **PASTIS-R parcels:** mIoU 50.68 with all labels, second after AlphaEarth (51.08); ahead of it at 1% (27.54 vs
  27.12), and well ahead of the fine-tuned patch models at 1%.
- **Tree species and regression:** best F1 on TreeSatAI-TS (77.96 vs 76.90 for AlphaEarth); lowest RMSE for biomass
  and canopy height.
- **Ablations** (Austrian Crop, validation F1 77.3 for the full model):
    - without pretraining (random weights): 43.8 (−33.5);
    - without shuffling −9.2, without mix-up −11.1, without both −14.7;
    - without Sentinel-1: −3.1;
    - without int8 quantisation: +0.6 (quantisation costs little).
- **Fine-tuning the encoder gave no gain** over the frozen embeddings with a head, for three learning-rate strategies.
- **Clouds:** scores hold until a pixel has fewer than about 20 valid Sentinel-2 observations per year and fall
  sharply at 10 or fewer.
- A model retrained on region-specific data was not better than the global model.

## What this research takes from it

- **An encoder of [E04](../experiments/E04_frozen_geofm.md):** TESSERA is one of the two pixel time-series models
  (with Presto). Its 128-d embedding of every pixel of a tile goes to the same UPerNet head as the image models
  ([Foundation models](../foundation_models/tessera.md)).
- **Evidence for the frozen-encoder question (RQ2, [E05](../experiments/E05_prithvi_strategies.md)):** for a pixel
  time-series model the authors found no benefit from fine-tuning, while pretraining itself was worth 33.5 F1 points.
  This is the opposite of the Prithvi result of
  [Tomotaki-Dawoud et al. (2025)](tomotaki-dawoud-2025-prithvi-finetuning.md); E05 tests the question for Prithvi on
  the irrigated / rainfed task.
- **Label efficiency:** the largest gains over other models were at 1% of the labels, which matters for regions with
  few surveyed parcels.

## Differences and limitations

- **Model version:** this research uses **TESSERA v2** (Medium student, 21 M parameters, distilled from a 2 B teacher;
  Feng et al. 2026, arXiv:2607.03949), not the v1 model of this paper. The ideas (per-pixel sets of dated observations,
  128-d embedding) are the same; sizes and training differ.
- **Input mismatch:** TESSERA is trained on many single acquisitions per year (40 drawn). Here it receives at most 12
  monthly composites and, as every model, Sentinel-2 only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)).
  The paper shows scores falling below about 20 valid Sentinel-2 observations, so the monthly input may cost accuracy.
- **No spatial context in the embedding:** each pixel is encoded alone; spatial patterns must come from the head
  (here UPerNet).
- **Evaluation design:** the benchmarks use their own splits, mostly in Europe; no irrigated / rainfed task and no
  block-bootstrap confidence intervals. Most numbers are single runs.
- The text describes the temporal pooling as a GRU in the main paper and as attention pooling in the supplementary
  material.
