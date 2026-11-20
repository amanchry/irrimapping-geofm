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

- **Pixel time series instead of image patches.** For each pixel the model reads the valid (cloud-free) observations
  of one year, without compositing, so the phenology is kept. Each observation carries its day of year.
    - Sentinel-2: 10 bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12);
    - Sentinel-1: VV and VH backscatter.
- **Architecture** (45.7 M parameters in the encoder):
    - one branch per sensor: linear projection of each observation + learnt day-of-year encoding, 4 transformer
      blocks, then pooling over time into a 128-d vector;
    - the Sentinel-1 and Sentinel-2 vectors are concatenated and fused by a 2-layer MLP into the final **128-d
      embedding**, stored as **int8** (quantisation-aware training, about 4× less storage);
    - 40 observations per sensor are drawn from the year (with replacement if fewer are valid).
- **Self-supervised training (Barlow Twins):** two "views" of the same pixel are two independent random draws of 40
  valid dates. The loss makes their embeddings agree (invariance) while keeping the embedding dimensions
  uncorrelated (redundancy reduction). So the embedding should describe the pixel's land surface, not the particular
  dates that happened to be cloud-free. Two regularisers:
    - **global shuffling:** pixels from all tiles are mixed before batching, so a batch does not hold neighbouring,
      near-identical pixels;
    - **mix-up:** mixed views of two pixels must give the matching mix of their embeddings.
- **Pretraining data:** about 800 million pixels from 3,012 MGRS tiles worldwide, 2017–2024; one epoch, batch
  32,768, 16 AMD MI300X GPUs.
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

- **An encoder of [E03](../experiments/E03_frozen_geofm.md):** TESSERA is one of the two pixel time-series models
  (with Presto). Its 128-d embedding of every pixel of a tile goes to the same UPerNet head as the image models
  ([Foundation models](../foundation_models.md#tessera-v2)).
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
