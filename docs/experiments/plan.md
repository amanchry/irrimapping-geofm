# Pixel-wise experiment plan

The experiments compare inputs and models for classifying **every 10 m pixel** as irrigated, rainfed or not
cropland, so that the best model can produce the annual continuous maps (Phase 5). They replace the parcel-wise
experiments E01–E05 ([D16](../decisions.md#d16-pixel-wise-classification-for-continuous-maps); archive:
[results](../archive/parcel_wise_classification/results.md)). Experiments run one at a time on the cluster; the
results are copied to the Mac for the reports.

## 1. Common set-up (identical for every experiment)

| Item | Choice |
| --- | --- |
| **Samples** | the labelled pixels of `04c_make_pixel_samples.py` ([Pixel samples](../phase1_data/pixel_samples.md)): inner pixels of the surveyed parcels (≤ 100 per parcel) and not-cropland pixels (0 votes in the [cropland mask](../phase1_data/cropland_mask.md)) |
| **Classes** | 1 irrigated · 2 rainfed · 3 not cropland ([D17](../decisions.md#d17-cropland-mask-from-four-products-and-a-not-cropland-class)) |
| **Regions and years** | Morocco: 2023/24 and 2024/25 pooled, each pixel with its own year (as in [D2](../decisions.md#d2-two-agricultural-years-per-parcel-season)). Utah: train and test on 2023/24 (WRLU 2024); 2020/21 only for the temporal test (RQ3) |
| **Input imagery** | Sentinel-2 only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)): 12 monthly composites of the agricultural year × 12 bands, from the same tile store for every model |
| **Validation** | PASTIS rotation over the 5 spatial folds (20 km blocks): 3 folds train, the next fold validation (early stopping, hyperparameters), 1 fold test; 5 rounds, so every pixel is tested once. Pixels of fold −1 are not used |
| **Same test pixels** | every experiment is scored on the same out-of-fold pixels. Models that work on whole tiles (P04, P05) predict the tile and are read at the sampled pixels |
| **Class imbalance** | class weights inversely proportional to the training-pixel counts (rainfed ≈ 1 : 11 irrigated in Morocco) |
| **Seeds** | neural models: 3 seeds (42, 1042, 2042), mean probability; classical models: seed 42 |

**Metrics** (all from the out-of-fold predictions):

- **Headline: cropland macro-F1**, the mean F1 of irrigated and rainfed over the parcel pixels. This is the question
  of the study and is comparable across experiments whatever the not-cropland performance.
- **3-class macro-F1**, and per-class F1, precision and recall; confusion matrix.
- **Parcel-level macro-F1**: each parcel gets the majority class of its pixels, which allows comparison with the
  archived parcel-wise results (Morocco 0.91, Utah 0.79).
- 95% confidence intervals by **block bootstrap** over the 20 km blocks; two inputs or models are compared by the
  paired difference on the same pixels; a difference counts when its CI excludes 0 **and** Holm-corrected p < 0.05
  ([evaluation protocol](../phase2_models/evaluation.md)).
- Results per region, per year (Morocco) and per zone.

## 2. Experiments

| ID | Question | Input | Model | Compute | Order |
| --- | --- | --- | --- | --- | --- |
| **P01** | How well do the raw Sentinel-2 pixel time series separate the classes? (baseline) | 12 months × 12 bands per pixel (144 values) + NDVI, NDWI per month | random forest; histogram gradient boosting | CPU | 1 |
| **P02** | Does a temporal neural network beat the classical baseline on the same pixels? | same pixel time series | TempCNN (Pelletier et al. 2019); L-TAE on single pixels (Garnot & Landrieu 2020) | GPU | 2 |
| **P03** | Do precomputed pixel embeddings of a foundation model help? | AlphaEarth annual embedding (64-d, 10 m; Brown et al. 2025) at the pixel's x/y | random forest; MLP | CPU (Earth Engine sampling) | 3 |
| **P04** | Do frozen geospatial foundation models with a light decoder beat the pixel models? (RQ1) | 128 × 128 px tiles, 12 months | frozen Prithvi-EO-2.0, Clay v1.5, TerraMind, AgriFM + UPerNet decoder (PANGAEA; Marsocci et al. 2024), L-TAE over time | GPU | 5 |
| **P05** | Does a spatio-temporal network trained from scratch match the foundation models? | 128 × 128 px tiles, 12 months | U-TAE (Garnot & Landrieu 2021) | GPU | 4 |
| **P06** | Does fine-tuning the best foundation model improve on the frozen encoder? (RQ2) | as P04 | best P04 encoder with LoRA | GPU | 6 |

Run order: P01 first (it fixes the evaluation code and gives the reference numbers), then the other pixel models
(P02, P03), then the tile models (P05 before P04, because U-TAE tests the tile training pipeline without the large
encoders), then P06.

### Details per experiment

**P01 · Sentinel-2 pixel baseline.** The pixel's 12 × 12 band values in reflectance × 10,000, plus NDVI and NDWI per
month; months without a clear observation are filled by linear interpolation in time. Random forest (500 trees,
`class_weight = "balanced"`) and histogram gradient boosting (class-weighted samples), hyperparameters chosen on the
validation fold. Permutation importance per month and band. This answers how far a per-pixel time series alone
goes; random forests are the standard in operational irrigation mapping (Ketchum et al. 2020; Xie et al. 2021).

**P02 · temporal neural networks.** The same 12-step input, normalised per band with training-fold statistics.
TempCNN (three 1-D convolution layers) and L-TAE applied to single pixels; trained with weighted cross-entropy and
early stopping on the validation fold; 3 seeds. Tests whether learnt temporal features beat the random forest.

**P03 · AlphaEarth embeddings.** The 64-d annual embedding sampled in Earth Engine at each pixel's centre
(`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, years 2017–2025). The embedding covers a **calendar year**, while the
agricultural year runs September–August (Morocco) or October–September (Utah): the year in which the agricultural
year ends is used (2023/24 → 2024), and the mismatch is stated as a limitation. Random forest and a small MLP.
Optional: TESSERA embeddings (128-d) if the years are available.

**P04 · frozen foundation models + decoder.** The encoders see whole 128 px tiles (16 px patches for Prithvi and
AgriFM's settings, 8 px for Clay), one month at a time or all months as the model allows; the patch features are
combined over time with L-TAE and brought back to 10 m by a UPerNet decoder, trained with a masked loss on the
labelled pixels of each tile (irrigated, rainfed and the not-cropland samples; all other pixels ignored). Only the
decoder and temporal head are trained. Same normalisations and band choices as in the archived embedding scripts
(D12, D15).

**P05 · U-TAE from scratch.** The PASTIS segmentation baseline: U-Net encoder with L-TAE attention over the 12
months, trained on the same tiles with the same masked loss. Shows how much the pretrained encoders add over a
spatio-temporal network of the same input.

**P06 · fine-tuning.** LoRA adapters on the best P04 encoder, same decoder and data; answers RQ2.

## 3. After the experiments

1. **Temporal test (RQ3), Utah:** the best models trained on 2023/24 and applied to 2020/21 on the test folds.
2. **Maps (Phase 5):** the best model on the wall-to-wall tiles (`03c_build_cropland_tiles.py`), predictions kept
   inside the cropland mask; irrigated and rainfed area per region and year.
3. **Map accuracy:** a stratified random sample over the map (including not-cropland), area-adjusted accuracy and
   area estimates (Olofsson et al. 2014); comparison with LGRIP30 and national statistics.

## 4. Optional ablations

- Sentinel-1 added to P01 / P02 (the tile store has it).
- Fewer labels: 10 / 25 / 50% of the training blocks (as in the archived experiments).
- Two classes only (no not-cropland class): how much the third class costs on irrigated / rainfed.
- Stricter mask (≥ 2 votes) for the maps.

## 5. How an experiment runs

Each experiment gets a folder `experiments/<ID>_<name>/` with a `config.yaml` (input, model, hyperparameters, seeds)
and a page in the documentation. The training scripts read the pixel samples or the tiles on the cluster and write
`results/<region>/` (out-of-fold predictions per pixel, metrics, figures); only `results/` is copied to the Mac, where
the report and the comparison with earlier experiments are made. The scripts are written one experiment at a time,
starting with P01.

## References

- Brown, C. F. et al. (2025). AlphaEarth Foundations: an embedding field model for accurate and efficient global
  mapping from sparse label data. arXiv:2507.22291.
- Garnot, V. S. F. & Landrieu, L. (2020). Lightweight temporal self-attention for classifying satellite image time
  series (L-TAE). *AALTD*.
- Garnot, V. S. F. & Landrieu, L. (2021). Panoptic segmentation of satellite image time series with convolutional
  temporal attention networks (U-TAE, PASTIS). *ICCV*.
- Ketchum, D. et al. (2020). IrrMapper. *Remote Sensing* 12, 2328.
- Marsocci, V. et al. (2024). PANGAEA: a global and inclusive benchmark for geospatial foundation models. arXiv:2412.04204.
- Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change.
  *Remote Sensing of Environment* 148, 42–57.
- Pelletier, C., Webb, G. I. & Petitjean, F. (2019). Temporal convolutional neural network for the classification of
  satellite image time series (TempCNN). *Remote Sensing* 11, 523.
- Xie, Y., Gibbs, H. K. & Lark, T. J. (2021). LANID. *ESSD* 13, 5689–5710.
