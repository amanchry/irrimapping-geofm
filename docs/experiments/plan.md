# Pixel-wise experiment plan

The experiments compare inputs, models and adaptation strategies for classifying the **10 m pixels of the surveyed
parcels as irrigated or rainfed** ([D16](../decisions.md#d16-pixel-wise-classification-for-continuous-maps),
[D19](../decisions.md#d19-two-class-experiments-cropland-extent-in-the-mapping-phase)). The best model later produces
the annual maps (Phase 5), where the cropland extent is decided. All experiments run for **Morocco first**; Utah
follows with the same plan. Experiments run one at a time on the cluster; the results are copied to the Mac for the
reports.

## 1. Common set-up (identical for every experiment)

| Item | Choice |
| --- | --- |
| **Samples** | the labelled pixels of `04_make_pixel_samples.py` ([Pixel samples](../phase1_data/pixel_samples.md)): inner pixels of the surveyed parcels, at most 100 per parcel. Morocco: 473,675 pixels of 8,775 parcels (7,579 irrigated, 1,196 rainfed) in 2,015 tiles |
| **Classes** | 1 irrigated · 2 rainfed (no cropland mask, no not-cropland class; [D19](../decisions.md#d19-two-class-experiments-cropland-extent-in-the-mapping-phase)) |
| **Regions and years** | **Morocco first**: 2023/24 and 2024/25 pooled, each pixel with its own year (as in [D2](../decisions.md#d2-two-agricultural-years-per-parcel-season)). Utah afterwards: train and test on 2023/24 (WRLU 2024); 2020/21 only for the temporal test (RQ3) |
| **Input imagery** | Sentinel-2 only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)): 12 monthly composites of the agricultural year × 12 bands, from the same tile store for every model |
| **Validation** | PASTIS rotation over the 5 spatial folds (20 km blocks; Morocco about 1,500 irrigated and 240 rainfed parcels per fold): 3 folds train, the next fold validation (early stopping, hyperparameters), 1 fold test; 5 rounds, so every pixel is tested once |
| **Same test pixels** | every experiment is scored on the same out-of-fold pixels. Models that work on whole tiles (E03–E05) predict the tile and are read at the sampled pixels |
| **Class imbalance** | class weights inversely proportional to the training-pixel counts (Morocco: rainfed pixels ≈ 1 : 8 irrigated, so a rainfed pixel counts about 8 times more) |
| **Seeds** | neural models: 3 seeds (42, 1042, 2042), mean probability; classical models: seed 42 |

**Metrics** (all from the out-of-fold predictions):

- **Headline: macro-F1**, the mean F1 of irrigated and rainfed, so the minority class (rainfed) counts as much as
  the majority.
- Per-class F1, precision and recall (**rainfed recall**: how many rainfed pixels are found); overall accuracy;
  confusion matrix.
- 95% confidence intervals by **block bootstrap** over the 20 km blocks; two inputs or models are compared by the
  paired difference on the same pixels; a difference counts when its CI excludes 0 **and** Holm-corrected p < 0.05
  ([evaluation protocol](../phase2_models/evaluation.md)).
- Results per region, per year (Morocco) and per zone.

## 2. Experiments

| ID | Question | Input | Model | Compute | Order |
| --- | --- | --- | --- | --- | --- |
| **E01** | How well do the raw Sentinel-2 pixel time series separate irrigated from rainfed? (baseline) | 12 months × 12 bands per pixel (144 values) + NDVI, NDWI per month | random forest; histogram gradient boosting | CPU | 1 |
| **E02** | Does a temporal neural network beat the classical baseline on the same pixels? | same pixel time series | TempCNN (Pelletier et al. 2019) | GPU | 2 |
| **E03** | Does a spatio-temporal network trained from scratch on the whole tile beat the pixel models? | 128 × 128 px tiles, 12 months | U-TAE (Garnot & Landrieu 2021) | GPU | 3 |
| **E04** | Do frozen geospatial foundation models with a light decoder beat the pixel models? (RQ1) | 128 × 128 px tiles, 12 months | frozen Prithvi-EO-2.0, Clay v1.5, TerraMind, AgriFM and the pixel time-series models Presto, TESSERA + UPerNet decoder (PANGAEA; Marsocci et al. 2024), L-TAE over time for the single-date image models | GPU | 4 |
| **E05** | Does fine-tuning (LoRA or full) beat the frozen encoder and the same network trained from scratch, and how does this depend on the number of labelled parcels? (RQ2) | 128 × 128 px tiles, 12 months; 10 / 25 / 50 / 100% of the training parcels | Prithvi-EO-2.0 + L-TAE + UPerNet (as E04): frozen, LoRA, fully fine-tuned, from scratch | GPU | 5 |

Run order (Morocco): E01 first (it fixes the evaluation code and gives the reference numbers), then the other
pixel model (E02), then the tile models (E03 before E04, because U-TAE tests the tile training pipeline without the
large encoders), then E05. Jobs: 1 CPU + 24 GPU (E02 1, E03 1, E04 6, E05 16); the same set again for Utah
afterwards.

### Details per experiment

**E01 · Sentinel-2 pixel baseline.** The pixel's 12 × 12 band values in reflectance × 10,000, plus NDVI and NDWI per
month; months without a clear observation are filled by linear interpolation in time. Random forest (500 trees,
`class_weight = "balanced"`) and histogram gradient boosting (class-weighted samples), hyperparameters chosen on the
validation fold. Permutation importance per month and band. This answers how far a per-pixel time series alone
goes; random forests are the standard in operational irrigation mapping (Ketchum et al. 2020; Xie et al. 2021).

**E02 · temporal neural network.** The same 12-step input, normalised per band with training-fold statistics.
TempCNN (three 1-D convolution layers over the months); trained with weighted cross-entropy and early stopping on
the validation fold; 3 seeds. Tests whether learnt temporal features beat the random forest. (A pixel-wise L-TAE was
also run; it is on hold and not reported.) All runs are compared on the [comparison page](comparison.md).

**E03 · U-TAE from scratch.** The PASTIS segmentation baseline: U-Net encoder with L-TAE attention over the 12
months, trained on the same tiles with the same masked loss. Shows how much the pretrained encoders add over a
spatio-temporal network of the same input.

**E04 · frozen foundation models + decoder.** The encoders see whole 128 px tiles (16 px patches for Prithvi and
AgriFM's settings, 8 px for Clay), one month at a time or all months as the model allows; the patch features are
combined over time with L-TAE and brought back to 10 m by a UPerNet decoder, trained with a masked loss on the
labelled pixels of each tile (irrigated and rainfed; all other pixels ignored). Only the
decoder and temporal head are trained. Same normalisations and band choices as in the archived embedding scripts (loaders in `pixelwise_run/encoders.py`)
(D12, D15). The two pixel time-series models, Presto (Tseng et al. 2023) and TESSERA v2 (Feng et al. 2025, 2026),
encode the 12-month series of every pixel of the tile into a 128-d vector; the 10 m feature map goes to the same
UPerNet head ([Foundation models](../foundation_models/index.md)).

**E05 · Prithvi: frozen, LoRA, fine-tuned or from scratch.** The E04 network with Prithvi, trained four ways:
encoder frozen (E04), low-rank adapters (LoRA, rank 16, on the attention and MLP layers) trained on the frozen
encoder, every encoder weight trained from the pretrained weights (encoder learning rate 10⁻⁵, head 10⁻⁴), or every
weight trained from random weights. Each with 10, 25, 50 and 100% of the training-fold parcels (per class,
nested draws; validation and test folds complete). Loss and F1 per epoch, score against the share of parcels, and
paired tests per share ([E05](E05_prithvi_strategies.md)).

## 3. After the experiments

1. **Temporal test (RQ3), Utah:** the best models trained on 2023/24 and applied to 2020/21 on the test folds.
2. **Maps (Phase 5):** the best model on wall-to-wall Sentinel-2 composites; the cropland extent is decided here
   (candidate masks such as the union of products or Dynamic World, compared on the omission of surveyed irrigated
   and rainfed parcels and on commission; [D19](../decisions.md#d19-two-class-experiments-cropland-extent-in-the-mapping-phase));
   irrigated and rainfed area per region and year. The maps are predicted from the 10.24 km blocks of
   [`data_download/`](https://github.com/amanchry/irrimapping-geofm/tree/main/data_download) (Earth Engine, the same
   compositing as the training tiles, which are cut from the same blocks): each block is classified on its own, on
   its cropland pixels only (tile models read 128 px windows from the block, pixel models read each pixel's series),
   and the block maps are merged at the end ([D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)). Dynamic World is downloaded for
   every block of the mapping area and flags the 128 px tiles with ≥ 1% cropland; Sentinel-2 and Sentinel-1 cover
   every block with such a tile and every block holding training tiles. An earlier, separate
   STAC approach (Earth Search, scene-classification cloud mask; [D18](../decisions.md#d18-wall-to-wall-composites-from-stac-for-the-maps))
   was retired before its download was complete (20 Nov 2026: Utah 2023/24 17,278 of 18,472 tiles; Morocco on hold).
3. **Map accuracy:** a stratified random sample over the map (including not-cropland), area-adjusted accuracy and
   area estimates (Olofsson et al. 2014); comparison with LGRIP30 and national statistics.

## 4. Optional ablations

- Sentinel-1 added to E01 / E02 (the tile store has it).
- Fewer labels for the other models: 10 / 25 / 50% of the training parcels, as in E05 (Prithvi only).
- A third class "not cropland" (weak labels from the cropland products), for the mapping phase: does a 3-class
  model remove false cropland better than a mask?
- Stricter mask (≥ 2 votes) for the maps.

## 5. How an experiment runs

Each experiment has a folder `experiments/E0<N>_<name>/` with its `run.py` (settings at the top, no command-line
arguments) and a page in this section; the shared code is listed on the [overview](index.md#code). On the cluster
every run is a Slurm job (`jobs/step_cpu.sbatch` or `jobs/step_gpu.sbatch` with `SCRIPT=<run.py>`); region, encoder
and quick-test mode are passed as `EXPERIMENT_REGION`, `EXPERIMENT_ENCODER` and `EXPERIMENT_QUICK_TEST`. A run
writes `results/<region>/<run>/` (out-of-fold predictions per pixel, metrics with CI, per-group metrics, confusion
matrix, settings); only `results/` is copied to the Mac, where `pixelwise_run/05_compare_experiments.py` builds the
comparison table and the paired tests, and the experiment page is filled in.

## References

- Feng, Z. et al. (2025). TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis.
  arXiv:2506.20380. Feng, Z. et al. (2026). TESSERA v2: scaling pixel-wise Earth foundation models. arXiv:2607.03949.
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
- Tseng, G. et al. (2023). Lightweight, pre-trained transformers for remote sensing timeseries (Presto).
  arXiv:2304.14065.
- Xie, Y., Gibbs, H. K. & Lark, T. J. (2021). LANID. *ESSD* 13, 5689–5710.
