# Comparison of the experiments

!!! success "Status: Morocco done, all runs of E01–E04 (22 November 2026); Utah later"
    **Morocco answer to RQ1:** frozen foundation models give the best scores (TerraMind 0.927 macro-F1) and are
    significantly better than the same pipeline trained from scratch (U-TAE), but not significantly better than a
    random forest on the pixel time series. This page holds all runs of E01–E04 side by side. Each experiment page reports only
    its own runs; E05 (Prithvi frozen / LoRA / fine-tuned / from scratch) is compared within itself on its
    [own page](E05_prithvi_strategies.md).

## What is compared

All runs of a region are scored on the **same out-of-fold pixels**: Morocco 2023/24 and 2024/25 pooled, 473,675
labelled pixels of 8,775 surveyed parcels, two classes (irrigated, rainfed).

| Run | Experiment | Model |
| --- | --- | --- |
| E01 random forest | [E01](E01_s2_pixel_baseline.md) | pixel time series, random forest (**reference**) |
| E01 gradient boosting | [E01](E01_s2_pixel_baseline.md) | pixel time series, histogram gradient boosting |
| E02 TempCNN | [E02](E02_temporal_nn.md) | pixel time series, temporal CNN |
| E03 U-TAE | [E03](E03_utae.md) | whole tiles, trained from scratch (**reference** for E04) |
| E04 Prithvi, Clay, TerraMind, AgriFM, Presto, TESSERA | [E04](E04_frozen_geofm.md) | whole tiles, frozen foundation model + L-TAE + UPerNet |

## Tables and figures

| Content | How |
| --- | --- |
| **Scores** | macro-F1 (irrigated, rainfed; headline), rainfed F1, precision and recall, with 95% block-bootstrap CI |
| **Paired tests against the E01 random forest** | does a model beat the pixel baseline? Difference on the same pixels, 95% CI, Holm-corrected p |
| **Paired tests against E03 U-TAE** | does pretraining add anything beyond the spatial context of the tile? |
| **Figures** | scores of all runs with CIs; confusion matrices of the random forest, U-TAE and the best foundation model |

A difference counts when its 95% CI excludes 0 and the Holm-corrected p < 0.05 ([evaluation protocol](../phase2_models/evaluation.md)).
Tables and tests: `pixelwise_run/05_compare_experiments.py` (`experiments/comparison/<region>/`); figures:
`experiments/report_figures.py` (`docs/experiments/figures/`).

## Results

### Morocco: all runs

| Run | Macro-F1 [95% CI] | Rainfed F1 [95% CI] | vs E01 random forest: Δ macro-F1 [95% CI], p<sub>Holm</sub> | vs E03 U-TAE: Δ macro-F1 [95% CI], p<sub>Holm</sub> |
| --- | --- | --- | --- | --- |
| E04 TerraMind | **0.927** [0.880, 0.964] | **0.872** [0.783, 0.937] | +0.034 [+0.004, +0.076], 0.29 | **+0.027** [+0.013, +0.052], < 0.01 |
| E04 Clay | 0.923 [0.874, 0.960] | 0.864 [0.769, 0.930] | +0.029 [−0.002, +0.071], 0.45 | **+0.023** [+0.010, +0.042], 0.01 |
| E04 AgriFM | 0.921 [0.868, 0.956] | 0.861 [0.758, 0.922] | +0.028 [−0.001, +0.064], 0.45 | **+0.021** [+0.009, +0.036], < 0.01 |
| E04 Prithvi | 0.917 [0.865, 0.957] | 0.853 [0.749, 0.924] | +0.023 [−0.007, +0.061], 0.76 | **+0.017** [+0.004, +0.036], 0.04 |
| E02 TempCNN | 0.912 [0.848, 0.955] | 0.845 [0.722, 0.922] | +0.018 [−0.017, +0.053], 1.0 | +0.012 [−0.020, +0.044], 1.0 |
| E04 TESSERA | 0.912 [0.853, 0.958] | 0.844 [0.733, 0.928] | +0.018 [+0.000, +0.039], 0.37 | +0.012 [−0.020, +0.043], 1.0 |
| E04 Presto | 0.903 [0.848, 0.942] | 0.828 [0.720, 0.899] | +0.009 [−0.022, +0.044], 1.0 | +0.003 [−0.008, +0.011], 1.0 |
| E03 U-TAE | 0.900 [0.845, 0.941] | 0.823 [0.716, 0.896] | +0.006 [−0.027, +0.043], 1.0 | reference |
| E01 random forest | 0.894 [0.834, 0.940] | 0.810 [0.697, 0.894] | reference | −0.006 [−0.043, +0.027], 1.0 |
| E01 gradient boosting | 0.893 [0.822, 0.946] | 0.809 [0.673, 0.903] | −0.000 [−0.017, +0.014], 1.0 | −0.007 [−0.057, +0.035], 1.0 |

Bold: the difference counts (CI excludes 0 and p<sub>Holm</sub> < 0.05; Holm over the nine other runs per
reference). Tables: `experiments/comparison/morocco/`.

![Scores of E01–E04](figures/scores_e03_e04.png)

![Rainfed recall and F1 per season and zone](figures/groups_e03_e04.png)

![Confusion matrices](figures/confusion_e03_e04.png)

### Answers to the RQ1 questions (Morocco)

| Question | Answer | Evidence |
| --- | --- | --- |
| Q1 · Do frozen GeoFMs beat a classifier on the raw Sentinel-2 time series? | **Not significantly.** All six score above the random forest (+0.009 to +0.034 macro-F1); TerraMind's CI excludes 0 but not after the Holm correction | [E04](E04_frozen_geofm.md#paired-tests) |
| Q2 · Do they beat a network trained from scratch on the same tiles? | **Yes, for the four image encoders** (TerraMind, Clay, AgriFM, Prithvi: +0.017 to +0.027, significant); not for Presto and TESSERA | [E04](E04_frozen_geofm.md#paired-tests) |
| Q3 · Does a temporal neural network beat the random forest? | **Not significantly** (TempCNN +0.018 [−0.017, +0.053]) | [E02](E02_temporal_nn.md#results) |
| Q4 · Image models or pixel time-series models? | **Image models** score higher (0.917–0.927 vs 0.903–0.912) and are 3–14 times faster here; the four image encoders cannot be ranked among themselves | [E04](E04_frozen_geofm.md#interpretation) |
| Q5 · Second region | Utah, after Morocco | – |

### What the results show

- **All models are within 0.034 macro-F1 of each other** (0.893–0.927). The separation of irrigated and rainfed
  pixels is largely possible from the Sentinel-2 time series of a single pixel; the shortwave-infrared bands and the
  winter and summer months carry most of it ([E01](E01_s2_pixel_baseline.md#which-months-and-bands-matter)).
- **The differences are in the hard cases:** rainfed fields in the wetter year 2024/25 and in Souss, where few
  rainfed parcels exist. Rainfed recall in Souss rises from 0.32 (random forest) to 0.64–0.68 (TempCNN, U-TAE) and
  0.81–0.84 (TerraMind, Clay); in 2024/25 from 0.66 to 0.85.
- **Pretrained features help beyond spatial context:** with tiles, labels and head identical, the pretrained image
  encoders beat U-TAE trained from scratch. [E05](E05_prithvi_strategies.md) confirms this for Prithvi (pretrained vs
  from scratch: +0.08 to +0.11 at every share of the labels) and shows that fine-tuning the encoder adds nothing over
  keeping it frozen.
- **Why the comparisons with the random forest stay inconclusive:** the rainfed pixels lie in few 20 km blocks,
  mostly in Doukkala, and the gains of the tile models come from a few of them (fold 0: 0.76 → 0.86–0.88), while in
  fold 1 the random forest is best. The block bootstrap, which treats the block as the unit, therefore gives wide
  intervals. Utah, with rainfed land spread more widely, will test whether the ranking holds.
