# E01 · Sentinel-2 pixel baseline

!!! success "Status: done in both regions"
    Morocco (2023/24 and 2024/25 pooled, 522,003 pixels) and Utah (2023/24, 654,382 pixels), on the Earth Engine
    tiles. Results below; comparison with E02 on the [E02 page](E02_temporal_nn.md).

**Question.** How well do the Sentinel-2 time series of single pixels separate irrigated, rainfed and not-cropland
pixels? E01 is the reference for every later experiment; the paired tests compare each run with the E01 random
forest.

**Why.** Random forests on per-pixel time series are the standard in operational irrigation mapping (Ketchum et al.
2020; Xie et al. 2021). Any more complex model has to beat this reference on the same pixels.

## Set-up

| Item | Choice |
| --- | --- |
| **Samples** | the labelled pixels ([Pixel samples](../phase1_data/pixel_samples.md)); Morocco 2023/24 and 2024/25 pooled, Utah 2023/24; pixels of fold −1 left out |
| **Classes** | 1 irrigated · 2 rainfed · 3 not cropland |
| **Input** | 12 months × (12 Sentinel-2 bands + NDVI + NDWI) = 168 features. Months without a clear observation are filled by linear interpolation in time (first / last month: nearest observed month); a band never observed is filled with the training median |
| **Models** | random forest (500 trees, `class_weight = "balanced"`); histogram gradient boosting (`class_weight = "balanced"`, at most 1,000 iterations, early stopping on an internal 10% split) |
| **Hyperparameters** | grid evaluated on the validation fold of each round (3-class macro-F1): random forest `max_features` ∈ {√p, 0.33} × `min_samples_leaf` ∈ {1, 5}; gradient boosting learning rate ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {31, 63} |
| **Validation** | PASTIS rotation over the 5 spatial folds: round *k* tests fold *k*, validates on fold *k*+1 and trains on the other three |
| **Importance** | grouped permutation importance of the random forest: drop in 3-class macro-F1 when all features of one month, or one band over all months, are shuffled (20,000 test pixels per round) |
| **Seed** | 42 |

## How to run

Script: [`experiments/E01_s2_pixel_baseline/run.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E01_s2_pixel_baseline/run.py)
(shared code: [`pixelwise_run/pixel_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/pixel_common.py)).

```bash
sbatch --export=ALL,SCRIPT=experiments/E01_s2_pixel_baseline/run.py,EXPERIMENT_REGION=morocco jobs/step_cpu.sbatch
```

```bash
sbatch --export=ALL,SCRIPT=experiments/E01_s2_pixel_baseline/run.py,EXPERIMENT_REGION=utah jobs/step_cpu.sbatch
```

`EXPERIMENT_QUICK_TEST=1` runs the whole chain on 20,000 pixels with one grid point (output in `results_quick/`).
CPU only; the random-forest grid dominates the run time.

## Outputs

`experiments/E01_s2_pixel_baseline/results/<region>/<rf|hgb>/`; only `results/` is copied from the cluster.

| File | Content |
| --- | --- |
| `oof_predictions.parquet` | per pixel: year, sample index, true label, class probabilities, predicted class, fold, block, zone, tile, coordinates |
| `metrics.csv` | cropland macro-F1 (headline), 3-class macro-F1, per-class F1 / precision / recall, with 95% block-bootstrap CI |
| `metrics_by_group.csv` | the same per year and per zone |
| `confusion.png` | confusion matrix (row shares and counts) |
| `chosen_params.csv` | hyperparameters chosen in each round |
| `importance_month.csv`, `importance_band.csv` | permutation importance (random forest) |
| `run_config.json`, `run.log` | settings, code version, log |

## Results

### Samples

| Region | Irrigated pixels (parcels) | Rainfed pixels (parcels) | Not-cropland pixels | Blocks | Pixels |
| --- | --- | --- | --- | --- | --- |
| Morocco | 419,883 (7,579) | 53,792 (1,196) | 48,328 | 91 | 522,003 |
| Utah | 394,064 (4,998) | 75,559 (1,013) | 184,759 | 175 | 654,382 |

Morocco pools 2023/24 (402,659 pixels) and 2024/25 (119,344). Rainfed pixels in Morocco come from three zones only
(Doukkala 46,608, Souss 6,869, Statte 312); Gharb, Tadla and El Haouz have irrigated parcels only.

### Scores

Out-of-fold scores over the 5 spatial folds, with 95% block-bootstrap confidence intervals (20 km blocks).

| Region | Model | Cropland macro-F1 | 3-class macro-F1 | F1 irrigated | F1 rainfed | F1 not cropland | Overall accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Morocco | random forest | **0.876** [0.801, 0.929] | 0.838 [0.773, 0.887] | 0.959 | 0.784 | 0.770 | 0.922 |
| Morocco | gradient boosting | **0.881** [0.801, 0.937] | 0.850 [0.783, 0.904] | 0.962 | 0.795 | 0.792 | 0.927 |
| Utah | random forest | **0.729** [0.682, 0.765] | 0.746 [0.714, 0.772] | 0.870 | 0.559 | 0.809 | 0.817 |
| Utah | gradient boosting | **0.726** [0.679, 0.763] | 0.750 [0.718, 0.776] | 0.875 | 0.552 | 0.824 | 0.824 |

Paired difference gradient boosting − random forest on the same pixels (block bootstrap; Holm correction over all runs
compared with the E01 random forest so far):

| Region | Cropland macro-F1 | 3-class macro-F1 |
| --- | --- | --- |
| Morocco | +0.005 [−0.007, 0.021], p<sub>Holm</sub> 0.72 | **+0.012** [0.003, 0.024], p<sub>Holm</sub> 0.018 |
| Utah | −0.003 [−0.015, 0.008], p<sub>Holm</sub> 0.62 | +0.004 [−0.004, 0.013], p<sub>Holm</sub> 0.61 |

![Scores of E01 and E02](figures/scores_e01_e02.png)

The two classifiers are equivalent on the headline metric in both regions. Gradient boosting is slightly better on
the 3-class score in Morocco, because it separates not-cropland pixels better (F1 0.792 vs 0.770). The random forest
remains the reference for the paired tests of the later experiments, as planned.

### Errors

![Confusion matrices](figures/confusion_e01_e02.png)

- **Irrigated pixels** are recognised well in Morocco (recall 0.95) and less well in Utah (0.82); in Utah 13% of the
  irrigated pixels are predicted as not cropland.
- **Rainfed pixels** are the weak class. Morocco: recall 0.73, the errors split between irrigated (16%) and not
  cropland (12%). Utah: recall 0.52, with a quarter predicted as not cropland and 23% as irrigated.
- **Not cropland** has high recall (0.91–0.94) but low precision (0.67 Morocco, 0.71 Utah): cropland pixels are
  predicted as not cropland (Morocco 15,356 irrigated and 6,396 rainfed pixels; Utah 50,806 and 18,938). Part of the confusion is label noise:
  a check of the Utah samples found 3.6% of the not-cropland pixels (6,784 of 186,458) inside fields mapped as crops
  in WRLU, because the mask misses them; a fix of the sampling is planned before the final runs.

### By year and zone

| Region | Group | Cropland macro-F1 (RF / GB) | F1 rainfed (RF / GB) | Rainfed recall (RF / GB) |
| --- | --- | --- | --- | --- |
| Morocco | 2023/24 | 0.893 / 0.894 | 0.822 / 0.822 | 0.824 / 0.833 |
| Morocco | 2024/25 | 0.844 / 0.859 | 0.718 / 0.749 | 0.583 / 0.622 |
| Morocco | Doukkala | 0.891 / 0.899 | 0.849 / 0.864 | 0.817 / 0.852 |
| Morocco | Souss | 0.583 / 0.516 | 0.205 / 0.078 | 0.114 / 0.041 |
| Utah | Bear River | 0.736 / 0.719 | 0.641 / 0.615 | 0.661 / 0.633 |
| Utah | Southeast Colorado River | 0.644 / 0.706 | 0.817 / 0.844 | 0.697 / 0.730 |
| Utah | West Desert | 0.677 / 0.732 | 0.687 / 0.744 | 0.732 / 0.681 |
| Utah | Sevier River | 0.484 / 0.489 | 0.048 / 0.055 | 0.025 / 0.029 |

Zones are listed when they hold enough rainfed pixels (Morocco ≥ 5,000; Utah ≥ 4,000); in zones without rainfed
parcels the cropland macro-F1 is bounded at about 0.5 and not informative. The rainfed class works where it is
common (Doukkala, Bear River) and fails where it is a small minority among irrigated parcels (Souss, Sevier River):
there the model predicts almost every rainfed pixel as irrigated or not cropland. In Morocco the 2024/25 season is
harder for rainfed pixels (recall 0.58 vs 0.82), with fewer training pixels from that year (22,121 rainfed vs
31,671 in 2023/24).

### What the random forest uses

![Permutation importance](figures/e01_importance.png)

- **Bands:** the shortwave-infrared bands B12 and B11 matter most in both regions, followed by NDVI and B1. They
  respond to soil and canopy water content, which is what separates an irrigated from a rainfed field in the dry
  season. In Utah all importances are small, because neighbouring months and bands carry overlapping information.
- **Months, Morocco** (September–August): February, at the peak of the rainfed winter crop, and June–August, when
  only irrigated fields are green.
- **Months, Utah** (October–September): August and July (irrigated fields green, dryland wheat harvested) and
  December–January. The winter months are likely informative through snow cover and missing observations rather than
  through vegetation; this is to be checked.

### Hyperparameters and run time

- Random forest: `max_features = √p` was chosen in 8 of 10 rounds; `min_samples_leaf` varied in Morocco (1 in 3 of 5
  rounds) and was 5 in every Utah round. Gradient boosting: mixed choices in Morocco, learning rate 0.05 with 31
  leaves in every Utah round.
- The validation score varies strongly between rounds in Morocco (0.67–0.90, 3-class): the round validated on fold 0
  is much harder, which explains the wide confidence intervals.
- CPU time on the cluster: random forest 1 h 46 min (Morocco) and 3 h 00 min (Utah) for the 4-point grid and the
  importance; gradient boosting 18 min and 35 min.

### Conclusion

A per-pixel Sentinel-2 time series with a random forest reaches a cropland macro-F1 of **0.876 in Morocco** and
**0.729 in Utah**. Irrigated pixels are well separated in both regions; the difficulty is the rainfed class, especially
in Utah and in zones where rainfed fields are rare, and its confusion with not cropland. These are the reference
numbers for E02–E04.

## References

- Ketchum, D. et al. (2020). IrrMapper. *Remote Sensing* 12, 2328.
- Xie, Y., Gibbs, H. K. & Lark, T. J. (2021). LANID. *ESSD* 13, 5689–5710.
