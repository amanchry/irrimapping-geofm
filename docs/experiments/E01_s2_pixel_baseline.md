# E01 · Sentinel-2 pixel baseline

!!! info "Status: ready to run"
    Script written and checked on a 20,000-pixel subset. Pixel samples available for Morocco 2023/24, 2024/25 and
    Utah 2023/24.

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

*To be filled after the run.*

| Region | Model | Cropland macro-F1 | 3-class macro-F1 |
| --- | --- | --- | --- |
| Morocco | random forest | | |
| Morocco | gradient boosting | | |
| Utah | random forest | | |
| Utah | gradient boosting | | |

## References

- Ketchum, D. et al. (2020). IrrMapper. *Remote Sensing* 12, 2328.
- Xie, Y., Gibbs, H. K. & Lark, T. J. (2021). LANID. *ESSD* 13, 5689–5710.
