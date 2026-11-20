# E01 · Sentinel-2 pixel baseline

!!! info "Status: planned (Morocco first, Utah later)"
    Two classes (irrigated, rainfed) on the labelled pixels of the surveyed parcels: Morocco 2023/24 and 2024/25
    pooled, 473,675 pixels of 8,775 parcels. Code ready and tested; results follow after the run.

**Question.** How well do the Sentinel-2 time series of single pixels separate irrigated from rainfed pixels? E01
is the reference for every later experiment: the paired tests compare each run with the E01 random forest.

**Why.** Random forests on per-pixel time series are the standard in operational irrigation mapping (Ketchum et al.
2020; Xie et al. 2021). Any more complex model has to beat this reference on the same pixels.

## Set-up

| Item | Choice |
| --- | --- |
| **Samples** | the labelled pixels of the surveyed parcels ([Pixel samples](../phase1_data/pixel_samples.md)): inner pixels, at most 100 per parcel; Morocco 2023/24 and 2024/25 pooled |
| **Classes** | 1 irrigated · 2 rainfed |
| **Input** | 12 months × (12 Sentinel-2 bands + NDVI + NDWI) = 168 features. Months without a clear observation are filled by linear interpolation in time (first / last month: nearest observed month); a band never observed is filled with the training median |
| **Models** | random forest (500 trees, `class_weight = "balanced"`); histogram gradient boosting (`class_weight = "balanced"`, at most 1,000 iterations, early stopping on an internal 10% split) |
| **Hyperparameters** | grid evaluated on the validation fold of each round (macro-F1): random forest `max_features` ∈ {√p, 0.33} × `min_samples_leaf` ∈ {1, 5}; gradient boosting learning rate ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {31, 63} |
| **Validation** | PASTIS rotation over the 5 spatial folds: round *k* tests fold *k*, validates on fold *k*+1 and trains on the other three |
| **Importance** | grouped permutation importance of the random forest: drop in macro-F1 when all features of one month, or one band over all months, are shuffled (20,000 test pixels per round) |
| **Seed** | 42 |

??? info "How the random forest works"
    A **decision tree** splits the pixels step by step on single features ("NDVI in July > 0.45?", "B11 in March <
    0.2?"), each time choosing the split that best separates the classes, until the groups are pure or small. One
    tree overfits easily. A **random forest** (Breiman 2001) grows many trees (here 500), each on a bootstrap sample
    of the training pixels and, at every split, on a random subset of the features (`max_features`: √168 ≈ 13 or
    33%). The trees therefore make different mistakes, and the forest averages their votes into class
    probabilities. `min_samples_leaf` stops a tree from splitting groups smaller than 1 or 5 pixels.

    `class_weight = "balanced"` counts each rainfed pixel more, in inverse proportion to its share, so the forest
    does not simply favour the irrigated majority. The months are 168 separate features: the forest has no notion
    of their order, but it can still use any month that separates the classes.

??? info "How histogram gradient boosting works"
    **Gradient boosting** (Friedman 2001) also combines many trees, but builds them **one after another**: each new
    small tree is fitted to the errors (the gradient of the loss) of all trees before it, and its prediction is
    added with a small weight (the learning rate, 0.05 or 0.1). Many weak trees in sequence give a strong model.
    The **histogram** variant (as in LightGBM, Ke et al. 2017; scikit-learn `HistGradientBoostingClassifier`)
    first bins every feature into at most 255 value ranges, which makes training on hundreds of thousands of
    pixels fast. `max_leaf_nodes` (31 or 63) limits the size of each tree; training stops when an internal 10%
    validation split no longer improves for 20 iterations.

??? info "How the permutation importance works"
    After training, the values of one group of features are shuffled across the test pixels (e.g. all 14 values of
    July, or one band in all 12 months), which breaks their link to the label while keeping their distribution.
    The drop in macro-F1 compared with the unshuffled data is the importance of that group: a large drop means the
    forest relies on it. Grouping by month answers "which months matter", grouping by band "which bands matter".

## How to run

Script: [`experiments/E01_s2_pixel_baseline/run.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E01_s2_pixel_baseline/run.py)
(shared code: [`pixelwise_run/pixel_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/pixel_common.py)).

```bash
sbatch --export=ALL,SCRIPT=experiments/E01_s2_pixel_baseline/run.py,EXPERIMENT_REGION=morocco jobs/step_cpu.sbatch
```

`EXPERIMENT_QUICK_TEST=1` runs the whole chain on 20,000 pixels with one grid point (output in `results_quick/`).
CPU only; the random-forest grid dominates the run time.

## Outputs

`experiments/E01_s2_pixel_baseline/results/<region>/<rf|hgb>/`; only `results/` is copied from the cluster.

| File | Content |
| --- | --- |
| `oof_predictions.parquet` | per pixel: year, sample index, true label, class probabilities, predicted class, fold, block, zone, tile, coordinates |
| `metrics.csv` | macro-F1 (headline), rainfed and irrigated F1 / precision / recall, overall accuracy, with 95% block-bootstrap CI |
| `metrics_by_group.csv` | the same per year and per zone |
| `confusion.png` | confusion matrix (row shares and counts) |
| `chosen_params.csv` | hyperparameters chosen in each round |
| `importance_month.csv`, `importance_band.csv` | permutation importance (random forest) |
| `run_config.json`, `run.log` | settings, code version, log |

## Results

_To be filled after the Morocco run._ Planned: scores with 95% CI, confusion matrix, scores per year and zone,
permutation importance per month and band, chosen hyperparameters and run time. The comparison with the other
experiments is on the [comparison page](comparison.md).

## References

- Breiman, L. (2001). Random forests. *Machine Learning* 45, 5–32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics* 29,
  1189–1232.
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*.
- Ketchum, D. et al. (2020). IrrMapper. *Remote Sensing* 12, 2328.
- Xie, Y., Gibbs, H. K. & Lark, T. J. (2021). LANID. *ESSD* 13, 5689–5710.
