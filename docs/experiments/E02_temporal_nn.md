# E02 · Temporal neural network (TempCNN)

!!! success "Status: done in both regions"
    Same pixels as [E01](E01_s2_pixel_baseline.md): Morocco 522,003 (2023/24 and 2024/25), Utah 654,382 (2023/24);
    3 seeds per round on one GPU.

!!! note "Pixel-wise L-TAE on hold"
    A second model, L-TAE on single pixels, was also run. Its set-up is under review, so its results are not
    reported here or in the comparisons of E03 and E04 (`EXCLUDE` in `pixelwise_run/05_compare_experiments.py`).

**Question.** Does a temporal neural network beat the classical baseline (E01) on the same pixels and the same
input?

**What changes from E01.** Only the model: the input is the same 12-month Sentinel-2 series with NDVI and NDWI.

| | E01 | E02 |
| --- | --- | --- |
| Model | random forest, gradient boosting | TempCNN |
| Input format | 168 features (months concatenated) | sequence of 12 months × 14 channels |
| Missing months | interpolated | interpolated |

## Set-up

| Item | Choice |
| --- | --- |
| **TempCNN** | Pelletier et al. (2019): three 1-D convolutions over the months (64 filters, kernel 5, batch normalisation, ReLU, dropout 0.2), dense layer of 256, output 3 classes |
| **Normalisation** | per channel with the training pixels of each round; values still missing after interpolation set to 0 (the training mean) |
| **Training** | AdamW (learning rate 10⁻³, weight decay 10⁻⁴), batch 512, class-weighted cross-entropy, early stopping on the validation fold's 3-class macro-F1 (patience 10, at most 100 epochs); the best epoch is kept |
| **Seeds** | 3 per round (42, 1042, 2042); the out-of-fold probability is their mean |
| **Validation and metrics** | as E01 |

??? info "How TempCNN works"
    TempCNN (Temporal Convolutional Neural Network; Pelletier et al. 2019) is a convolutional network that slides
    along the **time axis** of one pixel, the way an image CNN slides over the two image axes. It was designed for
    pixel-wise classification of Sentinel-2 time series and is a standard baseline for crop mapping.

    **Input.** One pixel as a small table of 12 months × 14 channels (12 bands, NDVI, NDWI), normalised per channel;
    months without a clear observation are filled by linear interpolation, because a convolution needs a value at
    every step.

    ```mermaid
    flowchart LR
        A["one pixel<br/>12 months × 14 channels"] --> B["Conv 1-D, 64 filters,<br/>window 5 months<br/>+ BatchNorm, ReLU, dropout"]
        B --> C["Conv 1-D, 64 filters,<br/>window 5 months"]
        C --> D["Conv 1-D, 64 filters,<br/>window 5 months"]
        D --> E["flatten<br/>12 × 64 = 768 values"]
        E --> F["dense layer 256<br/>+ BatchNorm, ReLU, dropout"]
        F --> G["3 class scores<br/>irrigated · rainfed · not cropland"]
    ```

    **1. Convolutions over time.** A filter is a small set of weights over a window of 5 consecutive months and all
    14 channels. It is moved one month at a time along the year (padded at both ends, so the output still has 12
    steps) and at each position gives one number: how strongly this 5-month stretch matches the pattern the filter
    has learnt. Typical learnt patterns are a green-up (NDVI rising over a few months), a harvest drop, or wet soil
    (NDWI) during the dry season. 64 filters give 64 such pattern detectors per month.

    **2. Three layers in a row.** The second and third layers combine the patterns found by the layer before, so
    they detect longer and more complex shapes, e.g. "green-up in winter, harvest in spring, then green again in
    summer" (an irrigated double crop). Each layer looks 5 steps wide, so after three layers one output step sees
    1 + 3 × 4 = 13 months, the whole agricultural year. Batch normalisation keeps the values on a stable scale,
    ReLU keeps only positive matches, and dropout (20%) switches off random units during training against
    overfitting.

    **3. Flatten and classify.** The 12 × 64 outputs are laid out as one vector of 768 values, so the network still
    knows **when** in the year each pattern occurred (a green summer means something else than a green winter). A
    dense layer of 256 units and a final layer turn this into three class scores; softmax gives probabilities.

    **Size and training.** 0.24 M weights. Trained with class-weighted cross-entropy, AdamW (learning rate 10⁻³),
    batch 512 pixels, early stopping on the validation fold; 3 seeds per round.

    **TempCNN compared with the random forest.**

    | | Random forest (E01) | TempCNN |
    | --- | --- | --- |
    | Reads the months | as 168 unordered features | as a sequence, through sliding 5-month windows |
    | Learns | splits on single values (e.g. "NDVI in July > 0.4") | local temporal shapes and their combinations |
    | Missing months | interpolated | interpolated |
    | Uses position in the year | implicitly (each feature is one month) | yes, the flattened output keeps the order |

    Code: `TempCNN` in [`pixelwise_run/nn_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/nn_common.py).

## How to run

Script: [`experiments/E02_temporal_nn/run.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E02_temporal_nn/run.py)
(network: [`pixelwise_run/nn_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/nn_common.py)).

```bash
sbatch --export=ALL,SCRIPT=experiments/E02_temporal_nn/run.py,EXPERIMENT_REGION=morocco jobs/step_gpu.sbatch
```

## Outputs

`experiments/E02_temporal_nn/results/<region>/tempcnn/`: the E01 files, plus `training_history.csv`.

## Results

### Scores

Out-of-fold scores with 95% block-bootstrap CI; the E01 random forest is repeated for comparison.

| Region | Model | Cropland macro-F1 | 3-class macro-F1 | F1 irrigated | F1 rainfed | F1 not cropland |
| --- | --- | --- | --- | --- | --- | --- |
| Morocco | E01 random forest | 0.876 [0.801, 0.929] | 0.838 [0.773, 0.887] | 0.959 | 0.784 | 0.770 |
| Morocco | TempCNN | **0.886** [0.798, 0.944] | **0.851** [0.784, 0.903] | 0.960 | 0.807 | 0.786 |
| Utah | E01 random forest | **0.729** [0.682, 0.765] | **0.746** [0.714, 0.772] | 0.870 | 0.559 | 0.809 |
| Utah | TempCNN | 0.721 [0.674, 0.762] | 0.745 [0.712, 0.775] | 0.862 | 0.545 | 0.829 |

Paired difference to the E01 random forest on the same pixels (block bootstrap, 95% CI; Holm correction over the
seven runs of E01–E04 compared with the E01 random forest; **bold** = CI excludes 0 and p<sub>Holm</sub> < 0.05):

| Region | Model | Δ cropland macro-F1 | Δ 3-class macro-F1 |
| --- | --- | --- | --- |
| Morocco | TempCNN | +0.010 [−0.013, 0.032], p<sub>Holm</sub> 1.00 | +0.013 [−0.005, 0.030], p<sub>Holm</sub> 0.22 |
| Utah | TempCNN | −0.008 [−0.022, 0.007], p<sub>Holm</sub> 0.62 | −0.001 [−0.010, 0.009], p<sub>Holm</sub> 0.87 |

![Scores of E01 and E02](figures/scores_e01_e02.png)

- **TempCNN is on par with the random forest:** +0.010 in Morocco and −0.008 in Utah on the cropland macro-F1,
  neither significant; the 3-class score does not differ either.
- Reading the 12 months as a sequence therefore does not change the picture of E01: with the same per-pixel input,
  the temporal model is not the limit.

### Errors

![Confusion matrices](figures/confusion_e01_e02.png)

Compared with the random forest, TempCNN finds more rainfed pixels (recall Morocco 0.80 vs 0.73, Utah 0.61 vs
0.52) and, in Utah, sends fewer irrigated pixels to not cropland (10% vs 13%). In exchange it predicts more irrigated
pixels as rainfed (Morocco 2.3% vs 1.5%, Utah 10% vs 5%), so the rainfed precision drops (Morocco 0.82 vs 0.85, Utah
0.49 vs 0.60) and the rainfed F1 changes little (Morocco 0.807 vs 0.784, Utah 0.545 vs 0.559). The rainfed class
remains the main error in Utah.

| Region | Group | Cropland macro-F1 (RF / TempCNN) | F1 rainfed (RF / TempCNN) |
| --- | --- | --- | --- |
| Morocco | 2023/24 | 0.893 / 0.903 | 0.822 / 0.839 |
| Morocco | 2024/25 | 0.844 / 0.854 | 0.718 / 0.751 |
| Morocco | Doukkala | 0.891 / 0.902 | 0.849 / 0.873 |
| Morocco | Souss | 0.583 / 0.499 | 0.205 / 0.075 |
| Utah | Bear River | 0.736 / 0.713 | 0.641 / 0.628 |
| Utah | Southeast Colorado River | 0.644 / 0.779 | 0.817 / 0.939 |
| Utah | Sevier River | 0.484 / 0.519 | 0.048 / 0.127 |
| Utah | West Desert | 0.677 / 0.702 | 0.687 / 0.677 |

TempCNN is better than the random forest in most groups but worse in Souss (Morocco) and Bear River (Utah); the
rainfed class in Souss and Sevier River is poorly recognised by both (F1 at most 0.21).

### Training

- Early stopping on the validation fold selected the best epoch after on average 10 epochs in Morocco and 6 in
  Utah. In Utah the validation score peaks early: the model fits the training blocks quickly, and more epochs do
  not transfer to the held-out blocks.
- GPU time per region: 12–15 min in Morocco and 11–13 min in Utah (5 rounds × 3 seeds).

### Conclusion

A temporal neural network on the same pixel series gives **no gain** over the random forest: TempCNN reaches a
cropland macro-F1 of 0.886 in Morocco and 0.721 in Utah, within the uncertainty of the spatial folds. The limits of
the per-pixel input (rainfed class, zones with few rainfed parcels) remain. The tile models test whether **spatial
context** helps where the pixel model does not: it does in Utah ([E03](E03_utae.md) U-TAE +0.037,
[E04](E04_frozen_geofm.md) foundation models up to +0.068), not significantly in Morocco.

## References

- Pelletier, C., Webb, G. I. & Petitjean, F. (2019). Temporal convolutional neural network for the classification of
  satellite image time series. *Remote Sensing* 11, 523.
