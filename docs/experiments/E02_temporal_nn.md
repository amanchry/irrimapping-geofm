# E02 · Temporal neural network (TempCNN)

!!! info "Status: planned (Morocco first, Utah later)"
    Same pixels as [E01](E01_s2_pixel_baseline.md): Morocco 2023/24 and 2024/25 pooled, 473,675 pixels, two classes
    (irrigated, rainfed); 3 seeds per round on one GPU. Code ready and tested; results follow after the run.

**Question.** Does a temporal neural network beat the classical baseline (E01) on the same pixels and the same
input?

**What changes from E01.** Only the model: the input is the same 12-month Sentinel-2 series with NDVI and NDWI,
read as a sequence instead of 168 separate features.

| | E01 | E02 |
| --- | --- | --- |
| Model | random forest, gradient boosting | TempCNN |
| Input format | 168 features (months concatenated) | sequence of 12 months × 14 channels |
| Missing months | interpolated | interpolated |

## Set-up

| Item | Choice |
| --- | --- |
| **TempCNN** | Pelletier et al. (2019): three 1-D convolutions over the months (64 filters, kernel 5, batch normalisation, ReLU, dropout 0.2), dense layer of 256, output 2 classes |
| **Normalisation** | per channel with the training pixels of each round; values still missing after interpolation set to 0 (the training mean) |
| **Training** | AdamW (learning rate 10⁻³, weight decay 10⁻⁴), batch 512, class-weighted cross-entropy, early stopping on the validation fold's macro-F1 (patience 10, at most 100 epochs); the best epoch is kept |
| **Seeds** | 3 per round (42, 1042, 2042); the seed also sets the initial weights; the out-of-fold probability is their mean |
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
        F --> G["2 class scores<br/>irrigated · rainfed"]
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
    dense layer of 256 units and a final layer turn this into two class scores (irrigated, rainfed); softmax gives probabilities.

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

_To be filled after the Morocco run._ Planned: scores with 95% CI and the paired difference to the E01 random
forest, confusion matrix, scores per year and zone, training curves and run time. The comparison with the other
experiments is on the [comparison page](comparison.md).

## References

- Pelletier, C., Webb, G. I. & Petitjean, F. (2019). Temporal convolutional neural network for the classification of
  satellite image time series. *Remote Sensing* 11, 523.
