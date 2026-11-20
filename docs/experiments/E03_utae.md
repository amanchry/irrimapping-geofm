# E03 · U-TAE from scratch

!!! info "Status: planned (Morocco first, Utah later)"
    Same labelled pixels as [E01](E01_s2_pixel_baseline.md) (Morocco 2023/24 and 2024/25 pooled, 473,675 pixels in
    2,015 tiles; two classes), but the model sees the whole 128 px tile; 3 seeds per round on one GPU. Code ready
    and tested; results follow after the run.

**Question.** Does a spatio-temporal network trained from scratch on the whole tile beat the pixel models (E01,
E02)? It is also the reference for the frozen foundation models of [E04](E04_frozen_geofm.md): it shows how much the
pretrained encoders add beyond spatial context.

**What changes from E02.** The model sees the whole 128 px tile (spatial context) instead of one pixel. The
labelled pixels, folds and metrics are the same.

## Model

U-TAE (Garnot & Landrieu 2021), the PASTIS semantic-segmentation baseline: a convolutional encoder shared by all
months, a lightweight temporal attention encoder (L-TAE) at the coarsest resolution that produces one attention
mask per head and pixel, these masks aggregate the skip features of every resolution over time, and a
convolutional decoder brings the result back to 10 m.

| Item | Choice |
| --- | --- |
| **Input** | 12 Sentinel-2 bands × 12 months of the whole tile, standardised per band with the training pixels of the round; missing values 0, months without any clear pixel masked in the attention |
| **Widths** | encoder 64–64–64–128, decoder 32–32–64–128 (the PASTIS configuration) |
| **Temporal attention** | L-TAE at 1/8 resolution: 16 heads, d<sub>k</sub> 4, d<sub>model</sub> 256 |
| **Output** | 2 class scores (irrigated, rainfed) for every 10 m pixel of the tile |
| **Parameters** | 1.15 M, all trained from random weights |

??? info "How U-TAE works"
    U-TAE (U-Net with Temporal Attention Encoder; Garnot & Landrieu 2021) classifies every pixel of an image time
    series. It combines a **U-Net**, which sees the neighbourhood of each pixel at several resolutions, with
    **temporal attention**, which decides which months matter at each place.

    ```mermaid
    flowchart TD
        A["tile 128 × 128 px · 12 months · 12 bands"] --> B["CNN encoder, the same for every month:<br/>maps at 128², 64², 32², 16² px"]
        B --> C["L-TAE on the 16² map:<br/>month weights per position (16 heads)"]
        C --> D["month weights upsampled and applied at every size:<br/>12 months → one map per size"]
        D --> E["U-Net decoder: 16² → 128²,<br/>adding the map of each size (skip connections)"]
        E --> F["2 class scores per 10 m pixel"]
    ```

    1. **Encoder per month.** A small convolutional network describes the image of each month on its own (the same
       weights for all 12 months) and halves the resolution three times: 128, 64, 32 and 16 px.
    2. **Temporal attention at the coarsest level.** At each position of the 16 × 16 map, an L-TAE looks at the 12
       months and gives each one a weight (per head), from the data of that position: months without a clear
       observation get no weight.
    3. **The same month weights at every level.** The weights are upsampled and used at all four resolutions to
       collapse the 12 months into one map per resolution.
    4. **Decoder.** Starting from the coarsest map, the decoder upsamples step by step and adds the map of the same
       resolution from step 3 (skip connections), so the result keeps the 10 m detail of the image while using the
       context of the coarser maps.

    Everything is trained from random weights on our labels only; it is the counterpart of E04 without
    pretraining.

??? info "How L-TAE works (temporal attention)"
    L-TAE (Lightweight Temporal Attention Encoder; Garnot & Landrieu 2020) turns a sequence of 12 monthly feature
    vectors into one vector. Each month gets a code for its position in the agricultural year; each of the 16 heads
    has a learnt **master query** and scores every month against it; a softmax turns the scores into weights that
    sum to 1 (months without data get weight 0); each head takes the weighted average of the months, and the heads
    are joined. Unlike a convolution, the weights depend on the data of each pixel or position. The same module is
    the temporal head of the foundation models in E04 and E05.

## Training

| Item | Choice |
| --- | --- |
| **Labels** | the labelled pixels of the round's training folds; every other pixel of the tile is ignored in the loss. A tile with pixels of two folds contributes only its training-fold labels |
| **Loss** | class-weighted cross-entropy (weights from the training pixels: rainfed counts about 8 times more) |
| **Optimiser** | AdamW, learning rate 10⁻³, weight decay 10⁻⁴, batch 4 tiles |
| **Augmentation** | random flips and 90° rotations |
| **Early stopping** | macro-F1 of the validation pixels, patience 10, at most 100 epochs |
| **Seeds** | 3 per round (42, 1042, 2042), mean probability |
| **Scoring** | the test tiles are predicted and read at their labelled pixels: the same out-of-fold pixels as every other experiment |

## How to run

Script: [`experiments/E03_utae/run.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E03_utae/run.py)
(model and training loop: [`pixelwise_run/tile_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/tile_common.py)).

```bash
sbatch --export=ALL,SCRIPT=experiments/E03_utae/run.py,EXPERIMENT_REGION=morocco jobs/step_gpu.sbatch
```

`EXPERIMENT_QUICK_TEST=1` runs the chain on 60 tiles for 2 epochs (output in `results_quick/`; too short to learn
the rainfed class, it only checks the code).

## Outputs

`experiments/E03_utae/results/<region>/utae/`: the E01 files plus `training_history.csv` (per epoch: training loss,
validation loss, validation macro-F1 and rainfed F1, seconds).

## Results

_To be filled after the Morocco run._ Planned: scores with 95% CI and the paired difference to the E01 random
forest, confusion matrix, scores per year and zone, training curves and run time. The comparison with the other
experiments is on the [comparison page](comparison.md).

## References

- Garnot, V. S. F. & Landrieu, L. (2020). Lightweight temporal self-attention for classifying satellite images time
  series (L-TAE). *AALTD*.
- Garnot, V. S. F. & Landrieu, L. (2021). Panoptic segmentation of satellite image time series with convolutional
  temporal attention networks. *ICCV*.
