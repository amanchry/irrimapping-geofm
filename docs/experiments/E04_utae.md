# E04 · U-TAE from scratch

!!! info "Status: ready to run"
    Script written and checked on a small tile subset. Runs before E03: it tests the tile training pipeline
    without the large encoders.

**Question.** Does a spatio-temporal network of the same input, trained from scratch, match the frozen foundation
models (E03)? It shows how much the pretrained encoders add.

**What changes from E02.** The model sees the whole 128 px tile (spatial context) instead of one pixel. The
labelled pixels, folds and metrics are the same.

## Model

U-TAE (Garnot & Landrieu 2021), the PASTIS semantic-segmentation baseline: a convolutional encoder shared by all
months, a lightweight temporal attention encoder (L-TAE) at the coarsest resolution that produces one attention
mask per head and pixel, these masks aggregate the skip features of every resolution over time, and a
convolutional decoder brings the result back to 10 m.

| Item | Choice |
| --- | --- |
| **Input** | 12 Sentinel-2 bands × 12 months, standardised per band with the training pixels of the round; missing values 0, months without any clear pixel masked in the attention |
| **Widths** | encoder 64–64–64–128, decoder 32–32–64–128 (the PASTIS configuration) |
| **Temporal attention** | L-TAE at 1/8 resolution: 16 heads, d<sub>k</sub> 4, d<sub>model</sub> 256 |
| **Parameters** | 1.15 M, all trained |
| **Training** | as E03 (masked class-weighted cross-entropy at the training-fold pixels, flips and rotations, early stopping on the validation pixels), but learning rate 10⁻³, batch 4 tiles, patience 10, at most 100 epochs; 3 seeds |

## How to run

Script: [`experiments/E04_utae/run.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E04_utae/run.py).

```bash
sbatch --export=ALL,SCRIPT=experiments/E04_utae/run.py,EXPERIMENT_REGION=morocco jobs/step_gpu.sbatch
```

## Outputs

`experiments/E04_utae/results/<region>/utae/`: the E01 files plus `training_history.csv`.

## Results

*To be filled after the run.*

| Region | Cropland macro-F1 | 3-class macro-F1 | Δ vs E01 RF |
| --- | --- | --- | --- |
| Morocco | | | |
| Utah | | | |

## References

- Garnot, V. S. F. & Landrieu, L. (2021). Panoptic segmentation of satellite image time series with convolutional
  temporal attention networks. *ICCV*.
