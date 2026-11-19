# E02 · Temporal neural networks

!!! info "Status: ready to run"
    Script written and checked on a 20,000-pixel subset. Runs after E01.

**Question.** Does a temporal neural network beat the classical baseline (E01) on the same pixels and the same
input?

**What changes from E01.** Only the model: the input is the same 12-month Sentinel-2 series with NDVI and NDWI.

| | E01 | E02 |
| --- | --- | --- |
| Model | random forest, gradient boosting | TempCNN; L-TAE on single pixels |
| Input format | 168 features (months concatenated) | sequence of 12 months × 14 channels |
| Missing months | interpolated | TempCNN: interpolated; L-TAE: interpolated and masked in the attention |

## Set-up

| Item | Choice |
| --- | --- |
| **TempCNN** | Pelletier et al. (2019): three 1-D convolutions over the months (64 filters, kernel 5, batch normalisation, ReLU, dropout 0.2), dense layer of 256, output 3 classes |
| **L-TAE** | Garnot & Landrieu (2020): 16 heads, d<sub>k</sub> 8, d<sub>model</sub> 256, output MLP 256–128, sinusoidal encoding of the day in the agricultural year; months without a clear observation get no attention |
| **Normalisation** | per channel with the training pixels of each round; values still missing after interpolation set to 0 (the training mean) |
| **Training** | AdamW (learning rate 10⁻³, weight decay 10⁻⁴), batch 512, class-weighted cross-entropy, early stopping on the validation fold's 3-class macro-F1 (patience 10, at most 100 epochs); the best epoch is kept |
| **Seeds** | 3 per round (42, 1042, 2042); the out-of-fold probability is their mean |
| **Validation and metrics** | as E01 |

L-TAE also saves the mean attention of every month per class on the test folds (`attention_by_month.csv`), which
shows which months the model relies on.

## How to run

Script: [`experiments/E02_temporal_nn/run.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E02_temporal_nn/run.py)
(networks: [`pixelwise_run/nn_common.py`](https://github.com/amanchry/irrimapping-geofm/blob/main/pixelwise_run/nn_common.py)).

```bash
sbatch --export=ALL,SCRIPT=experiments/E02_temporal_nn/run.py,EXPERIMENT_REGION=morocco jobs/step_gpu.sbatch
```

## Outputs

`experiments/E02_temporal_nn/results/<region>/<tempcnn|ltae>/`: the E01 files, plus `training_history.csv` and
`attention_by_month.csv` (L-TAE).

## Results

*To be filled after the run.* Paired difference to the E01 random forest from `pixelwise_run/05_compare_experiments.py`.

| Region | Model | Cropland macro-F1 | 3-class macro-F1 | Δ vs E01 RF |
| --- | --- | --- | --- | --- |
| Morocco | TempCNN | | | |
| Morocco | L-TAE | | | |
| Utah | TempCNN | | | |
| Utah | L-TAE | | | |

## References

- Garnot, V. S. F. & Landrieu, L. (2020). Lightweight temporal self-attention for classifying satellite image time
  series. *AALTD*.
- Pelletier, C., Webb, G. I. & Petitjean, F. (2019). Temporal convolutional neural network for the classification of
  satellite image time series. *Remote Sensing* 11, 523.
