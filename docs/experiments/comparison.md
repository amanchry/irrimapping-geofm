# Comparison of the experiments

All pixel-wise runs of E01–E04 side by side, on the **same out-of-fold pixels** of each region, with the paired
tests that answer RQ1. Each experiment page reports only its own runs. E05 (Prithvi frozen / LoRA / fine-tuned /
from scratch) is compared within itself on its [own page](E05_prithvi_strategies.md).

Tables and tests: `pixelwise_run/05_compare_experiments.py` (`experiments/comparison/<region>/`); figures:
`experiments/report_figures.py`. The pixel-wise L-TAE run of E02 is on hold and not included.

## Scores

Cropland macro-F1 (mean F1 of irrigated and rainfed, headline) and 3-class macro-F1, out of fold, with 95%
block-bootstrap CI:

| Region | Run | Cropland macro-F1 | 3-class macro-F1 | F1 irrigated | F1 rainfed | F1 not cropland |
| --- | --- | --- | --- | --- | --- | --- |
| Morocco | E01 random forest | 0.876 [0.801, 0.929] | 0.838 [0.773, 0.887] | 0.959 | 0.784 | 0.770 |
| Morocco | E01 gradient boosting | 0.881 [0.801, 0.937] | 0.850 [0.783, 0.904] | 0.962 | 0.795 | 0.792 |
| Morocco | E02 TempCNN | 0.886 [0.798, 0.944] | 0.851 [0.784, 0.903] | 0.960 | 0.807 | 0.786 |
| Morocco | E03 U-TAE | 0.897 [0.843, 0.936] | 0.859 [0.815, 0.896] | 0.961 | 0.814 | 0.802 |
| Morocco | E04 Prithvi | 0.888 [0.822, 0.939] | 0.871 [0.812, 0.918] | 0.968 | 0.794 | 0.852 |
| Morocco | E04 Clay | 0.894 [0.822, 0.949] | 0.884 [0.812, 0.939] | 0.974 | 0.808 | 0.868 |
| Morocco | E04 TerraMind | 0.893 [0.816, 0.950] | 0.884 [0.806, 0.940] | 0.975 | 0.805 | 0.871 |
| Morocco | E04 AgriFM | 0.900 [0.842, 0.946] | 0.871 [0.816, 0.916] | 0.967 | 0.815 | 0.831 |
| Utah | E01 random forest | 0.729 [0.682, 0.765] | 0.746 [0.714, 0.772] | 0.870 | 0.559 | 0.809 |
| Utah | E01 gradient boosting | 0.726 [0.679, 0.763] | 0.750 [0.718, 0.776] | 0.875 | 0.552 | 0.824 |
| Utah | E02 TempCNN | 0.721 [0.674, 0.762] | 0.745 [0.712, 0.775] | 0.862 | 0.545 | 0.829 |
| Utah | E03 U-TAE | 0.766 [0.725, 0.797] | 0.766 [0.739, 0.791] | 0.883 | 0.600 | 0.816 |
| Utah | E04 Prithvi | 0.790 [0.748, 0.823] | 0.798 [0.769, 0.822] | 0.905 | 0.627 | 0.863 |
| Utah | E04 Clay | 0.795 [0.751, 0.829] | 0.805 [0.777, 0.831] | 0.909 | 0.634 | 0.872 |
| Utah | E04 TerraMind | 0.797 [0.757, 0.829] | 0.805 [0.777, 0.829] | 0.908 | 0.639 | 0.867 |
| Utah | E04 AgriFM | 0.784 [0.739, 0.816] | 0.791 [0.762, 0.815] | 0.900 | 0.621 | 0.852 |

![Scores of E01–E04](figures/scores_e03_e04.png)

## Paired tests

Difference to a reference run on the same pixels (block bootstrap, 95% CI, Holm correction over the seven runs
compared with the same reference; **bold** = CI excludes 0 and p<sub>Holm</sub> < 0.05). Each cell: difference
[95% CI], p<sub>Holm</sub>.

### Against the E01 random forest (does a model beat the pixel baseline?)

| Region | Run | Δ cropland macro-F1 | Δ 3-class macro-F1 |
| --- | --- | --- | --- |
| Morocco | E01 gradient boosting | +0.005 [−0.007, 0.021], 1.00 | **+0.012** [0.003, 0.024], 0.018 |
| Morocco | E02 TempCNN | +0.010 [−0.013, 0.032], 1.00 | +0.013 [−0.005, 0.030], 0.22 |
| Morocco | E03 U-TAE | +0.021 [−0.012, 0.063], 0.95 | +0.021 [−0.003, 0.048], 0.22 |
| Morocco | E04 Prithvi | +0.012 [−0.019, 0.041], 1.00 | **+0.033** [0.011, 0.053], 0.016 |
| Morocco | E04 Clay | +0.018 [−0.001, 0.038], 0.41 | **+0.046** [0.028, 0.068], < 0.001 |
| Morocco | E04 TerraMind | +0.017 [−0.002, 0.037], 0.45 | **+0.046** [0.026, 0.071], < 0.001 |
| Morocco | E04 AgriFM | +0.024 [0.002, 0.050], 0.24 | **+0.033** [0.017, 0.050], 0.010 |
| Utah | E01 gradient boosting | −0.003 [−0.015, 0.008], 0.62 | +0.004 [−0.004, 0.013], 0.61 |
| Utah | E02 TempCNN | −0.008 [−0.022, 0.007], 0.62 | −0.001 [−0.010, 0.009], 0.87 |
| Utah | E03 U-TAE | **+0.037** [0.019, 0.053], < 0.001 | **+0.020** [0.008, 0.032], < 0.001 |
| Utah | E04 Prithvi | **+0.061** [0.044, 0.080], < 0.001 | **+0.052** [0.039, 0.066], < 0.001 |
| Utah | E04 Clay | **+0.065** [0.047, 0.085], < 0.001 | **+0.059** [0.045, 0.073], < 0.001 |
| Utah | E04 TerraMind | **+0.068** [0.046, 0.091], < 0.001 | **+0.058** [0.043, 0.075], < 0.001 |
| Utah | E04 AgriFM | **+0.055** [0.038, 0.072], < 0.001 | **+0.045** [0.034, 0.056], < 0.001 |

### Against E03 U-TAE (does pretraining add anything beyond the spatial context of the tile?)

| Region | Run | Δ cropland macro-F1 | Δ 3-class macro-F1 |
| --- | --- | --- | --- |
| Morocco | E01 random forest | −0.021 [−0.063, 0.012], 1.00 | −0.021 [−0.048, 0.003], 0.77 |
| Morocco | E01 gradient boosting | −0.015 [−0.065, 0.022], 1.00 | −0.009 [−0.041, 0.019], 1.00 |
| Morocco | E02 TempCNN | −0.011 [−0.063, 0.022], 1.00 | −0.008 [−0.043, 0.019], 1.00 |
| Morocco | E04 Prithvi | −0.009 [−0.031, 0.009], 1.00 | +0.012 [−0.006, 0.028], 0.82 |
| Morocco | E04 Clay | −0.002 [−0.042, 0.023], 1.00 | +0.024 [−0.009, 0.051], 0.82 |
| Morocco | E04 TerraMind | −0.004 [−0.049, 0.027], 1.00 | +0.025 [−0.013, 0.054], 0.82 |
| Morocco | E04 AgriFM | +0.003 [−0.017, 0.020], 1.00 | +0.012 [−0.004, 0.027], 0.77 |
| Utah | E01 random forest | **−0.037** [−0.053, −0.019], < 0.001 | **−0.020** [−0.032, −0.008], < 0.001 |
| Utah | E01 gradient boosting | **−0.040** [−0.061, −0.016], < 0.001 | **−0.016** [−0.030, −0.0004], 0.048 |
| Utah | E02 TempCNN | **−0.044** [−0.066, −0.021], < 0.001 | **−0.021** [−0.035, −0.006], 0.028 |
| Utah | E04 Prithvi | **+0.025** [0.007, 0.045], 0.020 | **+0.032** [0.018, 0.046], < 0.001 |
| Utah | E04 Clay | **+0.029** [0.011, 0.047], < 0.001 | **+0.039** [0.026, 0.050], < 0.001 |
| Utah | E04 TerraMind | **+0.031** [0.013, 0.050], < 0.001 | **+0.038** [0.024, 0.052], < 0.001 |
| Utah | E04 AgriFM | +0.018 [−0.001, 0.038], 0.064 | **+0.024** [0.012, 0.038], < 0.001 |


## Errors

![Confusion matrices of the random forest, U-TAE and TerraMind](figures/confusion_e03_e04.png)

The random forest (pixels), U-TAE (tiles, from scratch) and TerraMind (tiles, frozen foundation model) as examples
of the three kinds of model; the confusion matrices of every run are on the experiment pages.

## Findings

- **Utah: the frozen foundation models are clearly better.** All four beat the random forest by 0.055–0.068
  cropland macro-F1, and three of them (Prithvi, Clay, TerraMind) also beat U-TAE by 0.025–0.031. Spatial context
  alone (U-TAE) adds +0.037 over the random forest; the pretrained features add about as much again.
- **Morocco: no significant difference on the headline metric.** All runs lie between 0.876 and 0.900, with CIs
  0.09–0.14 wide; no model differs significantly from the random forest or from U-TAE. On the 3-class score the four
  foundation models beat the random forest (+0.033 to +0.046), through the not-cropland class.
- **The temporal model on single pixels does not help:** TempCNN and gradient boosting are on par with the random
  forest in both regions.
- **The four foundation models are statistically indistinguishable** from each other in both regions
  ([E04](E04_frozen_geofm.md)).
