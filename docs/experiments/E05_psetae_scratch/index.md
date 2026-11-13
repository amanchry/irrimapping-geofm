# E05 · PSE-TAE trained from scratch on Sentinel-2 pixel sets (neural baseline)

*A temporal deep model trained only on our labelled parcels, with no pretraining: the neural baseline that a frozen GeoFM has to beat.*

| | |
| --- | --- |
| **Status** | 🟦 ready to run (after E02; code tested on both regions) |
| **Research question** | RQ1 — Q4 of the [experiment plan](../../phase2_models/experiment_plan.md#4-design-what-is-varied) |
| **Builds on** | [E02](../E02_frozen_pca/index.md) |
| **Config** | [`config.yaml`](https://github.com/amanchry/irrimapping-geofm/blob/main/experiments/E05_psetae_scratch/config.yaml) |
| **Report** | written when the results are in |

## What this experiment builds on

Every method choice of this experiment and the paper it comes from (full list: [Reference papers](../../references/index.md)).

| What this experiment takes | Reference |
| --- | --- |
| Label efficiency of foundation embeddings for cropland mapping | Mughees et al. (2026), [From foundation embeddings to cropland maps: label efficiency, temporal transferability …](https://arxiv.org/abs/2609.17138) |
| Embeddings built for mapping from sparse labels | Brown et al. (2025), [AlphaEarth Foundations … from sparse label data](https://arxiv.org/abs/2507.22291) |
| PSE-TAE: pixel-set encoder + temporal attention, 64 pixels per parcel | Garnot, Landrieu, Giordano & Chehata (2020), [PSE-TAE](https://arxiv.org/abs/1911.07757) |
| Validation rotation; early stopping on the validation fold | Garnot & Landrieu (2021), [PASTIS / U-TAE](https://arxiv.org/abs/2107.07933) · [summary](../../references/pastis-utae.md) |
| A neural time-series model beat RF for irrigated mapping | Bazzi et al. (2019), [Mapping irrigated areas using Sentinel-1 time series in Catalonia](https://doi.org/10.3390/rs11151836) |
| Temporal deep models vs GeoFM embeddings under geographic shift | Tong & Wang (2025), [Invariant features for global crop type classification](https://arxiv.org/abs/2509.03497) · [summary](../../references/tong-wang-2025.md) |
| Spatial blocks (20 km) instead of random splits | Roberts et al. (2017), [Cross-validation strategies for data with … spatial … structure](https://doi.org/10.1111/ecog.02881) |
| Why random CV overstates accuracy (shown in every run) | Ploton et al. (2020), [Spatial validation reveals poor predictive performance …](https://doi.org/10.1038/s41467-020-18321-y) |
| Confidence intervals by resampling whole blocks | Künsch (1989), [The jackknife and the bootstrap for general stationary observations](https://doi.org/10.1214/aos/1176347265) |
| Paired differences between inputs with CIs | Foody (2009), [Classification accuracy comparison …](https://doi.org/10.1016/j.rse.2009.03.014) |
| Correction for testing many input × head pairs | Holm (1979), A simple sequentially rejective multiple test procedure, *Scand. J. Stat.* 6 |
| No kappa coefficient | Pontius & Millones (2011), [Death to Kappa](https://doi.org/10.1080/01431161.2011.552923) |

## Question
Does a temporal deep model trained only on our labelled parcels, with no pretraining, match the frozen foundation
models?

## Why
A foundation model is only useful if it beats what a model trained from scratch on the same labels achieves.
PSE-TAE (Garnot et al. 2020) was built for parcel classification from satellite time series: it samples pixels
inside each parcel and learns the temporal pattern with attention. It is the neural baseline of RQ1, next to the
classical Sentinel-2 feature baseline.

## Setup
| Setting | Value |
| --- | --- |
| Input | pixel sets: inner pixels of the parcel (edge pixels within 10 m left out; parcels with < 4 inner pixels keep all), 10 Sentinel-2 bands × 12 months ([D14](../../decisions.md#d14-sentinel-2-only-for-every-input)) |
| Model | PSE-TAE: 64 pixels sampled per parcel; mean of 5 random draws at prediction |
| Training | Adam 1e-3, weight decay 1e-4, batch 128, ≤ 100 epochs, early stopping on validation macro-F1 (patience 15) |
| Validation | PASTIS rotation: 3 folds train, next fold validation, 1 fold test |
| Seeds | 3 (mean ± SD; out-of-fold prediction = mean probability) |
| Runs | `morocco_s2` (`views/`), `utah_s2` (`views_2023-24/`); `morocco_s2s1` only for the Sentinel-1 ablation (A5) |

## Label efficiency (part of this experiment)

Every input is also trained on **10, 25 and 50% of the training blocks**, so the report shows a learning curve next
to the main result (Q5 of the [experiment plan](../../phase2_models/experiment_plan.md#6-label-efficiency-inside-e02e05)).

| Setting | Value |
| --- | --- |
| Fractions | 10, 25, 50% of the training blocks of each round; 100% = the main run |
| Sampling | whole blocks: rainfed and irrigated-only blocks each put in a random order (per region, round and draw), and a fraction takes the first blocks of each list, so the draws are **nested** (10% ⊂ 25% ⊂ 50%); the order is accepted only if its first 10% hold ≥ 20 rainfed parcels. No fraction below 10%. |
| Draws | 3 per fraction; draw *k* is the same in E02–E05 and for every input (paired) |
| Seeds | 1 seed per draw (the main run keeps its 3 seeds) |
| Validation and test folds | unchanged |
| Runs | 1 input (pixel sets) × 3 fractions × 3 draws = **9** per region, for PSE-TAE |
| Result | macro-F1 and rainfed F1 against the training fraction, one line per input, Morocco and Utah |

## How to run

```bash
# on the Mac (all regions, main and label-efficiency runs, compare, report):
#   experiments/run_experiment.py:  EXPERIMENT = "E05_psetae_scratch"
python experiments/run_experiment.py
# on the cluster: one Slurm job per run, then one compare job per region (EXPERIMENT and DRY_RUN in the script)
python experiments/submit_slurm.py
```

- **Order:** after E02 (compares with its baseline and GeoFM runs).
- **Hardware:** GPU. `QUICK = True` checks the whole chain in a few minutes first.
- **Results:** `experiments/E05_psetae_scratch/results/<region>/`; the report page is written by the `report` step on the Mac.

## Outcome
To be filled in when the results are in.
