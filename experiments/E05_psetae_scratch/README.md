# E05 · PSE-TAE trained from scratch

| | |
| --- | --- |
| **Status** | 📝 planned — 3 seeds per region; setup page: `docs/experiments/E05_psetae_scratch/index.md` |
| **Config** | [`config.yaml`](config.yaml) |
| **Script** | `models_run/03b_train_psetae.py` |

## Question
Does a temporal deep model trained only on our labelled parcels (no pretraining) match the frozen foundation
models?

## Status
A first run on the 128 px tile store exists (outer 5 spatial folds, 3 folds train, next fold early stopping), in
`results/psetae_agyear_s2s1/`. It is kept here and not reported until the next phase.

## Runs (one config file each, in `configs/`)

Sentinel-2 only for every input ([D14](../../docs/decisions.md#d14-sentinel-2-only-for-every-input)):
`config.yaml` runs `morocco_s2` and `utah_s2`. `morocco_s2s1` is the optional Sentinel-1 ablation; `utah_s2s1`
is kept but not run (Sentinel-1 is thin in Utah after the loss of Sentinel-1B).

| Config | Region · views | Bands | Results |
| --- | --- | --- | --- |
| `morocco_s2.yaml` | Morocco · `views` (2023/24 + 2024/25) | 10 Sentinel-2 bands | `results/psetae_morocco_views_s2/` |
| `morocco_s2s1.yaml` | Morocco · `views` | + Sentinel-1 VV, VH (ascending) | `results/psetae_morocco_views_s2s1/` |
| `utah_s2.yaml` | Utah · `views_2023-24` | 10 Sentinel-2 bands | `results/psetae_utah_views_2023-24_s2/` |
| `utah_s2s1.yaml` | Utah · `views_2023-24` | + Sentinel-1 VV, VH (descending) | `results/psetae_utah_views_2023-24_s2s1/` |

## How to run
```bash
# all four in a row — experiments/run_experiment.py:  EXPERIMENT = "E05_psetae_scratch"
python experiments/run_experiment.py
# one run (e.g. in a Slurm job), without editing a script:
EXPERIMENT=E05_psetae_scratch PSETAE_CONFIG=utah_s2s1.yaml python models_run/03b_train_psetae.py
```
