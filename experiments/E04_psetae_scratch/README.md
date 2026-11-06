# E04 · PSE-TAE trained from scratch (parked)

| | |
| --- | --- |
| **Status** | ⏸ parked — presented in the next phase; not on the overview pages yet (`listed: false`) |
| **Config** | [`config.yaml`](config.yaml) |
| **Script** | `models_run/03b_train_psetae.py` |

## Question
Does a temporal deep model trained only on our labelled parcels (no pretraining) match the frozen foundation
models?

## Status
A first run on the 128 px tile store exists (outer 5 spatial folds, 3 folds train, next fold early stopping), in
`results/psetae_agyear_s2s1/`. It is kept here and not reported until the next phase.

## How to run
```bash
# experiments/run_experiment.py:  EXPERIMENT = "E04_psetae_scratch"
python experiments/run_experiment.py
```
