# models_run/

Part 2 · study: dataset inspection, embeddings, training and evaluation, comparison (scripts 01–05).

| Script | What it does |
| --- | --- |
| `01_inspect_dataset.py` | checks of the dataset (inventory, observations, purity, profiles) |
| `02a–02d_embed_*.py` | frozen foundation-model embeddings per parcel (Clay, Prithvi, TerraMind, AgriFM), per region |
| `03_train_evaluate.py` | LogReg / RF / SVM on the S2 baseline or on embeddings (E02) |
| `03b_train_psetae.py` | PSE-TAE from scratch on the pixel sets (E05) |
| `03c_train_ltae.py` | MLP head on the PCA input (E03) or L-TAE + MLP on the monthly sequence (E04) |
| `04_compare_models.py` | all inputs on the same parcels: CI, differences vs the baseline (Holm), learning curves |
| `05_breakdown_oof.py` | results by zone, season, patch purity and parcel area |
| `experiment_utils.py` | shared helpers: region paths, environment overrides, label-efficiency block draws |

Scripts 03–05 run per region (`REGION`) and are normally started by `experiments/run_experiment.py` (or
`experiments/submit_slurm.py` on the cluster).

Each script has its settings at the top and a description of what it does and writes.

**Documentation:** [https://amanchry.github.io/irrimapping-geofm/phase2_models/frozen_encoders/](https://amanchry.github.io/irrimapping-geofm/phase2_models/frozen_encoders/) (source: [`docs/phase2_models/frozen_encoders.md`](../docs/phase2_models/frozen_encoders.md))
