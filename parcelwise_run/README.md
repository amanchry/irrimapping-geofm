# parcelwise_run/ (archive)

The parcel-wise pipeline as it was run for the archived experiments E01–E05 (one label per parcel; experiments in
`../experiments/parcel_wise_classification/`). Scripts are numbered in run order; run them from the project folder,
settings at the top of each script.

| Step | Script | What it does |
| --- | --- | --- |
| 01–03 | `../data_training_prep/` (shared) | labels, spatial folds, training tiles cut from the blocks of `../data_download/` (formerly copies in this folder; the cropland-mask check of 01 is no longer run) |
| 04 | `04_make_views.py` | parcel features and patch-aligned chips per parcel |
| 05 | `05_inspect_dataset.py` | checks of the dataset (inventory, observations, purity, profiles) |
| 06 | `06a_embed_clay.py`, `06b_embed_prithvi.py`, `06c_embed_terramind.py`, `06d_embed_agrifm.py` | frozen foundation-model embeddings per parcel |
| 07 | `07_train_evaluate.py` | LogReg / RF / SVM on the S2 baseline or the embeddings (archived E01, E02) |
| 07b | `07b_train_psetae.py` | PSE-TAE from scratch on the pixel sets (archived E05) |
| 07c | `07c_train_ltae.py` | MLP head (archived E03) or L-TAE + MLP (archived E04) |
| 08 | `08_compare_models.py` | all inputs on the same parcels: CI, differences vs the baseline (Holm), learning curves |
| 09 | `09_breakdown_oof.py` | results by zone, season, patch purity and parcel area |

Steps 07–09 run per region and are normally started by `../experiments/parcel_wise_classification/run_experiment.py`
(or `submit_slurm.py` on the cluster). Helper modules: `experiment_utils.py` (region paths, environment overrides,
label-efficiency block draws), `regions.py`, `cropland_mask.py`. Steps 01–03 are copies of the scripts in
`../pixelwise_run/`. The 01 copies already contain the cropland-mask check (step 0 of the pixel-wise pipeline),
which was added after the parcel-wise runs.

**Documentation:** [Archive](https://amanchry.github.io/irrimapping-geofm/archive/)
