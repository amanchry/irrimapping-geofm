# Pixel-wise experiments

E01–E04 classify every 10 m pixel as irrigated, rainfed or not cropland. Each folder holds a `run.py` with its
settings at the top and writes `results/<region>/<run>/`; the shared code is in `../pixelwise_run/`
(`05_compare_experiments.py` there compares finished runs). The archived parcel-wise experiments are in
`parcel_wise_classification/` (scripts in `../parcelwise_run/`).

Foundation models in E04: Prithvi, Clay, TerraMind, AgriFM, Presto and
TESSERA (all six loaded by `pixelwise_run/encoders.py`; code cloned to `external/`, weights in
`checkpoints/<model>/`).

E05 (LoRA fine-tuning, RQ2) is deferred: `E05_lora/run.py` and its draft page `E05_lora/page_draft.md`
are kept but not run or published.

Documentation (set-up, how to run, results): https://amanchry.github.io/irrimapping-geofm/experiments/
