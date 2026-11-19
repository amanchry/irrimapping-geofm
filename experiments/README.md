# Pixel-wise experiments

E01–E05 classify every 10 m pixel as irrigated, rainfed or not cropland. Each folder holds a `run.py` with its
settings at the top and writes `results/<region>/<run>/`; the shared code is in `../pixelwise_run/`
(`05_compare_experiments.py` there compares finished runs). The archived parcel-wise experiments are in
`parcel_wise_classification/` (scripts in `../parcelwise_run/`).

Foundation models in E03 / E05: Prithvi, Clay, TerraMind, AgriFM, Presto and
TESSERA (all six loaded by `pixelwise_run/encoders.py`; code cloned to `external/`, weights in
`checkpoints/<model>/`).

Documentation (set-up, how to run, results): https://amanchry.github.io/irrimapping-geofm/experiments/
