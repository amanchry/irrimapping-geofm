# Archive

Finished work that is no longer the current approach. The pages, figures, result tables and scripts are kept
unchanged, so every archived result can still be read, checked and re-run.

| Archive | Period | What it is | Pages | Code and results |
| --- | --- | --- | --- | --- |
| **Parcel-wise classification** (E01–E05) | Sep – Nov 2026 | RQ1 on **one label per parcel**: frozen GeoFM embeddings (or Sentinel-2 features) averaged inside each parcel, then PCA or L-TAE and a classifier; PSE-TAE from scratch as the neural baseline. Morocco and Utah 2023/24. | [Results](parcel_wise_classification/results.md) · [Experiments](parcel_wise_classification/index.md) | [`experiments/parcel_wise_classification/`](https://github.com/amanchry/irrimapping-geofm/tree/main/experiments/parcel_wise_classification) |

## Why the parcel-wise experiments were archived

The aim is an **annual, continuous (wall-to-wall) map** of irrigated and rainfed cropland. A parcel-wise model
needs a parcel boundary for every field it classifies, and boundaries exist only for the surveyed parcels. The
experiments therefore move to **pixel-wise classification**: every 10 m pixel gets its own prediction, so the
model can be applied to the whole area ([pixel-wise experiments](../experiments/index.md)).

The parcel-wise results remain the RQ1 evidence for which inputs separate irrigated from rainfed parcels:

- **Morocco:** a tie at about 0.91 macro-F1 (PSE-TAE 0.911, S2 + MLP 0.908, Clay + L-TAE 0.906).
- **Utah 2023/24:** Clay ahead by about 0.03 (Clay + SVM 0.791 vs S2 + RF 0.764).
- The classifier and the temporal aggregation change little once the classifier is non-linear; AgriFM is weakest.

## Re-running an archived experiment

The scripts of the archive run from their new folder; the training scripts in `models_run/` are shared and write
into the archive's `results/` folders.

```bash
# experiments/parcel_wise_classification/run_experiment.py:  EXPERIMENT = "E02_frozen_pca"
python experiments/parcel_wise_classification/run_experiment.py
# on the cluster
python experiments/parcel_wise_classification/submit_slurm.py
```
