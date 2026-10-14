# Parcel feature extraction

One script per data source. Each writes one table with one row per parcel (`parcel_id` index),
so sources can be combined later with a join. No train / test logic here.

| Script | Source | Columns | Output |
|---|---|---|---|
| `extract_s2.py` | Sentinel-2 L2A, Planetary Computer | B02–B12, NDVI, EVI, NDWI, GCVI, NDRE per month: `NDVI_m09` … | `s2_<start>_<end>.parquet` |
| `extract_s1.py` | Sentinel-1 RTC, Planetary Computer | VV, VH, VHVV (dB) per month: `VV_m09` … | `s1_<start>_<end>.parquet` |
| `extract_alphaearth.py` | AlphaEarth Zarr, Source Cooperative | A00–A63 | `alphaearth_<year>.parquet` |

Every table also has `n_px`: the number of 10 m pixels averaged for that parcel.
Monthly columns use the calendar month (`_m09` = September), so tables from different seasons share column names.

## Run

```bash
pip install -r requirements.txt          # or the conda-forge equivalents, Python >= 3.11
# edit the SETTINGS block at the top of a script (paths, months, workers ...), then:
python extract_s2.py
python extract_s1.py
python extract_alphaearth.py
```

Long runs on a server: `nohup python extract_s2.py > s2.out 2>&1 &`

Quick test first: set `LIMIT_TILES = 3` in the settings block.

## How it works

- **Per parcel:** the mean of all 10 m pixels inside the parcel, after shrinking it by 5 m to drop mixed edge pixels
  (parcels too small for that use every pixel they touch).
- **Tiles:** parcels are grouped into 10 km tiles; each tile's data window is loaded once and all its parcels are
  processed together. This is far faster than one cloud request per parcel.
- **Sentinel-2:** SCL cloud mask (keeps vegetation, bare soil, water, unclassified), monthly median, the +1000
  reflectance offset of processing baseline ≥ 04.00 removed, indices computed from the monthly composite.
- **Sentinel-1:** monthly median γ⁰ in linear power, parcel mean in linear power, then dB.
- **AlphaEarth:** int8 values de-quantised `(q/127.5)² · sign(q)`, −128 = no-data.

## Progress, errors, resume

- Progress bar in parcels; a log per run in `outputs/features/logs/`.
- Each tile is written to `outputs/features/_parts/<run>/<tile>.parquet` as soon as it finishes.
- A failing tile (network, timeout …) is retried (`RETRIES`), then recorded in `<tile>.failed.json` with the
  error and traceback; the run continues.
- **Re-running the script resumes:** finished tiles are skipped, failed ones are retried.
- At the end: `<run>.parquet` (features), `<run>_summary.json` (counts, failed tiles, settings) and
  `<run>_failed_parcels.csv`.
- `MERGE_ONLY = True` rebuilds the final table from the finished tiles without downloading anything.

## Combine later

```python
import pandas as pd
s2 = pd.read_parquet("outputs/features/s2_2023-09_2024-08.parquet").drop(columns="n_px")
s1 = pd.read_parquet("outputs/features/s1_2023-09_2024-08.parquet").drop(columns="n_px")
features = s2.join(s1, how="inner")      # then join labels / folds on parcel_id
```

New sources (precipitation, ETa …) follow the same pattern: a `tile_features(parcels)` function that returns
one row per parcel, run through `common.run_extraction`.
