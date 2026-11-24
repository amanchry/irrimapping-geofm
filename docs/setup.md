# Getting started

This page specifies the software environment, the required input data and model weights, the execution order of
the scripts and the structure of the repository.

## Setup

### 1. Environment

```bash

# creates "condaml" (or: conda env update -n condaml -f environment.yml)
conda env create -f environment.yml                 
conda activate condaml

# Clay (--no-deps: avoids an einops conflict with TerraTorch)
pip install --no-deps git+https://github.com/Clay-foundation/model.git  

# AgriFM
pip install mmengine                               
```

A GPU (CUDA or Apple Silicon) is recommended for the tile models (experiments E02–E05). Everything else runs on a CPU.

### 2. Earth Engine

Every download of `data_download/` uses Earth Engine with a service account: the block downloads
(`02a_dynamic_world.py`, `02b_s2.py`, `02c_s1.py`), the union cropland mask (`00_union_cropland_mask.py`, kept for
later) and the annual Dynamic World reference layer (`04c_ref_dynamic_world_annual.py`). Put the service-account key in `secrets/` (path
set in `data_download/download_common.py` and at the top of the scripts that log in themselves). ESA WorldCover
(`04a_ref_esa_worldcover.py`) is read from the Planetary Computer STAC catalogue and LGRIP30
(`04b_ref_lgrip30.py`) from NASA Earthdata, which needs a token in `secrets/earthdata_token.txt` or in the
environment variable `EARTHDATA_TOKEN`.

### 3. Input data

| File | Source | Where |
| --- | --- | --- |
| Field survey, Morocco (parcels with land use and irrigation method) | Ouassanouan et al. (2026), *Scientific Data* | `data/Morocco_field_survey/lulc_fielddata.shp` |
| Water-Related Land Use (WRLU), Utah, one layer per year | Utah Division of Water Resources | `data/Utah_field_data/` |
| Country and state boundaries | FAO GAUL 2015 (Morocco), 2024 (Utah) | `data/admin_boundaries/` |

The imagery is not downloaded by hand. `data_download/` fetches every source once per region and agricultural
year on one grid of 10.24 km blocks (1024 × 1024 px at 10 m), one script per source; `data_training_prep/04_cut_tiles.py`
cuts the 128 px training tiles out of these blocks, and the maps are predicted from the same blocks
([D20](decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)).

### 4. Model weights

All weights are stored in **`checkpoints/<model>/`** and are **downloaded
automatically** on the first run of each experiment that uses the model; later runs reuse them. Folders and sources per model:
[Foundation models · Weights](foundation_models/in_this_study.md#weights).

Only the Clay code needs installing by hand (`pip install --no-deps git+https://github.com/Clay-foundation/model.git`,
see *Environment* above); TerraTorch (Prithvi, TerraMind) and mmengine (AgriFM) come with `environment.yml`. Presto and TESSERA need no package: their code is cloned into `external/presto` and `external/tessera` on first use.

---

## Running

Run every script from the project folder. Each script has its **settings at the top**. A description at the top of each script explains what it does and what it writes.

The scripts are grouped in folders and numbered in run order within each folder. `data_training_prep/` builds the
labels, the spatial folds and the training tiles; `data_download/` downloads all imagery and the cropland mask on
the 10.24 km block grid ([D20](decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)); `pixelwise_run/` makes the pixel samples and compares the
experiments (current pipeline); `parcelwise_run/` is the archived parcel-wise pipeline, which uses the labels, folds
and tiles of `data_training_prep/`. The block plan reads the tile plan (blocks with training tiles are always downloaded)
and the training tiles are cut from the downloaded blocks, so the two folders run interleaved:

```bash
# 1. Labels, folds and tile plan (data_training_prep/; no download)
python data_training_prep/01_prepare_labels_Morocco.py      # labelled parcels (01_prepare_labels_Utah.py for Utah)
python data_training_prep/02_make_spatial_folds.py          # spatial folds
python data_training_prep/03_plan_tiles.py                  # tile plan, label rasters, tile index

# 2. Downloads on the 10.24 km block grid (data_download/; Earth Engine)
python data_download/01_plan_blocks.py                 # every block of the mapping area (no mask); training tiles counted per block
python data_download/02a_dynamic_world.py              # binary Dynamic World cropland image for every block
python data_download/02b_s2.py                         # Sentinel-2 monthly composites (Cloud Score+), cropland/training blocks only
python data_download/02c_s1.py                         # Sentinel-1 monthly composites, ascending / descending, same blocks
python data_download/03_build_vrt.py                   # one VRT mosaic per source and year

# 3. Training tiles (data_training_prep/; local)
python data_training_prep/04_cut_tiles.py                   # 128 px training tiles cut from the blocks
python data_training_prep/05_tile_quicklooks.py             # visual check of selected tiles

# 4. Pixel-wise experiments (pixelwise_run/, experiments/)
python pixelwise_run/04_make_pixel_samples.py          # labelled pixels, irrigated / rainfed (04_pixel_samples/<season>/)
python experiments/E01_s2_pixel_baseline/run.py        # experiments, one at a time: E01, E02, E03, E04, E05
python pixelwise_run/05_compare_experiments.py         # comparison table and paired tests

# Reference maps (data_download/; documentation and map comparison only)
python data_download/04a_ref_esa_worldcover.py         # ESA WorldCover 2021
python data_download/04b_ref_lgrip30.py                # LGRIP30 (Earthdata token)
python data_download/04c_ref_dynamic_world_annual.py   # annual Dynamic World layer (candidate mask)
python data_download/00_union_cropland_mask.py         # union of 4 cropland products (kept for later; not used by the download)
```

On the cluster each run is a Slurm job (`jobs/step_cpu.sbatch` / `jobs/step_gpu.sbatch` with `SCRIPT=<script>`;
commands on each experiment page). `N_TEST_BLOCKS` in the download scripts and `N_TEST_TILES` in
`data_training_prep/03_plan_tiles.py` limit a first run to a few blocks or tiles; `EXPERIMENT_QUICK_TEST=1` checks a whole
experiment on a small subset first.
The archived parcel-wise pipeline (`parcelwise_run/`, steps 01–09; experiments in
`experiments/parcel_wise_classification/`) is described in `parcelwise_run/README.md` and the
[Archive](archive/index.md).

## Project structure

```
irrigated_area_mapping_code/
├── README.md                short project summary + link to this website
├── CLAUDE.md                rules for keeping the documentation up to date
├── mkdocs.yml               website configuration (MkDocs Material)
├── environment.yml          conda environment "condaml"
├── docs/                    ALL documentation (this website; navigation: docs/SUMMARY.md)
│   ├── introduction.md · research_plan.md · decisions.md · setup.md
│   ├── foundation_models/   overview, one page per model, in_this_study.md
│   ├── phase1_data/ … phase5_maps/   one folder per phase (data, models, transferability, fusion, maps)
│   ├── experiments/             plan.md, one page per experiment (E0N_<name>.md) + figures/
│   ├── data_source/ · references/   data catalogue · literature summaries
│   ├── weekly/                  one page per review meeting
│   ├── results.md · progress.md results overview · change log
│   ├── archive/                 archived parcel-wise classification
│   └── assets/                  style sheet and scripts of the website
├── data_download/           all downloads on the 10.24 km block grid (00–04c), regions.py, download_common.py
├── data_training_prep/           labels, spatial folds, tile plan, training tiles cut from the blocks (01–05)
├── pixelwise_run/           pixel samples (04), comparison (05) + shared code (six encoders, models, metrics)
├── experiments/             E01–E05: one folder per experiment with run.py + results/; comparison/
│   └── parcel_wise_classification/   archived parcel-wise experiments (configs, run_experiment.py, results)
├── parcelwise_run/          archived parcel-wise pipeline, steps 01–09 (labels to comparison)
├── jobs/                    Slurm job scripts for the cluster (step_cpu, step_gpu, …) and logs/
├── tools/                   extra scripts (viewing, side checks)
├── data/                    input data                          (not in git)
├── checkpoints/             model weights                       (not in git)
├── external/                third-party code: AgriFM, Presto, TESSERA (not in git)
├── secrets/                 Earth Engine service-account key    (never commit)
└── outputs/<region>/        outputs per region, morocco or utah (not in git)
    ├── data_training_prep/      01_labels · 02_folds · 03_tiles_128px/agyear · 04_pixel_samples/<season>
    ├── data_download/           grid/blocks (one grid per region) · <source>/<season>/<bx>_<by>.tif (one folder
    │                            per source) · reference/ (cropland mask, reference layers)
    ├── study/                   01_inspection · 02_embeddings (archived parcel-wise runs)
    └── analysis/                figures and pages from tools/ (Morocco)
```
