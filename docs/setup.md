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

Earth Engine is used by `pixelwise_run/03_build_tiles.py` (tile store), `cropland_extent/make_cropland_mask.py`
(cropland mask), `cropland_extent/download_dynamic_world.py` (annual Dynamic World layer) and the earlier
wall-to-wall route `pixelwise_run/06_build_cropland_tiles.py`. Put the
service-account key in `secrets/` (path set at the top of each script), or run `earthengine authenticate` once. The
STAC download `s2_download/` needs no account: it reads the public Earth Search catalogue.

### 3. Input data

| File | Source | Where |
| --- | --- | --- |
| Field survey, Morocco (parcels with land use and irrigation method) | Ouassanouan et al. (2026), *Scientific Data* | `data/Morocco_field_survey/lulc_fielddata.shp` |
| Water-Related Land Use (WRLU), Utah, one layer per year | Utah Division of Water Resources | `data/Utah_field_data/` |
| Country and state boundaries | FAO GAUL 2015 (Morocco), 2024 (Utah) | `data/admin_boundaries/` |

The imagery is not downloaded by hand. `03_build_tiles.py` fetches the Sentinel-1/2 tile store from Earth Engine;
`s2_download/02_download_s2_stac.py` builds the wall-to-wall Sentinel-2 composites from the Earth Search STAC
catalogue (`s2_download/README.md`).

### 4. Model weights

All weights are stored in **`checkpoints/<model>/`** and are **downloaded
automatically** on the first run of each experiment that uses the model; later runs reuse them. Folders and sources per model:
[Foundation models · Weights](foundation_models.md#weights).

Only the Clay code needs installing by hand (`pip install --no-deps git+https://github.com/Clay-foundation/model.git`,
see *Environment* above); TerraTorch (Prithvi, TerraMind) and mmengine (AgriFM) come with `environment.yml`. Presto and TESSERA need no package: their code is cloned into `external/presto` and `external/tessera` on first use.

---

## Running

Run every script from the project folder. Each script has its **settings at the top**. A description at the top of each script explains what it does and what it writes.

Two pipelines, each in its own folder with the scripts numbered in run order: `pixelwise_run/` (current) and
`parcelwise_run/` (archive); steps 01–03 are the same scripts in both. A third component, `s2_download/`, downloads
the wall-to-wall Sentinel-2 composites from STAC for every cropland tile (training, experiments and maps of the
STAC-based approach).

```bash
# Pixel-wise pipeline (pixelwise_run/; step table in pixelwise_run/README.md)
python pixelwise_run/01_prepare_labels_Morocco.py      # 01 labels (01_prepare_labels_Utah.py for Utah)
python pixelwise_run/02_make_spatial_folds.py          # 02 spatial folds
python pixelwise_run/03_build_tiles.py                 # 03 tile store (Earth Engine)
python pixelwise_run/04_make_pixel_samples.py          # 04 labelled pixels, irrigated / rainfed (04_pixel_samples/<season>/)
python experiments/E01_s2_pixel_baseline/run.py        # experiments, one at a time: E01, E02, E03, E04, E05
python pixelwise_run/05_compare_experiments.py         # 05 comparison table and paired tests

# Cropland extent for the mapping phase (cropland_extent/; not used by the steps above)
python cropland_extent/make_cropland_mask.py           # cropland mask per region and year (union of 4 products)
python cropland_extent/download_dynamic_world.py       # annual Dynamic World layer per year (candidate mask)
python pixelwise_run/06_build_cropland_tiles.py        # 06 earlier Earth Engine route for wall-to-wall tiles

# STAC download (s2_download/; details in s2_download/README.md)
EXPERIMENT_REGION=utah python s2_download/01_s2_grid_overview.py      # blocks and Sentinel-2 tiles over the cropland
EXPERIMENT_REGION=utah python s2_download/02_download_s2_stac.py      # monthly composites, every tile with ≥ 1% cropland
```

On the cluster each run is a Slurm job (`jobs/step_cpu.sbatch` / `jobs/step_gpu.sbatch` with `SCRIPT=<script>`;
commands on each experiment page; the STAC download runs as `s2_download/s2_dl.sbatch`). `EXPERIMENT_QUICK_TEST=1` checks a whole experiment on a small subset first.
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
│   ├── introduction.md · research_plan.md · decisions.md · foundation_models.md · setup.md
│   ├── phase1_data/ … phase5_maps/   one folder per phase (data, models, transferability, fusion, maps)
│   ├── experiments/             plan.md, one page per experiment (E0N_<name>.md) + figures/
│   ├── data_source/ · references/   data catalogue · literature summaries
│   ├── weekly/                  one page per review meeting
│   ├── results.md · progress.md results overview · change log
│   ├── archive/                 archived parcel-wise classification
│   └── assets/                  style sheet and scripts of the website
├── cropland_extent/         cropland mask and reference layers (WorldCover, LGRIP30, Dynamic World); mapping phase only
├── pixelwise_run/           pixel-wise pipeline, steps 01–06 + shared code (six encoders, models, metrics)
├── s2_download/             wall-to-wall Sentinel-2 composites from STAC (scripts 01–02, s2_dl.sbatch)
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
    ├── dataset/                 01_labels · 02_folds · 03_tiles_128px/agyear · 04_pixel_samples/<season>
    │                            (04c_pixel_samples/: earlier three-class samples, no longer used)
    ├── cropland_extent/         cropland mask per year, reference layers (mapping phase)
    ├── s2_download/             grid overview and STAC tiles per year
    ├── study/                   01_inspection · 02_embeddings (archived parcel-wise runs)
    └── analysis/                figures and pages from tools/ (Morocco)
```
