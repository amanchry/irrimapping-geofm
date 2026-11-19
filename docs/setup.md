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

A GPU (CUDA or Apple Silicon) is recommended for the embedding scripts. Everything else runs on a CPU.

### 2. Earth Engine

Only `pixelwise_run/03_build_tiles.py` uses Earth Engine. Put the service-account key in `secrets/` (path set at the top of the script), or run `earthengine authenticate` once.

### 3. Input data

| File | Source | Where |
| --- | --- | --- |
| Field survey (parcels with land use and irrigation method) | Ouassanouan et al. (2026), *Scientific Data* | `data/field_survey/lulc_fielddata.shp` |
| Morocco boundary | GADM 4.1 | `data/admin/gadm41_MAR_0.json` |

The Sentinel-1/2 imagery is not downloaded by hand. `03_build_tiles.py` fetches it from Earth Engine.

### 4. Model weights

All weights are stored in **`checkpoints/<model>/`** and are **downloaded
automatically** on the first run of each embedding script; later runs reuse them. Folders and sources per model:
[Foundation models · Weights](foundation_models.md#weights).

Only the Clay code needs installing by hand (`pip install --no-deps git+https://github.com/Clay-foundation/model.git`,
see *Environment* above); TerraTorch (Prithvi, TerraMind) and mmengine (AgriFM) come with `environment.yml`. Presto and TESSERA need no package: their code is cloned into `external/presto` and `external/tessera` on first use.

---

## Running

Run every script from the project folder. Each script has its **settings at the top**. A description at the top of each script explains what it does and what it writes.

Two independent pipelines, each in its own folder with the scripts numbered in run order: `pixelwise_run/`
(current) and `parcelwise_run/` (archive). Steps 01–03 are the same scripts in both.

```bash
# Pixel-wise pipeline (pixelwise_run/; step table in pixelwise_run/README.md)
python cropland_extent/make_cropland_mask.py           # 0  cropland mask per region and year
python pixelwise_run/01_prepare_labels_Morocco.py      # 01 labels (01_prepare_labels_Utah.py for Utah)
python pixelwise_run/02_make_spatial_folds.py          # 02 spatial folds
python pixelwise_run/03_build_tiles.py                 # 03 tile store (Earth Engine)
python pixelwise_run/04_make_pixel_samples.py          # 04 labelled pixels
python experiments/E01_s2_pixel_baseline/run.py        # experiments, one at a time: E01, E02, E04, E03, E05
python pixelwise_run/05_compare_experiments.py         # 05 comparison table and paired tests
python pixelwise_run/06_build_cropland_tiles.py        # 06 later, for the maps: wall-to-wall tiles
```

On the cluster each run is a Slurm job (`jobs/step_cpu.sbatch` / `jobs/step_gpu.sbatch` with `SCRIPT=<script>`;
commands on each experiment page). `EXPERIMENT_QUICK_TEST=1` checks a whole experiment on a small subset first.
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
│   ├── research/                introduction, method in brief, protocol, decisions, literature
│   ├── dataset/ · models/       the two pipelines, script by script
│   ├── experiments/             one folder per experiment: index.md (setup) + report.md + figures/
│   ├── weekly/                  one page per review meeting
│   ├── results.md · progress.md results overview · change log
│   └── assets/                  style sheet and diagram script of the website
├── cropland_extent/         cropland mask (step 0) and reference layers (WorldCover, LGRIP30)
├── pixelwise_run/           pixel-wise pipeline, steps 01–06 + shared code (six encoders, models, metrics)
├── experiments/             E01–E04: one folder per experiment with run.py + results/
│   └── parcel_wise_classification/   archived parcel-wise experiments (configs, run_experiment.py, results)
├── parcelwise_run/          archived parcel-wise pipeline, steps 01–09 (labels to comparison)
├── tools/                   extra scripts (viewing, side checks)
├── data/                    input data                          (not in git)
├── checkpoints/             model weights                       (not in git)
├── external/                third-party code: AgriFM, Presto, TESSERA (not in git)
├── secrets/                 Earth Engine service-account key    (never commit)
└── outputs/                 dataset and embeddings written by the scripts (not in git)
    ├── dataset/                 01_labels · 02_folds · 03_tiles_128px/agyear
    ├── study/                   01_inspection · 02_embeddings (shared by all experiments) · development runs
    └── analysis/                figures and pages from tools/
```
