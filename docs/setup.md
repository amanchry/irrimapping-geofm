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

Only `ml_dataset_prep/03_build_tiles.py` uses Earth Engine. Put the service-account key in `secrets/` (path set at the top of the script), or run `earthengine authenticate` once.

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
see *Environment* above); TerraTorch (Prithvi, TerraMind) and mmengine (AgriFM) come with `environment.yml`.

---

## Running

Run every script from the project folder. Each script has its **settings at the top**. A description at the top of each script explains what it does and what it writes.

```bash
# Phase 1 · dataset
python ml_dataset_prep/01_prepare_labels.py
python ml_dataset_prep/02_make_spatial_folds.py
python ml_dataset_prep/03_build_tiles.py       
python ml_dataset_prep/04_make_views.py

# Phase 2 · model evaluation
python models_run/01_inspect_dataset.py          # check the data before training
python models_run/02a_embed_clay.py              # 02a–02d: GPU recommended, resumable
python models_run/02b_embed_prithvi.py
python models_run/02c_embed_terramind.py
python models_run/02d_embed_agrifm.py

# Experiments · training, comparison and report from one config file
python experiments/run_experiment.py             # set EXPERIMENT at the top, e.g. "E02_frozen_pca_pastis"
```

For a quick check before a full run, set `N_TEST_TILES = 20` in `03_build_tiles.py` and `QUICK_TEST: true` in the experiment's `config.yaml`. How experiments work, and how to add one: [Experiments](experiments/index.md).

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
├── ml_dataset_prep/         PART 1 · dataset scripts 01–04
├── models_run/              PART 2 · study scripts 01–04
├── experiments/             one folder per experiment: config.yaml + results/ (written by the scripts)
│   ├── run_experiment.py        runs an experiment from its config
│   ├── make_report.py           tables + figures into docs/experiments/<name>/report.md
│   └── _template/               config, setup page and report page to start a new experiment
├── tools/                   extra scripts (viewing, side checks)
├── data/                    input data                          (not in git)
├── checkpoints/             model weights                       (not in git)
├── external/                third-party code: AgriFM            (not in git)
├── secrets/                 Earth Engine service-account key    (never commit)
└── outputs/                 dataset and embeddings written by the scripts (not in git)
    ├── dataset/                 01_labels · 02_folds · 03_tiles_128px/agyear
    ├── study/                   01_inspection · 02_embeddings (shared by all experiments) · development runs
    └── analysis/                figures and pages from tools/
```
