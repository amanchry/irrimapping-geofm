# Irrigated Cropland Mapping using Geospatial Foundation Models

MSc thesis code: parcel-level classification of **irrigated vs rainfed** cropland in Morocco,
comparing four geospatial foundation models (Clay, Prithvi-EO-2.0, TerraMind, AgriFM) with a
Sentinel-1/2 baseline under the same spatial cross-validation.

## Folder structure

```
irrigated_area_mapping_code/
├── README.md
├── environment.yml              conda environment "condaml" (everything the code needs)
├── pipeline/                    the main workflow, run in number order
│   ├── 01_prepare_labels.py         field survey -> labelled parcels
│   ├── 02_make_spatial_folds.py     20 km blocks -> 5 spatial folds (+ random folds for comparison)
│   ├── 03_extract_s1s2_features.py  Sentinel-1/2 features per parcel (baseline input, Earth Engine)
│   ├── 04_download_chips.py         monthly image chips per parcel, shared by all foundation models
│   ├── 05a_embed_clay.py            Clay v1.5 embeddings
│   ├── 05b_embed_prithvi.py         Prithvi-EO-2.0 embeddings
│   ├── 05c_embed_terramind.py       TerraMind embeddings (Sentinel-1 + Sentinel-2)
│   ├── 05d_embed_agrifm.py          AgriFM embeddings
│   ├── 06_train_evaluate.py         4 classifiers, nested spatial CV, metrics with 95% CI (any input)
│   └── 07_compare_models.py         all inputs on the same parcels, differences vs the baseline
├── analysis/                    supporting analyses (not needed for the main result)
│   ├── a1_gmia_next_points.py       GMIA-NEXT ground-truth points: quality, Morocco check points
│   └── a2_map_reference_products.py ESA WorldCover / LGRIP30 maps of Morocco
├── data/raw/                    input data (not in git)
│   ├── field_survey/                lulc_fielddata.* (Ouassanouan et al. 2026)
│   ├── admin/                       gadm41_MAR_*.json
│   ├── worldcover/                  ESA_WorldCover_10m_2021.tif
│   └── gmia_next/                   GMIA-NEXT ground-truth points (for analysis/a1)
├── checkpoints/                 model weights (not in git): clay/, agrifm/
├── external/                    third-party code (not in git): AgriFM/
├── secrets/                     Earth Engine service-account key (never commit)
├── outputs/                     everything the scripts write, one folder per step (not in git)
│   ├── 01_labels/  02_folds/  03_s1s2_features/  04_chips/  05_embeddings/
│   ├── 06_models/<run>/  07_comparison/  analysis/
├── docs/                        papers and dataset documentation
└── archive/                     old scripts, notebooks and outputs (kept for reference)
```

Every script is independent: settings are in a block at the top (no command-line arguments),
a description at the top explains what it does, its steps and its outputs, and paths are relative
to the project folder. Run scripts from the project folder.

## Setup

```bash
conda env create -f environment.yml          # or: conda env update -n condaml -f environment.yml
conda activate condaml
pip install --no-deps git+https://github.com/Clay-foundation/model.git
earthengine authenticate                      # only if you do not use the service-account key
```

Model weights:

| Model | Weights | How |
| --- | --- | --- |
| Clay v1.5 | `checkpoints/clay/clay-v1.5.ckpt`, `metadata.yaml` | `curl -L -o checkpoints/clay/clay-v1.5.ckpt https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt` and `curl -L -o checkpoints/clay/metadata.yaml https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml` |
| Prithvi-EO-2.0-300M-TL | downloaded automatically from Hugging Face | — |
| TerraMind-1.0-base | downloaded automatically from Hugging Face | — |
| AgriFM | `checkpoints/agrifm/AgriFM.pth` + code in `external/AgriFM` | downloaded / cloned automatically by `05d_embed_agrifm.py` (weights: https://glass.hku.hk/casual/AgriFM/AgriFM.pth); needs `pip install mmengine` |

## Running the pipeline

```bash
python pipeline/01_prepare_labels.py
python pipeline/02_make_spatial_folds.py
python pipeline/03_extract_s1s2_features.py      # Earth Engine, ~1-2 h
python pipeline/04_download_chips.py             # Earth Engine, a few hours; resumable
python pipeline/05a_embed_clay.py                # each 05 script: resumable, faster on a GPU
python pipeline/05b_embed_prithvi.py
python pipeline/05c_embed_terramind.py
python pipeline/05d_embed_agrifm.py
python pipeline/06_train_evaluate.py             # set INPUT at the top: s1s2, clay, prithvi, terramind, agrifm
python pipeline/07_compare_models.py
```

Tips: set `N_TEST_PARCELS = 20` (scripts 03, 04) or `QUICK_TEST = True` (script 06) for a quick
check before a full run. Scripts 03 and 04 skip parcels already downloaded if they are restarted.

## Method in short

- **Labels**: 8,781 cropland parcels (7,585 irrigated, 1,196 rainfed) surveyed Dec 2023 - Jan 2025.
  Each parcel gets satellite data from the agricultural year (1 Sep - 31 Aug) it was surveyed in.
- **Inputs**: Sentinel-1/2 features per 30-day period (bands, NDVI, NDWI, EVI, VV, VH), and frozen
  embeddings of four foundation models from the same 64 x 64-pixel monthly chips, averaged over the
  patches covering each parcel.
- **Validation**: nested spatial cross-validation on 20 km blocks (5 folds); hyperparameters tuned
  only on training folds; 95% confidence intervals from a block bootstrap; inputs compared with
  paired differences on the same parcels (Holm-adjusted).
- **Classifiers**: logistic regression (linear probe), random forest, histogram gradient boosting, RBF SVM.
