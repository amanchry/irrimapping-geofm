# Irrigated Cropland Mapping using Geospatial Foundation Models

MSc thesis code: parcel-level classification of **irrigated vs rainfed** cropland in Morocco,
comparing four geospatial foundation models (Clay, Prithvi-EO-2.0, TerraMind, AgriFM) with a
Sentinel-1/2 baseline under the same spatial cross-validation.

## Two parts

The code is split into two independent parts:

1. **`dataset/` – the ML-ready dataset** (can be published on its own as a dataset paper).
   Labelled parcels, spatial folds and one shared store of Sentinel-1/2 tiles, plus ready-to-use
   views for any model (parcel features, pixel sets, patch-aligned crops, full tiles).
2. **`study/` – the irrigated-area study.** Uses only the outputs of `dataset/`: checks the data,
   extracts foundation-model embeddings, trains and evaluates the classifiers and compares the inputs.

Detailed documentation of each part (methodology, every script, flow diagrams, file formats):
**[dataset/README.md](dataset/README.md)** and **[study/README.md](study/README.md)**.

## Folder structure

```
irrigated_area_mapping_code/
├── README.md  PROGRESS.md
├── environment.yml              conda environment "condaml" (everything the code needs)
├── dataset/                     PART 1: dataset generation, run in number order
│   ├── 01_prepare_labels.py         field survey -> labelled parcels
│   ├── 02_make_spatial_folds.py     20 km blocks -> 5 spatial folds (train / validation / test by fold)
│   ├── 03_build_tiles.py            tile store: 128 x 128 px monthly S1/S2 tiles + label / parcel-ID / edge rasters (Earth Engine)
│   └── 04_make_views.py             model inputs cut from the tiles: parcel features, pixel sets, crops_p8 / crops_p16
├── study/                       PART 2: irrigated-area study, run in number order
│   ├── 01_inspect_dataset.py        checks of the dataset: inventory, label QA, observations, patch purity, profiles, example crops
│   ├── 02a_embed_clay.py            Clay v1.5 embeddings
│   ├── 02b_embed_prithvi.py         Prithvi-EO-2.0 embeddings
│   ├── 02c_embed_terramind.py       TerraMind embeddings (Sentinel-1 + Sentinel-2)
│   ├── 02d_embed_agrifm.py          AgriFM embeddings
│   ├── 03_train_evaluate.py         4 classifiers, nested spatial CV, metrics with 95% CI (any input)
│   └── 04_compare_models.py         all inputs on the same parcels, differences vs the baseline
├── analysis/                    supporting analyses (not needed for the main result)
│   ├── a1_gmia_next_points.py       GMIA-NEXT ground-truth points: quality, Morocco check points
│   ├── a2_map_reference_products.py ESA WorldCover / LGRIP30 maps of Morocco
│   └── a3_export_chips_geotiff.py   single crops as GeoTIFF for QGIS
├── data/raw/                    input data (not in git)
│   ├── field_survey/                lulc_fielddata.* (Ouassanouan et al. 2026)
│   ├── admin/                       gadm41_MAR_*.json
│   ├── worldcover/                  ESA_WorldCover_10m_2021.tif
│   └── gmia_next/                   GMIA-NEXT ground-truth points (for analysis/a1)
├── checkpoints/                 model weights (not in git): clay/, agrifm/
├── external/                    third-party code (not in git): AgriFM/
├── secrets/                     Earth Engine service-account key (never commit)
├── outputs/                     everything the scripts write (not in git)
│   ├── dataset/   01_labels/  02_folds/  03_tiles/agyear/ (tiles, labels, QA, views/)
│   ├── study/     01_inspection/  02_embeddings/  03_models/<run>/  04_comparison/
│   ├── analysis/
│   └── legacy/    03_s1s2_features/  04_chips/   (earlier Earth Engine features and chips, for reference)
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
| AgriFM | `checkpoints/agrifm/AgriFM.pth` + code in `external/AgriFM` | downloaded / cloned automatically by `study/02d_embed_agrifm.py` (weights: https://glass.hku.hk/casual/AgriFM/AgriFM.pth); needs `pip install mmengine` |

## Running

```bash
# Part 1: dataset
python dataset/01_prepare_labels.py
python dataset/02_make_spatial_folds.py
python dataset/03_build_tiles.py               # Earth Engine, a few hours; resumable (N_TEST_TILES = 20 first)
python dataset/04_make_views.py                # local; parcel features, pixel sets, crops_p8 / crops_p16

# Part 2: study
python study/01_inspect_dataset.py             # check the dataset before training
python study/02a_embed_clay.py                 # each 02 script: resumable, faster on a GPU
python study/02b_embed_prithvi.py
python study/02c_embed_terramind.py
python study/02d_embed_agrifm.py
python study/03_train_evaluate.py              # set INPUT at the top: s1s2, clay, prithvi, terramind, agrifm
python study/04_compare_models.py
```

Tips: set `N_TEST_TILES = 20` (dataset/03) or `QUICK_TEST = True` (study/03) for a quick check before a full run.
dataset/03 skips tiles already downloaded if it is restarted.

## Method in short

- **Labels**: 8,781 cropland parcels (7,585 irrigated, 1,196 rainfed) surveyed Dec 2023 - Jan 2025.
  Each parcel gets satellite data from the agricultural year (1 Sep - 31 Aug) it was surveyed in.
- **Data**: one non-overlapping store of 128 x 128-pixel tiles (1.28 km, UTM 29N) with 12 monthly
  composites of the parcel's agricultural year (Sentinel-2 median of clear pixels + clear-observation
  count; Sentinel-1 ascending and descending, linear-power mean in dB) and label / parcel-ID / edge
  rasters (0 = ignore). Every model input is cut from these same pixels: parcel medians (baseline),
  pixel sets, and 64 x 64 crops shifted so the parcel centroid sits at the centre of a model patch.
- **Inputs**: Sentinel-1/2 parcel features per month (bands, NDVI, NDWI, EVI, VV, VH), and frozen
  embeddings of four foundation models from the patch-aligned crops, averaged over the patches
  covering each parcel.
- **Validation**: nested spatial cross-validation on 20 km blocks (5 folds); hyperparameters tuned
  only on training folds; 95% confidence intervals from a block bootstrap; inputs compared with
  paired differences on the same parcels (Holm-adjusted).
- **Classifiers**: logistic regression (linear probe), random forest, histogram gradient boosting, RBF SVM.
