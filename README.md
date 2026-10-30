# Irrigated Cropland Mapping using Geospatial Foundation Models

MSc thesis code. Parcels in Morocco are classified as **irrigated** or **rainfed**. Frozen embeddings of four geospatial foundation models (**Clay v1.5, Prithvi-EO-2.0, TerraMind, AgriFM**) are compared with a classic **Sentinel-1/2 feature baseline**. All inputs come from the same pixels and are scored on the same parcels, under the same nested spatial cross-validation.

| Part | Folder | What it does | Details |
| --- | --- | --- | --- |
| 1 · ML-ready dataset | [`ml_dataset_prep/`](ml_dataset_prep/) | labels, spatial folds, Sentinel-1/2 tile store, model inputs | [ml_dataset_prep/README.md](ml_dataset_prep/README.md) |
| 2 · Study | [`models_run/`](models_run/) | dataset checks, embeddings, training and evaluation, comparison | [models_run/README.md](models_run/README.md) |
| Tools | [`tools/`](tools/) | extra scripts for viewing and side checks (not needed for the results) | top of each script |

The two parts are independent. Part 1 produces the dataset, which can be used downstream ML tasks.

---

## Workflow

```mermaid
flowchart TD
    subgraph INPUTS["Inputs"]
        FS["Field survey<br/>~10,000 parcels, Dec 2023 – Jan 2025"]
        EE["Google Earth Engine<br/>Sentinel-2 L2A · Cloud Score+ · Sentinel-1 GRD"]
        W["Pretrained weights<br/>Clay · Prithvi · TerraMind · AgriFM"]
    end

    subgraph P1["Part 1 · ml_dataset_prep/"]
        L["01 Labels<br/>8,781 parcels: 7,585 irrigated · 1,196 rainfed"]
        F["02 Spatial folds<br/>20 km blocks → 5 folds"]
        T["03 Tile store<br/>128 × 128 px tiles · 12 months × 19 bands<br/>+ label / parcel-ID / edge rasters"]
        V["04 Model views<br/>parcel features · pixel sets · patch-aligned crops"]
        L --> F --> T --> V
    end

    subgraph P2["Part 2 · models_run/"]
        I["01 Inspect dataset<br/>QA · purity · seasonal profiles"]
        E["02a–02d Embeddings<br/>frozen foundation models"]
        C["03 Train & evaluate<br/>4 classifiers · nested spatial CV<br/>block-bootstrap 95% CI"]
        K["04 Compare inputs<br/>paired differences vs baseline"]
        I --> E --> C --> K
    end

    FS --> L
    EE --> T
    V -->|"checked"| I
    V -->|"crops"| E
    V -->|"parcel features (baseline)"| C
    F -->|"folds"| C
    W --> E
    K --> R(["Does a foundation model beat<br/>the Sentinel-1/2 baseline?"])

    T -.-> TOOLS["tools/<br/>virtual rasters · parcel explorer · GeoTIFF export"]
    V -.-> TOOLS

    classDef part1 fill:#1F5F8B,color:#fff,stroke:#1F5F8B
    classDef part2 fill:#B4541A,color:#fff,stroke:#B4541A
    classDef tool fill:#F4F1EA,stroke:#8A919A,color:#1C2733
    class L,F,T,V part1
    class I,E,C,K part2
    class TOOLS tool
```

**Method in one paragraph.** Each surveyed parcel gets the 12 monthly Sentinel-1/2 composites of its agricultural year (1 Sep – 31 Aug), cut from a shared, non-overlapping tile store.
- **Baseline:** a parcel's monthly medians of the Sentinel-2 bands, NDVI, NDWI, EVI and Sentinel-1 VV/VH.
- **Foundation models:** each embeds a 64 × 64 px crop, shifted so that the parcel centroid sits at the centre of a model patch. The patch embeddings over the parcel are then averaged.
- **Classifiers:** logistic regression, random forest, gradient boosting and an SVM, tuned and tested with nested spatial cross-validation on 20 km blocks.
- **Comparison:** inputs are compared on the same parcels with block-bootstrap confidence intervals, Holm-adjusted.

---

## Project structure

```
irrigated_area_mapping_code/
├── README.md                this file: setup and overview
├── PROGRESS.md              decisions, change log, open points
├── environment.yml          conda environment "condaml"
├── ml_dataset_prep/         PART 1 · dataset (README.md + scripts 01–04)
├── models_run/              PART 2 · study (README.md + scripts 01–04)
├── tools/                   extra scripts (see below)
├── data/                    input data                          (not in git)
│   ├── field_survey/            lulc_fielddata.shp (+ .dbf .shx .prj .cpg)
│   ├── admin/                   gadm41_MAR_0.json …  (GADM 4.1 Morocco)
│   └── worldcover/              ESA_WorldCover_10m_2021.tif
├── checkpoints/             model weights: clay/, agrifm/       (not in git)
├── external/                third-party code: AgriFM/           (not in git)
├── secrets/                 Earth Engine service-account key    (never commit)
├── docs/                    papers, dataset documentation
└── outputs/                 everything the scripts write        (not in git)
    ├── dataset/                 01_labels · 02_folds · 03_tiles/agyear (tiles, views, vrt)
    ├── study/                   01_inspection · 02_embeddings · 03_models · 04_comparison
    └── analysis/                figures and pages from tools/
```

---

## Setup

### 1. Environment

```bash
conda env create -f environment.yml                 # creates "condaml" (or: conda env update -n condaml -f environment.yml)
conda activate condaml
pip install --no-deps git+https://github.com/Clay-foundation/model.git   # Clay (--no-deps: avoids an einops conflict with TerraTorch)
pip install mmengine                                 # AgriFM
```

A GPU (CUDA or Apple Silicon) is recommended for the embedding scripts. Everything else runs on a CPU.

### 2. Earth Engine

Only `ml_dataset_prep/03_build_tiles.py` uses Earth Engine. Put the service-account key in `secrets/` (path set at the top of the script), or run `earthengine authenticate` once. **Never commit the key.** `secrets/` and `*service-account*.json` are in `.gitignore`.

### 3. Input data

| File | Source | Where |
| --- | --- | --- |
| Field survey (parcels with land use and irrigation method) | Ouassanouan et al. (2026), *Scientific Data* | `data/field_survey/lulc_fielddata.shp` |
| Morocco boundary | GADM 4.1 | `data/admin/gadm41_MAR_0.json` |
| ESA WorldCover 2021, 10 m (optional cropland check) | ESA WorldCover v200 | `data/worldcover/ESA_WorldCover_10m_2021.tif` |

The Sentinel-1/2 imagery is not downloaded by hand. `03_build_tiles.py` fetches it from Earth Engine.

### 4. Model weights

| Model | How |
| --- | --- |
| Clay v1.5 | `curl -L -o checkpoints/clay/clay-v1.5.ckpt https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt`<br/>`curl -L -o checkpoints/clay/metadata.yaml https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml` |
| Prithvi-EO-2.0-300M-TL | downloaded automatically from Hugging Face on the first run |
| TerraMind-1.0-base | downloaded automatically from Hugging Face on the first run |
| AgriFM | weights (`checkpoints/agrifm/AgriFM.pth`) and code (`external/AgriFM`) fetched automatically by `models_run/02d_embed_agrifm.py` |

---

## Running

Run every script from the project folder. Each script has its **settings at the top**: there are no command-line arguments. A description at the top of each script explains what it does and what it writes.

```bash
# Part 1 · dataset
python ml_dataset_prep/01_prepare_labels.py
python ml_dataset_prep/02_make_spatial_folds.py
python ml_dataset_prep/03_build_tiles.py        # Earth Engine; hours; resumable (N_TEST_TILES = 20 for a test)
python ml_dataset_prep/04_make_views.py

# Part 2 · study
python models_run/01_inspect_dataset.py          # check the data before training
python models_run/02a_embed_clay.py              # 02a–02d: GPU recommended, resumable
python models_run/02b_embed_prithvi.py
python models_run/02c_embed_terramind.py
python models_run/02d_embed_agrifm.py
python models_run/03_train_evaluate.py           # once per INPUT: s1s2, clay, prithvi, terramind, agrifm
python models_run/04_compare_models.py
```

For a quick check before a full run, set `N_TEST_TILES = 20` in `03_build_tiles.py` and `QUICK_TEST = True` in `03_train_evaluate.py`.

---

## Tools (`tools/`)

These scripts support the work but are not part of the pipeline. Run them whenever needed.

| Script | What it gives you |
| --- | --- |
| `a1_gmia_next_points.py` | GMIA-NEXT ground-truth points: quality check and possible independent test points |
| `a2_map_reference_products.py` | maps of ESA WorldCover and LGRIP30 over Morocco (Earth Engine) |
| `a3_export_chips_geotiff.py` | selected model crops as GeoTIFFs for QGIS |
| `a4_build_vrt.py` | the whole tile store as one virtual raster per agricultural year, for QGIS or testing |
| `a5_parcel_explorer.py` | interactive HTML map of all parcels; click one to see its monthly NDVI / NDWI / VH profile |

---

## Documentation

- **[ml_dataset_prep/README.md](ml_dataset_prep/README.md):** dataset design, every script, tiles and label rasters, the two agricultural years, file formats, flow diagrams.
- **[models_run/README.md](models_run/README.md):** research questions, inputs compared, embedding models, nested spatial CV, metrics, comparison.
- **[PROGRESS.md](PROGRESS.md):** current status, method decisions, open points, change log.
