# Progress log

Running record of decisions, code changes and open points for the thesis
*Irrigated vs rainfed cropland mapping in Morocco with geospatial foundation models*.
Update this file with every change to the code or the method (newest entry on top in the
change log).

---

## Current status (2026-10-29)

| Step | Script | Status |
| --- | --- | --- |
| Labels | `dataset/01_prepare_labels.py` | Run: 8,781 parcels (7,585 irrigated, 1,196 rainfed) |
| Spatial folds | `dataset/02_make_spatial_folds.py` | Run: 5 folds, 20 km blocks |
| Tile store | `dataset/03_build_tiles.py` | Plan + labels tested on all parcels; Earth Engine part not yet run |
| Model views | `dataset/04_make_views.py` | Tested end to end with synthetic imagery |
| Dataset inspection | `study/01_inspect_dataset.py` | Written 2026-10-29; tested on synthetic data |
| Clay embeddings | `study/02a_embed_clay.py` | Tested with a random encoder; not yet run |
| Prithvi embeddings | `study/02b_embed_prithvi.py` | Tested with random weights, not yet run |
| TerraMind embeddings | `study/02c_embed_terramind.py` | Tested with random weights, not yet run |
| AgriFM embeddings | `study/02d_embed_agrifm.py` | Tested with random weights; weights + repo still to download |
| Train / evaluate | `study/03_train_evaluate.py` | Tested on synthetic tile-store features |
| Compare inputs | `study/04_compare_models.py` | Tested on test runs |
| GMIA-NEXT check points | `analysis/a1_gmia_next_points.py` | Needs `data/raw/gmia_next/` |
| Reference maps | `analysis/a2_map_reference_products.py` | Converted from notebook |
| Crops to GeoTIFF (QGIS) | `analysis/a3_export_chips_geotiff.py` | Reads `views/crops_p16` |

Next steps
1. Run dataset/03 with `N_TEST_TILES = 20`, then dataset/04 and study/01; check the figures (and a3 in QGIS); then full runs.
2. Run study/02a-02d, then study/03 for each `INPUT`, then study/04.
3. Decide on the open points below.

---

## Method decisions

| Date | Decision | Reason / reference |
| --- | --- | --- |
| 2026-10-29 | All inputs (parcel features and foundation-model crops) cut from ONE tile store; baseline = per-pixel median of parcel pixels without edge pixels; Sentinel-1 orbit of the baseline = orbit of the crops | Identical pixels, compositing and masks for every model; tile design as in PASTIS (Garnot & Landrieu 2021), Sen4AgriNet (Sykas et al. 2022) |
| 2026-10-15 | Foundation models compared: Clay, Prithvi-EO-2.0, TerraMind, AgriFM (AlphaEarth only a pipeline test, not in the thesis) | Thesis proposal |
| 2026-10-15 | Main comparison with **frozen** encoders + the same 4 classifiers; fine-tuning only as a secondary experiment | Fair across models; Butsko et al. 2025, Marsocci et al. 2024 |
| 2026-10-15 | Time window = **agricultural year 1 Sep - 31 Aug**; growing season only (Nov-May) and calendar year as ablations | Ministry reports season rainfall from 1 Sep; sowing ~Nov, harvest May-Jul (FAO GIEWS; USDA FAS 2017); summer is when only irrigated land stays green |
| 2026-10-15 | Each parcel gets imagery from **its own survey season** (`ag_season`: 2023/24 or 2024/25); both years pooled in one dataset, periods aligned (p01 = September) | Survey ran Dec 2023 - Jan 2025; dropping 2024/25 would lose 42% of rainfed parcels |
| 2026-10-15 | Year and region partly confounded (Gharb, Tadla only 2023/24; Souss, Statte only 2024/25) -> no clean cross-year test; possible within Doukkala / El Haouz | Survey design |
| 2026-10-15 | Validation = **nested spatial CV** on 20 km blocks (5 folds); random CV only to show its optimism | Roberts et al. 2017; Ploton et al. 2020; Kattenborn et al. 2022 |
| 2026-10-15 | Metrics: macro-F1 (main), per-class F1/precision/recall, balanced accuracy, PR-AUC, Brier, ECE; 95% CI by block bootstrap; inputs compared by paired differences + Holm | Foody 2009; Pontius & Millones 2011 (no kappa) |
| 2026-10-15 | Classifiers: logistic regression (linear probe), random forest, histogram gradient boosting, RBF SVM | Alain & Bengio 2016; Breiman 2001; Gao et al. 2018; Elwan et al. 2022 |
| 2026-10-15 | Cloud mask: **Cloud Score+** (cs_cdf >= 0.60) instead of QA60; scene filter raised to 60% cloud | Pasquarella et al. 2023 (GEE dataset page: 0.50-0.65) |
| 2026-10-15 | S1/S2 empty periods filled by **linear interpolation in time** per parcel (15-day file: 79% of parcels had a gap) | Avoids constant median fill |
| 2026-10-15 | Parcels shrunk by **10 m** (one pixel) for S1/S2 feature extraction | Removes mixed edge pixels |
| 2026-10-15 | One shared **chip store** (64 x 64 px, 10 m, 12 months, 12 S2 bands + VV/VH) for all foundation models | Same imagery for every model; one download |
| 2026-10-15 | Parcel embedding = patches covering the parcel, weighted by the share inside the parcel | Parcel-level labels |
| 2026-10-15 | Clay: monthly chips one by one, 10 bands, Clay normalisation, week/lat-lon codes | Clay v1.5 docs and code |
| 2026-10-15 | Prithvi: 6 bands (B2 B3 B4 B8A B11 B12), 4 frames per pass, date + location; 10 m input (pretrained on 30 m HLS) | Model card config.json |
| 2026-10-15 | TerraMind: S2L2A (12 bands) + S1GRD, standardised with v1 pretraining stats, modalities averaged | TerraTorch terramind_register.py |
| 2026-10-15 | AgriFM: 12 months at once, 10 bands, normalisation from its cropland config; loaded without mmcv (mmengine registries) | AgriFM repo configs/cropland_mapping.py |

---

## Open points

- [ ] AgriFM normalisation: config mean ~4,180 (blue) vs ~1,100 in Earth Engine S2 L2A -> confirm the pixel scale its weights expect.
- [ ] AgriFM weights: download `AgriFM.pth` to `checkpoints/agrifm/` and clone the repo to `external/AgriFM`.
- [ ] Prithvi on 10 m input (pretrained on 30 m HLS): state in thesis; optional 30 m ablation.
- [ ] Neighbouring fields inside a chip (mixed 160 m patches, transformer context): add parcel "purity" + accuracy by parcel size; optional masked-chip ablation (`MASK_OUTSIDE`).
- [ ] Per-year results (2023/24 vs 2024/25) from the out-of-fold predictions; year-transfer test within Doukkala / El Haouz.
- [ ] Tunisia: label source and transfer experiment (zero-shot and few-shot).
- [ ] Prediction / wall-to-wall mapping step (later).
- [ ] Optional: ET (WaPOR), LST, precipitation (CHIRPS) as extra baseline variables, as ablations only.

---

## Change log (newest first)

### 2026-10-29 (documentation)
- New `dataset/README.md` and `study/README.md`: full methodology of each part, every script (purpose, steps,
  settings, outputs), Mermaid flow diagrams in the markdown, file formats, limitations and references.
  The main README links to both. TODO: complete the Ouassanouan et al. (2026) reference (title, DOI).
- Docstrings corrected: embedding output names `<model>_agyear_tiles.npz` (02a-02d), baseline source in study/03,
  a3 output folder.

### 2026-10-29 (restructure)
- **Code split into two parts.** `dataset/` builds the ML-ready dataset (publishable separately);
  `study/` uses only its outputs. `pipeline/` is no longer used.
- Renamed scripts: `pipeline/01`, `02` -> `dataset/01`, `02`; `pipeline/04_build_tiles.py` -> `dataset/03_build_tiles.py`;
  `pipeline/04b_make_views.py` -> `dataset/04_make_views.py`; `pipeline/05a-05d` -> `study/02a-02d`;
  `pipeline/06_train_evaluate.py` -> `study/03_train_evaluate.py`; `pipeline/07_compare_models.py` -> `study/04_compare_models.py`.
  `pipeline/03_extract_s1s2_features.py` (Earth Engine features) moved to `archive/legacy_pipeline/`.
- Outputs moved: `outputs/01_labels`, `02_folds` -> `outputs/dataset/`; tile store in `outputs/dataset/03_tiles/agyear/`;
  `05_embeddings`, `06_models`, `07_comparison` -> `outputs/study/02_embeddings`, `03_models`, `04_comparison`;
  old `03_s1s2_features` and `04_chips` -> `outputs/legacy/`. All paths in the scripts updated.
- **New `study/01_inspect_dataset.py`:** inventory per class / season / fold, label QA (overlaps, tiny parcels),
  clear observations per month, patch purity aligned vs centred, seasonal NDVI / NDWI / VH profiles,
  example crops per class, NDVI agreement with the legacy Earth Engine features. Output: `outputs/study/01_inspection/`.
- a3 now reads `views/crops_p16` and writes to `outputs/study/01_inspection/chips_geotiff/`.
- README rewritten for the two-part structure. Backups: `archive/scripts_2026-10-29/`.

### 2026-10-29
- **New data source for all inputs: shared tile store.** `04_build_tiles.py` (non-overlapping 128 x 128-px tiles,
  label / parcel-ID / edge rasters, S1 per orbit in linear power) and `04b_make_views.py` (parcel features, pixel sets,
  patch-aligned crops `crops_p8` / `crops_p16`) replace `04_download_chips.py` and the Earth Engine features of
  script 03. Old scripts and results are kept for reference; old runs keep their names.
- **05a-05d:** new settings `CHIP_SOURCE = "tiles"` and `CROP_PATCH_PX` (8 for Clay, 16 for Prithvi, TerraMind,
  AgriFM). Output files are now `<model>_agyear_tiles.npz` (old: `<model>_agyear.npz`).
- **06:** new settings `S1S2_SOURCE = "tiles"`, `TILE_FEATURES = "inner"`, `S1_ORBIT = "crops"`, `EMB_SOURCE = "tiles"`.
  The parcel-feature table of 04b is read directly: bookkeeping columns (fold, n_px_used, used_inner) and the
  observation counts (S2_NCLEAR, S1_N_*) are dropped, the chosen Sentinel-1 orbit is renamed to VV/VH (same orbit
  as the foundation-model crops by default), months are labelled Sep ... Aug. New run names:
  `s1s2-all_agyear_tiles-inner-<orbit>` and `<model>_agyear_tiles_concat`.
- **07:** `BASELINE_RUN` and `RUNS` set to the five tile-store runs (old settings in a comment).
- Tested 06 end to end on a synthetic table in the 04b format: 192 features (216 with both orbits), all four
  classifiers, importance by variable and month.
- Backups of the changed scripts: `archive/scripts_2026-10-29/`.

### 2026-10-27
- **New chip design (tile store)** replacing `04_download_chips.py` (moved to `archive/scripts_2026-10-27/`):
  - `04_build_tiles.py`: fixed non-overlapping 128 x 128 px tiles (1.28 km) in EPSG:32629; only tiles needed by the parcels
    (1,755 home + 302 outline + 930 context = 2,987 tiles); 12 monthly composites x 19 bands: S2 median (Cloud Score+),
    S2 clear-observation count, S1 ascending / descending separately as linear-power mean -> dB, S1 scene counts;
    label (0 ignore / 1 irrigated / 2 rainfed), parcel_idx and edge rasters; scene IDs per tile and month; QA per tile;
    GeoParquet tile index with fold (87 tiles hold parcels of two folds -> `mixed_folds`).
  - `04b_make_views.py`: parcel median features (all pixels / without edge pixels), pixel sets (PSE-TAE), and 64 x 64
    crops per patch size (8, 16) shifted so the parcel centroid is at a patch centre; same format as the old chips.
  - Why: old chips overlapped (median 8 other chips per chip, 6.3% across folds), baseline and foundation models used
    different pixels / compositing, small parcels were split over 4 patches (purity 0.37 / 0.10 for 8 / 16 px patches;
    aligned crops: 0.62 / 0.36 in the test sample), S1 was averaged in dB with orbits mixed, no observation counts.
  - Label QA from the parcel geometry: 79 overlapping parcel pairs (2 with different labels, 25 covering > 50% of the
    smaller parcel) -> `parcel_overlaps.csv`; 3,684 shared pixels set to ignore; 65 parcels have no pixel after the
    10 m inner buffer.
  - Design follows PASTIS, Sen4AgriNet, S2-Agri (PSE-TAE), Fields of The World, ClarkCGA / Prithvi crop chips.
- Next: run 04 with `N_TEST_TILES = 20`, inspect tiles in QGIS, full run, then 04b; point the 05 scripts to
  `outputs/04_tiles/agyear/views/crops_p8` (Clay) or `crops_p16` (Prithvi, TerraMind; AgriFM to check).

### 2026-10-16
- `05d_embed_agrifm.py`: downloads the weights automatically from https://glass.hku.hk/casual/AgriFM/AgriFM.pth and clones the AgriFM code if missing.
- `05d_embed_agrifm.py`: checks for mmengine, the AgriFM code (`external/AgriFM`) and the weights before starting, with a clear message for each.
- `06_train_evaluate.py`: `INPUT` reset to `"s1s2"` (it was set to `"clay"` before the Clay embeddings existed).
- S1/S2 features extracted: `outputs/03_s1s2_features/s1s2_features_agyear_30d.csv` (8,781 parcels, 192 features, no empty values).
- Note (macOS): if a script stops with `OMP: Error #15` (two OpenMP copies), install PyTorch from conda-forge instead of pip, or set `KMP_DUPLICATE_LIB_OK=TRUE` as a workaround.

### 2026-10-15
- **Added** `analysis/a3_export_chips_geotiff.py`: writes chips of selected parcels as GeoTIFFs (14 bands per month, 12-month NDVI, parcel mask and outline) for QGIS.
- **Restructured the project** into `pipeline/`, `analysis/`, `data/raw/`, `checkpoints/`, `external/`, `secrets/`, `outputs/<step>/`, `docs/`, `archive/`. Old scripts, notebooks and outputs moved to `archive/`.
- **Converted all notebooks to scripts**: `1_training_data_inspect` -> `01_prepare_labels.py`; `3_split_train_test` -> `02_make_spatial_folds.py`; `1b_gmia_next_inspect` -> `analysis/a1_gmia_next_points.py`; `map_esa_lgrip_morocco` -> `analysis/a2_map_reference_products.py`.
- **New** `04_download_chips.py` (shared chip store) and `05a-05d` embedding scripts (Clay, Prithvi, TerraMind, AgriFM) with a common output format `outputs/05_embeddings/<model>_<window>.npz`.
- **New** `06_train_evaluate.py`: one script for every input (`INPUT` setting), nested spatial CV, 4 classifiers, block-bootstrap CI, paired classifier differences, spatial vs random CV, grouped importance, final models.
- **New** `07_compare_models.py`: all inputs on the same parcels, differences vs the S1/S2 baseline with bootstrap p-values (Holm-adjusted).
- **New** `environment.yml` (conda env `condaml`), README, .gitignore. Clay installed with `--no-deps` (einops conflict with TerraTorch).
- S1/S2 extraction: per-parcel agricultural year, Cloud Score+, 10 m inner buffer, `ag_season` column in the output.
- Training: gap filling by interpolation; SVM added as 4th classifier; spatial-vs-random comparison at fixed hyperparameters.
- Methodology document drafted (Claude Docs: "Thesis Methodology: Irrigated vs Rainfed Mapping with GeoFMs") and reading list of verified references.

### Before 2026-10-15 (summary)
- Field survey inspected; irrigated / rainfed labels defined (pivot irrigation included; cropland only).
- 20 km spatial blocks chosen (random split: 79% of test parcels within 250 m of a training parcel; spatial: ~2%).
- Feature extraction moved from open-source STAC to Google Earth Engine (speed).
- AlphaEarth tested as a first embedding input (not part of the thesis).
