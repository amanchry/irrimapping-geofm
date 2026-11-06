# Progress log

Running record of decisions, code changes and open points for the thesis
*Irrigated vs rainfed cropland mapping in Morocco with geospatial foundation models*.
Update this file with every change to the code or the method (newest entry on top in the
change log).

---

## Current status (2026-11-05)

| Step | Where | Status |
| --- | --- | --- |
| Labels, spatial folds | `ml_dataset_prep/01`, `02` | Done: 8,781 parcels (7,585 irrigated, 1,196 rainfed), 5 folds of 20 km blocks |
| Tile store, model views | `ml_dataset_prep/03`, `04` | Done: 2,987 tiles; parcel features, pixel sets, chips p8 / p16 |
| Dataset inspection | `models_run/01` | Done (`outputs/study/01_inspection/`) |
| Embeddings | `models_run/02a–02d` | Done for Clay, Prithvi, TerraMind, AgriFM (`outputs/study/02_embeddings/`); AgriFM scaling still to confirm |
| E01 · frozen + PCA, nested CV | `experiments/E01_frozen_pca_nested` | Done: no GeoFM beats the baseline (best Clay + SVM 0.910 vs 0.906) |
| E02 · frozen + PCA, PASTIS | `experiments/E02_frozen_pca_pastis` | Ready to run (main RQ1 result) |
| E03 · frozen + L-TAE, PASTIS | `experiments/E03_frozen_ltae_pastis` | Planned: needs `models_run/03c_train_ltae.py` |
| E04 · PSE-TAE from scratch | `experiments/E04_psetae_scratch` | Run exists; parked for the next phase |
| Documentation | `docs/`, `mkdocs.yml` | Thesis design, overview, protocol, decisions, results, weekly notes, literature notes; website (MkDocs, GitHub Pages) |

Next steps
1. Run E02 (`python experiments/run_experiment.py`, `EXPERIMENT = "E02_frozen_pca_pastis"`).
2. Confirm the AgriFM normalisation; re-embed if needed.
3. Write `models_run/03c_train_ltae.py`, run E03.
4. Then RQ2 (LoRA fine-tuning).

---

## Method decisions

| Date | Decision | Reason / reference |
| --- | --- | --- |
| 2026-11-05 | Results that go into the thesis are produced as **experiments** (config per experiment, `experiments/`); two temporal-aggregation pipelines: concat + PCA 128 (E01, E02) and L-TAE (E03) | Reproducibility; PANGAEA (Marsocci et al. 2024) |
| 2026-11-04 | Validation = **PASTIS rotation** (3 folds train / next fold validation / 1 fold test); nested CV kept as sensitivity check (E01) | Garnot & Landrieu 2021; one design for classifiers and neural networks |
| 2026-10-29 | All inputs (parcel features and foundation-model chips) cut from ONE tile store; baseline = per-pixel median of parcel pixels without edge pixels; Sentinel-1 orbit of the baseline = orbit of the chips | Identical pixels, compositing and masks for every model; tile design as in PASTIS (Garnot & Landrieu 2021), Sen4AgriNet (Sykas et al. 2022) |
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
- [x] ~~Complete the Ouassanouan et al. (2026) reference~~ (done 2026-11-05).
- [ ] Prithvi on 10 m input (pretrained on 30 m HLS): state in thesis; optional 30 m ablation.
- [ ] Neighbouring fields inside a chip (mixed 160 m patches, transformer context): add parcel "purity" + accuracy by parcel size; optional masked-chip ablation (`MASK_OUTSIDE`).
- [ ] Per-year results (2023/24 vs 2024/25) from the out-of-fold predictions; year-transfer test within Doukkala / El Haouz.
- [ ] Tunisia: label source and transfer experiment (zero-shot and few-shot).
- [ ] Prediction / wall-to-wall mapping step (later).
- [ ] Optional: ET (WaPOR), LST, precipitation (CHIRPS) as extra baseline variables, as ablations only.
- [ ] **Phase 2 (after Morocco is complete): US transfer study** - see the change-log entry of 2026-11-04 (future phase). First step: one-day data check of the original state label sources.

---

## Change log (newest first)

### 2026-11-06 (documentation website with MkDocs; D11 heads and mapping)
- **MkDocs Material website** replaces the hand-built HTML (`docs/build_site.py`, `docs/site/` moved to
  `tmp_trash/old_html_site/`). `mkdocs.yml` (root) + `mkdocs/gen_pages.py` (mkdocs-gen-files): the markdown files stay
  where they are; at build time they are mirrored into the site at the same paths, so links work the same on GitHub
  and on the site; images copied; links to scripts / tables / folders point to GitHub; navigation (tabs: Home,
  Research, Results, Meetings, Code reference) written from the experiment configs (mkdocs-literate-nav).
  Anchors use GitHub's slugs (pymdownx.slugs). Mermaid rendered by `mkdocs/pages/javascripts/mermaid.js` (keeps the
  diagram colours). `.github/workflows/docs.yml` rebuilds and publishes to GitHub Pages on every push to main.
  `environment.yml`: mkdocs<2, mkdocs-material, mkdocs-gen-files, mkdocs-literate-nav. `.gitignore`: `site/`.
  `make_report.py`: `<details markdown="1">` so the collapsed table renders on both.
  Checked: build without warnings; 19 pages, 26 Mermaid diagrams, no broken internal links or anchors.
- **D11** (docs/04_decisions.md): parcel benchmark with classical heads for RQ1; for the maps (RQ2–RQ4) frozen GeoFM +
  UPerNet on the tile label pixels, RF on per-pixel S1/S2 features, and neural baselines TempCNN (pixel) and U-TAE
  (tile); pixel predictions aggregated per parcel for the same scoring. Literature table added to
  docs/research/README.md (references checked against the papers).

### 2026-11-05 (project organised: experiments, documentation hierarchy, HTML site)
- **Experiments.** New `experiments/`: one folder per experiment with `config.yaml` (all settings, same names as the
  scripts), `README.md` (question, why, what changes, how to run), `REPORT.md` (answer first; tables and figures
  generated) and `results/`. `run_experiment.py` runs the steps of a config (train_evaluate per input, compare,
  report), skips finished runs and saves config + git commit + dates in `results/run_info.json`.
  `make_report.py` writes the tables (macro-F1 per input × classifier, differences vs baseline with Holm verdict,
  per-class metrics with CI, macro-F1 per fold) and two figures into REPORT.md, and refreshes
  `experiments/README.md` and `docs/05_results.md`. Template in `experiments/_template/`.
- **Experiment mode in the scripts.** `models_run/03_train_evaluate.py`, `03b_train_psetae.py`, `04_compare_models.py`:
  new setting `EXPERIMENT` (None = as before, results in `outputs/study/`). When set (or by `run_experiment.py`), the
  config's section replaces the settings, unknown names stop the run, results go to
  `experiments/<name>/results/`; finished runs are skipped. 03: derived paths moved below the override; no
  `_nested` suffix inside an experiment. 04: compares all runs of the experiment, baseline found automatically,
  `EXTRA_RUNS` adds runs of other experiments (PCA vs L-TAE); figure title names the experiment.
- **Runs moved:** the five runs of 30 Oct (nested CV) from `outputs/study/03_models/` to
  `experiments/E01_frozen_pca_nested/results/`, compared and reported (8,775 common parcels); the PSE-TAE run to
  `experiments/E04_psetae_scratch/results/` (parked, `listed: false`).
- **E01 result:** no GeoFM differs from the baseline after Holm for Clay, Prithvi, TerraMind; AgriFM worse (RF, SVM).
  Best: Clay + SVM 0.910, baseline SVM 0.906. Fold 0 (rainfed in Doukkala + Souss): baseline 0.80, Clay and Prithvi
  higher with every classifier.
- **Planned:** E02 (PASTIS validation, ready) and E03 (L-TAE pipeline; design in its README and in the protocol §5).
- **Documentation hierarchy:** `docs/README.md` (index with map), `01_thesis_design.md`, `02_overview.md`,
  `03_protocol.md`, `04_decisions.md` (D1–D10: departures from the proposal), `05_results.md`, `weekly/` (one page
  per meeting; first: 2026-W45), `research/README.md` (literature notes). Breadcrumbs and next / previous links on
  every page; main README, `ml_dataset_prep/README.md` and `models_run/README.md` linked into the hierarchy.
- **HTML site:** `docs/build_site.py` (standard library) writes `docs/site/*.html` from the markdown: sidebar by
  hierarchy, page contents, Mermaid diagrams, working links between pages.
- Fixed: Ouassanouan et al. (2026) reference completed (title, DOI, from the proposal); leftover of the chips rename
  in `ml_dataset_prep/README.md` ("land use, chip and irrigation method" -> "crop").
- `.gitignore`: `*.pt` and the final models in experiment results.
- **Fixed tile-store path:** `ml_dataset_prep/03`/`04` write to `outputs/dataset/03_tiles_128px/`, but the readers still
  used `03_tiles/` -> updated in `models_run/01, 02a-02d, 03, 03b`, `tools/a3, a4, a5`, docstrings and READMEs
  (24 places). Without it every models_run script stopped with "views_info.json not found".
- Checked: config override and typo stop (03); `run_experiment.py` on E01 (finished runs skipped, compare + report
  rerun, same numbers); a QUICK_TEST run of the baseline in experiment mode with the PASTIS rotation on the real data
  (5 rounds, 1.4 min); site: 19 pages, 26 Mermaid diagrams, no console errors, no broken page or section links;
  all markdown links and anchors resolve.

### 2026-11-05 (US parcel data: geodatabases converted to shapefiles; IrrMap code checked)
- Data in `IrrMap/US_Parcel_Data` (outside this repo). New script there: `gdb_to_shp.py` (settings at top,
  resumable in chunks). Output next to each source in `<gdb name>_shp/`, original CRS, checked against the source
  (same feature count and total area for every layer).
  - Arizona folder = USGS *Verified irrigated agricultural lands 2002-17*: 14 state layers (AZ, CA, CO, FL, GA, ID, IL,
    MS, MT, NM, TX, UT, WA, WY), **irrigated only** (IRG_STATUS, IRG_SYS, CROP_TYPE, DATA_DATE, DATA_YEAR).
  - Washington = WSDA crop distribution 2025, 222,520 fields; crop names joined from the WSDACrop table.
    **137,185 fields have Irrigation = "None"** -> candidate rainfed/dryland class (to verify: fallow, pasture, CRP).
  - Colorado CRDSS Div 5 (2000-2023) and Utah WRLU (2017-2024) were already shapefiles.
  - Field names shortened to 10 characters (e.g. InitialSurveyDate -> InitSurvey); dates stored as text YYYY-MM-DD.
- IrrMap code check: random 80/20 split of single-date patches; ~100% of test patch locations also appear in train on
  other dates (WA 2018: 31,056 of 31,064) -> their scores are not spatially independent. Test set also used for
  per-epoch evaluation. Minor bugs (missing argparse import, undefined src_path, constant LAND_TYPE band).
  Reusable for phase 2: the label sources only, not their patches.

- Usefulness check (both classes needed):
  - **Washington WSDA 2025 - most useful.** Field crops (cereal, oilseed, vegetable, hay, seed): 53,723 rainfed
    (Irrigation = None) vs 33,172 irrigated; rainfed mainly in the dry east (Whitman, Spokane, Lincoln, Adams).
    Use eastern WA only (semi-arid, winter rain, dryland wheat next to irrigated fields = closest US match to Morocco).
  - **Utah WRLU 2017-2025 - useful, adds years.** 2023: 15,611 "Dry Crop" vs 110,907 irrigated (fallow/pasture removed).
    Yearly maps -> temporal transfer test.
  - USGS verified irrigated lands 2002-17 (Arizona folder) and Colorado CRDSS Div 5: irrigated only -> no rainfed class;
    at most extra irrigated examples.
  - Label definitions: Utah metadata defines "Dry Crop" = no irrigation method applied, crop watered by natural
    processes (rain); "None" there = non-agricultural land. WSDA has no public definition of Irrigation = "None"
    (WSDA's own "Dryland Crops" layer = crops grown without irrigation water). 34,235 of the 53,723 WA "None" crop
    fields come from NASS (CDL), not WSDA ground survey -> irrigation probably not observed. **Use DataSource = WSDA
    only: 19,390 rainfed vs 23,705 irrigated.**

### 2026-11-04 (future phase: US data for spatial and temporal transfer — decision deferred)
Discussed after reading IrrMap (Mandal et al. 2025, arXiv:2505.08273). **Decision: complete Morocco first; come back to this later.**
- **IrrMap itself is not usable as the study area.** It labels irrigation *method* (flood / sprinkler / drip, all else
  "other"); there is **no rainfed class**, so irrigated vs rainfed cannot be studied with it. Its labels come from
  state agency maps (not field survey); inputs are single-season 224 px patches; validation is a random 80/20 split
  per state, with no validation split and no spatial test; only a U-Net (ResNet-34) is benchmarked, no GeoFMs.
  -> Keep Morocco as the AOI (field-surveyed labels, data-scarce region, 12-month time series, spatial folds).
  Cite IrrMap as related work and as a model for the dataset paper (Part 1).
- **Idea for a second phase:** use the **original field boundaries and labels** behind IrrMap (state datasets), build
  chips from scratch with the same pipeline (labels -> 20 km block folds -> tile store -> chips -> embeddings ->
  same classifiers and validation), to test **spatial and temporal transferability**.
  - Why it is valuable: state data often covers many years over the same fields -> clean temporal transfer
    (train earlier years, test later years), which Morocco cannot offer (year and region overlap); a second
    continent tests whether GeoFMs transfer better than S1/S2 features (Morocco <-> US).
  - Conditions: (1) the source data must contain **non-irrigated / dryland parcels** (deal-breaker; e.g. Washington
    agricultural land-use data includes dryland wheat — to be checked state by state); (2) small scope: 1-2 states,
    3-4 years, ~10-20k parcels; (3) only after Morocco is finished.
  - Pipeline changes needed: new label reader in `ml_dataset_prep/01_prepare_labels.py`; **support other UTM zones** in
    `03_build_tiles.py` (now fixed to UTM 29N; western US spans zones 10-13); US season window via `START_MONTH`;
    optional fold by year in 02; a new transfer evaluation (train region/year A, test B). `models_run/` unchanged.
  - Risks: administrative (noisier) labels; different crop systems (Morocco <-> US transfer may be poor for all models);
    ~4-8 extra weeks.
  - First step when resuming: one-day check of which state sources IrrMap used, whether they include
    dryland / non-irrigated fields, which years, and whether field boundaries are downloadable.
  - Fallback if the check fails: keep as future work; use GMIA-NEXT ground-truth points (`tools/a1`) or Tunisia as
    the external test.

### 2026-11-04 (validation design: PASTIS scheme)
- `models_run/03_train_evaluate.py`: new setting `VALIDATION = "pastis"` (default). Per round: 3 folds train,
  the next fold (k + 1) is the validation fold that chooses the hyperparameters, fold k is the test fold; the
  model is trained on the 3 training folds. Same scheme as PASTIS (Garnot & Landrieu 2021) and
  `03b_train_psetae.py`, so classifiers and the neural-network baseline are trained, tuned and tested identically.
  `VALIDATION = "nested"` keeps the earlier design (inner spatial CV on 4 folds, refit on 4) for a sensitivity
  check; its runs get the suffix `_nested`, so run names of the default stay the ones `04_compare_models.py` expects.
- READMEs (main, ml_dataset_prep §5.4, models_run §6.3 + workflow diagram) updated with the rotation table.
- Results produced before this change used the nested design.

### 2026-10-30 (PSE-TAE moved to next phase)
- PSE-TAE removed from the READMEs (main + models_run: workflow diagram, tables, section 6b, run order, settings,
  reference); it will be presented in the next phase. Code kept unchanged: `models_run/03b_train_psetae.py` and the
  PSE-TAE entry in `04_compare_models.py` (RUNS skips it until the run exists). Previous README versions in `tmp_trash/`.

### 2026-10-30 (models_run workflow diagram)
- `models_run/README.md` section 1: new single Mermaid workflow diagram, from the Part 1 outputs to the comparison:
  six inputs with their data, bands, patch size and time handling; embedding pooling + PCA; the four classifiers
  with their grids; PSE-TAE training; nested spatial CV; metrics; comparison; optional sensitivity runs.

### 2026-10-30 (terminology: chips)
- One term for the 64 x 64-px windows cut from the tiles around each parcel: **chips** (was "crops").
  Code: `crops_p8/16` -> `chips_p8/16`, `CROP_PX` -> `CHIP_PX`, `CROP_PATCH_SIZES/PX` -> `CHIP_PATCH_SIZES/PX`,
  index column `n_px_in_crop` -> `n_px_in_chip`, views_info key `s1_orbit_for_crops_and_pixelsets` ->
  `s1_orbit_for_chips_and_pixelsets`, study/03 option `S1_ORBIT = "crops"` -> `"chips"`; the old separately
  downloaded store is now called "legacy". Agricultural uses (cropland, crop type, Seasonal Crop) unchanged.
- Existing outputs renamed in place (folders, CSV headers, JSON keys): no re-run needed. Tested study/01 and 03b.

### 2026-10-30 (deep-learning baseline)
- Classifier choice: keep the 4 classifiers of study/03 (logistic regression, RF, HGB, SVM); headline = RF for all
  inputs + logistic regression (linear probe) for the foundation models; HGB / SVM as robustness checks.
- New `models_run/03b_train_psetae.py`: PSE-TAE (Garnot et al. 2020) trained from scratch on the pixel sets
  (10 S2 bands + S1 VV/VH of the chips' orbit, inner pixels), paper-default architecture, class-weighted
  cross-entropy, early stopping on the next fold, same 5 outer spatial folds, same output format as 03.
  Tested on synthetic pixel sets (learns the signal: macro-F1 0.998; early stop after 16-22 epochs).
- `04_compare_models.py`: run `psetae_agyear_s2s1` added to RUNS; runs with their own model are compared with the
  baseline's `BASELINE_CLASSIFIER = "rf"` (new column `baseline_classifier`). Tested.
- READMEs updated (models_run section 6b with diagrams, run order, settings; main README).

### 2026-10-30 (folders renamed, main README)
- Folders renamed by the user: `dataset/` -> `ml_dataset_prep/`, `study/` -> `models_run/`, `analysis/` -> `tools/`;
  raw data moved from `data/raw/<x>` to `data/<x>`; `archive/` removed. Output folders unchanged (`outputs/dataset`,
  `outputs/study`, `outputs/analysis`).
- Fixed the remaining `data/raw` paths (ml_dataset_prep/01, tools/a1, tools/a2) and `.gitignore` (`data/`).
  Script descriptions, folder READMEs and this file updated to the new folder names (text only).
- Main README rewritten as an overview: the two parts, overall workflow diagram (Mermaid), project structure,
  setup (environment, Earth Engine, input data, model weights), run order, tools, links to the folder READMEs.

### 2026-10-30 (dataset check + parcel explorer)
- Checked the full dataset run (dataset/01-04) and models_run/01: 2,987 tiles, 8,781 parcels in both chip views, 8,775 in the
  feature tables (6 parcels fully covered by overlapping polygons have no own pixel; excluded automatically by
  models_run/04), no missing values, every parcel-month has a clear S2 view, NDVI r = 0.97-0.99 vs the legacy EE table,
  S1 orbit ASC. Purity (median) p8 0.36 -> 0.59, p16 0.09 -> 0.30 with patch alignment. Ready for models_run/02-04.
- New `tools/a5_parcel_explorer.py` -> `outputs/analysis/parcel_explorer.html`: parcels on a satellite / street
  basemap, filters (class, year, zone, fold), colour by class or fold, search; click -> monthly NDVI / NDWI / VH / VV /
  clear-observation profile against class median + IQR of the same agricultural year; pin parcels to compare.
  Standard library only; tested in a headless browser (no errors).

### 2026-10-30 (README: two agricultural years)
- `ml_dataset_prep/README.md`: new section 6.2 explaining how the two agricultural years are handled (season from the
  survey date, tiles keyed by season + grid position, season-only label burning, one Earth Engine image per year,
  same band layout, folds by location so a place in both years stays in one fold, 47 tile positions in both years),
  with a flow diagram; the season split added to the overview diagram. Section 6 renumbered (6.3-6.6).

### 2026-10-30
- New `tools/a4_build_vrt.py`: all tiles combined into GDAL virtual rasters, one per agricultural year (the years
  overlap in space): labels, true colour, false colour and Sentinel-1 ascending for each month; optional 228-band
  full stack (`FULL_STACK`, ~230 MB XML for 2023/24, off by default). Standard library only; relative paths.
  Tested: VRT pixels and band names identical to the source tiles. Built for the current store: 2,183 tiles
  (2023/24) and 804 tiles (2024/25) -> 74 VRTs in `outputs/dataset/03_tiles/agyear/vrt/`.

### 2026-10-29 (documentation)
- New `ml_dataset_prep/README.md` and `models_run/README.md`: full methodology of each part, every script (purpose, steps,
  settings, outputs), Mermaid flow diagrams in the markdown, file formats, limitations and references.
  The main README links to both. TODO: complete the Ouassanouan et al. (2026) reference (title, DOI).
- Docstrings corrected: embedding output names `<model>_agyear_tiles.npz` (02a-02d), baseline source in models_run/03,
  a3 output folder.

### 2026-10-29 (restructure)
- **Code split into two parts.** `ml_dataset_prep/` builds the ML-ready dataset (publishable separately);
  `models_run/` uses only its outputs. `pipeline/` is no longer used.
- Renamed scripts: `pipeline/01`, `02` -> `dataset/01`, `02`; `pipeline/04_build_tiles.py` -> `ml_dataset_prep/03_build_tiles.py`;
  `pipeline/04b_make_views.py` -> `ml_dataset_prep/04_make_views.py`; `pipeline/05a-05d` -> `models_run/02a-02d`;
  `pipeline/06_train_evaluate.py` -> `models_run/03_train_evaluate.py`; `pipeline/07_compare_models.py` -> `models_run/04_compare_models.py`.
  `pipeline/03_extract_s1s2_features.py` (Earth Engine features) moved to `archive/legacy_pipeline/`.
- Outputs moved: `outputs/01_labels`, `02_folds` -> `outputs/dataset/`; tile store in `outputs/dataset/03_tiles/agyear/`;
  `05_embeddings`, `06_models`, `07_comparison` -> `outputs/study/02_embeddings`, `03_models`, `04_comparison`;
  old `03_s1s2_features` and `04_chips` -> `outputs/legacy/`. All paths in the scripts updated.
- **New `models_run/01_inspect_dataset.py`:** inventory per class / season / fold, label QA (overlaps, tiny parcels),
  clear observations per month, patch purity aligned vs centred, seasonal NDVI / NDWI / VH profiles,
  example chips per class, NDVI agreement with the legacy Earth Engine features. Output: `outputs/study/01_inspection/`.
- a3 now reads `views/chips_p16` and writes to `outputs/study/01_inspection/chips_geotiff/`.
- README rewritten for the two-part structure. Backups: `archive/scripts_2026-10-29/`.

### 2026-10-29
- **New data source for all inputs: shared tile store.** `04_build_tiles.py` (non-overlapping 128 x 128-px tiles,
  label / parcel-ID / edge rasters, S1 per orbit in linear power) and `04b_make_views.py` (parcel features, pixel sets,
  patch-aligned chips `chips_p8` / `chips_p16`) replace `04_download_chips.py` and the Earth Engine features of
  script 03. Old scripts and results are kept for reference; old runs keep their names.
- **05a-05d:** new settings `CHIP_SOURCE = "tiles"` and `CHIP_PATCH_PX` (8 for Clay, 16 for Prithvi, TerraMind,
  AgriFM). Output files are now `<model>_agyear_tiles.npz` (old: `<model>_agyear.npz`).
- **06:** new settings `S1S2_SOURCE = "tiles"`, `TILE_FEATURES = "inner"`, `S1_ORBIT = "chips"`, `EMB_SOURCE = "tiles"`.
  The parcel-feature table of 04b is read directly: bookkeeping columns (fold, n_px_used, used_inner) and the
  observation counts (S2_NCLEAR, S1_N_*) are dropped, the chosen Sentinel-1 orbit is renamed to VV/VH (same orbit
  as the foundation-model chips by default), months are labelled Sep ... Aug. New run names:
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
    chips per patch size (8, 16) shifted so the parcel centroid is at a patch centre; same format as the old chips.
  - Why: old chips overlapped (median 8 other chips per chip, 6.3% across folds), baseline and foundation models used
    different pixels / compositing, small parcels were split over 4 patches (purity 0.37 / 0.10 for 8 / 16 px patches;
    aligned chips: 0.62 / 0.36 in the test sample), S1 was averaged in dB with orbits mixed, no observation counts.
  - Label QA from the parcel geometry: 79 overlapping parcel pairs (2 with different labels, 25 covering > 50% of the
    smaller parcel) -> `parcel_overlaps.csv`; 3,684 shared pixels set to ignore; 65 parcels have no pixel after the
    10 m inner buffer.
  - Design follows PASTIS, Sen4AgriNet, S2-Agri (PSE-TAE), Fields of The World, ClarkCGA / Prithvi crop chips.
- Next: run 04 with `N_TEST_TILES = 20`, inspect tiles in QGIS, full run, then 04b; point the 05 scripts to
  `outputs/04_tiles/agyear/views/chips_p8` (Clay) or `chips_p16` (Prithvi, TerraMind; AgriFM to check).

### 2026-10-16
- `05d_embed_agrifm.py`: downloads the weights automatically from https://glass.hku.hk/casual/AgriFM/AgriFM.pth and clones the AgriFM code if missing.
- `05d_embed_agrifm.py`: checks for mmengine, the AgriFM code (`external/AgriFM`) and the weights before starting, with a clear message for each.
- `06_train_evaluate.py`: `INPUT` reset to `"s1s2"` (it was set to `"clay"` before the Clay embeddings existed).
- S1/S2 features extracted: `outputs/03_s1s2_features/s1s2_features_agyear_30d.csv` (8,781 parcels, 192 features, no empty values).
- Note (macOS): if a script stops with `OMP: Error #15` (two OpenMP copies), install PyTorch from conda-forge instead of pip, or set `KMP_DUPLICATE_LIB_OK=TRUE` as a workaround.

### 2026-10-15
- **Added** `tools/a3_export_chips_geotiff.py`: writes chips of selected parcels as GeoTIFFs (14 bands per month, 12-month NDVI, parcel mask and outline) for QGIS.
- **Restructured the project** into `pipeline/`, `analysis/`, `data/raw/`, `checkpoints/`, `external/`, `secrets/`, `outputs/<step>/`, `docs/`, `archive/`. Old scripts, notebooks and outputs moved to `archive/`.
- **Converted all notebooks to scripts**: `1_training_data_inspect` -> `01_prepare_labels.py`; `3_split_train_test` -> `02_make_spatial_folds.py`; `1b_gmia_next_inspect` -> `tools/a1_gmia_next_points.py`; `map_esa_lgrip_morocco` -> `tools/a2_map_reference_products.py`.
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
