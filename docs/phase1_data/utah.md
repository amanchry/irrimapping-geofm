# Utah training data

This page describes the **Utah dataset**, the second, independent study area
([D13](../decisions.md#d13-utah-as-a-second-independent-study-area)). It is built with the same pipeline as the
[Morocco training data](morocco.md): only the label script (`data_training_prep/01_prepare_labels_Utah.py`) is
region-specific; the other scripts of `data_training_prep/`, `data_download/` and `pixelwise_run/` are the same files,
run with `REGION = "utah"`. As in Morocco, the experiments classify the labelled
10 m pixels of the crop fields as irrigated or rainfed
([D16](../decisions.md#d16-pixel-wise-classification-for-continuous-maps)); no cropland mask is used. The
experiments run in Morocco first; Utah follows later with the same plan.

!!! info "Status"
    Labels (01), spatial folds (02) and the tile plan with label rasters (03) are complete for WRLU 2024
    (2023/24) and WRLU 2021 (2020/21). For 2023/24 the tile images (03) are complete;
    the [pixel samples](pixel_samples.md) (04) are to be rebuilt on the cluster with the two-class script
    (`04_pixel_samples/`), after which E01–E05 follow the Morocco runs. The earlier three-class samples are kept in
    `04c_pixel_samples/` and are no longer used. The 2020/21 data are prepared for the temporal test (RQ3). The
    wall-to-wall imagery is downloaded per 10.24 km block by `data_download/` (Dynamic World from Earth Engine,
    Sentinel-1/2 from open STAC catalogues, [D21](../decisions.md#d21-sentinel-2-and-sentinel-1-from-open-stac-catalogues-earth-engine-only-for-dynamic-world)), which replaces the
    retired STAC download ([D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them));
    see [Cropland extent §2](cropland_extent.md#2-blocks-and-downloads).

**Contents**

1. [Overview](#1-overview)
2. [Study design](#2-study-design)
3. [Why Utah](#3-why-utah)
4. [Source data](#4-source-data)
5. [Which years, and how the labels change between years](#5-which-years-and-how-the-labels-change-between-years)
6. [Script 01 — labels](#6-script-01--labels)
7. [Script 01 — sampling of cells](#7-script-01--sampling-of-cells)
8. [Script 02 — spatial folds](#8-script-02--spatial-folds)
9. [Script 03 — tile store](#9-script-03--tile-store)
10. [Temporal test design](#10-temporal-test-design)
11. [Differences from Morocco](#11-differences-from-morocco)
12. [Running, one year at a time](#12-running-one-year-at-a-time)
13. [Known limitations](#13-known-limitations)

---

## 1. Overview

| Step | Script | What comes out |
| --- | --- | --- |
| 1 | `01_prepare_labels_Utah.py` | 12,015 labelled crop fields in 550 sampled cells: WRLU 2024 (5,000 irrigated, 1,013 rainfed) and WRLU 2021 (4,987 irrigated, 1,015 rainfed); plus all 245,648 labelled crop fields of both years before sampling |
| 2 | `02_make_spatial_folds.py` (`REGION = "utah"`) | a fold (0–4) per field, from 183 spatial blocks of 20 km |
| 3 | `data_training_prep/03_plan_tiles.py`, `data_download/01_plan_blocks.py` … `02c_s1.py`, `data_training_prep/04_cut_tiles.py` (`REGION = "utah"`) | tile plan: 5,436 tiles (2,720 for 2023/24, 2,716 for 2020/21); label rasters written; images cut from the downloaded blocks |
| 4 | `04_make_pixel_samples.py` (`REGION = "utah"`) | labelled pixels of the crop fields (irrigated, rainfed) per year (`04_pixel_samples/<season>/`): 2023/24 to be rebuilt with the two-class script |

All outputs are in `outputs/utah/data_training_prep/` (`01_labels/`, `02_folds/`, `03_tiles_128px/agyear/`,
`04_pixel_samples/`), with the same
file names and columns as `outputs/morocco/data_training_prep/`.

---

## 2. Study design

- **Utah is an independent second study area.** There is no Morocco → Utah transfer: each region is trained and
  tested on its own data, with the same pipeline, foundation models, classifiers and evaluation, and the results
  are reported side by side. This tests whether the conclusions from Morocco hold in a second region with other
  crops, climate and labels.
- **Two questions in Utah:**
    1. how well the models separate irrigated and rainfed pixels (spatial cross-validation, as in Morocco);
    2. **temporal transferability:** whether a model trained on one year still works in another year
       ([section 10](#10-temporal-test-design)).
- **Training year:** WRLU 2024 (agricultural year `2023/24`). **Temporal test year:** WRLU 2021 (`2020/21`).

---

## 3. Why Utah

The candidates were compared in [D10](../decisions.md#d10-transfer-region-under-review) (Tunisia, eastern
Washington, Utah, the IrrMap states):

- **An explicit rainfed class.** The WRLU metadata defines `IRR_Method = "Dry Crop"` as a field with no irrigation,
  watered by natural processes (rain). IrrMap labels only the irrigation method (no rainfed class), and Colorado
  (CRDSS) and most layers of the USGS verified irrigated lands are irrigated-only.
- **Enough rainfed fields:** 13,000–17,000 rainfed crop fields every year (figure below). Tunisia has 10 rainfed
  GMIA-NEXT points.
- **Every year from 2017 to 2025**, mapped by one agency with one method, so the same pipeline also gives a
  temporal test.

<figure markdown>
  ![WRLU crop fields per year](figures/utah/source_years.png){ width="760" }
  <figcaption>WRLU crop fields (grain/seeds, hay/turf, field crops) per year and class, before sampling.</figcaption>
</figure>

---

## 4. Source data

| Data | Use | Details |
| --- | --- | --- |
| Water-Related Land Use (WRLU), Utah Division of Water Resources | labels | one polygon layer per year, 2017–2025; all agricultural fields of Utah and of the parts of its river basins in Idaho, Wyoming, Colorado, Nevada and Arizona; crop (`Descriptio`, `CropGroup`), irrigation method (`IRR_Method`), basin, county, state. In `data/Utah_field_data/` |
| FAO GAUL 2024 (level 1) | maps | state outlines (`data/admin_boundaries/`) |
| Sentinel-2 L2A (SCL cloud mask), Sentinel-1 RTC, Dynamic World | imagery | as for Morocco: Sentinel-2 from Earth Search and Sentinel-1 from Planetary Computer (open STAC catalogues), Dynamic World from Google Earth Engine, per 10.24 km block (`data_download/`), for the training tiles and the wall-to-wall maps ([D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)); the models use Sentinel-2 only (D14) |

---

## 5. Which years, and how the labels change between years

**Agricultural year.** WRLU year *Y* describes the crops of the growing season *Y*. The dataset uses the US water
year, 1 October (*Y*−1) to 30 September (*Y*), written as for Morocco: WRLU 2024 → `2023/24`, WRLU 2021 → `2020/21`.
It covers winter wheat from emergence to the July harvest, the whole summer irrigation season and June–September,
when irrigated and dry crops differ most. A calendar year would start in snow and cut off the autumn crops. As in
Morocco, every field gets the 12 monthly composites of its own year.

**Check against two crop calendars.** The window was compared with the GGCMI Phase 3 crop calendar (Jägermeyr et al.,
2021; 0.5° grid; median over the 92 cells inside Utah) and the FAO AQUASTAT irrigated crop calendar (FAO, 2023), which
gives one calendar for the US Southwest (Arizona, California, Hawaii, Nevada, New Mexico and Utah).

| Crops | GGCMI Phase 3 (Utah cells) | FAO AQUASTAT (US Southwest) | Inside Oct–Sep? |
| --- | --- | --- | --- |
| Winter wheat | irrigated sown about 1 September (day 244), rainfed 19–29 September (day 262–272); harvest July (sources: MIRCA2000, SAGE) | November – May | growth and harvest yes; the September sowing falls in m12 of the previous year |
| Spring wheat, barley | sown April, harvest July – August | barley November – April | yes |
| Maize | sown about 10 May (day 130), mature about mid-October (155 days; source: SAGE) | April – August | yes, except the last days before harvest |
| Vegetables, potatoes, sugar beet, sunflower | gap-filled values only | April – August | yes |
| Temporary fodder | not in GGCMI | November – March | yes |

GGCMI places Utah's winter-wheat sowing in September, before 1 October, so the window starts with the wheat already
sown or emerging rather than at sowing; bare, freshly sown fields in September are not part of a field's year. Maize
can be harvested in the first half of October, the first month of the next year. No 12-month window avoids both
cuts; October – September keeps the whole summer irrigation season, including September. The FAO calendar mixes
Utah with the low deserts of Arizona and California (winter cereals and fodder from November to April) and so
describes Utah poorly. Alfalfa and other hay are in neither dataset.

**How much changes between years.** All labelled crop fields of each year (the whole of WRLU, not only the sample)
were compared with WRLU 2024. A field counts as the same field when its outline overlaps a 2024 outline with an
intersection over union above 0.8 (`outputs/utah/data_training_prep/01_labels/year_change_vs_2024.csv`):

| WRLU year | Labelled crop fields | Same outline as in 2024 | Crop changed | Irrigation method changed | Class changed (irrigated ↔ rainfed) |
| --- | --- | --- | --- | --- | --- |
| 2024 | 122,624 (15,276 rainfed) | – | – | – | – |
| 2025 | 123,722 | 121,811 (98.5%) | 6.3% | 17.4%¹ | 28 (0.02%) |
| 2023 | 121,029 | 116,492 (96.3%) | 27.1% | 0.1% | 17 (0.01%) |
| **2021** | 123,024 | 110,821 (90.1%) | **35.7%** | 0.7% | **83 (0.07%)** |
| 2019 | 121,063 | 100,491 (83.0%) | 38.0% | 2.7% | 1,352 (1.35%) |
| 2017 | 117,901 | 92,054 (78.1%) | 40.6% | 3.0% | 1,307 (1.42%) |

¹ Mostly a change of coding: in 2025 WRLU split *Sprinkler* into *Sprinkler*, *Pivot* and *Wheel* (wheel line).

- **Field outlines** change slowly (78–98% unchanged).
- **Crops** change a lot through rotation (27–41% after 1–7 years).
- **Irrigated vs rainfed** hardly ever changes (0.01–0.07% since 2021). In Utah, irrigation depends on a water right
  and on canals, pivots and wells tied to the land; part of the stability may also come from labels carried over
  between map updates, which the data cannot separate.

**Choice of years:**

- **WRLU 2024 = training year:** recent, actively updated (27% of the crops changed from 2023), and it overlaps the
  Moroccan survey years.
- **WRLU 2021 = temporal test year:** three years earlier; 36% of the crops differ from 2024 while the labels stay
  consistent (83 class changes in 123,024 fields), so the test isolates the change of year (crops, weather,
  timing). 2021 was a drought year in Utah (drought emergency declared in March 2021), a realistic year-to-year
  shift that is reported as context. It still has both Sentinel-1 satellites (Sentinel-1B failed in December 2021).
- **Not 2025:** 94% of its crops repeat 2024, so it looks largely carried over and is no independent test.
- **Not 2019** (for now): the jump from 83 class changes (2021) to 1,352 (2019) is sudden and may come from a
  revision of the mapping rather than from real change; it would need checking first.
- **Not all years:** the labels are near-copies, so more years add volume but little information, at about 10 GB
  of tiles per year.

---

## 6. Script 01 — labels

1. **Source inspection** for every year in `FILES`: fields and area per crop group and irrigation method
   (`source_counts.csv`, `source_years.png`).
2. **Read** the years in `YEARS` (agricultural fields only) and clean the geometry as for Morocco: repair invalid
   polygons, drop empty and duplicate geometries.
3. **Label:**

    | `IRR_Method` | Label |
    | --- | --- |
    | Flood, Sprinkler, Drip; from 2025 also Pivot and Wheel | 1 · irrigated |
    | Dry Crop | 2 · rainfed |
    | Sub-irrigated (natural high water table) | dropped: neither class |

4. **Crop fields only** (`CROP_GROUPS`): grain/seeds, hay/turf and field crops. Fallow/idle land and pasture are
   dropped (no crop to observe, or grazing land rather than cropland). Garden, orchard and small fruit are dropped
   too: they are small groups, and orchards are almost all irrigated in Utah (19 "Dry Crop" orchards in 2023), so
   they would only add an easy irrigated class.
5. **All labelled crop fields before sampling** (`EXPORT_ALL_FIELDS = True`): every labelled crop field of `YEARS`
   goes to `all_crop_fields.gpkg` (for QGIS) with a column `sampled` that marks the fields of the sampled cells,
   plus summary tables and figures, so the sample can be compared with the whole population:

    | WRLU year | Irrigated | Rainfed | In the sample |
    | --- | --- | --- | --- |
    | 2024 | 107,348 | 15,276 | 5,000 irrigated (4.7%), 1,013 rainfed (6.6%) |
    | 2021 | 107,035 | 15,989 | 4,987 irrigated (4.7%), 1,015 rainfed (6.3%) |

6. **Attributes** with the Morocco names: `parcel_id` (`UT<year>_<feature id>`), `ag_season`, `label`,
   `class_name`, `crop`, `zone` (= river basin), `area_ha`, `nn_dist_m` (nearest field of the same year). Utah
   extras: `year`, `state`, `county`, `crop_group`, `irr_method`, `cell_id`. WRLU has no survey date, so `date` is
   empty.

**How many fields are dropped on the way** (WRLU 2024): 283,781 polygons in the file → 190,269 agricultural (urban,
water, wetland and other land removed) → 122,624 labelled crop fields (fallow, pasture, sub-irrigated, orchard,
garden and small fruit removed) → 6,013 in the sampled cells.

<figure markdown>
  ![All labelled crop fields](figures/utah/all_fields_map.png){ width="820" }
  <figcaption>All labelled crop fields of WRLU 2024 and 2021 before sampling.</figcaption>
</figure>

<figure markdown>
  ![Crops of all fields](figures/utah/all_fields_crops_by_class.png){ width="760" }
  <figcaption>Main crops per class, all labelled crop fields of both years. Alfalfa is the main crop of both classes.</figcaption>
</figure>

---

## 7. Script 01 — sampling of cells

**Why a sample and not all fields.** All labelled crop fields would be about 122,000 fields per year: roughly
20,000 tiles (~75 GB) per year and 15–20 times the model run time of Morocco, for
little extra *independent* information. The statistics are limited by the rainfed class and by the number of
spatial blocks, not by the number of fields: fields in the same block are strongly alike (89% of field pairs
closer than 1 km share their class), and the block bootstrap counts blocks. The sample also keeps the Utah dataset
close to the size of the Moroccan one (8,781 parcels), so model differences are not a matter of data volume.

**Three nested units:**

| Unit | Size | Role |
| --- | --- | --- |
| Field | median 2.5 ha | one sample with its label |
| **Cell** | 1.28 × 1.28 km = one 128-px tile of script 03 | the sampling unit: a cell is sampled with all its fields, or not at all |
| **Block** | 20 × 20 km (~240 cells) | the fold unit of script 02; also used to spread the sample |

Both grids start at the origin of UTM 12N: cell = (⌊x / 1,280 m⌋, ⌊y / 1,280 m⌋), block = (⌊x / 20,000 m⌋,
⌊y / 20,000 m⌋). A field belongs to the cell and to the block that contain its centroid.

**Algorithm** (`sample_cells`: stratified random sampling with a cap per block):

1. **Cells** with at least one labelled crop field in any of `YEARS` (both years pooled): 12,424 cells.
2. **Two strata:** "with rainfed" = at least one rainfed field in either year; "irrigated only" = the rest.
3. **Random order** within each stratum (fixed seed `SEED = 42`, so every run gives the same cells).
4. **Cap:** going down the shuffled list, a cell is skipped once its 20 km block already has `MAX_CELLS_PER_BLOCK`
   (6) cells of that stratum. Without the cap the densest valleys (Cache Valley, Bear River) would take most of the
   sample.
5. **The first `N_RAINFED_CELLS` (400) and `N_IRRIGATED_CELLS` (150)** cells are kept: 550 cells in 176 blocks.
6. **The same cells are used for every year** in `YEARS`; every labelled crop field inside a sampled cell is kept.

**What is labelled inside a sampled tile:**

| In the tile | Label raster |
| --- | --- |
| crop fields (grain, hay, field crops), irrigated or Dry Crop, whose centroid lies in the cell | 1 irrigated / 2 rainfed |
| fallow, pasture, orchards, sub-irrigated fields | 0 = ignore |
| roads, towns, natural land | 0 = ignore |
| a field centred in a neighbouring, not sampled cell that reaches into the tile | 0 = ignore |

A labelled field is always complete: when its outline crosses into the next tile, that tile is built as well
("outline" tile). Within a 20 km block at most 12 cells (6 + 6) are sampled, i.e. at most ~5% of its area is
labelled. Labels are therefore dense within a sampled cell and sparse within a block, as in Morocco, where the
survey did not label every field either.

**Result:**

| Agricultural year | Irrigated | Rainfed | Rainfed share |
| --- | --- | --- | --- |
| 2020/21 (WRLU 2021, test year) | 4,987 | 1,015 | 16.9% |
| 2023/24 (WRLU 2024, training year) | 5,000 | 1,013 | 16.8% |
| **Total** | **9,987** | **2,028** | **16.9%** |

Rainfed fields are over-represented on purpose (16.9% of the sample against 12.5–13% of all fields), which makes
the rainfed F1 more reliable. Median field size is 2.5 ha in both classes, three times the Moroccan median
(0.83 ha). By state: 8,234 + 1,057 fields in Utah, 1,707 + 849 in Idaho (Bear River basin), the rest in Colorado
and Wyoming (irrigated + rainfed). 1,171 of the 2,028 rainfed fields (58%) are in the Bear River basin.

<figure markdown>
  ![Sampled cells](figures/utah/sampling_map.png){ width="520" }
  <figcaption>Cells of 1.28 km with crop fields in WRLU 2024 or 2021 (grey) and the sampled cells: with a rainfed field in either year (red), irrigated only (blue). The same cells are used for both years.</figcaption>
</figure>

<figure markdown>
  ![Reference fields](figures/utah/parcels_map.png){ width="520" }
  <figcaption>Sampled reference fields, WRLU 2024 and 2021.</figcaption>
</figure>

<figure markdown>
  ![Crops per class](figures/utah/crops_by_class.png){ width="760" }
  <figcaption>Main crops per class in the sample. Rainfed fields are mostly dryland alfalfa, winter wheat and grass hay.</figcaption>
</figure>

<figure markdown>
  ![Class balance and size](figures/utah/class_balance_and_size.png){ width="760" }
</figure>

**Larger samples** are a change of `N_RAINFED_CELLS` / `N_IRRIGATED_CELLS` and a re-run of 01–03:

| Option | Cells | Fields per year (approx.) | Cost relative to now |
| --- | --- | --- | --- |
| now | 400 + 150 | ~6,000 | 1× |
| double | 800 + 300 | ~12,000 | ~2× |
| all cells with rainfed + matching irrigated-only | ~4,800 + 1,500 | ~60,000 | ~10× |
| all fields (`SAMPLE = False`) | ~12,400 | ~122,000 | ~20× |

**Outputs** (`outputs/utah/data_training_prep/01_labels/`): `parcels_labelled.geojson` (the sample), `source_counts.csv`,
`summary.csv`, `summary_by_zone.csv`, `sampled_cells.csv`, `year_change_vs_2024.csv`; before sampling:
`all_crop_fields.gpkg`, `all_fields_summary.csv`, `all_fields_by_zone.csv`, `all_fields_*.png`; figures.

---

## 8. Script 02 — spatial folds

The same script as for Morocco (`REGION = "utah"`). The class-similarity curve uses only pairs of fields of the
**same** agricultural year: the same field in two years is not two independent samples.

| Distance between fields | ≤ 0.25 km | 1 km | 3 km | 10 km | 15 km | 20 km | random placement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pairs with the same class | 92.1% | 89.2% | 80.2% | 80.2% | 70.5% | 70.7% | 71.9% |

Class similarity reaches the random-placement level at about 15 km, so the **20 km block** of Morocco is kept.

| Split (average over the 5 rounds) | Rainfed in smallest fold | Test with train ≤ 250 m | Test with train ≤ 1 km |
| --- | --- | --- | --- |
| random | 405 | 94.3% | 99.4% |
| blocks 5 km | 405 | 6.5% | 27.7% |
| blocks 20 km | 405 | 2.2% | 9.8% |
| blocks 50 km | 336 | 0.3% | 2.1% |

**Folds:** 183 blocks, about 2,000 irrigated and 405 rainfed fields per fold. A field belongs to the block of its
centroid, so **the same place falls in the same block, and therefore in the same fold, in every year**: a model is
never trained on a place it is tested on, in either year. The greedy assignment places the blocks with the most
rainfed fields first; the many small blocks with few fields come last, so one fold (here fold 0, 59 blocks)
collects more, smaller blocks than the others (30–33). Morocco shows the same pattern (fold 2: 42 of 91 blocks).

<figure markdown>
  ![Class similarity](figures/utah/class_similarity_by_distance.png){ width="620" }
</figure>

<figure markdown>
  ![Spatial folds](figures/utah/spatial_folds.png){ width="820" }
  <figcaption>Fields coloured by fold (left) and class counts per fold (right).</figcaption>
</figure>

<figure markdown>
  ![Spatial blocks and folds](figures/utah/spatial_folds_blocks.png){ width="820" }
  <figcaption>The 20 km blocks coloured by fold, with enlargements of the two densest areas: A on the Utah–Idaho border (Cache Valley, Bear River), B in central Utah. Whole blocks go to one fold.</figcaption>
</figure>

---

## 9. Script 03 — tile store

The same scripts as for Morocco (`data_training_prep/03_plan_tiles.py`, the `data_download/` blocks,
`data_training_prep/04_cut_tiles.py`; `REGION = "utah"`). Grid, bands, cloud mask and file formats are unchanged
([Morocco, section 6](morocco.md#6-script-03--build_tiles)). Two settings come from `regions.py`: the projection
(UTM 12N, EPSG:32612) and the first month (October, so `m01` = October).

**One tile per place and year.** The tile name starts with the agricultural year, so the same place gets one tile
per year, on the same pixel grid:

| | Training year (WRLU 2024) | Test year (WRLU 2021) |
| --- | --- | --- |
| tile | `2023-24_184_3383` | `2020-21_184_3383` |
| image `_img.tif` | Sentinel-1/2, Oct 2023 – Sep 2024 | Sentinel-1/2, Oct 2020 – Sep 2021 |
| label `_lab.tif` | the 2024 fields and labels | the 2021 fields and labels |
| field in `parcels.csv` | own `parcel_id` / `parcel_idx` | own `parcel_id` / `parcel_idx` |
| fold | e.g. fold 3 | the same fold 3 |

The tiles store Sentinel-1 and Sentinel-2 bands, but every model uses the Sentinel-2 bands only
([D14](../decisions.md#d14-sentinel-2-only-for-every-input)).

`parcels.csv` holds both years (6,013 fields of 2023/24 and 6,002 of 2020/21); their numbers never change when
another year is downloaded.

**Tile plan** (`DOWNLOAD = False`): 12,015 fields; 6 smaller than one pixel centre; 280 with no pixel left after the
10 m inner buffer; no overlapping fields. 5,436 tiles (2,720 for 2023/24, 2,716 for 2020/21; 1,042 home tiles),
about 10 GB of images per year. 120 tiles hold fields of two folds: a 20 km block is 15.6 cells wide, so some cells
straddle a block border (pixel samples take the fold of their parcel; tile models need the loss mask, as in
Morocco).

---

## 10. Temporal test design

Because the training year and the test year use the same cells, the test fold holds the **same fields** in both
years. Per round of the 5-fold rotation:

| Score | Train | Test | Measures |
| --- | --- | --- | --- |
| **A · same year** | 2024, training folds | 2024, test fold | spatial accuracy, as in Morocco |
| **B · new year, new places** | 2024, training folds | **2021, test fold** | **temporal transferability** (main result) |
| C · new year, seen places | 2024, training folds | 2021, training-fold fields | change of imagery only (upper bound: places seen in training) |

- **B − A** is the cost of the change of year, measured on identical fields (a paired comparison, so the block
  bootstrap gives a narrow interval).
- Within B, the fields whose crop changed between the two years show where a new year matters most.
- The test asks whether a model recognises irrigation in a **different year with other crops and weather**. It
  cannot test the detection of changes in irrigation (83 class changes in the whole of WRLU).
- The experiments read their years from `SEASONS` in `pixelwise_run/pixel_common.py` (Utah: 2023/24 only); the
  2021 test needs the 2020/21 tile images and pixel samples first.

---

## 11. Differences from Morocco

| | Morocco | Utah |
| --- | --- | --- |
| Labels | field survey, Dec 2023 – Jan 2025 | agency land-use map (WRLU), mostly image interpretation; 1,987 fields field-checked in 2025 |
| Rainfed definition | "rainfed" in the survey | `IRR_Method = "Dry Crop"` |
| Crops | seasonal crops and trees | grain, hay and field crops (no orchards) |
| Selection | all surveyed parcels | 550 sampled 1.28 km cells, all fields inside |
| Years | 2023/24 and 2024/25, pooled | 2023/24 (training) and 2020/21 (temporal test), kept apart |
| Agricultural year | 1 Sep – 31 Aug | 1 Oct – 30 Sep (water year) |
| Projection | UTM 29N | UTM 12N |
| `zone` | irrigation district | river basin |
| Median field size | 0.83 ha | 2.5 ha |
| Rainfed share | 13.6% | 16.9% |

---

## 12. Running, one year at a time

```bash
conda activate condaml
python data_training_prep/01_prepare_labels_Utah.py      # ~1 min; YEARS = [2024, 2021]
# in every further script set REGION = "utah" at the top, then:
python data_training_prep/02_make_spatial_folds.py       # seconds
python data_training_prep/03_plan_tiles.py               # tile plan + label rasters
python data_download/01_plan_blocks.py              # every block of the mapping area (no mask, no labels needed)
python data_download/02a_dynamic_world.py           # binary Dynamic World cropland of every block (resumable)
python data_download/02b_s2.py                      # cropland / training blocks; SEASONS per region in the script; first N_TEST_BLOCKS = 1; resumable
python data_download/02c_s1.py
python data_download/03_build_vrt.py
python data_training_prep/04_cut_tiles.py                # training tiles cut from the blocks
python pixelwise_run/04_make_pixel_samples.py       # SEASONS per region in the script -> 04_pixel_samples/<season>/
```

Run everything in the `condaml` environment: another GDAL / GEOS version once gave one field more.

| Step | Runs for | Why |
| --- | --- | --- |
| 01 labels, 02 folds | **all years in `YEARS`** at once (local, ~1 min) | sampled cells, parcel numbers and folds are fixed once, the same for every year |
| 03 plan + label rasters | all years (local, seconds) | the parcel numbers in the label rasters never change |
| block download (`data_download/02a`–`02c`), cut-out (`data_training_prep/04`) | **only the years in `SEASONS`** of the download scripts | the training year 2023/24 and the test year 2020/21 can be run one at a time; finished blocks and existing tiles are skipped |
| 04 pixel samples | **only the years in `SEASONS`** (each needs its tiles) | each year in its own folder: `04_pixel_samples/2023-24/`, `04_pixel_samples/2020-21/` |

!!! note "Archived: parcel-wise set-up"
    The archived parcel-wise pipeline ran `parcelwise_run/04_make_views.py` (views A–C, `views_2023-24/`) and
    `parcelwise_run/05_inspect_dataset.py` at this step; see the [Archive](../archive/index.md).

Running 01 with one year only and adding the other year later would redraw the cells and rebalance the folds, and
the two years would no longer be paired.

Region settings (`data_download/regions.py`): `crs_epsg = 32612`, `start_month = 10`, `block_km = 20`, map
boundaries. Utah settings in `01_prepare_labels_Utah.py`: `YEARS`, `CROP_GROUPS`, `EXCLUDE_CROPS`,
`IRRIGATED_METHODS`, `RAINFED_METHODS`, `STATES`, `EXPORT_ALL_FIELDS`, `SAMPLE`, `CELL_M`, `N_RAINFED_CELLS`,
`N_IRRIGATED_CELLS`, `MAX_CELLS_PER_BLOCK`, `SEED`.

---

## 13. Known limitations

- **Labels are not a field survey.** WRLU is mapped mainly from imagery with field checks on part of the area, and
  lists the Cropland Data Layer among its inputs. Its errors are not known per field.
- **Labels barely change between years** ([section 5](#5-which-years-and-how-the-labels-change-between-years)): the
  temporal test measures robustness to a new year, not change detection, and it must hold out space (score B).
- **Unequal inclusion probabilities.** Rainfed areas are favoured, and cells in crowded blocks are less likely to be
  sampled. This is fine for comparing models (all see the same fields), but the scores are not unbiased accuracy
  estimates for all Utah fields; that would need sampling weights. The same holds for Morocco, whose survey is not
  a probability sample either.
- **Dryland alfalfa** is the largest rainfed crop (1,122 of 2,028). It is perennial and green in spring, unlike the
  Moroccan rainfed cereals. `EXCLUDE_CROPS = ["Alfalfa"]` gives a sensitivity run.
- **Class and place:** 58% of the rainfed fields lie in the Bear River basin (northern Utah and south-east Idaho),
  so results are also reported per basin, as per zone in Morocco.
- **Snow:** December–February composites in northern Utah often show snow (the SCL cloud mask keeps the snow class 11); both
  classes are affected alike.
- **Sentinel-1:** 2021 has both Sentinel-1 satellites, 2024 only Sentinel-1A, so about twice as many Sentinel-1
  scenes per month in the test year.
- **No cropland mask in the labels:** every WRLU field is cropland, so no mask check is made; the annual Dynamic World
  cropland serves the download and the mapping phase only ([Cropland extent](cropland_extent.md)).
