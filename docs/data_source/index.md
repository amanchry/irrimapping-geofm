# Data sources

Catalogue of every dataset used in this research or planned for later phases: provider, content, access and status. How each dataset is processed into training data is described per study area: [Morocco training data](../phase1_data/morocco.md) and [Utah training data](../phase1_data/utah.md).

**Status:** ✅ used · 🟦 used in a side check · 📝 planned · 🔍 candidate (under review)

## Ground truth

| Dataset | Provider | Content | Period | Use here | Access | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **Crop and irrigation ground-truth dataset for Moroccan agricultural regions** | Ouassanouan et al. (2026), *Scientific Data* | ~10,000 geolocated parcels in El Gharb, Tadla, Doukkala, El Haouz, Souss; 45 crop types, 6 irrigation systems, a photo per parcel | Dec 2023 – Jan 2025 | labels of the first study area ([Morocco training data, §3](../phase1_data/morocco.md#3-source-data)) | open, citation required · [paper](https://doi.org/10.1038/s41597-026-06993-y) | ✅ |
| **GMIA-NEXT** reference points ([summary](../references/kebede-2026-gmia-next.md)) | Kebede et al. (2026) | 380,696 global irrigated / non-irrigated point labels (paper: 230,683 + 153,194) | 2023/24 | independent test points (Morocco check; Tunisia: 54 points, 44 irrigated, 10 rainfed) | open · [maps](https://doi.org/10.5281/zenodo.17627111) · [points](https://zenodo.org/records/20732681) | 🟦 |
| **WSDA Agricultural Land Use** (Washington, USA) | Washington State Department of Agriculture | field polygons with crop and irrigation method; `Irrigation = None` for non-irrigated fields | 2025 (yearly) | candidate transfer region (eastern Washington) | open · [WSDA](https://agr.wa.gov/departments/land-and-water/natural-resources/agricultural-land-use) | 🔍 |
| **Water-Related Land Use** (Utah, USA) | Utah Division of Water Resources | field polygons; `IRR_Method` incl. *Dry Crop* (rainfed) | 2017 – 2025 (yearly) | **second study area** and temporal test (labels of the [Utah training data](../phase1_data/utah.md): WRLU 2024 and 2021); `data/Utah_field_data/` | open · [WRLU](https://dwre-utahdnr.opendata.arcgis.com/pages/wrlu-data) | ✅ |
| Verified irrigated agricultural lands, 2002–2017 | USGS | irrigated fields only, 14 US states | 2002 – 2017 | not usable (no rainfed class) | open · [ScienceBase](https://www.sciencebase.gov/catalog/item/5ece996d82ce30fd9808527b) | – |
| CRDSS irrigated parcels, Division 5 (Colorado, USA) | Colorado Division of Water Resources | irrigated parcels only | 2000 – 2023 | not usable (no rainfed class) | open · [CDSS GIS](https://cdss.colorado.gov/gis-data/gis-data-by-category) | – |

## Satellite imagery (Google Earth Engine)

| Collection | Provider | Bands / variables | Resolution | Use here | Earth Engine ID | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **Sentinel-2 Level-2A** (surface reflectance, harmonised) | ESA Copernicus | B1–B12 (12 bands) | 10–60 m, resampled to 10 m; 5-day revisit | monthly composites: input of every model | `COPERNICUS/S2_SR_HARMONIZED` | ✅ |
| **Cloud Score+** for Sentinel-2 | Google (Pasquarella et al., 2023) | `cs_cdf` (clear-sky probability) | 10 m | cloud mask of the Sentinel-2 composites | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | ✅ |
| **Sentinel-1 GRD** | ESA Copernicus | VV, VH backscatter, ascending and descending | 10 m; 6–12-day revisit | monthly composites in the tile store; not used by the models for now ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)) | `COPERNICUS/S1_GRD` | 🟦 |

Access: open (Copernicus data policy); Earth Engine account or service account needed. Compositing, cloud
threshold and tiles: [Morocco training data, §6](../phase1_data/morocco.md#6-script-03--build_tiles).

## Auxiliary and reference maps

| Dataset | Provider | Content | Resolution | Use here | Access | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **FAO GAUL** (2015 level 0 Morocco; 2024 level 1 Utah) | FAO | country and state boundaries | vector | outlines on the maps | open · [FAO GAUL](https://data.apps.fao.org/catalog/dataset/global-administrative-unit-layers-gaul) | ✅ |
| **ESA WorldCover 2021 v200** | Zanaga et al. (2022), ESA | 11 land-cover classes (cropland = 40) | 10 m | cropland check of the survey parcels; reference map (`tools/a7_download_esa_worldcover.py`) | CC-BY 4.0 · [Zenodo](https://doi.org/10.5281/zenodo.7254221) | ✅ |
| **LGRIP30** | Teluguntla et al. (2023), USGS / NASA LP DAAC | irrigated and rainfed cropland | 30 m, nominal 2015 | reference map (`tools/a8_download_lgrip30.py`); comparison of the final maps | open · [LP DAAC](https://doi.org/10.5067/COMMUNITY/LGRIP/LGRIP30.001) | 🟦 |
| **GMIE** | Wu et al. (2023) | global maximum irrigation extent | 30 m | comparison of the final maps | open, citation required · [paper](https://doi.org/10.1016/j.gloenvcha.2023.102652) | 📝 |
| **WorldCereal** cropland / irrigation | Van Tricht et al. (2023) | cropland and irrigation products | 10 m | comparison; cropland mask for mapping | open · [ESSD paper](https://doi.org/10.5194/essd-15-5491-2023) | 📝 |
| **GLAD global cropland** | Potapov et al. (2021) | cropland extent and change | 30 m | cropland mask for mapping | open · [paper](https://doi.org/10.1038/s43016-021-00429-z) | 📝 |
| **FAO WaPOR** | FAO | evapotranspiration, water productivity | 20–250 m | interpretation of irrigation patterns; optional extra variables | open (registration) · [WaPOR](https://www.fao.org/in-action/remote-sensing-for-water-productivity/en) | 📝 |
| **FAO AQUASTAT** | FAO | national irrigated-area statistics | country | check of the annual map series (RQ3) | open · [AQUASTAT](https://www.fao.org/aquastat/en/) | 📝 |

## Pretrained models

Clay v1.5, Prithvi-EO-2.0-300M-TL, TerraMind-1.0-base and AgriFM, with open weights. Developers, papers and download
locations are listed in [Foundation models](../foundation_models.md).
