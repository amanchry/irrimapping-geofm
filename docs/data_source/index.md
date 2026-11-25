# Data sources

Catalogue of every dataset used in this research or planned for later phases: provider, content, access and status. How each dataset is processed into training data is described per study area: [Morocco training data](../phase1_data/morocco.md) and [Utah training data](../phase1_data/utah.md).

**Status:** ✅ used · 🟦 used in a side check · 📝 planned · 🔍 candidate (under review)

## Ground truth

| Dataset | Provider | Content | Period | Use here | Access | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **Crop and irrigation ground-truth dataset for Moroccan agricultural regions** | Ouassanouan et al. (2026), *Scientific Data* | ~10,000 geolocated parcels in El Gharb, Tadla, Doukkala, El Haouz, Souss; 45 crop types, 6 irrigation systems, a photo per parcel | Dec 2023 – Jan 2025 | labels of the first study area ([Morocco training data, §3](../phase1_data/morocco.md#3-source-data)) | open, citation required · [paper](https://doi.org/10.1038/s41597-026-06993-y) | ✅ |
| **Water-Related Land Use (WRLU)** (Utah, USA) | Utah Division of Water Resources | field polygons; `IRR_Method` incl. *Dry Crop* (rainfed) | 2017 – 2025 (yearly) | **second study area** and temporal test (labels of the [Utah training data](../phase1_data/utah.md): WRLU 2024 and 2021); `data/Utah_field_data/` | open · [WRLU data portal](https://dwre-utahdnr.opendata.arcgis.com/pages/wrlu-data) | ✅ |
| **GMIA-NEXT** reference points ([summary](../references/kebede-2026-gmia-next.md)) | Kebede et al. (2026) | 380,696 global irrigated / non-irrigated point labels (paper: 230,683 + 153,194) | 2023/24 | independent test points (Morocco check; Tunisia: 54 points, 44 irrigated, 10 rainfed) | open · [maps](https://doi.org/10.5281/zenodo.17627111) · [points](https://zenodo.org/records/20732681) | 🟦 |
| **WSDA Agricultural Land Use** (Washington, USA) | Washington State Department of Agriculture | field polygons with crop and irrigation method; `Irrigation = None` for non-irrigated fields | 2025 (yearly) | not used: candidate region of the earlier transfer design; Utah was chosen as second study area ([D13](../decisions.md#d13-utah-as-a-second-independent-study-area)) | open · [WSDA](https://agr.wa.gov/departments/land-and-water/natural-resources/agricultural-land-use) | – |
| Verified irrigated agricultural lands, 2002–2017 | USGS | irrigated fields only, 14 US states | 2002 – 2017 | not usable (no rainfed class) | open · [ScienceBase](https://www.sciencebase.gov/catalog/item/5ece996d82ce30fd9808527b) | – |
| CRDSS irrigated parcels, Division 5 (Colorado, USA) | Colorado Division of Water Resources | irrigated parcels only | 2000 – 2023 | not usable (no rainfed class) | open · [CDSS GIS](https://cdss.colorado.gov/gis-data/gis-data-by-category) | – |

## Satellite imagery

Since 24 Nov 2026 Sentinel-2 and Sentinel-1 are read from open STAC catalogues; Google Earth Engine is used only
for Dynamic World ([D21](../decisions.md#d21-sentinel-2-and-sentinel-1-from-open-stac-catalogues-earth-engine-only-for-dynamic-world)).

| Collection | Provider | Bands / variables | Resolution | Use here | Catalogue / collection ID | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **Sentinel-2 Level-2A** (surface reflectance, ESA Collection 1) | ESA Copernicus; Earth Search STAC (Element 84, AWS), no account | B1–B12 (12 bands) and the scene classification (SCL) | 10–60 m, read onto the 10 m block grid (nearest neighbour); 5-day revisit | monthly composites per 10.24 km block (`data_download/02b_s2.py`): scenes with `eo:cloud_cover` < 60, offset of Collection 1 removed (reflectance × 10,000), cloud mask from SCL (classes 0–3 and 8–10 removed); input of every model and of the maps | `sentinel-2-c1-l2a` (Earth Search) | ✅ |
| **Cloud Score+** for Sentinel-2 | Google (Pasquarella et al., 2023) | `cs_cdf` (clear-sky probability) | 10 m | no longer used: cloud mask of the Earth Engine Sentinel-2 composites until 24 Nov 2026, replaced by the scene classification (SCL) of the STAC download ([D21](../decisions.md#d21-sentinel-2-and-sentinel-1-from-open-stac-catalogues-earth-engine-only-for-dynamic-world)) | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` (Earth Engine) | – |
| **Sentinel-1 RTC** (GRD, radiometrically terrain corrected) | ESA Copernicus; Microsoft Planetary Computer (terrain correction by Catalyst), signed anonymous URLs | VV, VH backscatter as linear gamma0, ascending and descending | 10 m; 6–12-day revisit | monthly composites per block (`data_download/02c_s1.py`: IW scenes with VV and VH, mean in linear power, stored as dB × 100) and in the training tiles; not used by the models for now ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)) | `sentinel-1-rtc` (Planetary Computer) | 🟦 |

Access: open (Copernicus data policy); Earth Search needs no account and Planetary Computer gives signed anonymous
URLs; an Earth Engine service account is needed only for Dynamic World. Every source is downloaded once per
region and agricultural year on one grid of 10.24 km blocks (`data_download/`); the training tiles are cut from
these blocks and the maps are predicted from them ([D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)). Compositing, cloud
threshold and tiles: [Morocco training data, §6](../phase1_data/morocco.md#6-script-03--build_tiles); extent of the
download: [Cropland extent §2](../phase1_data/cropland_extent.md#2-blocks-and-downloads).

!!! note "Retired: Sentinel-2 from STAC"
    Wall-to-wall Sentinel-2 composites were at first downloaded from the Earth Search STAC catalogue (Element 84,
    collection `sentinel-2-c1-l2a`, SCL cloud mask; [D18](../decisions.md#d18-wall-to-wall-composites-from-stac-for-the-maps)).
    This per-tile route (`s2_download/`) was retired in favour of the blocks of `data_download/`
    ([D20](../decisions.md#d20-download-once-on-1024-km-blocks-from-earth-engine-training-tiles-cut-from-them)),
    which read Sentinel-2 from the same Earth Search collection since 24 Nov 2026 ([D21](../decisions.md#d21-sentinel-2-and-sentinel-1-from-open-stac-catalogues-earth-engine-only-for-dynamic-world)).

## Auxiliary and reference maps

The cropland layer serves the download and the mapping phase only: the experiments classify the surveyed parcels
as irrigated or rainfed and use no cropland mask. The maps use the annual Dynamic World cropland; ESA WorldCover,
GLAD, Esri LULC and GCEP30 were the four products of the archived union cropland mask
([Cropland extent](../phase1_data/cropland_extent.md)).

| Dataset | Provider | Content | Resolution | Use here | Access | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **FAO GAUL** (2015 level 0 Morocco; 2024 level 1 Utah) | FAO | country and state boundaries | vector | outlines on the maps | open · [FAO GAUL](https://data.apps.fao.org/catalog/dataset/global-administrative-unit-layers-gaul) | ✅ |
| **ESA WorldCover 2021 v200** | Zanaga et al. (2022), ESA | 11 land-cover classes (cropland = 40) | 10 m | reference map (`data_download/04a_ref_esa_worldcover.py`, [Cropland extent §5](../phase1_data/cropland_extent.md#5-reference-layers)); earlier one of the 4 products of the [archived union cropland mask](../archive/cropland_union_mask.md) | CC-BY 4.0 · [Zenodo](https://doi.org/10.5281/zenodo.7254221) | ✅ |
| **LGRIP30** | Teluguntla et al. (2023), USGS / NASA LP DAAC | irrigated and rainfed cropland | 30 m, nominal 2015 | reference map (`data_download/04b_ref_lgrip30.py`); comparison of the final maps | open · [LP DAAC](https://doi.org/10.5067/COMMUNITY/LGRIP/LGRIP30.001) | 🟦 |
| **GMIE** | Wu et al. (2023) | global maximum irrigation extent | 30 m | comparison of the final maps | open, citation required · [paper](https://doi.org/10.1016/j.gloenvcha.2023.102652) | 📝 |
| **WorldCereal** cropland / irrigation | Van Tricht et al. (2023) | cropland and irrigation products | 10 m | comparison; cropland mask for mapping | open · [ESSD paper](https://doi.org/10.5194/essd-15-5491-2023) | 📝 |
| **GLAD global cropland** | Potapov et al. (2022) | cropland extent and change (2003–2019) | 30 m | used only by the [archived union cropland mask](../archive/cropland_union_mask.md); not used now | open · [paper](https://doi.org/10.1038/s43016-021-00429-z) · [GLAD](https://glad.umd.edu/dataset/croplands) | – |
| **Esri / Impact Observatory LULC** | Karra et al. (2021) | annual land use / land cover (crops = class 5), 2017–2025 | 10 m | used only by the [archived union cropland mask](../archive/cropland_union_mask.md); not used now | open · [GEE community catalogue](https://gee-community-catalog.org/projects/S2TSLULC/) | – |
| **GCEP30** (GFSAD) | Thenkabail et al. (2021), USGS | global cropland extent, nominal 2015 | 30 m | used only by the [archived union cropland mask](../archive/cropland_union_mask.md); not used now | open · [USGS PP 1868](https://doi.org/10.3133/pp1868) · [GEE community catalogue](https://gee-community-catalog.org/projects/gcep30/) | – |
| **Dynamic World V1** | Brown et al. (2022) | 9 land-cover classes with per-class probabilities, per Sentinel-2 image | 10 m | cropland layer that selects the downloads and the map pixels: for every 10.24 km block of the mapping area, a binary cropland image of the whole region per agricultural year (crops the most frequent class in at least 2 of its 12 months) (`data_download/02a_dynamic_world.py`); annual layer per calendar year (most frequent class, mean crop probability, image count; `04c_ref_dynamic_world_annual.py`); the only cropland layer of the maps (the union of 4 products is archived) | open · [GEE catalogue](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) | ✅ |
| **FAO WaPOR** | FAO | evapotranspiration, water productivity | 20–250 m | interpretation of irrigation patterns; optional extra variables | open (registration) · [WaPOR](https://www.fao.org/in-action/remote-sensing-for-water-productivity/en) | 📝 |
| **FAO AQUASTAT** | FAO | national irrigated-area statistics | country | check of the annual map series (RQ3) | open · [AQUASTAT](https://www.fao.org/aquastat/en/) | 📝 |
| **FAO AQUASTAT irrigated crop calendars** | FAO (2023) | months of growth per irrigated crop and country (about 60 crops, 165 countries; one Excel file) | country (USA in four regions) | check of the agricultural year ([Morocco](../phase1_data/morocco.md), [Utah](../phase1_data/utah.md)) | open · [AQUASTAT crop calendars](https://www.fao.org/aquastat/en/databases/crop-calendar) | ✅ |
| **GGCMI Phase 3 crop calendar** | Jägermeyr et al. (2021) | planting day, maturity day and season length of 18 crops, irrigated and rainfed, with the source used per cell | 0.5° | check of the agricultural year ([Morocco](../phase1_data/morocco.md), [Utah](../phase1_data/utah.md)) | open, CC BY 4.0 · [Zenodo](https://doi.org/10.5281/zenodo.5062513) | ✅ |

## Pretrained models

Six foundation models with open weights: Prithvi-EO-2.0-300M-TL, Clay v1.5, TerraMind-1.0-base, AgriFM, Presto
and TESSERA v2 (Medium). Developers, papers and download
locations are listed in [Foundation models](../foundation_models/index.md).
