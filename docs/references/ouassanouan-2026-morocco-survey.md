# Moroccan crop and irrigation survey · Ouassanouan et al. (2026)

!!! quote "Citation"
    Ouassanouan, Y., Elfarkh, J., Grich, S., Liblab, A., & Chehbouni, A. (2026). Crop and irrigation types
    ground-truth dataset for Moroccan agricultural regions. *Scientific Data*, 13, 746.
    [doi:10.1038/s41597-026-06993-y](https://doi.org/10.1038/s41597-026-06993-y) ·
    data: [figshare, doi:10.6084/m9.figshare.28486022.v2](https://doi.org/10.6084/m9.figshare.28486022.v2)

**In one line:** the first openly available, systematically collected parcel-level ground-truth dataset of crop
types and irrigation systems for North Africa, and the source of all labels in this research.

## What they did

- **Regions:** five major agricultural regions of Morocco with contrasting climates and water sources:
  El Gharb (humid to sub-humid; rice, vegetables; surface water), Tadla (semi-arid to sub-humid; large-scale
  irrigation; sugar beet, cereals), Doukkala (Mediterranean, ~400 mm; mixed irrigated and rainfed systems),
  El Haouz (semi-arid, ~250 mm; olives, citrus; groundwater) and Souss (arid; drip-irrigated export crops).
- **Period:** December 2023 – January 2025, i.e. two agricultural seasons.
- **Protocol:** a standardised survey form built in QGIS and deployed offline in the QField mobile application,
  with Sentinel-2 false-colour composites (B11, B8, B2) as background. A team of four (the same persons throughout)
  digitised each parcel on site; two collectors recorded the parcels on either side of the route.
- **Attributes per parcel:** polygon, crop type (45 types), agricultural class (seasonal crops, trees, trees +
  seasonal crops, weeds, bare soil, water bodies), irrigation system, geotagged photograph; for trees also height,
  crown diameter and row and tree spacing; contextual notes from farmer interviews where possible.
- **Irrigation systems:** rainfed and six irrigated systems: flood (traditional surface), basin (structured
  flood), drip, sprinkler and centre pivot.

## Data records

| Record | Content |
| --- | --- |
| Shapefile | about 10,000 parcel polygons with crop type, class and irrigation system |
| Photographs | one geotagged photo per parcel |
| Sentinel-2 table | mean Sentinel-2 bands (B2–B12) and NDVI per parcel for 2024 |

## Quality control

- Daily review of the collected data by the team; debriefing after each mission.
- Post-processing: automated filtering and manual cleaning, removal of overlaps between polygons of different
  collectors, harmonisation of misspelled and overlapping class labels (e.g. "fruit trees" and "apple trees" merged
  into *Trees*), correction of clearly mislabelled parcels.
- Consistency check with 2024 NDVI time series per class: bare soil 0.1–0.3 all year; seasonal crops 0.7–0.9 in the
  growing season and 0.1–0.3 from June to October; trees 0.7–0.8, falling to 0.5–0.6 in summer; weeds 0.2–0.4 in
  February–April.

## What this research takes from it

- **Labels:** the irrigation-system attribute defines the binary target. Any irrigated system = irrigated, *rainfed*
  = rainfed; only the cropland classes (seasonal crops, trees, trees + seasonal) are kept. After cleaning,
  8,781 parcels remain (7,585 irrigated, 1,196 rainfed) ([ML-ready dataset §4](../phase1_data/morocco.md#4-script-01--prepare_labels)).
- **Agricultural year:** the survey date assigns each parcel to the 2023/24 or 2024/25 season, and the imagery is
  taken from that season ([D2](../decisions.md#d2-two-agricultural-years-per-parcel-season)).
- **Spatial validation:** parcels were recorded along survey routes, so neighbouring parcels are clustered and
  usually share a class, which motivates the 20 km block cross-validation
  ([dataset §5](../phase1_data/morocco.md#5-script-02--make_spatial_folds)).
- **Imagery:** the provided 2024 Sentinel-2 averages are not used; monthly composites are computed for each
  parcel's own agricultural year from Earth Engine ([dataset §6](../phase1_data/morocco.md#6-script-03--build_tiles)).

## Differences and limitations

- **Class imbalance and regional concentration:** rainfed parcels form 14% of the cleaned cropland sample, and
  1,122 of the 1,196 rainfed parcels lie in Doukkala; rainfed accuracy therefore largely reflects one region.
- **Region and season are confounded:** El Gharb and Tadla were surveyed only in 2023/24 and Souss only in 2024/25.
- **Route-based sampling:** parcels close to the survey routes are over-represented; the sample is not a
  probability sample of Moroccan cropland, so it cannot be used directly for unbiased area estimation (Phase 5).
- **Single-season labels:** irrigation status is observed once and may change between years with water
  availability.
- **Validation of the labels** is qualitative (photographs, NDVI consistency); no independent accuracy figure for
  the labels is reported. Remaining polygon overlaps were found in this research (six parcels fully covered by
  others; [dataset §6.4](../phase1_data/morocco.md#64-label-rasters-local-no-earth-engine)).
