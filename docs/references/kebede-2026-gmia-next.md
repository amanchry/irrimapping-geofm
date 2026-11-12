# GMIA-NEXT · Kebede et al. (2026)

!!! quote "Citation"
    Kebede, E. A., et al. [and Davis, K. F.] (2026). GMIA-NEXT: next-generation global map of irrigated areas.
    *Research Square* (preprint, Data Note). [doi:10.21203/rs.3.rs-10085674/v1](https://doi.org/10.21203/rs.3.rs-10085674/v1) ·
    [PDF](https://assets-eu.researchsquare.com/files/rs-10085674/v1_covered_aea166e8-1c19-4364-a3a3-65009e657220.pdf?c=1782125281) ·
    maps: [Zenodo, doi:10.5281/zenodo.17627111](https://zenodo.org/records/17627111) ·
    reference points: [Zenodo, doi:10.5281/zenodo.20732681](https://zenodo.org/records/20732681) ·
    code: [GitHub](https://github.com/EndiKebede/GMIA-NEXT-Next-Generation-Global-Map-of-Irrigated-Areas-Using-Data-Fusion)

**In one line:** a 30 m global map of irrigated cropland for the 2023/24 growing season, produced with random
forests from Landsat and environmental predictors, together with an openly released global set of irrigated and
non-irrigated reference points.

## What they did

- **Task:** binary classification (irrigated / non-irrigated) of every 30 m pixel inside a cropland mask (maximum
  GLAD cropland extent 2020–2024, to include fallow years).
- **Season:** the major irrigation season of each agro-ecological zone (AEZ), from the FAO irrigated crop calendar
  and the GGCMI Phase III calendars.
- **Predictors (21):** seasonal maximum, minimum and mean of EVI, GI, NDVI and NDWI from Landsat 8/9 (Collection 2
  Level-2, QA cloud mask); elevation and slope (SRTM); MODIS evapotranspiration, potential evapotranspiration and land
  surface temperature; SMAP soil moisture; CHIRTS temperature; ERA5-Land vapour pressure deficit; distance to rivers
  (HydroRIVERS) and to irrigation canals.
- **Reference points:** 24,777 newly curated points (8,672 irrigated, 16,105 non-irrigated), sampled at random within
  cropland stratified by AEZ and labelled by image interpretation (dry-season greenness relative to neighbouring
  fields, seasonal NDVI, proximity to water, centre-pivot geometry); supplemented by national and regional datasets
  (LANID, USGS Upper Colorado, LUCAS, MapBiomas, GMIE centre pivots, studies in Zambia, Nigeria and India). In total
  230,683 irrigated and 153,194 non-irrigated points; labels from other years were harmonised to 2023/24 with NDVI.
- **Models:** random forest per AEZ tile (5 × 2 tiles per continental AEZ, pooled with neighbours where samples are
  few) or per continent (South America, Africa, Oceania); fixed 50% and tile-optimised probability thresholds;
  3 × 3 majority filter.
- **Validation:** 60/20/20 split into training, evaluation and test points, with points grouped into 45 m clusters;
  comparison with MIRCA-OS, AEI and regional products (ECIRA, LANID, CIrrMap250, GMIE, CLUM and others).

## Key results

| Region | Accuracy (%) | Precision (%) | Recall (%) |
| --- | ---: | ---: | ---: |
| Global | 80.5 ± 2.1 | 72.5 ± 3.2 | 74.9 ± 3.3 |
| Africa | 78.7 ± 6.0 | 54.9 ± 7.6 | 74.1 ± 8.4 |
| Europe | 89.8 ± 0.2 | 94.2 ± 0.2 | 83.4 ± 0.4 |
| North America | 92.7 ± 4.5 | 82.2 ± 1.0 | 97.0 ± 1.2 |

- Low precision in Africa and South America: non-irrigated cropland is often mapped as irrigated.
- At 5 arc-minutes the map correlates with MIRCA-OS, AEI and Meier et al. (r = 0.74–0.76) and generally shows more
  irrigated area. Morocco is not discussed; in North Africa only Egypt is compared (with AIM).

## Data records

| Record | Content |
| --- | --- |
| Maps ([17627111](https://zenodo.org/records/17627111)) | irrigation probability (0–100%) and binary maps (fixed and dynamic threshold), 30 m, WGS84, one GeoTIFF per continent; 15.4 GB; CC BY 4.0 |
| Reference points ([20732681](https://zenodo.org/records/20732681)) | `GMIA-NEXT_GTPs_2024_v0_1` shapefile with the label (`irrigated`: 1 / 0) and the source of each point; CC BY 4.0 |
| Code | Earth Engine scripts (AEZ-tile models) and Python (continental models) |

## What this research takes from it

- **Independent test points** for the transfer test (RQ4) and for the accuracy assessment of the annual maps
  ([Phase 5](../phase5_maps/index.md)); never used for training. Tunisia has 54 points (44 irrigated, 10 rainfed)
  ([D10](../decisions.md#d10-transfer-region-under-review)); `tools/a1_gmia_next_points.py` checks the points of a
  country and compares the Moroccan points with the survey parcels.
- **Comparison product** for the maps of Phase 5, alongside LGRIP30, GMIE and WorldCereal.
- **Season definition:** irrigation is defined for one growing season, consistent with the agricultural-year design
  of this research ([dataset §6.2](../phase1_data/morocco.md#62-two-agricultural-years)).
- **Benchmark level:** about 80% accuracy globally and 79% in Africa at 30 m with hand-crafted features; a useful
  reference for the accuracy that this research reaches at 10 m.

## Differences and limitations

- **Resolution and sensor:** 30 m Landsat; the authors note spectral mixing for smallholder fields and point to
  Sentinel or Planet data for such landscapes. This research uses 10 m Sentinel-1/2.
- **Labels by image interpretation,** not field visits, for the curated points; harmonisation of other years with an
  NDVI tolerance adds label noise.
- **"Non-irrigated" is not strictly "rainfed cropland":** the label depends on the cropland mask; the check with
  ESA WorldCover in `tools/a1_gmia_next_points.py` shows whether a point lies on cropland.
- **Validation design:** 45 m clustering is much weaker than the 20 km spatial blocks used here, so the reported
  accuracies are probably optimistic for unseen areas.
- **Single major season** per AEZ: supplementary irrigation in the wet season may be missed.
- **Preprint** (v1, June 2026), not yet peer-reviewed.
