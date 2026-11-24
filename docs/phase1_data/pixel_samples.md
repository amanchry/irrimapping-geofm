# Pixel samples for pixel-wise classification

The continuous maps classify every 10 m pixel on its own, so the models are trained and tested on **pixels**, not
on parcels ([D16](../decisions.md#d16-pixel-wise-classification-for-continuous-maps)).
`pixelwise_run/04_make_pixel_samples.py` turns the labelled parcels into labelled pixels. It creates no new
labels: the label of a pixel is the survey label of its parcel, which `data_training_prep/03_plan_tiles.py` already wrote into the
label raster of each tile. The script picks pixels and writes **one row per pixel**: its 12-month time series, its
label, parcel, fold and position. Every pixel-wise experiment starts from this table, so all models see the same
pixels.

![From labelled parcels to labelled pixels](figures/pixel_samples_explainer.png)

*One tile in Doukkala (2023/24, 15 surveyed parcels): (1) Sentinel-2 tile with the surveyed parcels; (2) label raster of
03: one label per parcel on all its pixels, edge pixels marked and skipped; (3) inner pixels and the sampled pixels:
at most 100 per parcel, so the 6 larger parcels are cut to 100; (4) every sampled pixel takes its parcel's fold and
20 km block (all parcels of this tile lie in one block, so one fold); (5) NDVI of the sampled pixels per class;
(6) the output table, one row per pixel.*

## Classes

| Code | Class | Pixels taken from |
| --- | --- | --- |
| 1 | Irrigated | inner pixels of irrigated parcels |
| 2 | Rainfed | inner pixels of rainfed parcels |

All surveyed parcels are cropland, so the experiments compare models on the irrigated / rainfed decision alone ([D19](../decisions.md#d19-two-class-experiments-cropland-extent-in-the-mapping-phase)). No
cropland mask is used to select pixels. Pixels outside the surveyed parcels are not used: their class is unknown.
Where cropland lies, and how land that is not cropland is handled, is decided in the mapping phase
([Cropland extent and mask](cropland_mask.md), [Phase 5](../phase5_maps/index.md)).

## Sampling rules

- **Inner pixels only.** Pixels within 10 m of a parcel outline (`edge` band of the label raster) mix the field
  with its neighbours and are skipped; a parcel without any inner pixel falls back to all its pixels
  (`edge_fallback`).
- **At most 100 pixels per parcel**, drawn at random (seed 42), so a few large irrigated fields do not dominate and
  the pixels of one parcel stay a small group.
- **Folds.** Every pixel takes its parcel's fold and 20 km block of the [spatial folds](morocco.md), so a parcel is
  never split between training and test.

## Output

`outputs/<region>/data_training_prep/04_pixel_samples/<season>/`

| File | Content |
| --- | --- |
| `pixels_X.npy` | int16 [pixels, 12 months, 19 bands]: the bands of the tile store (12 Sentinel-2 bands, clear-observation count, Sentinel-1 VV / VH and scene counts per orbit), NODATA −32768; the experiments use the Sentinel-2 bands only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)) |
| `pixels.parquet` | one row per pixel, same order: `label`, `class_name`, `parcel_idx`, `parcel_id`, `fold`, `block_id`, `zone`, `tile_id`, `row`, `col`, `x`, `y` (UTM pixel centre), `lon`, `lat`, `edge`, `edge_fallback` |
| `summary.csv` | pixels and parcels per class and fold |
| `samples_info.json` | settings, band order, class codes, CRS |

The earlier three-class samples (with not-cropland pixels) are kept unchanged in `04c_pixel_samples/`; no
experiment reads them.

## Sample sets

| Region | Year | Pixels | Irrigated (parcels) | Rainfed (parcels) | Tiles |
| --- | --- | --- | --- | --- | --- |
| Morocco | 2023/24 | 367,882 | 336,211 (6,183) | 31,671 (698) | 1,490 |
| Morocco | 2024/25 | 105,793 | 83,672 (1,396) | 22,121 (498) | 525 |
| **Morocco** | **pooled** | **473,675** | **419,883 (7,579)** | **53,792 (1,196)** | **2,015** |
| Utah | 2023/24 | to be rebuilt with the two-class script on the cluster | – | – | – |

Counts from `summary.csv` of each set; tiles from `pixels.parquet`. Every pixel has a fold (0–4). The experiments
pool the two Morocco years, each pixel with its own year: 473,675 pixels of 8,775 parcels in 2,015 tiles.

### Morocco per fold

| Fold | 2023/24 irrigated | 2023/24 rainfed | 2024/25 irrigated | 2024/25 rainfed |
| --- | --- | --- | --- | --- |
| 0 | 69,104 | 4,130 | 15,559 | 6,814 |
| 1 | 62,922 | 6,154 | 16,616 | 4,666 |
| 2 | 72,720 | 10,134 | 22,557 | 1,173 |
| 3 | 65,290 | 8,444 | 20,426 | 1,986 |
| 4 | 66,175 | 2,809 | 8,514 | 7,482 |

**Limitations.**

- Rainfed is the minority class (about 1 rainfed pixel per 8 irrigated pixels when the two years are pooled); the
  models need class weights.
- The rainfed pixels are unevenly spread over the folds and years (e.g. 1,173 in fold 2 of 2024/25 but 10,134 in
  fold 2 of 2023/24), so the rainfed scores vary between test folds.
- The tiles lie around the surveyed parcels, so accuracy measured on these pixels holds for similar areas; the
  accuracy of the final map needs an independent sample over the map.

## Status

| Region | Year | Status |
| --- | --- | --- |
| Morocco | 2023/24 | ✅ done (two classes); input of E01–E05 (planned, Morocco first) |
| Morocco | 2024/25 | ✅ done (two classes); pooled with 2023/24 |
| Utah | 2023/24 | ⏳ to be rebuilt with the two-class script on the cluster; experiments follow Morocco |
| Utah | 2020/21 | prepared for the temporal test (RQ3); not yet built |
