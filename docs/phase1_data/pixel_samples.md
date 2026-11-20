# Pixel samples for pixel-wise classification

The continuous maps classify every 10 m pixel on its own, so the models are trained and tested on **pixels**, not
on parcels ([D16](../decisions.md#d16-pixel-wise-classification-for-continuous-maps)).
`pixelwise_run/04_make_pixel_samples.py` turns the labelled parcels into labelled pixels. It creates no new
labels: the label of a pixel is the survey label of its parcel, which `03_build_tiles.py` already wrote into the
label raster of each tile. The script picks pixels and writes **one row per pixel**: its 12-month time series, its
label, parcel, fold and position. Every pixel-wise experiment starts from this table, so all models see the same
pixels.

![From labelled parcels to labelled pixels](figures/pixel_samples_explainer.png)

*One tile in Doukkala (2023/24): (1) Sentinel-2 tile with the surveyed parcels; (2) label raster of 03: one label per
parcel on all its pixels, edge pixels marked; (3) votes of the cropland mask; (4) the sampled pixels; (5) NDVI of the
sampled pixels per class; (6) the output table.*

## Classes

| Code | Class | Pixels taken from |
| --- | --- | --- |
| 1 | Irrigated | inner pixels of irrigated parcels |
| 2 | Rainfed | inner pixels of rainfed parcels |
| 3 | Not cropland | pixels that **none of the 4 cropland products calls cropland** (0 votes in the [cropland mask](cropland_mask.md)) **and that are not built-up, snow/ice or permanent water** (WorldCover 2021 classes 50, 70, 80); in practice natural vegetation, scrub, grassland, bare soil or rock. Of these, at most 100 random pixels per tile are taken that lie outside every surveyed parcel and at least 2 pixels (20 m) from any parcel pixel, so that no mixed edge pixel or unsurveyed part of a field is labelled not cropland |
| – | Built-up, snow/ice, water | **not sampled**: removed from the mask by rule (`excluded` band), so they never reach the model and are not cropland on every map |

Both the class-3 pixels and the excluded pixels lie outside the cropland mask, so both are "not cropland" on the
final map by rule. Class 3 is sampled for a different reason: the model predicts only **inside** the mask, and the
union of four products also includes false cropland (scrub, bare or natural land that one product took for fields).
The class-3 pixels are examples of that kind of land, so the model learns to recognise it inside the mask. Built-up
and water never reach the model, so they are not used as examples: they would only teach an easy distinction the
model never has to make.

```
all land
├── built-up / water / snow (excluded)  → not cropland by rule; never seen by the model
├── 0 votes, not excluded               → not cropland by rule; ALSO the training examples of class 3
└── cropland mask (≥ 1 vote)            → the model decides: irrigated · rainfed · not cropland
```

Cropland pixels outside the surveyed parcels are not used: their class is unknown.

## Sampling rules

- **Inner pixels only.** Pixels within 10 m of a parcel outline (`edge` band of the label raster) mix the field
  with its neighbours and are skipped; a parcel without any inner pixel falls back to all its pixels
  (`edge_fallback`).
- **At most 100 pixels per parcel**, drawn at random (seed 42), so a few large irrigated fields do not dominate and
  the pixels of one parcel stay a small group.
- **At most 100 not-cropland pixels per tile**, drawn at random from every tile of the year.
- **Folds.** A parcel pixel takes its parcel's fold of the [spatial folds](morocco.md) (20 km blocks), so a parcel is
  never split between training and test. A not-cropland pixel takes the fold of its 20 km block; pixels in a block
  without labelled parcels get fold −1 and are left out of cross-validation.

## Output

`outputs/<region>/dataset/04c_pixel_samples/<season>/`

| File | Content |
| --- | --- |
| `pixels_X.npy` | int16 [pixels, 12 months, 19 bands]: the bands of the tile store (12 Sentinel-2 bands, clear-observation count, Sentinel-1 VV / VH and scene counts per orbit), NODATA −32768; the experiments use the Sentinel-2 bands only ([D14](../decisions.md#d14-sentinel-2-only-for-every-input)) |
| `pixels.parquet` | one row per pixel, same order: `label`, `class_name`, `parcel_idx`, `parcel_id`, `fold`, `block_id`, `zone`, `tile_id`, `row`, `col`, `x`, `y` (UTM pixel centre), `lon`, `lat`, `edge`, `edge_fallback`, `mask_cropland`, `mask_votes` |
| `summary.csv` | pixels and parcels per class and fold |
| `samples_info.json` | settings, band order, class codes, CRS |


## Sample sets

| Region | Year | Pixels | Irrigated (parcels) | Rainfed (parcels) | Not cropland (tiles) | Of which fold −1 |
| --- | --- | --- | --- | --- | --- | --- |
| Morocco | 2023/24 | 402,756 | 336,211 (6,183) | 31,671 (698) | 34,874 (455) | 97 |
| Morocco | 2024/25 | 119,369 | 83,672 (1,396) | 22,121 (498) | 13,576 (158) | 25 |
| Utah | 2023/24 | 656,081 | 394,064 (4,998) | 75,559 (1,013) | 186,458 (1,911) | 1,699 |

Counts from `summary.csv` of each set. The experiments leave out the pixels of fold −1: Morocco pools both years
(522,003 pixels in 2,230 tiles), Utah 2023/24 gives 654,382 pixels in 2,405 tiles
([E01](../experiments/E01_s2_pixel_baseline.md), [E04](../experiments/E04_frozen_geofm.md)).

### Morocco 2023/24 in detail

402,756 pixels (184 MB): 336,211 irrigated from 6,183 parcels, 31,671 rainfed from 698 parcels and 34,874 not
cropland from 455 tiles.

| Fold | Irrigated | Rainfed | Not cropland |
| --- | --- | --- | --- |
| 0 | 69,104 | 4,130 | 8,009 |
| 1 | 62,922 | 6,154 | 11,004 |
| 2 | 72,720 | 10,134 | 8,338 |
| 3 | 65,290 | 8,444 | 2,445 |
| 4 | 66,175 | 2,809 | 4,981 |
| −1 (no fold) | – | – | 97 |

Checks: all 2,000 tested parcel pixels lie inside their own parcel; no Sentinel-2 value is missing; 0.2% of the
parcel pixels lie outside the cropland mask. Median NDVI: irrigated peaks at 0.71 in February–March, rainfed at
0.36 in April (2023/24 was a dry year), not cropland stays near 0.15.

**Limitations.**

- Rainfed is the minority class (about 1 rainfed pixel per 11 irrigated); the models need class weights.
- The not-cropland pixels come from 455 of 2,183 tiles: in the cropland belt nearly every pixel has at least one
  cropland vote. They are mostly bare land and rangeland, but also evergreen riverside vegetation (example tile
  above). False cropland inside the union mask, the hard case, has no label yet.
- The tiles lie around the surveyed parcels, so accuracy measured on these pixels holds for similar areas; the
  accuracy of the final map needs an independent sample over the map.

## Status

| Region | Year | Status |
| --- | --- | --- |
| Morocco | 2023/24 | ✅ done; used in E01–E04 |
| Morocco | 2024/25 | ✅ done; used in E01–E04 (pooled with 2023/24) |
| Utah | 2023/24 | ✅ done; used in E01–E05 |
| Utah | 2020/21 | prepared for the temporal test (RQ3); not yet built |
