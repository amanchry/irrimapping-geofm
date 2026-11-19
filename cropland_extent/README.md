# cropland_extent/

Cropland products and the cropland mask: where the wall-to-wall irrigated / rainfed maps are made. `make_cropland_mask.py` is
step 0 of the pixel-wise pipeline: run it before `pixelwise_run/01_prepare_labels_*.py`; steps 01, 04 and 06 of
`pixelwise_run/` read its mask.

All three write to **`outputs/<region>/cropland_extent/`** and use the same area: the region's mapping area in
`pixelwise_run/regions.py` (Morocco without Western Sahara; Utah).

| Script | What it does | Output (in `outputs/<region>/cropland_extent/`) |
| --- | --- | --- |
| `make_cropland_mask.py` | cropland mask per region and agricultural year: union of 4 cropland products in Earth Engine (ESA WorldCover 2021, Esri LULC, GCEP30, GLAD); vote count and per-product flags kept | `cropland_mask/<season>/` |
| `download_esa_worldcover.py` | ESA WorldCover 10 m GeoTIFF of the region (Planetary Computer) + PNG map with legend | `esa_worldcover/` |
| `download_lgrip30.py` | LGRIP30 irrigated / rainfed cropland 30 m GeoTIFF of the region (NASA Earthdata token) + PNG map with legend | `lgrip30/` |

The two download scripts make reference layers for maps and documentation; the mask itself reads the products
directly in Earth Engine. Settings are at the top of each script (`REGION`); `regions.py` in `pixelwise_run/` holds the region
settings shared by all scripts.
