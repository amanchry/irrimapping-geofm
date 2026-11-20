# cropland_extent/

Cropland products and the cropland mask, for the **mapping phase**: where the wall-to-wall irrigated / rainfed maps
will be made. These scripts are independent of the experiments: no step of `pixelwise_run/` (labels, samples,
experiments) uses a cropland mask; only the mapping tiles (`pixelwise_run/06_build_cropland_tiles.py`,
`s2_download/`) read the mask. The mask will be chosen later by comparing products (e.g. the union of products of
`make_cropland_mask.py` against Dynamic World alone) on the surveyed parcels.

All scripts write to **`outputs/<region>/cropland_extent/`** and use the same area: the region's mapping area in
`pixelwise_run/regions.py` (Morocco without Western Sahara; Utah).

| Script | What it does | Output (in `outputs/<region>/cropland_extent/`) |
| --- | --- | --- |
| `make_cropland_mask.py` | cropland mask per region and agricultural year: union of 4 cropland products in Earth Engine (ESA WorldCover 2021, Esri LULC, GCEP30, GLAD); vote count and per-product flags kept | `cropland_mask/<season>/` |
| `download_esa_worldcover.py` | ESA WorldCover 10 m GeoTIFF of the region (Planetary Computer) + PNG map with legend | `esa_worldcover/` |
| `download_lgrip30.py` | LGRIP30 irrigated / rainfed cropland 30 m GeoTIFF of the region (NASA Earthdata token) + PNG map with legend | `lgrip30/` |
| `download_dynamic_world.py` | annual Dynamic World 10 m land cover per agricultural year (Earth Engine): most frequent class, mean crop probability, number of images; GeoTIFF on the project grid + PNG map with legend | `dynamic_world/` |

The download scripts make reference layers for maps and documentation; the mask itself reads the products
directly in Earth Engine. Settings are at the top of each script (`REGION`); `regions.py` in `pixelwise_run/` holds the region
settings shared by all scripts.
