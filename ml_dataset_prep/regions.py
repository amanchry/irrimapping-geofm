"""
regions.py  —  the few settings that differ between study regions (shared by scripts 01-04)
===========================================================================================

Each script of ml_dataset_prep/ has REGION = "morocco" or "utah" at the top. Everything that depends on the
region is read from here, so the same scripts build both datasets and the two can never get out of step
(projection, agricultural year, boundaries for the maps). All outputs of a region go to
outputs/<region>/dataset/ (01_labels, 02_folds, 03_tiles_128px).

Region settings
---------------
crs_epsg       metric projection for areas, distances, spatial blocks and the tile grid
               (Morocco: UTM 29N; Utah: UTM 12N, which covers 108-114 W)
start_month    first month of the agricultural year (12 monthly composites from this month)
               Morocco: 1 September - 31 August (sowing of rainfed cereals with the autumn rains,
               harvest May-June; irrigated summer crops up to August)
               Utah: 1 October - 30 September, the US water year; covers the sowing of winter wheat
               (Sep-Oct) up to the last alfalfa cuts and harvests in September
block_km       size of the spatial blocks for the folds (02_make_spatial_folds.py checks it)
admin_file     boundary file in data/admin_boundaries/ (FAO GAUL), only for the maps
admin_where    which features of the boundary file to draw (OGR SQL filter)
admin_name     name column of those features
"""
import os
from pathlib import Path

os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")      # GAUL polygons are larger than the GDAL default limit

ROOT = Path(__file__).resolve().parents[1]                  # project folder
ADMIN_DIR = ROOT / "data" / "admin_boundaries"

REGIONS = {
    "morocco": {
        "crs_epsg": 32629,
        "start_month": 9,
        "block_km": 20,
        "admin_file": "FAO_GAUL_2015_AdminL0.geojson",
        "admin_where": "ADM0_NAME = 'Morocco'",
        "admin_name": "ADM0_NAME",
    },
    "utah": {
        "crs_epsg": 32612,
        "start_month": 10,
        "block_km": 20,
        "admin_file": "FAO_GAUL_2024_AdminL1.geojson",
        "admin_where": "gaul0_name = 'United States of America' AND gaul1_name IN "
                       "('Utah', 'Idaho', 'Wyoming', 'Nevada', 'Colorado', 'Arizona')",
        "admin_name": "gaul1_name",
    },
}


def settings(region):
    if region not in REGIONS:
        raise ValueError(f"REGION must be one of {list(REGIONS)}, not {region!r}")
    return REGIONS[region]


def dataset_dir(region):
    """outputs/<region>/dataset/ — all dataset outputs of one region."""
    return ROOT / "outputs" / region / "dataset"


def admin_boundaries(region):
    """Boundary polygons for the maps (EPSG:4326). The global GAUL file is large, so the region's features are
    read once and cached in outputs/<region>/dataset/admin_boundaries.geojson."""
    import geopandas as gpd
    import pyogrio
    cache = dataset_dir(region) / "admin_boundaries.geojson"
    if cache.exists():
        return gpd.read_file(cache)
    s = settings(region)
    gdf = pyogrio.read_dataframe(ADMIN_DIR / s["admin_file"], where=s["admin_where"]).to_crs(4326)
    gdf = gdf[[s["admin_name"], "geometry"]].rename(columns={s["admin_name"]: "name"})
    gdf["geometry"] = gdf.geometry.simplify(0.001)                  # ~100 m: plenty for maps, much smaller file
    cache.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache, driver="GeoJSON")
    return gdf
