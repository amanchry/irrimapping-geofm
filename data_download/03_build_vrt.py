"""
03_build_vrt.py  —  one virtual mosaic (VRT) per source and agricultural year over all downloaded blocks
======================================================================================================

What it does
------------
Writes outputs/<REGION>/data_download/<source>/<season>/mosaic.vrt over all block files of that source and year, so a
whole region can be opened in QGIS, read with rasterio by coordinates (any window, e.g. a 128 px map window across
two blocks), or exported. A VRT is a small XML file that points to the blocks: it adds no copy of the data.
Re-run it after more blocks are downloaded.

Run (from the project folder; after the downloads)
    python data_download/03_build_vrt.py
"""
from download_common import download_dir, season_tag

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
SOURCES = ["s2", "s1"]                 # Dynamic World cropland is one region-wide file already (02a)
SEASONS = {"morocco": ["2023/24", "2024/25"],
           "utah": ["2023/24", "2020/21"]}[REGION]
# ==========================================================================


def main():
    from osgeo import gdal
    gdal.UseExceptions()
    for source in SOURCES:
        for ag_season in SEASONS:
            folder = download_dir(REGION) / source / season_tag(ag_season)
            files = sorted(str(p) for p in folder.glob("*.tif") if not p.name.endswith(".part"))
            if not files:
                continue
            vrt = folder / "mosaic.vrt"
            gdal.BuildVRT(str(vrt), files)
            print(f"{source} {ag_season}: {len(files):,} blocks -> {vrt}")


if __name__ == "__main__":
    main()
