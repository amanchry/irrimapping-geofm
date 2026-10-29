"""
a3_export_chips_geotiff.py  —  write some parcels' chips as GeoTIFFs to look at in QGIS
=======================================================================================

What it does
------------
The crops of dataset/04_make_views.py are stored as compressed NumPy arrays (fast for the models, not
viewable in GIS). This script turns the chips of a few parcels back into georeferenced GeoTIFFs,
using the UTM zone and chip corner saved in index.csv, so they overlay exactly on the parcels.

For each selected parcel it writes a folder outputs/study/01_inspection/chips_geotiff/<class>_<parcel_id>/ with:
  m01_Sep.tif ... m12_Aug.tif   one file per month, 14 bands: B1 ... B12 (Sentinel-2 reflectance x 10000)
                                and VV, VH (Sentinel-1 dB x 100); no-data = -32768 (cloudy month/pixel)
  ndvi_12months.tif             NDVI for the 12 months (float, one band per month) - quick seasonal view
  parcel_mask.tif               1 = pixel inside the parcel (the mask the models use)
  parcel.geojson                the parcel outline (in the chip's UTM zone)

Viewing in QGIS
---------------
* Drag a month file in; Properties > Symbology > Multiband color: Red = band 4 (B4), Green = band 3 (B3),
  Blue = band 2 (B2), min/max about 0-3000 for true colour; or R = B8, G = B4, B = B3 for false colour
  (vegetation red).
* ndvi_12months.tif: single band, pseudocolour -0.1 to 0.9; step through bands 1-12 (Sep ... Aug) to
  see the season. Irrigated parcels should stay green in summer (bands 10-12).
* Load parcel.geojson on top (outline only).

Steps
-----
1. Read chips_info.json and index.csv of the chips, and the labelled parcels.
2. Pick the parcels: PARCEL_IDS if given, otherwise N_PER_CLASS random irrigated and rainfed parcels.
3. Find each parcel's part file, read its chip and mask, and write the GeoTIFFs and the outline.

Run (from the project folder, after 04_download_chips.py)
---------------------------------------------------------
    python analysis/a3_export_chips_geotiff.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
WINDOW = "agyear"                                          # as in dataset/03_build_tiles.py
CHIP_DIR = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW / "views" / "crops_p16"   # crops of dataset/04_make_views.py
PARCELS_FILE = ROOT / "outputs" / "dataset" / "01_labels" / "parcels_labelled.geojson"
OUT_DIR = ROOT / "outputs" / "study" / "01_inspection" / "chips_geotiff"

PARCEL_IDS = None             # e.g. ["1234", "5678"] for specific parcels; None = random selection
N_PER_CLASS = 5               # random parcels per class when PARCEL_IDS is None
SEED = 42
# ==========================================================================

CLASS_NAMES = {1: "irrigated", 2: "rainfed"}


def month_names(info):
    start = pd.Timestamp(year=2023, month=info["start_month"], day=1)
    return [(start + pd.DateOffset(months=k)).strftime("%b") for k in range(info["n_months"])]


def write_tif(path, array, transform, epsg, nodata, descriptions):
    """Write [bands, H, W] as a compressed GeoTIFF with band names."""
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
                       count=array.shape[0], dtype=array.dtype, crs=f"EPSG:{epsg}", transform=transform,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(array)
        for i, name in enumerate(descriptions, start=1):
            dst.set_band_description(i, name)


def select_parcels(index):
    if PARCEL_IDS:
        missing = set(map(str, PARCEL_IDS)) - set(index.index)
        if missing:
            print(f"Not in the chips (skipped): {sorted(missing)}")
        return [p for p in map(str, PARCEL_IDS) if p in index.index]
    rng = np.random.default_rng(SEED)
    chosen = []
    for label in CLASS_NAMES:
        ids = index.index[index["label"] == label].to_numpy()
        chosen += list(rng.choice(ids, size=min(N_PER_CLASS, len(ids)), replace=False))
    return chosen


def main():
    info = json.loads((CHIP_DIR / "chips_info.json").read_text())
    index = pd.read_csv(CHIP_DIR / "index.csv", dtype={"parcel_id": str}).drop_duplicates("parcel_id").set_index("parcel_id")
    parcels = gpd.read_file(PARCELS_FILE)
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    parcels = parcels.set_index("parcel_id")
    months = month_names(info)
    bands = info["bands"]
    nodata = info["nodata"]
    i_red, i_nir = bands.index("B4"), bands.index("B8")
    gsd = info["gsd_m"]

    chosen = select_parcels(index)
    print(f"Exporting {len(chosen)} parcels to {OUT_DIR}")
    for part, ids in index.loc[chosen].groupby("part").groups.items():
        d = np.load(CHIP_DIR / part)
        position = {pid: i for i, pid in enumerate(d["parcel_id"].astype(str))}
        for pid in ids:
            row = index.loc[pid]
            chip, mask = d["chips"][position[pid]], d["mask"][position[pid]]       # [months, bands, H, W], [H, W]
            transform = Affine(gsd, 0, row["x0"], 0, -gsd, row["y0"])             # top-left corner, 10 m pixels
            folder = OUT_DIR / f"{CLASS_NAMES[int(row['label'])]}_{pid}"
            folder.mkdir(parents=True, exist_ok=True)

            # One 14-band file per month.
            for k, name in enumerate(months):
                write_tif(folder / f"m{k + 1:02d}_{name}.tif", chip[k], transform, row["epsg"], nodata, bands)

            # NDVI per month (float; NaN where cloudy).
            red = chip[:, i_red].astype(np.float32)
            nir = chip[:, i_nir].astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi[(chip[:, i_red] == nodata) | (chip[:, i_nir] == nodata)] = np.nan
            write_tif(folder / "ndvi_12months.tif", ndvi.astype(np.float32), transform, row["epsg"], np.nan,
                      [f"NDVI {m}" for m in months])

            write_tif(folder / "parcel_mask.tif", mask[None].astype(np.uint8), transform, row["epsg"], None, ["parcel"])
            outline = parcels.loc[[pid], ["label", "class_name", "ag_season", "area_ha", "geometry"]]
            (folder / "parcel.geojson").write_text(outline.to_crs(int(row["epsg"])).reset_index().to_json(drop_id=True))
            print(f"  {folder.name}: {row['ag_season']}, {len(months)} months")


if __name__ == "__main__":
    main()
