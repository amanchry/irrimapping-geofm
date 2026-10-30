"""
a4_build_vrt.py  —  combine all tiles of the tile store into virtual rasters (VRT) for viewing and testing
=========================================================================================================

What it does
------------
The tile store of ml_dataset_prep/03_build_tiles.py holds thousands of small 128 x 128-pixel GeoTIFFs. This
script writes GDAL virtual rasters (.vrt) that show them as ONE seamless raster per agricultural year,
so the whole study area can be opened in QGIS (or read with rasterio) as a single layer. A VRT is a
small XML file that only points to the tiles: nothing is copied, and the tiles stay the only data.

One VRT per agricultural year: the two years cover the same places, so mixing them in one raster
would hide one year under the other.

Products (per agricultural year, see PRODUCTS)
----------------------------------------------
  <season>_labels.vrt                     label, parcel_idx, edge (from the _lab.tif tiles)
  <season>_truecolor_m01_Sep.vrt ...      B4 B3 B2 for each month (reflectance x 10000)
  <season>_falsecolor_m01_Sep.vrt ...     B8 B4 B3 for each month
  <season>_s1asc_m01_Sep.vrt ...          VV_ASC VH_ASC for each month (dB x 100)
  <season>_full.vrt                       all 228 bands (only if FULL_STACK = True; large file, slow to open)
Values are the stored ones (no rescaling); NODATA = -32768 is set on image bands.

Viewing in QGIS
---------------
Drag a .vrt into QGIS. True colour: Min/Max 0-2500 (cumulative count cut works too); false colour:
0-4500; Sentinel-1: VV about -2500 to 0, VH about -3000 to -500 (dB x 100). Labels: band 1 with
"Paletted/Unique values" (0 = ignore, 1 = irrigated, 2 = rainfed); set 0 as transparent if wished.
Overlay outputs/dataset/02_folds/parcels_folds.geojson to check alignment.

Testing in Python
-----------------
    import rasterio
    with rasterio.open("outputs/dataset/03_tiles/agyear/vrt/2023-24_labels.vrt") as src:
        lab = src.read(1, window=rasterio.windows.from_bounds(x0, y0, x1, y1, src.transform))

Notes
-----
* Only standard Python is used (no GDAL needed to BUILD the VRTs; QGIS / rasterio read them).
* Tile positions come from the tile names (<season>_<ix>_<iy>) and the grid in tiles_info.json.
* Paths inside the VRTs are relative, so the project folder can be moved.
* Re-run after more tiles have been downloaded.

Run (from the project folder, after ml_dataset_prep/03_build_tiles.py)
--------------------------------------------------------------
    python tools/a4_build_vrt.py
"""
import json
import os
import re
from pathlib import Path
from calendar import month_abbr
from xml.sax.saxutils import escape

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                         # project folder
WINDOW = "agyear"                                                   # as in ml_dataset_prep/03_build_tiles.py
TILE_ROOT = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW
TILE_DIR = TILE_ROOT / "tiles"
OUT_DIR = TILE_ROOT / "vrt"

PRODUCTS = {                       # name -> bands taken from every month (names as in tiles_info.json)
    "truecolor": ["B4", "B3", "B2"],
    "falsecolor": ["B8", "B4", "B3"],
    "s1asc": ["VV_ASC", "VH_ASC"],
}
MONTHS = None                      # None = all 12; or e.g. ["Jan", "Jun"]
MAKE_LABELS = True                 # <season>_labels.vrt
FULL_STACK = False                 # <season>_full.vrt with all 228 bands (large XML, slow in QGIS)
# ==========================================================================

INFO = json.loads((TILE_ROOT / "tiles_info.json").read_text())
TILE_PX, GSD, EPSG = INFO["tile_px"], INFO["gsd_m"], INFO["crs_epsg"]
TILE_M = TILE_PX * GSD
MB, N_MONTHS, NODATA = INFO["month_bands"], INFO["n_months"], INFO["nodata"]
MONTH_NAMES = [month_abbr[(INFO["start_month"] - 1 + k) % 12 + 1] for k in range(N_MONTHS)]   # Sep ... Aug
NAME = re.compile(r"^(?P<season>\d{4}-\d{2})_(?P<ix>-?\d+)_(?P<iy>-?\d+)_img\.tif$")


def find_tiles():
    """Downloaded tiles per season: {season: [(ix, iy, img_path, lab_path), ...]}."""
    seasons = {}
    for f in sorted(TILE_DIR.glob("*_img.tif")):
        m = NAME.match(f.name)
        if m:
            lab = f.with_name(f.name.replace("_img.tif", "_lab.tif"))
            seasons.setdefault(m["season"], []).append((int(m["ix"]), int(m["iy"]), f, lab if lab.exists() else None))
    return seasons


def band_index(band, month):
    """1-based band number in an _img.tif: month-major, <band>_m01 ... (see tiles_info.json band_order)."""
    return month * len(MB) + MB.index(band) + 1


def write_vrt(path, tiles, bands, dtype, nodata, use_lab=False):
    """bands: list of (description, source band number). Tiles are placed on the fixed grid."""
    xs, ys = [t[0] for t in tiles], [t[1] for t in tiles]
    ix0, ix1, iy0, iy1 = min(xs), max(xs), min(ys), max(ys)
    width, height = (ix1 - ix0 + 1) * TILE_PX, (iy1 - iy0 + 1) * TILE_PX
    x_min, y_max = ix0 * TILE_M, (iy1 + 1) * TILE_M                     # top-left corner of the mosaic
    sources = []
    for ix, iy, img, lab in tiles:
        src = lab if use_lab else img
        if src is None:
            continue
        rel = escape(os.path.relpath(src, path.parent).replace(os.sep, "/"))
        col, row = (ix - ix0) * TILE_PX, (iy1 - iy) * TILE_PX
        sources.append((rel, col, row))
    out = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
           f'  <SRS>EPSG:{EPSG}</SRS>',
           f'  <GeoTransform>{x_min:.1f}, {GSD:.1f}, 0.0, {y_max:.1f}, 0.0, {-GSD:.1f}</GeoTransform>']
    for k, (desc, b) in enumerate(bands, start=1):
        out.append(f'  <VRTRasterBand dataType="{dtype}" band="{k}">')
        out.append(f'    <Description>{escape(desc)}</Description>')
        if nodata is not None:
            out.append(f'    <NoDataValue>{nodata}</NoDataValue>')
        for rel, col, row in sources:
            out.append(f'    <SimpleSource><SourceFilename relativeToVRT="1">{rel}</SourceFilename>'
                       f'<SourceBand>{b}</SourceBand>'
                       f'<SourceProperties RasterXSize="{TILE_PX}" RasterYSize="{TILE_PX}" DataType="{dtype}"/>'
                       f'<SrcRect xOff="0" yOff="0" xSize="{TILE_PX}" ySize="{TILE_PX}"/>'
                       f'<DstRect xOff="{col}" yOff="{row}" xSize="{TILE_PX}" ySize="{TILE_PX}"/></SimpleSource>')
        out.append('  </VRTRasterBand>')
    out.append('</VRTDataset>')
    path.write_text("\n".join(out))
    size = path.stat().st_size / 1e6
    print(f"  {path.name:40s} {len(bands):3d} band(s), {len(sources):,} tiles, "
          f"{width:,} x {height:,} px, {size:.1f} MB")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seasons = find_tiles()
    if not seasons:
        raise SystemExit(f"No *_img.tif tiles found in {TILE_DIR}: run ml_dataset_prep/03_build_tiles.py first")
    months = [k for k in range(N_MONTHS) if MONTHS is None or MONTH_NAMES[k] in MONTHS]
    for season, tiles in seasons.items():
        print(f"\n{season}: {len(tiles):,} downloaded tiles")
        if MAKE_LABELS:
            write_vrt(OUT_DIR / f"{season}_labels.vrt", tiles,
                      [("label (0 ignore, 1 irrigated, 2 rainfed)", 1), ("parcel_idx", 2), ("edge", 3)],
                      "Int32", None, use_lab=True)
        for name, bands in PRODUCTS.items():
            for k in months:
                tag = f"m{k + 1:02d}_{MONTH_NAMES[k]}"
                write_vrt(OUT_DIR / f"{season}_{name}_{tag}.vrt", tiles,
                          [(f"{b}_m{k + 1:02d}", band_index(b, k)) for b in bands], "Int16", NODATA)
        if FULL_STACK:
            write_vrt(OUT_DIR / f"{season}_full.vrt", tiles,
                      [(f"{b}_m{k + 1:02d}", band_index(b, k)) for k in range(N_MONTHS) for b in MB],
                      "Int16", NODATA)
    print(f"\nSaved in {OUT_DIR}")


if __name__ == "__main__":
    main()
