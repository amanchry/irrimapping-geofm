"""
04_cut_tiles.py  —  cut the 128 px training tiles out of the downloaded 10.24 km blocks
======================================================================================

What it does
------------
For every tile of the training-tile plan (03_plan_tiles.py: the tiles around the labelled parcels) the tile is cut
out of its block in data_download/ (block (bx, by) = (ix // 8, iy // 8); tiles and blocks share one grid, so the
cut-out is exact, without resampling) and written as the file every model reads, together with the QA of the tile.
Only the training area gets tile files; the wall-to-wall maps read windows directly from the blocks.

Per tile and month (12 months, month-major: all bands of m01, then m02, ...)
  B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12   Sentinel-2 (reflectance x 10,000)          from download/s2/
  S2_NCLEAR                                clear Sentinel-2 observations              from download/s2/
  VV_ASC VH_ASC S1_N_ASC                   Sentinel-1 ascending (dB x 100, scenes)    from download/s1/
  VV_DSC VH_DSC S1_N_DSC                   Sentinel-1 descending                      from download/s1/
= 19 bands x 12 months = 228 bands, int16, NODATA -32768: the layout of the tiles downloaded before, so all
pixelwise_run/ scripts and experiments read them unchanged. Dynamic World (data_download/dynamic_world/cropland_<season>.tif) is written as
a separate file where downloaded.

Tiles that already exist are kept (OVERWRITE = False): the earlier training tiles stay as they are.

Output (outputs/<REGION>/data_training_prep/03_tiles_128px/agyear/)
  tiles/<tile_id>_img.tif      int16, 228 bands (descriptions "B4_m03", ...)
  tiles/<tile_id>_dw.tif       uint8, 1 band CROPLAND (1 cropland, 0 other, 255 no data) of
                               data_download/02a_dynamic_world.py (where it was downloaded)
  tiles/<tile_id>_scenes.json  scene IDs per month of the tile's block: s2, s1_asc, s1_dsc
  tiles_qa.jsonl               QA per tile (no-data share, clear observations on parcels, Sentinel-1 months)
  tiles_info.json              bands, months, scales, NODATA (added to the grid settings of 03)
  tiles.parquet                refreshed: tiles cut, QA

Run (from the project folder; after data_download/02a, 02b and 02c for the region and years)
    python data_training_prep/04_cut_tiles.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from tile_store import N_MONTHS, NODATA, S2_BANDS, TILE_PX, load_plan, out_dir, tile_dir, tile_file, write_tif
from download_common import DW_NODATA, TILE_M, block_file, block_name, block_of_tile, dw_cropland_file, read_dw_window  # noqa: E402  (data_download/, via tile_store)
from regions import settings  # noqa: E402

# ================================ SETTINGS ================================
REGION = "morocco"             # "morocco" | "utah" (regions.py)
OVERWRITE = False              # True = cut again even where the tile file exists
# ==========================================================================

S2_PART = S2_BANDS + ["S2_NCLEAR"]                                  # bands per month in download/s2/
S1_PART = ["VV_ASC", "VH_ASC", "S1_N_ASC", "VV_DSC", "VH_DSC", "S1_N_DSC"]   # bands per month in download/s1/
MONTH_BANDS = S2_PART + S1_PART
COUNT_BANDS = ["S2_NCLEAR", "S1_N_ASC", "S1_N_DSC"]
NAMES = [f"{b}_m{k:02d}" for k in range(1, N_MONTHS + 1) for b in MONTH_BANDS]
CRS_EPSG = settings(REGION)["crs_epsg"]


def cut(path, row, col):
    with rasterio.open(path) as src:
        return src.read(window=Window(col, row, TILE_PX, TILE_PX))


def tile_qa(cube, lab):
    """QA numbers of one tile: share of NODATA, clear Sentinel-2 observations on labelled pixels."""
    c = cube.reshape(N_MONTHS, len(MONTH_BANDS), TILE_PX, TILE_PX)
    s2_missing = c[:, S2_BANDS.index("B4")] == NODATA
    n_clear = c[:, MONTH_BANDS.index("S2_NCLEAR")]
    on = lab[1] > 0
    qa = {"s2_nodata_pct": round(100 * float(s2_missing.mean()), 2),
          "s1_asc_months": int((c[:, MONTH_BANDS.index("S1_N_ASC")].max(axis=(1, 2)) > 0).sum()),
          "s1_dsc_months": int((c[:, MONTH_BANDS.index("S1_N_DSC")].max(axis=(1, 2)) > 0).sum())}
    if on.any():
        qa["s2_nodata_parcel_pct"] = round(100 * float(s2_missing[:, on].mean()), 2)
        qa["s2_nclear_parcel_median"] = [float(np.median(n_clear[k][on])) for k in range(N_MONTHS)]
    return qa


def scenes(ag_season, bx, by):
    """Scene IDs per month of the tile's block (download/s2/, download/s1/ provenance files)."""
    out = {}
    for source in ("s2", "s1"):
        p = block_file(REGION, source, ag_season, bx, by).with_name(f"{block_name(bx, by)}_scenes.json")
        out[source] = json.loads(p.read_text()) if p.exists() else {}
    return {f"m{k:02d}": {"s2": out["s2"].get(f"m{k:02d}", []), "s1_asc": out["s1"].get(f"m{k:02d}", {}).get("asc", []),
                          "s1_dsc": out["s1"].get(f"m{k:02d}", {}).get("dsc", [])} for k in range(1, N_MONTHS + 1)}


def refresh_index():
    """Re-run the index step of 03_plan_tiles.py (tiles cut, QA)."""
    spec = importlib.util.spec_from_file_location("plan", Path(__file__).with_name("03_plan_tiles.py"))
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    if plan.REGION != REGION:
        print(f"Index not refreshed: REGION of 03_plan_tiles.py is {plan.REGION}; re-run it for {REGION}")
        return
    tiles = load_plan(REGION)
    parcels = pd.read_csv(out_dir(REGION) / "parcels.csv")
    keep = [c for c in ["ag_season", "ix", "iy", "tile_id", "x0", "y0", "geometry"] if c in tiles.columns]
    index = plan.tile_index(tiles[keep], parcels, out_dir(REGION) / "tiles_qa.jsonl")
    index.to_parquet(out_dir(REGION) / "tiles.parquet")
    print(f"Index refreshed: {int(index['has_img'].sum()):,} of {len(index):,} training tiles ready")


def main():
    info_file = out_dir(REGION) / "tiles_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    info.update({"month_bands": MONTH_BANDS, "s2_bands": S2_BANDS, "count_bands": COUNT_BANDS,
                 "s2_scale": "surface reflectance x 10000",
                 "s1": "Sentinel-1 GRD IW, mean in linear power per orbit direction, stored as dB x 100",
                 "s1_scale": 100, "nodata": NODATA, "band_order": "month-major: <band>_m01 ... <band>_m12",
                 "source": "cut from the 10.24 km blocks of data_download/ (02a Dynamic World, 02b Sentinel-2, "
                           "02c Sentinel-1)"})
    info_file.write_text(json.dumps(info, indent=2))
    tiles = load_plan(REGION)
    qa_file = out_dir(REGION) / "tiles_qa.jsonl"
    done, missing, dw = 0, 0, 0
    for t in tiles.itertuples():
        bx, by, row, col = block_of_tile(int(t.ix), int(t.iy))
        img = tile_file(REGION, t.tile_id, "img")
        if not img.exists() or OVERWRITE:
            s2_block = block_file(REGION, "s2", t.ag_season, bx, by)
            s1_block = block_file(REGION, "s1", t.ag_season, bx, by)
            if not (s2_block.exists() and s1_block.exists()):
                missing += 1
            else:
                s2 = cut(s2_block, row, col).reshape(N_MONTHS, len(S2_PART), TILE_PX, TILE_PX)
                s1 = cut(s1_block, row, col).reshape(N_MONTHS, len(S1_PART), TILE_PX, TILE_PX)
                cube = np.concatenate([s2, s1], axis=1).reshape(len(NAMES), TILE_PX, TILE_PX).astype(np.int16)
                write_tif(img, cube, int(t.ix), int(t.iy), NAMES, CRS_EPSG, nodata=NODATA)
                (tile_dir(REGION) / f"{t.tile_id}_scenes.json").write_text(json.dumps(scenes(t.ag_season, bx, by)))
                with rasterio.open(tile_file(REGION, t.tile_id, "lab")) as src:
                    qa = tile_qa(cube, src.read())
                with open(qa_file, "a") as fh:
                    fh.write(json.dumps({"tile_id": t.tile_id, **qa}) + "\n")
                done += 1
        dw_file = dw_cropland_file(REGION, t.ag_season)
        dw_tile = tile_file(REGION, t.tile_id, "dw")
        if dw_file.exists() and (not dw_tile.exists() or OVERWRITE):
            with rasterio.open(dw_file) as src:                     # region-wide binary cropland of 02a (uint8)
                arr = read_dw_window(src, int(t.ix) * TILE_M, (int(t.iy) + 1) * TILE_M, TILE_PX)
            write_tif(dw_tile, arr[None], int(t.ix), int(t.iy), ["CROPLAND"], CRS_EPSG, nodata=DW_NODATA)
            dw += 1
    print(f"Cut {done:,} image tiles and {dw:,} Dynamic World tiles; {missing:,} tiles wait for their block "
          f"(run data_download/02a, 02b and 02c)")
    refresh_index()


if __name__ == "__main__":
    main()
