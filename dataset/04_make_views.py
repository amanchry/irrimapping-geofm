"""
04_make_views.py  —  model inputs cut from the tile store (stage 2 of 2; local, no Earth Engine)
==================================================================================================

What it does
------------
Reads the tile store of 03_build_tiles.py and writes every input the models need. Because all
views come from the same downloaded pixels, differences between models cannot come from
different imagery, compositing or masks.

Views
-----
A. Parcel feature tables (random forest / gradient boosting baseline, simple baselines)
   Per parcel and month: MEDIAN over the parcel's pixels of the 12 Sentinel-2 bands, NDVI, NDWI,
   EVI, Sentinel-1 VV/VH (ascending and descending, dB), and the clear-observation counts.
   Median, not mean, as in Sen4AgriNet / EuroCropsML (robust to residual cloud and edge pixels).
     parcel_features_all.csv     all parcel pixels
     parcel_features_inner.csv   without edge pixels (INNER_BUFFER_M of 03_build_tiles.py); parcels with no
                                 inner pixel fall back to all pixels (column used_inner = False)
B. Pixel sets (PSE-TAE / LTAE-type models, Garnot et al. 2020)
     pixelsets.npz   pixels [N_pixels, months, bands] int16, offsets [parcels + 1], edge [N_pixels],
                     parcel_id, label, bands; parcel i = pixels[offsets[i]:offsets[i + 1]]
C. Patch-aligned crops for the foundation models, one folder per patch size (CROP_PATCH_SIZES)
     crops_p8/, crops_p16/   same format as the legacy 04_download_chips.py (part_*.npz, index.csv,
                             chips_info.json), so scripts 02a-02d only need CHIP_DIR changed.
   The CROP_PX x CROP_PX crop is shifted by half a patch so the parcel centroid lies at the CENTRE
   of one model patch instead of on the corner of four patches (centred crops with a crop size
   that is a multiple of the patch size split small parcels over 4 mostly-neighbour patches).
   index.csv adds `purity` (share of the pooled patches inside the parcel, the weighting used by
   the study/02 embedding scripts), `purity_centred` (same for the old centred crop), n_px, area_ha, fold, zone.
D. Full tiles (segmentation models such as U-TAE): use outputs/dataset/03_tiles/<WINDOW>/tiles/ directly;
   the label band has 0 = ignore for unlabelled pixels.

Sentinel-1 orbit for B and C: S1_ORBIT = "auto" picks, over the whole study area, the orbit
direction with more valid parcel-months (written to views_info.json); "ASC" / "DSC" force one.
Feature tables (A) keep both orbits as separate columns.

Output (outputs/dataset/03_tiles/<WINDOW>/views/)
-----------------------------------------
parcel_features_all.csv, parcel_features_inner.csv, pixelsets.npz, crops_p8/, crops_p16/,
qa_purity.csv, qa_clear_obs.csv, views_info.json

Run (from the project folder, after 03_build_tiles.py)
------------------------------------------------------
    python dataset/04_make_views.py
"""
import json
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                         # project folder
WINDOW = "agyear"
TILE_ROOT = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW                  # from 03_build_tiles.py
OUT_DIR = TILE_ROOT / "views"

MAKE_FEATURES = True           # view A
MAKE_PIXELSETS = True          # view B
CROP_PATCH_SIZES = [8, 16]     # view C: 8 = Clay v1.5; 16 = Prithvi-EO-2.0, TerraMind (check AgriFM)
CROP_PX = 64                   # crop size (must be <= CROP_PX of 03_build_tiles.py)
S1_ORBIT = "auto"              # "auto" | "ASC" | "DSC" for views B and C
BATCH_PARCELS = 64             # parcels per crop part file
CACHE_TILES = 48               # tiles kept in memory while cutting views
# ==========================================================================

INFO = json.loads((TILE_ROOT / "tiles_info.json").read_text())
TILE_DIR = TILE_ROOT / "tiles"
TILE_PX, GSD, NODATA = INFO["tile_px"], INFO["gsd_m"], INFO["nodata"]
N_MONTHS, MB = INFO["n_months"], INFO["month_bands"]
S2_BANDS = INFO["s2_bands"]
S1_SCALE = INFO["s1_scale"]
warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="Mean of empty slice")


# ------------------------------------------------------------ reading the tile store
def tile_name(ag_season, ix, iy):
    return f"{ag_season.replace('/', '-')}_{ix}_{iy}"


class TileStore:
    """Reads windows of the fixed grid across tile borders. Global pixel coordinates:
    col = floor(x / GSD), row = floor(-y / GSD) (rows grow southwards)."""

    def __init__(self, cache=CACHE_TILES):
        self.cache, self.size, self.missing = OrderedDict(), cache, set()

    def _tile(self, season, ix, iy):
        key = (season, ix, iy)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        name = tile_name(season, ix, iy)
        img_f, lab_f = TILE_DIR / f"{name}_img.tif", TILE_DIR / f"{name}_lab.tif"
        if not (img_f.exists() and lab_f.exists()):
            self.missing.add(name)
            data = None
        else:
            with rasterio.open(img_f) as src:
                img = src.read().reshape(N_MONTHS, len(MB), TILE_PX, TILE_PX)
            with rasterio.open(lab_f) as src:
                lab = src.read()
            data = (img, lab)
        self.cache[key] = data
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return data

    def window(self, season, row0, col0, h, w):
        """img int16 [months, bands, h, w] (NODATA outside the store), lab int32 [3, h, w] (0 outside)."""
        img = np.full((N_MONTHS, len(MB), h, w), NODATA, dtype=np.int16)
        lab = np.zeros((3, h, w), dtype=np.int32)
        for trow in range(row0 // TILE_PX, (row0 + h - 1) // TILE_PX + 1):
            for tcol in range(col0 // TILE_PX, (col0 + w - 1) // TILE_PX + 1):
                data = self._tile(season, tcol, -trow - 1)
                if data is None:
                    continue
                r_a, r_b = max(row0, trow * TILE_PX), min(row0 + h, (trow + 1) * TILE_PX)
                c_a, c_b = max(col0, tcol * TILE_PX), min(col0 + w, (tcol + 1) * TILE_PX)
                src = (slice(r_a - trow * TILE_PX, r_b - trow * TILE_PX), slice(c_a - tcol * TILE_PX, c_b - tcol * TILE_PX))
                dst = (slice(r_a - row0, r_b - row0), slice(c_a - col0, c_b - col0))
                img[(..., *dst)] = data[0][(..., *src)]
                lab[(slice(None), *dst)] = data[1][(slice(None), *src)]
        return img, lab


def parcel_window(p):
    """Global (row0, col0, h, w) covering the parcel outline (bounds in parcels.csv), one pixel margin."""
    row0, col0 = int(np.floor(-p.maxy / GSD)) - 1, int(np.floor(p.minx / GSD)) - 1
    return row0, col0, int(np.floor(-p.miny / GSD)) + 2 - row0, int(np.floor(p.maxx / GSD)) + 2 - col0


def centroid_pixel(p):
    return int(np.floor(-p.cy / GSD)), int(np.floor(p.cx / GSD))


# ------------------------------------------------------------ view A: features
def parcel_pixels(img, lab, idx):
    """Pixels of one parcel -> int16 [pixels, months, bands], edge flag [pixels]."""
    on = lab[1] == idx
    return np.moveaxis(img[:, :, on], -1, 0), lab[2][on].astype(bool)


def to_physical(px):
    """int16 pixels [N, months, bands] -> dict of float arrays [N, months]: reflectance 0-1, dB, counts."""
    out = {}
    for j, b in enumerate(MB):
        v = px[:, :, j].astype(np.float32)
        if b in INFO["count_bands"]:
            out[b] = v
            continue
        v[px[:, :, j] == NODATA] = np.nan
        out[b] = v / 10000 if b in S2_BANDS else v / S1_SCALE
    nir, red, blue, swir = out["B8"], out["B4"], out["B2"], out["B11"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["NDVI"] = (nir - red) / (nir + red)
        out["NDWI"] = (nir - swir) / (nir + swir)
        out["EVI"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    return out


FEATURE_BANDS = S2_BANDS + ["NDVI", "NDWI", "EVI", "VV_ASC", "VH_ASC", "VV_DSC", "VH_DSC"]


def feature_row(px):
    """Median over pixels per band and month; counts: mean clear S2 obs, max S1 scenes."""
    phys = to_physical(px)
    row = {}
    for b in FEATURE_BANDS:
        med = np.nanmedian(phys[b], axis=0)
        row.update({f"{b}_m{k + 1:02d}": float(med[k]) for k in range(N_MONTHS)})
    for b, fn in (("S2_NCLEAR", np.mean), ("S1_N_ASC", np.max), ("S1_N_DSC", np.max)):
        v = fn(phys[b], axis=0)
        row.update({f"{b}_m{k + 1:02d}": float(v[k]) for k in range(N_MONTHS)})
    return row


# ------------------------------------------------------------ view C: crops
def purity(mask, patch_px):
    """Weighted share of the pooled patches inside the parcel (study/02 scripts: weights = parcel share per patch)."""
    n = mask.shape[0] // patch_px
    w = mask[: n * patch_px, : n * patch_px].astype(np.float32).reshape(n, patch_px, n, patch_px).mean(axis=(1, 3))
    w = w[w > 0]
    return float((w * w).sum() / w.sum()) if w.size else 0.0


def choose_orbit(parcels, store):
    if S1_ORBIT in ("ASC", "DSC"):
        return S1_ORBIT
    tiles = pd.read_parquet(TILE_ROOT / "tiles.parquet")
    tiles = tiles[tiles["role"] != "context"]
    asc, dsc = tiles["s1_asc_months"].sum(), tiles["s1_dsc_months"].sum()
    print(f"Sentinel-1 orbit: ascending {asc} vs descending {dsc} valid tile-months -> "
          f"{'ASC' if asc >= dsc else 'DSC'}")
    return "ASC" if asc >= dsc else "DSC"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parcels = pd.read_csv(TILE_ROOT / "parcels.csv", dtype={"parcel_id": str})
    tiles = pd.read_parquet(TILE_ROOT / "tiles.parquet")
    have = set(tiles.loc[tiles["downloaded"], "tile_id"])
    complete = parcels["tiles_needed"].str.split(";").apply(lambda ts: all(t in have for t in ts))
    print(f"{int(complete.sum()):,} of {len(parcels):,} parcels have all their tiles downloaded")
    parcels = parcels[complete].sort_values(["home_tile", "parcel_idx"]).reset_index(drop=True)   # tile order: cache hits
    store = TileStore()
    orbit = choose_orbit(parcels, store)
    crop_bands = S2_BANDS + [f"VV_{orbit}", f"VH_{orbit}", "S2_NCLEAR"]
    crop_idx = [MB.index(b) for b in crop_bands]

    feats = {"all": [], "inner": []}
    px_list, edge_list, offsets, px_ids = [], [], [0], []
    crops = {p: {"chips": [], "mask": [], "rows": [], "n_part": 0} for p in CROP_PATCH_SIZES}
    for p in CROP_PATCH_SIZES:
        (OUT_DIR / f"crops_p{p}").mkdir(exist_ok=True)

    def flush(p, final=False):
        c = crops[p]
        if not c["rows"] or (len(c["rows"]) < BATCH_PARCELS and not final):
            return
        folder = OUT_DIR / f"crops_p{p}"
        part = f"part_{c['n_part']:05d}.npz"
        np.savez_compressed(folder / part, parcel_id=np.array([r["parcel_id"] for r in c["rows"]]),
                            chips=np.stack(c["chips"]), mask=np.stack(c["mask"]))
        rows = pd.DataFrame([{**r, "part": part} for r in c["rows"]])
        rows.to_csv(folder / "index.csv", mode="a", header=c["n_part"] == 0, index=False)
        c.update(chips=[], mask=[], rows=[], n_part=c["n_part"] + 1)

    for p in CROP_PATCH_SIZES:                                       # fresh index files
        f = OUT_DIR / f"crops_p{p}" / "index.csv"
        f.unlink(missing_ok=True)
        for old in (OUT_DIR / f"crops_p{p}").glob("part_*.npz"):
            old.unlink()

    reach = CROP_PX // 2 + max(CROP_PATCH_SIZES) // 2
    for i, p in enumerate(parcels.itertuples()):
        crow, ccol = centroid_pixel(p)
        # One window around the centroid holds the parcel crop of every patch size; the parcel
        # outline itself may reach further, so read its own bounding window too.
        img_c, lab_c = store.window(p.ag_season, crow - reach, ccol - reach, 2 * reach, 2 * reach)
        img_p, lab_p = store.window(p.ag_season, *parcel_window(p))
        px, edge = parcel_pixels(img_p, lab_p, p.parcel_idx)
        meta = {"parcel_id": p.parcel_id, "label": p.label, "ag_season": p.ag_season, "fold": p.fold}

        if MAKE_FEATURES:
            if len(px):
                feats["all"].append({**meta, "n_px_used": len(px), **feature_row(px)})
                inner = px[~edge] if (~edge).any() else px
                feats["inner"].append({**meta, "n_px_used": len(inner), "used_inner": bool((~edge).any()),
                                       **feature_row(inner)})
        if MAKE_PIXELSETS and len(px):
            px_ids.append((p.parcel_id, p.label))
            px_list.append(px)
            edge_list.append(edge)
            offsets.append(offsets[-1] + len(px))

        for ps in CROP_PATCH_SIZES:
            r0 = reach - CROP_PX // 2 - ps // 2                        # centroid at the centre of a patch
            centred = reach - CROP_PX // 2                             # old rule: centroid on a patch corner
            sl = (slice(r0, r0 + CROP_PX), slice(r0, r0 + CROP_PX))
            mask = (lab_c[1][sl] == p.parcel_idx).astype(np.uint8)
            mask_c = lab_c[1][centred:centred + CROP_PX, centred:centred + CROP_PX] == p.parcel_idx
            c = crops[ps]
            c["chips"].append(img_c[:, crop_idx][(..., *sl)])
            c["mask"].append(mask)
            c["rows"].append({**meta, "lat": p.lat, "lon": p.lon, "epsg": INFO["crs_epsg"],
                              "x0": (ccol - reach + r0) * GSD, "y0": -(crow - reach + r0) * GSD,
                              "zone": p.zone, "area_ha": round(p.area_ha, 4), "n_px": p.n_px,
                              "n_px_in_crop": int(mask.sum()), "purity": round(purity(mask, ps), 4),
                              "purity_centred": round(purity(mask_c, ps), 4)})
            flush(ps)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,}/{len(parcels):,} parcels")
    for ps in CROP_PATCH_SIZES:
        flush(ps, final=True)
        info = {"bands": crop_bands, "s2_bands": S2_BANDS, "s1_bands": [f"VV_{orbit}", f"VH_{orbit}"],
                "s1_scale": S1_SCALE, "s2_scale": "surface reflectance x 10000", "nodata": NODATA,
                "chip_px": CROP_PX, "gsd_m": GSD, "patch_px": ps, "start_month": INFO["start_month"],
                "n_months": N_MONTHS, "window": WINDOW, "crs_epsg": INFO["crs_epsg"],
                "alignment": f"crop shifted by {ps // 2} px: parcel centroid at the centre of a {ps}-px patch",
                "source": "03_build_tiles.py tile store"}
        # The study/02 embedding scripts read Sentinel-1 as "VV" / "VH": name the chosen orbit's bands that way.
        info["bands"] = S2_BANDS + ["VV", "VH", "S2_NCLEAR"]
        info["s1_bands"], info["s1_orbit"] = ["VV", "VH"], orbit
        (OUT_DIR / f"crops_p{ps}" / "chips_info.json").write_text(json.dumps(info, indent=2))

    if MAKE_FEATURES:
        for k, rows in feats.items():
            pd.DataFrame(rows).to_csv(OUT_DIR / f"parcel_features_{k}.csv", index=False)
    if MAKE_PIXELSETS and px_list:
        np.savez(OUT_DIR / "pixelsets.npz", pixels=np.concatenate(px_list), edge=np.concatenate(edge_list),
                 offsets=np.array(offsets), parcel_id=np.array([i for i, _ in px_ids]),
                 label=np.array([l for _, l in px_ids]), bands=np.array(MB))

    # QA tables.
    qa = []
    for ps in CROP_PATCH_SIZES:
        idx = pd.read_csv(OUT_DIR / f"crops_p{ps}" / "index.csv")
        for col in ("purity_centred", "purity"):
            qa.append({"patch_px": ps, "crop": "centred" if col == "purity_centred" else "aligned",
                       "median_purity": round(idx[col].median(), 3),
                       "pct_parcels_purity_below_0.5": round(100 * (idx[col] < 0.5).mean(), 1)})
    pd.DataFrame(qa).to_csv(OUT_DIR / "qa_purity.csv", index=False)
    print(f"\nPurity of pooled patches (median):\n{pd.DataFrame(qa).to_string(index=False)}")
    if MAKE_FEATURES and feats["all"]:
        f = pd.DataFrame(feats["all"]).merge(parcels[["parcel_id", "zone", "class_name"]], on="parcel_id")
        cols = [f"S2_NCLEAR_m{k + 1:02d}" for k in range(N_MONTHS)]
        obs = f.groupby(["zone", "ag_season", "class_name"])[cols].mean().round(1)
        obs.to_csv(OUT_DIR / "qa_clear_obs.csv")
        print(f"\nMean clear Sentinel-2 observations per month (Sep..Aug):\n{obs.to_string()}")
    (OUT_DIR / "views_info.json").write_text(json.dumps({
        "s1_orbit_for_crops_and_pixelsets": orbit, "crop_px": CROP_PX, "crop_patch_sizes": CROP_PATCH_SIZES,
        "n_parcels": len(parcels), "missing_tiles": sorted(store.missing)}, indent=2))
    if store.missing:
        print(f"WARNING: {len(store.missing)} tiles were missing (crops near them are partly NODATA)")
    print(f"Saved views in {OUT_DIR}")


if __name__ == "__main__":
    main()
