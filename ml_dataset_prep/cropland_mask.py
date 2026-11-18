"""
cropland_mask.py  —  read the cropland mask of 00_make_cropland_mask.py for labelled parcels (used by scripts 01)
===============================================================================================================

The labelled parcels are checked against the cropland mask of THEIR agricultural year, but never dropped: a
parcel outside the mask is a real cropland parcel that the cropland products miss, so it shows the omission of
the mask (and the parcel stays in training, otherwise the training set would depend on the mask).

Columns added to the parcels
----------------------------
  mask_frac       share of the parcel's 10 m pixels (pixel centre inside the polygon; all touched pixels for
                  parcels smaller than one pixel) that are cropland in the mask; NaN outside the mapping area
  mask_votes      mean number of products that call the parcel's pixels cropland (0 - number of products)
  in_mask         mask_frac >= IN_MASK_FRAC (True / False; False also outside the mapping area)
  mask_<product>  share of the parcel's pixels that one product calls cropland (e.g. mask_worldcover)
"""
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask

from regions import dataset_dir

IN_MASK_FRAC = 0.5                 # a parcel counts as "in the mask" if at least half of its pixels are
NODATA = 65535


def mask_dir(region, ag_season):
    return dataset_dir(region) / "00_cropland_mask" / ag_season.replace("/", "-")


def parcel_mask_stats(parcels, region):
    """Add mask_frac, mask_votes, in_mask, mask_<product> to the parcels (GeoDataFrame with ag_season)."""
    out = parcels.copy()
    seasons = sorted(out["ag_season"].unique())
    missing = [s for s in seasons if not (mask_dir(region, s) / "cropland_mask.vrt").exists()]
    if missing:
        raise FileNotFoundError(f"no cropland mask for {missing}: run 00_make_cropland_mask.py first "
                                f"(REGION = {region!r}, SEASONS including {missing})")
    info = json.loads((mask_dir(region, seasons[0]) / "mask_info.json").read_text())
    products = {p: v["bit"] for p, v in info["products"].items()}
    cols = ["mask_frac", "mask_votes"] + [f"mask_{p}" for p in products]
    for c in cols:
        out[c] = np.nan
    for season in seasons:
        rows = np.flatnonzero((out["ag_season"] == season).to_numpy())
        with rasterio.open(mask_dir(region, season) / "cropland_mask.vrt") as src:
            geoms = out.iloc[rows].to_crs(src.crs).geometry
            for i, geom in zip(rows, geoms):
                v = None
                for all_touched in (False, True):                  # tiny parcels: all touched pixels
                    try:
                        arr, _ = rio_mask(src, [geom], crop=True, filled=False, all_touched=all_touched)
                    except ValueError:                             # parcel outside the mask blocks
                        break
                    keep = ~np.ma.getmaskarray(arr[0]) & (arr[0].filled(NODATA) != NODATA)
                    if keep.any():
                        v = arr[:, keep].data
                        break
                if v is None:
                    continue
                crop, votes, bits = v[0], v[1], v[2]
                out.iat[i, out.columns.get_loc("mask_frac")] = float((crop == 1).mean())
                out.iat[i, out.columns.get_loc("mask_votes")] = float(votes.mean())
                for p, b in products.items():
                    out.iat[i, out.columns.get_loc(f"mask_{p}")] = float(((bits >> b) & 1).mean())
    for c in cols:
        out[c] = out[c].round(3)
    out["in_mask"] = out["mask_frac"] >= IN_MASK_FRAC
    return out


def mask_check_table(parcels):
    """Per agricultural year and class: parcels inside the mapping area, share in the mask and per product."""
    products = [c for c in parcels.columns if c.startswith("mask_") and c not in ("mask_frac", "mask_votes")]
    rows = []
    for (season, cls), g in parcels.groupby(["ag_season", "class_name"]):
        a = g[g["mask_frac"].notna()]
        row = {"ag_season": season, "class_name": cls, "parcels": len(g), "in_mapping_area": len(a),
               "in_mask_pct": round(100 * a["in_mask"].mean(), 1) if len(a) else np.nan,
               "mean_mask_frac": round(a["mask_frac"].mean(), 3) if len(a) else np.nan}
        for c in products:
            row[f"{c[5:]}_pct"] = round(100 * (a[c] >= IN_MASK_FRAC).mean(), 1) if len(a) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
