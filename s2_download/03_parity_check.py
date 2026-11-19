"""
03_parity_check.py  —  STAC tiles vs Earth Engine tiles on the same 128 px tiles
=================================================================================

Question: are the monthly Sentinel-2 composites from the STAC download (02_download_s2_stac.py, TILE_SET = "parity")
close enough to the Earth Engine tiles of the parcel store (03_build_tiles.py) that models trained on the Earth
Engine tiles can map with the STAC tiles?

What it compares (tile by tile, same tile_id = same pixels)
-----------------------------------------------------------
1. Reflectance, per band and month, on pixels that are clear in both: bias (STAC - EE), mean absolute difference,
   RMSE, correlation, bias relative to the EE mean.
2. Cloud masks: share of pixel-months clear in both, only in Earth Engine (Cloud Score+), only in STAC (SCL),
   and the mean number of clear observations per month (S2_NCLEAR) of each.

Reading the result: bias and RMSE of a few percent of the reflectance and similar clear-month shares mean the
sources are interchangeable. The decisive check is the model score on both inputs (next step, once a trained
model is saved); if it drops, the training tiles are rebuilt from STAC.

Outputs (outputs/<region>/s2_download/parity/<season>/comparison/)
------------------------------------------------------------------
by_band.csv, by_band_month.csv, cloud_mask.csv, parity.png

Run
---
    python s2_download/03_parity_check.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pixelwise_run"))   # regions.py is shared
from regions import dataset_dir  # noqa: E402

# ================================ SETTINGS ================================
REGION = "morocco"                       # "morocco" | "utah"
SEASON = "2023/24"
SAMPLE_PX = 2000                         # pixels per tile kept for the scatter plots
SEED = 42
# ==========================================================================

ROOT = Path(__file__).resolve().parents[1]
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
N_MONTHS, NODATA = 12, -32768
TAG = SEASON.replace("/", "-")
STAC_DIR = ROOT / "outputs" / REGION / "s2_download" / "parity" / TAG / "tiles"
EE_DIR = dataset_dir(REGION) / "03_tiles_128px" / "agyear" / "tiles"
OUT = ROOT / "outputs" / REGION / "s2_download" / "parity" / TAG / "comparison"


def read(path):
    """[12 months, 12 bands, H, W] float (NaN = no clear observation) and [12, H, W] S2_NCLEAR."""
    with rasterio.open(path) as src:
        desc = list(src.descriptions)
        idx = [desc.index(f"{b}_m{m:02d}") + 1 for m in range(1, N_MONTHS + 1) for b in S2_BANDS]
        ncl = [desc.index(f"S2_NCLEAR_m{m:02d}") + 1 for m in range(1, N_MONTHS + 1)]
        x = src.read(idx).astype(np.float32).reshape(N_MONTHS, len(S2_BANDS), src.height, src.width)
        n = src.read(ncl).astype(np.float32)
    x[x == NODATA] = np.nan
    return x, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [(p, EE_DIR / p.name) for p in sorted(STAC_DIR.glob("*_img.tif")) if (EE_DIR / p.name).exists()]
    if not pairs:
        raise FileNotFoundError(f"no tiles in both {STAC_DIR} and {EE_DIR}")
    rng = np.random.default_rng(SEED)
    acc = {}                                        # (band, month) -> sums
    mask_rows, samples = [], []
    for sp, ep in pairs:
        s, ns = read(sp)
        e, ne = read(ep)
        vs, ve = ~np.isnan(s[:, 3]), ~np.isnan(e[:, 3])             # clear month-pixels (B4)
        mask_rows.append({"tile_id": sp.name[:-8], "both": float((vs & ve).mean()), "ee_only": float((ve & ~vs).mean()),
                          "stac_only": float((vs & ~ve).mean()), "neither": float((~vs & ~ve).mean()),
                          "nclear_ee": float(ne.mean()), "nclear_stac": float(ns.mean())})
        for m in range(N_MONTHS):
            for bi, b in enumerate(S2_BANDS):
                ok = ~np.isnan(s[m, bi]) & ~np.isnan(e[m, bi])
                if not ok.any():
                    continue
                a, c = s[m, bi][ok].astype(np.float64), e[m, bi][ok].astype(np.float64)
                k = acc.setdefault((b, m + 1), np.zeros(6))
                k += [len(a), (a - c).sum(), np.abs(a - c).sum(), ((a - c) ** 2).sum(), c.sum(), (a * c).sum()]
                acc.setdefault((b, m + 1, "sq"), np.zeros(4))
                acc[(b, m + 1, "sq")] += [a.sum(), (a ** 2).sum(), (c ** 2).sum(), 0]
        both = np.argwhere(vs & ve)                                   # [k, 3] month, row, col
        if len(both):
            pick = both[rng.choice(len(both), min(SAMPLE_PX, len(both)), replace=False)]
            samples.append(np.stack([s[pick[:, 0], :, pick[:, 1], pick[:, 2]], e[pick[:, 0], :, pick[:, 1], pick[:, 2]]], 1))

    rows = []
    for (b, m), v in [(k, v) for k, v in acc.items() if len(k) == 2]:
        n, sd, sad, ssd, se, sse = v
        sa, saa, scc, _ = acc[(b, m, "sq")]
        r = (sse / n - sa / n * se / n) / np.sqrt(max(saa / n - (sa / n) ** 2, 1e-9) * max(scc / n - (se / n) ** 2, 1e-9))
        rows.append({"band": b, "month": m, "n_px": int(n), "bias": sd / n, "mad": sad / n, "rmse": np.sqrt(ssd / n),
                     "ee_mean": se / n, "bias_pct": 100 * sd / se, "r": r})
    bm = pd.DataFrame(rows).sort_values(["band", "month"])
    bm.to_csv(OUT / "by_band_month.csv", index=False)
    w = bm.assign(wb=bm["bias"] * bm["n_px"], wm=bm["mad"] * bm["n_px"], ws=bm["rmse"] ** 2 * bm["n_px"],
                  we=bm["ee_mean"] * bm["n_px"], wr=bm["r"] * bm["n_px"]).groupby("band", sort=False)
    bb = pd.DataFrame({"n_px": w["n_px"].sum(), "bias": w["wb"].sum() / w["n_px"].sum(),
                       "mad": w["wm"].sum() / w["n_px"].sum(), "rmse": np.sqrt(w["ws"].sum() / w["n_px"].sum()),
                       "ee_mean": w["we"].sum() / w["n_px"].sum(), "r_mean": w["wr"].sum() / w["n_px"].sum()})
    bb["bias_pct"] = 100 * bb["bias"] / bb["ee_mean"]
    bb = bb.reindex(S2_BANDS).reset_index()
    bb.to_csv(OUT / "by_band.csv", index=False)
    cm = pd.DataFrame(mask_rows)
    cm.to_csv(OUT / "cloud_mask.csv", index=False)

    smp = np.concatenate(samples)                                     # [N, 2, 12]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    for a, b in zip(ax[:3], ["B4", "B8", "B11"]):
        i = S2_BANDS.index(b)
        a.hexbin(smp[:, 1, i], smp[:, 0, i], gridsize=60, bins="log", cmap="viridis", extent=(0, 6000, 0, 6000))
        a.plot([0, 6000], [0, 6000], "r-", lw=0.8)
        row = bb[bb["band"] == b].iloc[0]
        a.set(xlabel="Earth Engine (reflectance x 10000)", ylabel="STAC", xlim=(0, 6000), ylim=(0, 6000),
              title=f"{b}: bias {row['bias']:+.0f} ({row['bias_pct']:+.1f}%), RMSE {row['rmse']:.0f}, r {row['r_mean']:.3f}")
    piv = bm.pivot(index="band", columns="month", values="bias_pct").reindex(S2_BANDS)
    im = ax[3].imshow(piv.values, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
    ax[3].set(yticks=range(len(S2_BANDS)), yticklabels=S2_BANDS, xticks=range(N_MONTHS),
              xticklabels=range(1, N_MONTHS + 1), xlabel="month of the agricultural year",
              title="bias STAC - EE (% of EE mean)")
    fig.colorbar(im, ax=ax[3], shrink=0.8)
    fig.suptitle(f"{REGION} {SEASON}: {len(pairs)} tiles. Clear in both {cm['both'].mean():.0%}, only EE "
                 f"{cm['ee_only'].mean():.0%}, only STAC {cm['stac_only'].mean():.0%}; clear obs. per month "
                 f"EE {cm['nclear_ee'].mean():.1f}, STAC {cm['nclear_stac'].mean():.1f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "parity.png", dpi=120)
    plt.close(fig)

    summary = {"tiles": len(pairs), **{k: round(float(cm[k].mean()), 4) for k in
                                         ["both", "ee_only", "stac_only", "neither", "nclear_ee", "nclear_stac"]}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(bb.round(3).to_string(index=False))
    print(f"Saved in {OUT}")


if __name__ == "__main__":
    main()
