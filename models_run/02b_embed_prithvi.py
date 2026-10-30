"""
02b_embed_prithvi.py  —  Prithvi-EO-2.0 embeddings for every parcel (frozen encoder)
====================================================================================

What it does
------------
Turns each parcel's chips (ml_dataset_prep/04_make_views.py) into Prithvi-EO-2.0 embeddings: one
1024-number vector per month of its agricultural year. The encoder is frozen; the embeddings are
classified in 03_train_evaluate.py like every other input.

The model
---------
Prithvi-EO-2.0-300M-TL (IBM & NASA; Szwarcman et al. 2024, arXiv:2412.02732;
https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL): a multi-temporal vision
transformer pretrained on Harmonized Landsat Sentinel-2 (HLS) time series. From its model card:
6 bands (Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2 = Sentinel-2 B2 B3 B4 B8A B11 B12),
4 frames per sample, 16 x 16-pixel patches, 1024-dimensional tokens, reflectance x 10000
normalised with its own band mean/std, plus optional time (year, day of year) and location
(lat, lon) inputs.

Choices made here (state them in the thesis)
---------------------------------------------
* Frames: the 12 months are fed in FRAMES_PER_PASS consecutive months at a time (default 4, as in
  pretraining): Sep-Dec, Jan-Apr, May-Aug. Each frame keeps its own tokens, so the output is still
  one embedding per month.
* Pixel size: Prithvi was pretrained on 30 m HLS. The chips are 10 m Sentinel-2, used as they are
  (patch = 160 m, closer to parcel size than 480 m). Prithvi's position encoding is rescaled to the
  64 x 64 input automatically.

Steps
-----
1. Load the frozen encoder through TerraTorch (downloads the weights from Hugging Face once).
2. For each batch of parcels: take bands B2 B3 B4 B8A B11 B12, normalise with Prithvi's mean/std,
   cloudy pixels -> 0 (= band mean).
3. Run the encoder on each group of FRAMES_PER_PASS months with their dates and the parcel location.
4. Per month: weighted mean of the 16 x 16-pixel patches covering the parcel (parcel mask stored with the chips).
5. Save in batches (resume possible) and merge into one file.

Install (once, in the condaml environment)
------------------------------------------
    pip install terratorch          (already in environment.yml)

Output
------
outputs/study/02_embeddings/prithvi_<WINDOW>_tiles.npz  (same layout as the other models)
    embeddings [N, 12, 1024] float16, valid [N, 12], steps = month names

Run (from the project folder)
-----------------------------
    python models_run/02b_embed_prithvi.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
MODEL_NAME = "prithvi"
WINDOW = "agyear"                                          # must match 03_build_tiles.py / 04_make_views.py
BACKBONE = "prithvi_eo_v2_300_tl"                          # TerraTorch name of Prithvi-EO-2.0-300M-TL
FRAMES_PER_PASS = 4                                        # months per forward pass (pretraining used 4)
USE_COORDS = True                                          # give date and location to the model
BATCH_PARCELS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================================

# ============================ SHARED CHIP CODE ============================
# Reading the chips of ml_dataset_prep/04_make_views.py, pooling patch embeddings inside the parcel and
# saving. This block is identical in the four 02*_embed_*.py scripts.
# Input chips. "tiles" = patch-aligned chips cut by 04_make_views.py from the shared tile store
# (same pixels as the parcel-feature baseline); "legacy" = the older, separately downloaded chip store (no longer kept).
CHIP_SOURCE = "tiles"
CHIP_PATCH_PX = 16                                         # 16 x 16-pixel patches (final AgriFM stride is also 16 px)
if CHIP_SOURCE == "tiles":
    CHIP_DIR = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW / "views" / f"chips_p{CHIP_PATCH_PX}"
    TAG = f"{WINDOW}_tiles"                                 # output name: <model>_agyear_tiles.npz
else:
    CHIP_DIR = ROOT / "outputs" / "legacy" / "04_chips" / WINDOW
    TAG = WINDOW                                            # output name: <model>_agyear.npz (old results)
OUT_DIR = ROOT / "outputs" / "study" / "02_embeddings"
PART_DIR = OUT_DIR / f"_parts_{MODEL_NAME}_{TAG}"
OUT_FILE = OUT_DIR / f"{MODEL_NAME}_{TAG}.npz"


def read_chip_info():
    """Band names, months, NODATA value and scales written with the chips (chips_info.json), plus the parcel index."""
    info = json.loads((CHIP_DIR / "chips_info.json").read_text())
    index = pd.read_csv(CHIP_DIR / "index.csv", dtype={"parcel_id": str}).drop_duplicates("parcel_id")
    return info, index.set_index("parcel_id")


def iter_chip_parts(done):
    """Yield (parcel_ids, chips, masks) for every chip part file, without parcels already embedded."""
    for f in sorted(CHIP_DIR.glob("part_*.npz")):
        d = np.load(f)
        keep = ~np.isin(d["parcel_id"], list(done))
        if keep.any():
            yield d["parcel_id"][keep].astype(str), d["chips"][keep], d["mask"][keep]


def select_bands(chips, info, names, scale=1.0):
    """Pick bands by name from int16 chips [B, months, bands, H, W] -> float32 with NaN where NODATA."""
    idx = [info["bands"].index(n) for n in names]
    sub = chips[:, :, idx].astype(np.float32)
    sub[chips[:, :, idx] == info["nodata"]] = np.nan
    return sub / scale


def month_is_valid(x):
    """A month counts as observed if at least one pixel of the chip has data. x: [B, months, bands, H, W]."""
    return ~np.isnan(x).all(axis=(2, 3, 4))


def patch_weights(masks, patch_px):
    """Share of each patch covered by the parcel. masks [B, H, W] (1 = parcel) -> [B, n, n].
    A parcel that covers no patch at all gets the patch under the chip centre (the parcel centroid)."""
    b, h, w = masks.shape
    n = h // patch_px
    m = masks[:, : n * patch_px, : n * patch_px].astype(np.float32)
    weights = m.reshape(b, n, patch_px, n, patch_px).mean(axis=(2, 4))
    empty = weights.sum(axis=(1, 2)) == 0
    weights[empty, n // 2, n // 2] = 1.0
    return weights


def pool_patches(grid, weights):
    """Weighted mean of patch embeddings inside the parcel. grid [B, n, n, D], weights [B, n, n] -> [B, D]."""
    w = weights / weights.sum(axis=(1, 2), keepdims=True)
    return np.einsum("bij,bijd->bd", w, grid)


def save_part(n_part, ids, embeddings, valid):
    np.savez(PART_DIR / f"part_{n_part:05d}.npz", parcel_id=np.array(ids),
             embeddings=embeddings.astype(np.float16), valid=valid)


def done_parcels():
    done = set()
    for f in PART_DIR.glob("part_*.npz"):
        done |= set(np.load(f)["parcel_id"].astype(str))
    return done, len(list(PART_DIR.glob("part_*.npz")))


def merge_parts(index, steps):
    """All part files -> one file: parcel_id, label, ag_season, steps, embeddings [N, steps, D], valid [N, steps]."""
    ids, embs, valid = [], [], []
    for f in sorted(PART_DIR.glob("part_*.npz")):
        d = np.load(f)
        ids += list(d["parcel_id"].astype(str))
        embs.append(d["embeddings"])
        valid.append(d["valid"])
    meta = index.loc[ids]
    np.savez_compressed(OUT_FILE, parcel_id=np.array(ids), label=meta["label"].to_numpy(),
                        ag_season=meta["ag_season"].to_numpy().astype(str), steps=np.array(steps),
                        embeddings=np.concatenate(embs), valid=np.concatenate(valid),
                        model=np.array(MODEL_NAME))
    e = np.concatenate(embs)
    print(f"Saved {OUT_FILE.name}: {e.shape[0]} parcels x {e.shape[1]} steps x {e.shape[2]} dimensions")


def month_dates(ag_season, info):
    """Middle of every month of the parcel's agricultural year (the composites have no single date)."""
    start = pd.Timestamp(year=int(ag_season[:4]), month=info["start_month"], day=1)
    return [start + pd.DateOffset(months=k) + pd.Timedelta(days=14) for k in range(info["n_months"])]


def month_names(info):
    start = pd.Timestamp(year=2023, month=info["start_month"], day=1)
    return [(start + pd.DateOffset(months=k)).strftime("%b") for k in range(info["n_months"])]


def run(embed_batch_fn, steps_fn):
    """Main loop shared by all models: read chip parts, embed, save parts, merge."""
    PART_DIR.mkdir(parents=True, exist_ok=True)
    info, index = read_chip_info()
    done, n_part = done_parcels()
    total = len(index)
    print(f"{MODEL_NAME}: {total} parcels with chips, {len(done)} already embedded | device {DEVICE}")
    t_start = time.time()
    for ids, chips, masks in iter_chip_parts(done):
        for s in range(0, len(ids), BATCH_PARCELS):
            t0 = time.time()
            sl = slice(s, s + BATCH_PARCELS)
            emb, valid = embed_batch_fn(ids[sl], chips[sl], masks[sl], info, index)
            save_part(n_part, ids[sl], emb, valid)
            n_part += 1
            done |= set(ids[sl])
            print(f"  {len(done)}/{total} parcels ({time.time() - t0:.0f}s for {len(ids[sl])})")
    merge_parts(index, steps_fn(info))
    print(f"Done in {(time.time() - t_start) / 60:.1f} min")
# ========================== END SHARED CHIP CODE ==========================


# ========================== PRITHVI-SPECIFIC CODE ==========================
# Prithvi-EO-2.0 bands and normalisation (model card config.json):
# HLS B02 B03 B04 B05(narrow NIR) B06(SWIR1) B07(SWIR2) = Sentinel-2 B2 B3 B4 B8A B11 B12
S2_NAMES = ["B2", "B3", "B4", "B8A", "B11", "B12"]
MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)[None, None, :, None, None]
STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)[None, None, :, None, None]
PATCH_PX = 16


def load_encoder():
    """Frozen Prithvi-EO-2.0 encoder built for FRAMES_PER_PASS frames (weights from Hugging Face)."""
    from terratorch.registry import BACKBONE_REGISTRY
    model = BACKBONE_REGISTRY.build(BACKBONE, pretrained=True, num_frames=FRAMES_PER_PASS)
    return model.eval().to(DEVICE)


ENCODER = None


@torch.no_grad()
def encode(x, temporal, location):
    """x [B, 6, T, H, W], temporal [B, T, 2] (year, day of year), location [B, 2] (lat, lon)
    -> patch grid per frame [B, T, n, n, 1024]. Tokens come in (frame, row, column) order after the class token."""
    b, _, t, h, w = x.shape
    n = h // PATCH_PX
    kwargs = {}
    if USE_COORDS:
        kwargs = {"temporal_coords": torch.tensor(temporal, dtype=torch.float32, device=DEVICE),
                  "location_coords": torch.tensor(location, dtype=torch.float32, device=DEVICE)}
    features = ENCODER.forward_features(torch.from_numpy(x).to(DEVICE), **kwargs)
    tokens = features[-1][:, 1:, :]                          # last (normalised) layer, without class token
    return tokens.reshape(b, t, n, n, -1).float().cpu().numpy()


def embed_batch(ids, chips, masks, info, index):
    """Chips of a batch of parcels -> embeddings [B, months, 1024] and valid [B, months]."""
    x = select_bands(chips, info, S2_NAMES)                     # [B, M, 6, H, W]
    valid = month_is_valid(x)
    x = np.nan_to_num((x - MEAN) / STD, nan=0.0)
    b, m = x.shape[:2]
    dates = [month_dates(index.loc[pid, "ag_season"], info) for pid in ids]          # per parcel, per month
    location = np.array([[index.loc[pid, "lat"], index.loc[pid, "lon"]] for pid in ids])
    weights = patch_weights(masks, PATCH_PX)
    parts = []
    for f0 in range(0, m, FRAMES_PER_PASS):
        frames = slice(f0, f0 + FRAMES_PER_PASS)
        clip = np.ascontiguousarray(x[:, frames].transpose(0, 2, 1, 3, 4))        # [B, 6, T, H, W]
        temporal = np.array([[[d.year, d.dayofyear] for d in dd[frames]] for dd in dates])
        grid = encode(clip, temporal, location)                                     # [B, T, n, n, D]
        t = grid.shape[1]
        pooled = pool_patches(grid.reshape(b * t, *grid.shape[2:]), np.repeat(weights, t, axis=0))
        parts.append(pooled.reshape(b, t, -1))
    emb = np.concatenate(parts, axis=1)
    return emb, valid


def main():
    global ENCODER
    ENCODER = load_encoder()
    run(embed_batch, month_names)


if __name__ == "__main__":
    main()
