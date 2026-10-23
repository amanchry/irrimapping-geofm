"""
05a_embed_clay.py  —  Clay v1.5 embeddings for every parcel (frozen encoder)
============================================================================

What it does
------------
Turns each parcel's chips (04_download_chips.py) into Clay embeddings: one 1024-number vector
per month of its agricultural year. The encoder is frozen (no training here); the embeddings are
classified in 06_train_evaluate.py with the same models and spatial validation as every other input.

The model
---------
Clay v1.5 (Clay Foundation; https://huggingface.co/made-with-clay/Clay, doi:10.57967/hf/1537):
a vision transformer trained as a masked autoencoder on Sentinel-2, Landsat, Sentinel-1, NAIP and
other imagery. It cuts an image into 8 x 8-pixel patches (80 m at 10 m) and returns one embedding
per patch plus a class token. Clay takes ONE date per image, so the 12 monthly chips are embedded
one by one. Besides the pixels it receives each band's wavelength, the pixel size, the date (week
of year) and the location (lat/lon), encoded as in the Clay tutorials.

Steps
-----
1. Load the frozen Clay encoder (checkpoints/clay/clay-v1.5.ckpt + metadata.yaml).
2. For each batch of parcels: take the 10 Sentinel-2 bands Clay knows (blue ... swir22 =
   B2 B3 B4 B5 B6 B7 B8 B8A B11 B12), normalise with Clay's band mean/std; cloudy pixels -> 0
   (= the band mean after normalisation).
3. Run the encoder on every monthly chip; put the patch embeddings back on their 8 x 8 grid.
4. Parcel embedding per month = mean of the patches covering the parcel, weighted by the share of
   each patch inside the parcel (mask from script 04).
5. Save in batches (resume possible) and merge into one file.

Install (once, in the condaml environment)
------------------------------------------
    pip install git+https://github.com/Clay-foundation/model.git
    mkdir -p checkpoints/clay
    curl -L -o checkpoints/clay/clay-v1.5.ckpt https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
    curl -L -o checkpoints/clay/metadata.yaml https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml

Output
------
outputs/05_embeddings/clay_<WINDOW>.npz
    parcel_id [N], label [N], ag_season [N], steps [12] (month names),
    embeddings [N, 12, 1024] float16, valid [N, 12] (False = month without any clear pixel)

Run (from the project folder)
-----------------------------
    python pipeline/05a_embed_clay.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
MODEL_NAME = "clay"
WINDOW = "agyear"                                          # must match 04_download_chips.py
CHECKPOINT = ROOT / "checkpoints" / "clay" / "clay-v1.5.ckpt"
METADATA = ROOT / "checkpoints" / "clay" / "metadata.yaml"
BATCH_PARCELS = 16            # parcels per step (x 12 months = chips per step)
EMBED_BATCH = 64              # chips per forward pass (lower it if memory runs out)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================================

# ============================ SHARED CHIP CODE ============================
# Reading the chips of 04_download_chips.py, pooling patch embeddings inside the parcel and
# saving. This block is identical in the four 05*_embed_*.py scripts.
CHIP_DIR = ROOT / "outputs" / "04_chips" / WINDOW
OUT_DIR = ROOT / "outputs" / "05_embeddings"
PART_DIR = OUT_DIR / f"_parts_{MODEL_NAME}_{WINDOW}"
OUT_FILE = OUT_DIR / f"{MODEL_NAME}_{WINDOW}.npz"


def read_chip_info():
    """Band names, months, NODATA value and scales written by 04_download_chips.py, plus the parcel index."""
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


# ============================ CLAY-SPECIFIC CODE ===========================
# Clay's Sentinel-2 L2A description (configs/metadata.yaml, sentinel-2-l2a):
# (Clay band name, Sentinel-2 band, mean, std, central wavelength in micrometres)
CLAY_BANDS = [
    ("blue", "B2", 1105.0, 1809.0, 0.493), ("green", "B3", 1355.0, 1757.0, 0.560),
    ("red", "B4", 1552.0, 1888.0, 0.665), ("rededge1", "B5", 1887.0, 1870.0, 0.704),
    ("rededge2", "B6", 2422.0, 1732.0, 0.740), ("rededge3", "B7", 2630.0, 1697.0, 0.783),
    ("nir", "B8", 2743.0, 1742.0, 0.842), ("nir08", "B8A", 2785.0, 1648.0, 0.865),
    ("swir16", "B11", 2388.0, 1470.0, 1.610), ("swir22", "B12", 1835.0, 1379.0, 2.190),
]
S2_NAMES = [b[1] for b in CLAY_BANDS]
MEAN = np.array([b[2] for b in CLAY_BANDS], dtype=np.float32)[None, None, :, None, None]
STD = np.array([b[3] for b in CLAY_BANDS], dtype=np.float32)[None, None, :, None, None]
WAVES = [b[4] for b in CLAY_BANDS]
GSD_M = 10.0
PATCH_PX = 8                  # Clay v1.5 patch size


def load_encoder():
    """Frozen Clay encoder; mask_ratio=0 and shuffle=False keep every patch, in grid order."""
    from claymodel.module import ClayMAEModule
    module = ClayMAEModule.load_from_checkpoint(str(CHECKPOINT), metadata_path=str(METADATA),
                                                mask_ratio=0.0, shuffle=False)
    module.eval()
    return module.model.encoder.to(DEVICE)


ENCODER = None


def time_code(date):
    """Week of year and hour as sine/cosine (Clay / stacchip convention); composites -> 11:00 UTC."""
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = 11 * 2 * np.pi / 24
    return [np.sin(week), np.cos(week), np.sin(hour), np.cos(hour)]


def latlon_code(lat, lon):
    """Latitude and longitude as sine/cosine (Clay convention)."""
    lat, lon = np.radians(lat), np.radians(lon)
    return [np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)]


@torch.no_grad()
def encode(chips, times, latlons):
    """chips [N, 10, H, W] -> patch grid [N, n, n, 1024]."""
    n = chips.shape[-1] // PATCH_PX
    out = []
    for s in range(0, len(chips), EMBED_BATCH):
        sl = slice(s, s + EMBED_BATCH)
        datacube = {
            "pixels": torch.from_numpy(chips[sl]).to(DEVICE),
            "time": torch.tensor(np.array(times[sl]), dtype=torch.float32, device=DEVICE),
            "latlon": torch.tensor(np.array(latlons[sl]), dtype=torch.float32, device=DEVICE),
            "gsd": torch.tensor(GSD_M, device=DEVICE),
            "waves": torch.tensor(WAVES, device=DEVICE),
        }
        tokens, kept_idx, _, _ = ENCODER(datacube)               # tokens[:, 0] = class token
        patches = tokens[:, 1:, :]
        grid = torch.zeros(patches.shape[0], n * n, patches.shape[2], device=DEVICE)
        grid.scatter_(1, kept_idx.long().unsqueeze(-1).expand_as(patches), patches)   # back to grid order
        out.append(grid.view(-1, n, n, patches.shape[2]).float().cpu().numpy())
    return np.concatenate(out)


def embed_batch(ids, chips, masks, info, index):
    """Chips of a batch of parcels -> embeddings [B, months, 1024] and valid [B, months]."""
    x = select_bands(chips, info, S2_NAMES)                     # [B, M, 10, H, W] reflectance x 10000
    valid = month_is_valid(x)
    x = np.nan_to_num((x - MEAN) / STD, nan=0.0)                # Clay normalisation; cloud -> band mean
    b, m = x.shape[:2]
    times, latlons = [], []
    for pid in ids:
        row = index.loc[pid]
        times += [time_code(d) for d in month_dates(row["ag_season"], info)]
        latlons += [latlon_code(row["lat"], row["lon"])] * m
    grid = encode(x.reshape(b * m, *x.shape[2:]), times, latlons)          # [B*M, n, n, D]
    weights = np.repeat(patch_weights(masks, PATCH_PX), m, axis=0)          # same mask every month
    emb = pool_patches(grid, weights).reshape(b, m, -1)
    return emb, valid


def main():
    global ENCODER
    ENCODER = load_encoder()
    run(embed_batch, month_names)


if __name__ == "__main__":
    main()
