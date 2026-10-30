"""
02c_embed_terramind.py  —  TerraMind embeddings for every parcel (frozen encoder, Sentinel-1 + Sentinel-2)
===========================================================================================================

What it does
------------
Turns each parcel's crops (ml_dataset_prep/04_make_views.py) into TerraMind embeddings: one 768-number vector
per month, computed from Sentinel-2 AND Sentinel-1 together. The encoder is frozen; the embeddings
are classified in 03_train_evaluate.py like every other input.

The model
---------
TerraMind-1.0-base (IBM & ESA; Jakubik et al. 2025, arXiv:2504.11171;
https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base): a multimodal transformer pretrained
on nine Earth-observation modalities. Used here as a backbone through TerraTorch with two inputs:
  S2L2A  12 Sentinel-2 bands B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12 (reflectance x 10000)
  S1GRD  Sentinel-1 VV, VH in dB
Each input is standardised with TerraMind's pretraining mean/std (TerraTorch terramind_register.py);
the backbone does not do this itself. Patches are 16 x 16 pixels (160 m); the position encoding is
interpolated to the 64 x 64 chips. Tokens of the two modalities are averaged (merge_method "mean").
TerraMind takes one date per input, so the 12 months are embedded one by one.

Steps
-----
1. Load the frozen TerraMind backbone (downloads the weights from Hugging Face once).
2. For each batch of parcels and each month: standardise S2 and S1, cloudy/missing pixels -> 0.
3. Run the encoder; take the last layer's tokens on their 4 x 4 grid.
4. Weighted mean of the patches covering the parcel (parcel mask stored with the crops).
5. Save in batches (resume possible) and merge into one file.

Install: terratorch (already in environment.yml).

Output
------
outputs/study/02_embeddings/terramind_<WINDOW>_tiles.npz  (same layout as the other models)
    embeddings [N, 12, 768] float16, valid [N, 12], steps = month names

Run (from the project folder)
-----------------------------
    python models_run/02c_embed_terramind.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
MODEL_NAME = "terramind"
WINDOW = "agyear"                                          # must match 03_build_tiles.py / 04_make_views.py
BACKBONE = "terramind_v1_base"
MODALITIES = ["S2L2A", "S1GRD"]                            # ["S2L2A"] = optical only (ablation)
BATCH_PARCELS = 16
EMBED_BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================================

# ============================ SHARED CHIP CODE ============================
# Reading the crops of ml_dataset_prep/04_make_views.py, pooling patch embeddings inside the parcel and
# saving. This block is identical in the four 05*_embed_*.py scripts.
# Input chips. "tiles" = patch-aligned crops cut by 04_make_views.py from the shared tile store
# (same pixels as the parcel-feature baseline); "chips" = the older 04_download_chips.py store.
CHIP_SOURCE = "tiles"
CROP_PATCH_PX = 16                                         # 16 x 16-pixel patches (final AgriFM stride is also 16 px)
if CHIP_SOURCE == "tiles":
    CHIP_DIR = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW / "views" / f"crops_p{CROP_PATCH_PX}"
    TAG = f"{WINDOW}_tiles"                                 # output name: <model>_agyear_tiles.npz
else:
    CHIP_DIR = ROOT / "outputs" / "legacy" / "04_chips" / WINDOW
    TAG = WINDOW                                            # output name: <model>_agyear.npz (old results)
OUT_DIR = ROOT / "outputs" / "study" / "02_embeddings"
PART_DIR = OUT_DIR / f"_parts_{MODEL_NAME}_{TAG}"
OUT_FILE = OUT_DIR / f"{MODEL_NAME}_{TAG}.npz"


def read_chip_info():
    """Band names, months, NODATA value and scales written with the crops (chips_info.json), plus the parcel index."""
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


# ========================= TERRAMIND-SPECIFIC CODE =========================
# TerraMind v1 pretraining statistics (TerraTorch terramind_register.py, untok_sen2l2a@224 and untok_sen1grd@224)
S2_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S2_MEAN = [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648]
S2_STD = [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311]
S1_NAMES = ["VV", "VH"]
S1_MEAN = [-12.599, -20.293]
S1_STD = [5.195, 5.890]
PATCH_PX = 16


def standardise(x, mean, std):
    """(x - mean) / std per band; missing pixels -> 0 (= the pretraining mean)."""
    mean = np.array(mean, dtype=np.float32)[None, None, :, None, None]
    std = np.array(std, dtype=np.float32)[None, None, :, None, None]
    return np.nan_to_num((x - mean) / std, nan=0.0)


def load_encoder():
    from terratorch.registry import BACKBONE_REGISTRY
    model = BACKBONE_REGISTRY.build(BACKBONE, pretrained=True, modalities=MODALITIES, merge_method="mean")
    return model.eval().to(DEVICE)


ENCODER = None


@torch.no_grad()
def encode(inputs):
    """inputs: modality -> [N, C, H, W] -> patch grid [N, n, n, 768] (last layer, modalities averaged)."""
    n_chips, _, h, _ = next(iter(inputs.values())).shape
    n = h // PATCH_PX
    out = []
    for s in range(0, n_chips, EMBED_BATCH):
        batch = {k: torch.from_numpy(v[s:s + EMBED_BATCH]).to(DEVICE) for k, v in inputs.items()}
        tokens = ENCODER(batch)[-1]                               # [b, n*n, 768], row-major patch order
        out.append(tokens.reshape(tokens.shape[0], n, n, -1).float().cpu().numpy())
    return np.concatenate(out)


def embed_batch(ids, chips, masks, info, index):
    """Chips of a batch of parcels -> embeddings [B, months, 768] and valid [B, months]."""
    s2 = select_bands(chips, info, S2_NAMES)                                    # reflectance x 10000
    s1 = select_bands(chips, info, S1_NAMES, scale=info["s1_scale"])           # dB
    valid = month_is_valid(s2) | month_is_valid(s1)
    b, m = s2.shape[:2]
    inputs = {}
    if "S2L2A" in MODALITIES:
        inputs["S2L2A"] = standardise(s2, S2_MEAN, S2_STD).reshape(b * m, len(S2_NAMES), *s2.shape[3:])
    if "S1GRD" in MODALITIES:
        inputs["S1GRD"] = standardise(s1, S1_MEAN, S1_STD).reshape(b * m, len(S1_NAMES), *s1.shape[3:])
    grid = encode(inputs)                                                       # [B*M, n, n, D]
    weights = np.repeat(patch_weights(masks, PATCH_PX), m, axis=0)
    emb = pool_patches(grid, weights).reshape(b, m, -1)
    return emb, valid


def main():
    global ENCODER
    ENCODER = load_encoder()
    run(embed_batch, month_names)


if __name__ == "__main__":
    main()
