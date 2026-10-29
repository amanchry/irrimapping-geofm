"""
02d_embed_agrifm.py  —  AgriFM embeddings for every parcel (frozen encoder)
===========================================================================

What it does
------------
Turns each parcel's 12 monthly Sentinel-2 crops (dataset/04_make_views.py) into ONE AgriFM embedding
for the whole agricultural year. The encoder is frozen; the embeddings are classified in
03_train_evaluate.py like every other input.

The model
---------
AgriFM (Li et al. 2025, arXiv:2505.21357; code https://github.com/flyakon/AgriFM): a modified
Video Swin Transformer pretrained on MODIS, Landsat and Sentinel-2 time series for agricultural
mapping. Unlike the other three models it takes the whole time series at once and compresses time
inside the network, so it returns one embedding per parcel (not one per month). Settings follow
the repository's configs/cropland_mapping.py: Sentinel-2 encoder, 10 bands (Blue, Green, Red,
Red-Edge 1-3, NIR B08, NIR B08A, SWIR 1, SWIR 2 = B2 B3 B4 B5 B6 B7 B8 B8A B11 B12), patch size
(4 time steps, 2 x 2 pixels), normalised with the mean/std given in that config. The final feature
map has a stride of 16 pixels (160 m), so a 64 x 64 chip gives a 4 x 4 grid.

Check before reporting results: the config's normalisation mean (e.g. 4179 for blue) is much
higher than typical Earth Engine Sentinel-2 L2A values (~1100). Confirm with the AgriFM authors /
repository which pixel scale the pretrained weights expect, and adjust MEAN/STD if needed.

Steps
-----
1. Build the AgriFM Sentinel-2 encoder from the cloned repository's code and load the pretrained weights.
2. For each batch of parcels: take the 10 bands for all 12 months, normalise; cloudy pixels -> 0.
3. Run the encoder on the [batch, 12 months, 10 bands, 64, 64] stack.
4. Weighted mean of the final feature-map cells covering the parcel (parcel mask stored with the crops).
5. Save in batches (resume possible) and merge into one file.

Install (once, in the condaml environment)
    pip install mmengine
    The AgriFM code (github.com/flyakon/AgriFM -> external/AgriFM) and the weights
    (https://glass.hku.hk/casual/AgriFM/AgriFM.pth -> checkpoints/agrifm/AgriFM.pth) are fetched
    automatically on the first run. Only the code's encoder file is used; the mmseg registries it
    needs are replaced by mmengine ones, so the repository's mmcv / mmseg requirements are not needed.

Output
------
outputs/study/02_embeddings/agrifm_<WINDOW>_tiles.npz  (same layout as the other models)
    embeddings [N, 1, D] float16 (one step: the whole year), valid [N, 1], steps = ["year"]

Run (from the project folder)
-----------------------------
    python study/02d_embed_agrifm.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
MODEL_NAME = "agrifm"
WINDOW = "agyear"                                          # must match 03_build_tiles.py / 04_make_views.py
AGRIFM_REPO = ROOT / "external" / "AgriFM"                 # git clone of github.com/flyakon/AgriFM
WEIGHTS = ROOT / "checkpoints" / "agrifm" / "AgriFM.pth"
WEIGHTS_URL = "https://glass.hku.hk/casual/AgriFM/AgriFM.pth"   # downloaded automatically if WEIGHTS is missing
REPO_URL = "https://github.com/flyakon/AgriFM"                 # cloned automatically if AGRIFM_REPO is missing
BATCH_PARCELS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================================

# ============================ SHARED CHIP CODE ============================
# Reading the crops of dataset/04_make_views.py, pooling patch embeddings inside the parcel and
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


# =========================== AGRIFM-SPECIFIC CODE ==========================
# Bands, normalisation and architecture from AgriFM configs/cropland_mapping.py (Sentinel-2 encoder)
S2_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
MEAN = np.array([4179.192015478227, 4065.9106675194444, 3957.274910960156, 5207.452475253116, 4327.12234687,
                 4873.16102239, 5049.1637925, 5111.07806856, 3056.86349163, 2490.9675032],
                dtype=np.float32)[None, None, :, None, None]
STD = np.array([4041.5212325268735, 3691.003119315892, 3629.331318356375, 2973.5178530908756, 3569.73343885,
                3085.9151435, 2937.56005119, 2806.04462314, 1808.30013156, 1694.20220774],
               dtype=np.float32)[None, None, :, None, None]
STRIDE_PX = 16                # final feature map: one cell per 16 x 16 pixels


def import_agrifm_encoder():
    """Load AgriFM's video_swin_transformer.py on its own. The file only needs two registries from
    mmseg; they are replaced by plain mmengine registries, so no mmcv / mmseg install is needed."""
    import importlib.util
    import types
    from mmengine.registry import Registry
    registry = Registry("agrifm")
    for name in ["mmseg", "mmseg.models", "mmseg.models.builder", "mmseg.registry", "mmseg.registry.registry"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["mmseg.models.builder"].BACKBONES = registry
    sys.modules["mmseg.registry.registry"].MODELS = registry
    sys.modules["mmseg.registry.registry"].TRANSFORMS = Registry("agrifm_transforms")
    path = AGRIFM_REPO / "AgriFM" / "models" / "video_swin_transformer.py"
    spec = importlib.util.spec_from_file_location("agrifm_video_swin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PretrainingSwinTransformer3DEncoder


def download_weights(path):
    """Download AgriFM.pth from WEIGHTS_URL with a progress line; written to a temporary file first."""
    import urllib.request
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading AgriFM weights from {WEIGHTS_URL}")

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            print(f"\r  {min(done, total) / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(WEIGHTS_URL, tmp, reporthook=progress)
    tmp.rename(path)
    print(f"\n  saved {path}")


def clone_repo():
    """git clone the AgriFM code into AGRIFM_REPO."""
    import subprocess
    print(f"Cloning {REPO_URL} -> {AGRIFM_REPO}")
    AGRIFM_REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(AGRIFM_REPO)], check=True)


def check_requirements(weights):
    """Fetch the AgriFM code and weights if missing; stop with a clear message if mmengine is missing."""
    try:
        import mmengine  # noqa: F401
    except ImportError:
        raise SystemExit("AgriFM needs the mmengine package -> pip install mmengine")
    if not (AGRIFM_REPO / "AgriFM" / "models" / "video_swin_transformer.py").exists():
        clone_repo()
    if weights and not Path(weights).exists():
        download_weights(weights)


def load_encoder(weights=WEIGHTS):
    """AgriFM Sentinel-2 encoder (PretrainingSwinTransformer3DEncoder) with the pretrained weights."""
    check_requirements(weights)
    Encoder = import_agrifm_encoder()
    encoder = Encoder(
        patch_emd_cfg=dict(type="SwinPatchEmbed3D", patch_size=(4, 2, 2), in_chans=10, embed_dim=128),
        backbone_cfg=dict(type="SwinTransformer3D", pretrained=None, pretrained2d=False, patch_size=(4, 2, 2),
                          embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=(8, 7, 7),
                          out_indices=(0, 1, 2, 3), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                          attn_drop_rate=0.0, drop_path_rate=0.0, patch_norm=False, frozen_stages=-1,
                          use_checkpoint=False, downsample_steps=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                          feature_fusion="cat", mean_frame_down=True),
        init_cfg=dict(type="pretrained", checkpoint=str(weights)) if weights else None,
    )
    return encoder.eval().to(DEVICE)


ENCODER = None


@torch.no_grad()
def encode(x):
    """x [B, T, 10, H, W] -> feature grid [B, n, n, D] (time already merged inside AgriFM)."""
    features = ENCODER(torch.from_numpy(x).to(DEVICE))["encoder_features"]      # [B, D, n, n]
    return features.permute(0, 2, 3, 1).float().cpu().numpy()


def embed_batch(ids, chips, masks, info, index):
    """Chips of a batch of parcels -> embeddings [B, 1, D] and valid [B, 1]."""
    x = select_bands(chips, info, S2_NAMES)                     # [B, 12, 10, H, W]
    valid = month_is_valid(x).any(axis=1, keepdims=True)         # at least one observed month
    x = np.ascontiguousarray(np.nan_to_num((x - MEAN) / STD, nan=0.0))
    grid = encode(x)
    emb = pool_patches(grid, patch_weights(masks, STRIDE_PX))
    return emb[:, None, :], valid


def main():
    global ENCODER
    ENCODER = load_encoder()
    run(embed_batch, lambda info: ["year"])


if __name__ == "__main__":
    main()
