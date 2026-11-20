"""
encoders.py  —  the six geospatial foundation models of the pixel-wise experiments (not run on its own)
=======================================================================================================

Loads each pretrained encoder with its bands, normalisation and weights. Used by tile_common.GeoFMSegmenter (E04
frozen, E05 LoRA). Weights are kept in checkpoints/<model>/ and downloaded on the first use; model code that is not
a pip package is cloned into external/<model>.

Image models (encode patches of a 128 px tile, one or several months per pass)
-----------------------------------------------------------------------------
ENCODER    weights / code                                         bands                         patch   months per pass
prithvi    Prithvi-EO-2.0-300M-TL via TerraTorch (Hugging Face)   B2 B3 B4 B8A B11 B12          16 px   4, with date + location
clay       Clay v1.5 (checkpoints/clay/clay-v1.5.ckpt)            10 bands (no B1, B9)          8 px    1, with week + location
terramind  TerraMind-1.0-base via TerraTorch, S2L2A only (D14)    12 bands                      16 px   1
agrifm     AgriFM (checkpoints/agrifm/AgriFM.pth, external/AgriFM) 10 bands                     4 px    all 12 (pyramid to 32 px)
Bands, normalisation statistics and loading are those of the archived embedding scripts
(parcelwise_run/06a-06d), so the encoders are identical. AgriFM is normalised with the region's band
statistics (D15; computed in tile_common.region_band_stats).

Pixel time-series models (encode the 12-month series of every pixel)
--------------------------------------------------------------------
Presto (Tseng et al. 2023, arXiv:2304.14065; github.com/nasaharvest/presto, MIT licence).
Code: external/presto (git clone); weights: checkpoints/presto/default_model.pt (copied from the repository's
data/default_model.pt). Input per month: 17 channels in Presto's groups (S1 VV/VH, S2 B2 B3 B4 | B5 B6 B7 | B8 |
B8A | B11 B12, ERA5 temperature / precipitation, SRTM elevation / slope, NDVI), Sentinel-2 = reflectance / 10,000,
plus a Dynamic World class per month, lat / lon and the first month. Sentinel-2 only (D14): the S1, ERA5 and SRTM
groups and Dynamic World are masked (the model removes masked tokens). Presto requires the same tokens to be masked
for every pixel of a batch, so months without a clear observation are filled by linear interpolation in time
(as the E01 features) instead of being masked. Output: mean of the encoder tokens, 128-d. Up to 24 months; 12 used.

TESSERA v2 (Feng et al. 2025, arXiv:2506.20380; v2 preprint arXiv:2607.03949; github.com/ucam-eo/tessera, code MIT,
weights CC0). Code: external/tessera (git clone); weights: the Medium pixel student (21 M parameters), Hugging Face
geotessera/TESSERA-V-2.0-2B-M, file checkpoints/tessera/ckpt/student_medium.pt. Input: Sentinel-2 observations
of one pixel as a set of (10 bands in the order B4 B2 B3 B8 B8A B5 B6 B7 B11 B12, raw reflectance, z-scored with
the model's statistics; day of year 1-365). TESSERA was trained on single acquisitions; here each observation is
one monthly composite, dated the 15th of its month. Only clear months are used; as in the reference inference code
(student/infer.py), the observations are padded or subsampled to the size bin of their count (8 or 16 for 12
months). Sentinel-2 only (D14): the Sentinel-1 branch gets one empty observation, as the reference code does when
no Sentinel-1 data is given. Output: 128-d (Matryoshka; all 128 used).

Both pixel encoders run in chunks of pixels; when the encoder is trained (LoRA, E05), each chunk is gradient-
checkpointed, so memory holds only the 128-d outputs of a tile, not the activations of all its pixels.
"""
import importlib.util
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pixel_common import ROOT, S2_BANDS

PRESTO_REPO = ROOT / "external" / "presto"
PRESTO_URL = "https://github.com/nasaharvest/presto"
PRESTO_WEIGHTS = ROOT / "checkpoints" / "presto" / "default_model.pt"
TESSERA_REPO = ROOT / "external" / "tessera"
TESSERA_URL = "https://github.com/ucam-eo/tessera"
TESSERA_HF = ("geotessera/TESSERA-V-2.0-2B-M", "ckpt/student_medium.pt")
TESSERA_WEIGHTS = ROOT / "checkpoints" / "tessera" / "ckpt" / "student_medium.pt"
CHUNK = 16384                     # pixels per encoder call (one 128 px tile)


def _clone(url, path):
    if not path.exists():
        print(f"Cloning {url} -> {path}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================== image models
def _download(url, path):
    """Download url to path (via a .part file, so a broken download is redone)."""
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)


def _hf_cache(folder):
    """Hugging Face downloads go to checkpoints/<model>/ instead of the hidden cache (set before TerraTorch loads)."""
    import os
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(folder)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(folder)


class Prithvi:
    """Prithvi-EO-2.0-300M-TL (Szwarcman et al. 2024); model card bands HLS B02-B07 = Sentinel-2 B2 B3 B4 B8A B11 B12."""
    S2_NAMES = ["B2", "B3", "B4", "B8A", "B11", "B12"]
    MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
    STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)
    PATCH_PX, DIM, FRAMES_PER_PASS, USE_COORDS = 16, 1024, 4, True
    WEIGHTS_DIR = ROOT / "checkpoints" / "prithvi"

    @classmethod
    def load_encoder(cls, pretrained=True):
        """pretrained = False: the same architecture with random initial weights (E05, training from scratch)."""
        _hf_cache(cls.WEIGHTS_DIR)
        from terratorch.registry import BACKBONE_REGISTRY
        return BACKBONE_REGISTRY.build("prithvi_eo_v2_300_tl", pretrained=pretrained,
                                       num_frames=cls.FRAMES_PER_PASS).eval()


class Clay:
    """Clay v1.5 (Clay Foundation 2024); band means / stds and wavelengths from Clay's metadata.yaml (sentinel-2-l2a)."""
    BANDS = [("B2", 1105.0, 1809.0, 0.493), ("B3", 1355.0, 1757.0, 0.560), ("B4", 1552.0, 1888.0, 0.665),
             ("B5", 1887.0, 1870.0, 0.704), ("B6", 2422.0, 1732.0, 0.740), ("B7", 2630.0, 1697.0, 0.783),
             ("B8", 2743.0, 1742.0, 0.842), ("B8A", 2785.0, 1648.0, 0.865), ("B11", 2388.0, 1470.0, 1.610),
             ("B12", 1835.0, 1379.0, 2.190)]
    S2_NAMES = [b[0] for b in BANDS]
    MEAN = np.array([b[1] for b in BANDS], dtype=np.float32)
    STD = np.array([b[2] for b in BANDS], dtype=np.float32)
    WAVES = [b[3] for b in BANDS]
    PATCH_PX, DIM, GSD_M = 8, 1024, 10.0
    CHECKPOINT = ROOT / "checkpoints" / "clay" / "clay-v1.5.ckpt"
    METADATA = ROOT / "checkpoints" / "clay" / "metadata.yaml"
    CHECKPOINT_URL = "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
    METADATA_URL = "https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml"

    @classmethod
    def load_encoder(cls):
        """Frozen Clay encoder; mask_ratio = 0 and shuffle = False keep every patch, in grid order."""
        if not cls.CHECKPOINT.exists():
            _download(cls.CHECKPOINT_URL, cls.CHECKPOINT)
        if not cls.METADATA.exists():
            _download(cls.METADATA_URL, cls.METADATA)
        from claymodel.module import ClayMAEModule
        module = ClayMAEModule.load_from_checkpoint(str(cls.CHECKPOINT), metadata_path=str(cls.METADATA),
                                                    mask_ratio=0.0, shuffle=False)
        return module.model.encoder.eval()


class TerraMind:
    """TerraMind-1.0-base (Jakubik et al. 2025), Sentinel-2 L2A only; pretraining statistics of TerraTorch's
    terramind_register.py (untok_sen2l2a@224)."""
    S2_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    MEAN = np.array([1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234, 3132.22,
                     3162.988, 2424.884, 1857.648], dtype=np.float32)
    STD = np.array([2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829,
                    1797.379, 1434.261, 1334.311], dtype=np.float32)
    PATCH_PX, DIM = 16, 768
    WEIGHTS_DIR = ROOT / "checkpoints" / "terramind"

    @classmethod
    def load_encoder(cls):
        _hf_cache(cls.WEIGHTS_DIR)
        from terratorch.registry import BACKBONE_REGISTRY
        return BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, modalities=["S2L2A"],
                                       merge_method="mean").eval()


class AgriFM:
    """AgriFM (Li et al. 2025): the Sentinel-2 encoder of the repository's configs/cropland_mapping.py
    (PretrainingSwinTransformer3DEncoder) with the AgriFM.pth weights. MEAN / STD are set per region (D15)."""
    S2_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    MEAN = STD = None
    PATCH_SIZE = (4, 4, 4)            # (months, pixels, pixels) of the pretrained patch embedding
    REPO = ROOT / "external" / "AgriFM"
    REPO_URL = "https://github.com/flyakon/AgriFM"
    WEIGHTS = ROOT / "checkpoints" / "agrifm" / "AgriFM.pth"
    WEIGHTS_URL = "https://glass.hku.hk/casual/AgriFM/AgriFM.pth"

    @classmethod
    def _encoder_class(cls):
        """AgriFM's video_swin_transformer.py loaded on its own; its two mmseg registries are replaced by plain
        mmengine registries, so no mmcv / mmseg install is needed."""
        import types
        from mmengine.registry import Registry
        registry = Registry("agrifm")
        for name in ["mmseg", "mmseg.models", "mmseg.models.builder", "mmseg.registry", "mmseg.registry.registry"]:
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["mmseg.models.builder"].BACKBONES = registry
        sys.modules["mmseg.registry.registry"].MODELS = registry
        sys.modules["mmseg.registry.registry"].TRANSFORMS = Registry("agrifm_transforms")
        return _import(cls.REPO / "AgriFM" / "models" / "video_swin_transformer.py",
                       "agrifm_video_swin").PretrainingSwinTransformer3DEncoder

    @classmethod
    def load_encoder(cls):
        try:
            import mmengine  # noqa: F401
        except ImportError:
            raise SystemExit("AgriFM needs the mmengine package -> pip install mmengine")
        _clone(cls.REPO_URL, cls.REPO)
        if not cls.WEIGHTS.exists():
            _download(cls.WEIGHTS_URL, cls.WEIGHTS)
        encoder = cls._encoder_class()(
            patch_emd_cfg=dict(type="SwinPatchEmbed3D", patch_size=cls.PATCH_SIZE, in_chans=10, embed_dim=128),
            backbone_cfg=dict(type="SwinTransformer3D", pretrained=None, pretrained2d=False, patch_size=cls.PATCH_SIZE,
                              embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=(8, 7, 7),
                              out_indices=(0, 1, 2, 3), mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                              attn_drop_rate=0.0, drop_path_rate=0.0, patch_norm=False, frozen_stages=-1,
                              use_checkpoint=False, downsample_steps=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                              feature_fusion="cat", mean_frame_down=True),
            init_cfg=None)
        # Checkpoint names: "encoder.backbone.*" -> "backbone.*", "encoder.S2_patch_emd.weights/bias" ->
        # "patch_emd.proj.weight/bias". Stops if any encoder weight is missing or has another shape.
        state = torch.load(cls.WEIGHTS, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        renamed = {"encoder.S2_patch_emd.weights": "patch_emd.proj.weight",
                   "encoder.S2_patch_emd.bias": "patch_emd.proj.bias"}
        source = {renamed.get(k, k.removeprefix("encoder.")): v for k, v in state.items()
                  if k in renamed or k.startswith("encoder.backbone.")}
        own = encoder.state_dict()
        bad = [k for k, v in own.items() if (k not in source or source[k].shape != v.shape)
               and not k.endswith("relative_position_index")]
        if bad:
            raise SystemExit(f"AgriFM weights: {len(bad)} encoder tensors missing or of another shape, e.g. {bad[:3]}")
        encoder.load_state_dict({k: v for k, v in source.items() if k in own}, strict=False)
        return encoder.eval()


IMAGE_ENCODERS = {"prithvi": Prithvi, "clay": Clay, "terramind": TerraMind, "agrifm": AgriFM}


# =============================================================== pixel time-series models
def fill_time_torch(a, valid):
    """Linear interpolation over months of a [N, T, C] tensor where valid [N, T] is False; first / last months take
    the nearest observed month; pixels never observed -> 0."""
    n, t, _ = a.shape
    steps = torch.arange(t, device=a.device).expand(n, t)
    prev = torch.cummax(torch.where(valid, steps, torch.full_like(steps, -1)), dim=1).values
    nxt = torch.flip(torch.cummin(torch.flip(torch.where(valid, steps, torch.full_like(steps, t)), [1]), dim=1).values, [1])
    a0 = torch.nan_to_num(a)
    pv = torch.gather(a0, 1, prev.clamp(0, t - 1)[..., None].expand_as(a0))
    nv = torch.gather(a0, 1, nxt.clamp(0, t - 1)[..., None].expand_as(a0))
    hp, hn = (prev >= 0)[..., None], (nxt < t)[..., None]
    w = ((steps - prev) / (nxt - prev).clamp(min=1))[..., None].to(a.dtype)
    filled = torch.where(hp & hn, pv + (nv - pv) * w, torch.where(hp, pv, torch.where(hn, nv, torch.zeros_like(pv))))
    return torch.where(valid[..., None], a0, filled)


class PrestoEncoder(nn.Module):
    """Presto encoder: s2 [N, T, 12] (S2_BANDS order, reflectance x 10,000, NaN = missing) -> [N, 128]."""
    S2_IN = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]       # Presto channels 2-11
    dim = 128

    def __init__(self):
        super().__init__()
        _clone(PRESTO_URL, PRESTO_REPO)
        presto = _import(PRESTO_REPO / "single_file_presto.py", "single_file_presto")
        model = presto.Presto.construct()
        if not PRESTO_WEIGHTS.exists():
            PRESTO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            PRESTO_WEIGHTS.write_bytes((PRESTO_REPO / "data" / "default_model.pt").read_bytes())
        model.load_state_dict(torch.load(PRESTO_WEIGHTS, map_location="cpu"))
        self.encoder = model.encoder
        self.s2_idx = [S2_BANDS.index(b) for b in self.S2_IN]
        mask = torch.ones(17)
        mask[2:12] = 0                                    # keep S2 (2-11) and NDVI (16); mask S1, ERA5, SRTM
        mask[16] = 0
        self.register_buffer("band_mask", mask)

    def _encode(self, s2, valid, latlon, month):
        n, t, _ = s2.shape
        x = fill_time_torch(s2[:, :, self.s2_idx], valid) / 10000.0
        b4, b8 = x[:, :, 2], x[:, :, 6]
        ndvi = ((b8 - b4) / (b8 + b4).clamp(min=1e-6)).clamp(-1, 1)
        full = torch.zeros(n, t, 17, device=s2.device, dtype=x.dtype)
        full[:, :, 2:12] = x
        full[:, :, 16] = ndvi
        dw = torch.full((n, t), 9, dtype=torch.long, device=s2.device)          # 9 = no Dynamic World class
        mask = self.band_mask.expand(n, t, 17)
        return self.encoder(full, dw, latlon, mask=mask, month=month, eval_task=True)

    def forward(self, s2, valid, latlon, start_month, doys):
        month = int(start_month) - 1                                            # 0 = January
        outs = []
        for s in range(0, len(s2), CHUNK):
            args = (s2[s:s + CHUNK], valid[s:s + CHUNK], latlon[s:s + CHUNK])
            if torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()):
                outs.append(checkpoint(lambda a, b, c: self._encode(a, b, c, month), *args, use_reentrant=False))
            else:
                outs.append(self._encode(*args, month))
        return torch.cat(outs)


class TesseraEncoder(nn.Module):
    """TESSERA v2 Medium pixel student: s2 [N, T, 12] (S2_BANDS order, reflectance x 10,000, NaN = missing) -> [N, 128]."""
    S2_IN = ["B4", "B2", "B3", "B8", "B8A", "B5", "B6", "B7", "B11", "B12"]
    dim = 128

    def __init__(self):
        super().__init__()
        _clone(TESSERA_URL, TESSERA_REPO)
        if not TESSERA_WEIGHTS.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(TESSERA_HF[0], TESSERA_HF[1], local_dir=str(TESSERA_WEIGHTS.parents[1]))
        student = TESSERA_REPO / "tessera_infer_v2" / "student"
        tm = _import(student / "model.py", "tessera_student_model")
        self.model = tm.load_model(str(TESSERA_WEIGHTS), torch.device("cpu"))
        self.s2_idx = [S2_BANDS.index(b) for b in self.S2_IN]
        self.register_buffer("mean", torch.tensor(tm.S2_BAND_MEAN))
        self.register_buffer("std", torch.tensor(tm.S2_BAND_STD))
        # PyTorch's fast inference path of nn.TransformerEncoderLayer reads the weights directly and would skip LoRA
        # adapters on linear1 / linear2; it is switched off (same results without adapters).
        torch.backends.mha.set_fastpath_enabled(False)

    def _encode(self, s2, valid, doys):
        """One group of pixels with the same observation-count bin."""
        n, t, _ = s2.shape
        x = (torch.nan_to_num(s2[:, :, self.s2_idx]) - self.mean) / (self.std + 1e-9)
        v = valid.cpu().numpy()
        bins = _bin_size(v.sum(1))
        out = torch.zeros(n, self.dim, device=s2.device, dtype=x.dtype)
        for b in np.unique(bins):
            rows = np.flatnonzero(bins == b)
            if b == 0:
                continue                                           # never observed: zero vector
            src = _source_indices(v[rows], int(b))                 # [G, b] month index per slot
            src_t = torch.from_numpy(src).to(s2.device)
            r = torch.from_numpy(rows).to(s2.device)
            g = torch.gather(x[r], 1, src_t[..., None].expand(-1, -1, x.shape[2]))
            d = doys.to(s2.device)[src_t].to(x.dtype)
            s2_in = torch.cat([g, d[..., None]], -1)
            s1_in = torch.zeros(len(rows), 1, 3, device=s2.device, dtype=x.dtype)      # no Sentinel-1 (D14)
            out[r] = self.model.encode(s2_in, s1_in).to(out.dtype)
        return out

    def forward(self, s2, valid, latlon, start_month, doys):
        outs = []
        for s in range(0, len(s2), CHUNK):
            args = (s2[s:s + CHUNK], valid[s:s + CHUNK])
            if torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()):
                outs.append(checkpoint(lambda a, b: self._encode(a, b, doys), *args, use_reentrant=False))
            else:
                outs.append(self._encode(*args, doys))
        return torch.cat(outs)


# ---- observation-count bins and padding, as in tessera_infer_v2/student/infer.py (pad_to_bin of training)
BIN_EDGES = list(range(8, 257, 8))


def _bin_size(n_obs):
    out = np.full_like(n_obs, BIN_EDGES[-1])
    out[n_obs <= 0] = 0
    for b in reversed(BIN_EDGES):
        out = np.where((n_obs > 0) & (n_obs <= b), b, out)
    return out


def _pad_pattern(n, size):
    if n >= size:
        return np.linspace(0, n - 1, size, dtype=np.int64)
    remain = size - n
    if remain <= n:
        fill = np.array([g[len(g) // 2] for g in np.array_split(np.arange(n), remain)], dtype=np.int64)
    else:
        fill = (np.arange(remain) % n).astype(np.int64)
    return np.concatenate([np.arange(n, dtype=np.int64), fill])


def _source_indices(valid, size):
    """[G, size] month indices: the clear months of each pixel, padded / subsampled to the bin size."""
    sorted_pos = np.argsort(~valid, axis=1, kind="stable")
    n_per = valid.sum(1)
    src = np.zeros((len(valid), size), dtype=np.int64)
    for n in np.unique(n_per):
        rows = np.flatnonzero(n_per == n)
        src[rows] = sorted_pos[rows][:, _pad_pattern(int(n), size)]
    return src


PIXEL_ENCODERS = {"presto": PrestoEncoder, "tessera": TesseraEncoder}
