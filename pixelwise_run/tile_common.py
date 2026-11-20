"""
tile_common.py  —  shared code of the tile experiments E03 (frozen GeoFMs + decoder), E04 (U-TAE),
                   E05 (Prithvi: frozen / fully fine-tuned / from scratch, with fewer labels)
=============================================================================================================

The tile models see whole 128 x 128 px tiles (12 months) of the parcel tile store (03_build_tiles.py) and predict a
class for every pixel. They are trained and scored on exactly the labelled pixels of 04_make_pixel_samples.py, like the pixel models:
the loss is computed only at the sampled pixels of the training folds (all other pixels are ignored), and the
out-of-fold prediction of a pixel is read from the prediction of its tile. A tile can hold pixels of two folds
(a tile may cross a 20 km block border); only the labels of the round's training folds enter the loss, so no test
label is used, but the image context of a training tile can contain test pixels (unlabelled).

Contents
--------
TileDataset       reads <tile_id>_img.tif, keeps the 12 Sentinel-2 bands of the 12 months (reflectance x 10,000;
                  NaN where the month has no clear observation) and builds the target map of the allowed folds
                  (-1 = ignored); random flips / 90 degree rotations for training.
GeoFMSegmenter    frozen, LoRA-adapted or fully trained (fine-tuned or, for Prithvi, randomly initialised)
                  foundation-model encoder + L-TAE over the months at every patch
                  position + simple feature pyramid + UPerNet head (Xiao et al. 2018), as in PANGAEA's
                  multi-temporal set-up (Marsocci et al. 2024). AgriFM merges the months itself and gives a
                  multi-scale pyramid, so it goes straight to the UPerNet head. Encoder loading and normalisation
                  are loaded by pixelwise_run/encoders.py (weights, bands and normalisation statistics).
                  Presto and TESSERA (pixel time-series models, pixelwise_run/encoders.py) encode every pixel
                  of the tile over the 12 months; the 10 m feature map is pooled to 4 scales for the UPerNet head.
UTAE              U-Net with temporal attention (Garnot & Landrieu 2021), trained from scratch.
add_lora          low-rank adapters (Hu et al. 2022) on the attention projections (qkv / to_qkv) of an encoder; for
                  TESSERA on the feed-forward layers (linear1 / linear2), because nn.MultiheadAttention packs q, k, v
                  in one weight that an adapter cannot wrap.
run_tile_experiment   PASTIS rotation, seeds, early stopping on the validation pixels, out-of-fold predictions,
                  metrics (pixelwise_run/pixel_common.py). Optional: a share of the training parcels only
                  (label_subset; validation and test folds unchanged), a separate learning rate for the encoder,
                  learning-rate warm-up and gradient clipping (E05).
"""
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from nn_common import DEVICE, LTAE
from encoders import IMAGE_ENCODERS, PIXEL_ENCODERS
from pixel_common import (CLASSES, ROOT, S2_BANDS, class_weights, get_logger, load_samples, metrics_from_confusion,
                          month_dates, month_positions, results_dir, rounds, s2_series, save_run, standardiser)

AMP = DEVICE == "cuda"


# ---------------------------------------------------------------- data
def tile_root(region):
    return ROOT / "outputs" / region / "dataset" / "03_tiles_128px" / "agyear"


class TileTable:
    """The labelled pixels (04_make_pixel_samples.py) grouped by tile."""

    def __init__(self, df, start_month):
        self.df = df
        self.groups = df.groupby("tile_id").indices                      # tile_id -> row numbers of df
        first = df.groupby("tile_id").agg(season=("season", "first"), lat=("lat", "mean"), lon=("lon", "mean"))
        self.meta = first
        self.start_month = start_month

    def tiles_with(self, folds):
        f = self.df["fold"].to_numpy()
        return sorted(t for t, idx in self.groups.items() if np.isin(f[idx], folds).any())


class TileDataset(torch.utils.data.Dataset):
    """keep: optional boolean array over the rows of table.df; only rows with keep = True enter the target map
    (E05: a share of the training parcels; the pixels of the other parcels stay in the image, unlabelled)."""

    def __init__(self, region, table, tiles, folds, augment=False, keep=None):
        self.dir = tile_root(region) / "tiles"
        info = json.loads((tile_root(region) / "tiles_info.json").read_text())
        mb = info["month_bands"]
        self.n_months, self.n_bands, self.nodata = info["n_months"], len(mb), info["nodata"]
        self.s2_idx = [mb.index(b) for b in S2_BANDS]
        self.nclear_idx = mb.index("S2_NCLEAR")
        self.table, self.tiles, self.folds, self.augment = table, list(tiles), list(folds), augment
        d = table.df
        self.rows, self.cols = d["row"].to_numpy(), d["col"].to_numpy()
        self.labels, self.fold = d["label"].to_numpy(), d["fold"].to_numpy()
        self.keep = keep

    def __len__(self):
        return len(self.tiles)

    def read(self, tile_id):
        import rasterio
        with rasterio.open(self.dir / f"{tile_id}_img.tif") as src:
            a = src.read()
        a = a.reshape(self.n_months, self.n_bands, *a.shape[1:])
        s2 = a[:, self.s2_idx].astype(np.float32)
        bad = (a[:, self.nclear_idx] <= 0) | (s2 == self.nodata).any(1)
        s2[np.broadcast_to(bad[:, None], s2.shape)] = np.nan
        return s2                                                          # [T, 12, H, W]

    def __getitem__(self, i):
        tile_id = self.tiles[i]
        x = self.read(tile_id)
        target = np.full(x.shape[-2:], -1, dtype=np.int64)
        idx = self.table.groups[tile_id]
        idx = idx[np.isin(self.fold[idx], self.folds)]
        if self.keep is not None:
            idx = idx[self.keep[idx]]
        target[self.rows[idx], self.cols[idx]] = self.labels[idx] - 1
        if self.augment:
            k = np.random.randint(4)
            x, target = np.rot90(x, k, axes=(2, 3)), np.rot90(target, k)
            if np.random.rand() < 0.5:
                x, target = x[..., ::-1], target[..., ::-1]
        m = self.table.meta.loc[tile_id]
        dates = month_dates(m["season"], self.table.start_month, self.n_months)
        meta = {"latlon": np.array([m["lat"], m["lon"]], dtype=np.float32),
                "year_doy": np.array([[d.year, d.dayofyear] for d in dates], dtype=np.float32),
                "week": np.array([d.isocalendar().week for d in dates], dtype=np.float32)}
        return np.ascontiguousarray(x), np.ascontiguousarray(target), meta, i


# ---------------------------------------------------------------- decoder
def conv_gn(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.GroupNorm(min(16, cout // 4), cout),
                         nn.ReLU(inplace=True))


class UPerHead(nn.Module):
    """UPerNet head (Xiao et al. 2018): pyramid pooling on the coarsest level, FPN top-down path, fusion of all
    levels, 1 x 1 classifier. feats: list of maps from fine to coarse. Pool sizes 1, 2, 4 (instead of 1, 2, 3, 6):
    the coarsest map of a 128 px tile has only 4 x 4 or 8 x 8 cells."""

    def __init__(self, in_chs, ch=128, n_classes=3, pool_scales=(1, 2, 4)):
        super().__init__()
        self.lateral = nn.ModuleList([conv_gn(c, ch, 1) for c in in_chs[:-1]])
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), conv_gn(in_chs[-1], ch, 1)) for s in pool_scales])
        self.bottleneck = conv_gn(in_chs[-1] + len(pool_scales) * ch, ch)
        self.fpn = nn.ModuleList([conv_gn(ch, ch) for _ in in_chs[:-1]])
        self.fuse = conv_gn(len(in_chs) * ch, ch)
        self.cls = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(ch, n_classes, 1))

    def forward(self, feats):
        top = feats[-1]
        pooled = [F.interpolate(p(top), size=top.shape[-2:], mode="bilinear", align_corners=False) for p in self.ppm]
        lat = [l(f) for l, f in zip(self.lateral, feats[:-1])] + [self.bottleneck(torch.cat([top] + pooled, 1))]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="bilinear", align_corners=False)
        outs = [f(l) for f, l in zip(self.fpn, lat[:-1])] + [lat[-1]]
        size = outs[0].shape[-2:]
        outs = [o if o.shape[-2:] == size else F.interpolate(o, size=size, mode="bilinear", align_corners=False) for o in outs]
        return self.cls(self.fuse(torch.cat(outs, 1)))


class ViTPyramid(nn.Module):
    """Single-scale patch features -> 4 scales (x4, x2, x1, x0.5), as for plain ViTs in PANGAEA / ViTDet."""

    def __init__(self, cin, ch):
        super().__init__()
        self.up4 = nn.Sequential(nn.ConvTranspose2d(cin, ch, 2, 2), nn.GroupNorm(16, ch), nn.GELU(), nn.ConvTranspose2d(ch, ch, 2, 2))
        self.up2 = nn.ConvTranspose2d(cin, ch, 2, 2)
        self.same = nn.Conv2d(cin, ch, 1)
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(cin, ch, 1))

    def forward(self, x):
        return [self.up4(x), self.up2(x), self.same(x), self.down(x)]


# ---------------------------------------------------------------- foundation-model encoders
_ENCODERS = {}


def cached_encoder(name, pretrained=True):
    """(module, encoder) of a foundation model, loaded once per process; every model gets its own deep copy, so
    LoRA adapters or fine-tuned weights of one round never reach the next. pretrained = False (Prithvi only, E05):
    a new random initialisation for every model, drawn from the seed set before the model is built."""
    import copy
    if not pretrained:
        if name != "prithvi":
            raise SystemExit(f"{name}: training from scratch is only set up for prithvi")
        spec = IMAGE_ENCODERS[name]
        return spec, spec.load_encoder(pretrained=False).cpu()
    if name not in _ENCODERS:
        if name in PIXEL_ENCODERS:
            _ENCODERS[name] = (None, PIXEL_ENCODERS[name]().cpu())
        else:
            spec = IMAGE_ENCODERS[name]
            _ENCODERS[name] = (spec, spec.load_encoder().cpu())
    mod, enc = _ENCODERS[name]
    return mod, copy.deepcopy(enc)


def region_band_stats(region, bands, n_tiles=200, seed=0):
    """Per-band mean / std of the region's tile store (valid pixels, every 4th pixel of n_tiles random tiles;
    imagery only, no labels). Used for AgriFM (D15: region statistics instead of the config's)."""
    import rasterio
    info = json.loads((tile_root(region) / "tiles_info.json").read_text())
    mb = info["month_bands"]
    files = sorted((tile_root(region) / "tiles").glob("*_img.tif"))
    files = [files[i] for i in np.random.default_rng(seed).choice(len(files), min(n_tiles, len(files)), replace=False)]
    idx = [mb.index(b) for b in bands]
    total, total_sq, count = np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx))
    for f in files:
        with rasterio.open(f) as src:
            a = src.read()[:, ::4, ::4].reshape(info["n_months"], len(mb), -1)
        clear = a[:, mb.index("S2_NCLEAR")] > 0
        v = a[:, idx].astype(np.float64)
        ok = clear[:, None] & (v != info["nodata"])
        total += np.where(ok, v, 0).sum((0, 2))
        total_sq += np.where(ok, v ** 2, 0).sum((0, 2))
        count += ok.sum((0, 2))
    mean = total / count
    return mean, np.sqrt(total_sq / count - mean ** 2)


class PixelPyramid(nn.Module):
    """10 m pixel features -> 4 scales (x1, x1/2, x1/4, x1/8) by average pooling, for the UPerNet head."""

    def forward(self, x):
        return [x] + [F.avg_pool2d(x, k) for k in (2, 4, 8)]


LORA_TARGETS = {"tessera": ("linear1", "linear2")}


class GeoFMSegmenter(nn.Module):
    """Foundation-model encoder + temporal L-TAE + UPerNet head -> logits [B, 3, H, W].
    Encoder modes: frozen = True (E03: pretrained, locked); lora_rank > 0 (pretrained, locked, low-rank adapters
    trained); frozen = False (every encoder weight trained: fine-tuning with pretrained = True, training from
    scratch with pretrained = False)."""

    def __init__(self, name, region, frozen=True, lora_rank=0, lora_alpha=16, ltae_dim=256, decoder_ch=128,
                 pretrained=True, lora_targets=None):
        super().__init__()
        if not pretrained and (frozen or lora_rank):
            raise SystemExit("pretrained = False needs frozen = False and no LoRA (a frozen random encoder is useless)")
        self.name = name
        self.start_month = json.loads((tile_root(region) / "tiles_info.json").read_text())["start_month"]
        mod, self.encoder = cached_encoder(name, pretrained)
        self.mod = mod
        full = not frozen and not lora_rank                 # every encoder weight trained
        for p in self.encoder.parameters():
            p.requires_grad = full
        self.adapted = 0
        if lora_rank:
            targets = lora_targets or LORA_TARGETS.get(name, ("qkv", "to_qkv"))
            self.adapted = add_lora(self.encoder, lora_rank, lora_alpha, targets)
        self.frozen = frozen and not lora_rank
        if name in PIXEL_ENCODERS:                     # Presto / TESSERA: raw reflectance in, per-pixel features out
            self.pyramid = PixelPyramid()
            self.head = UPerHead([self.encoder.dim] * 4, decoder_ch)
            return
        self.bands = [S2_BANDS.index(b) for b in mod.S2_NAMES]
        if name == "agrifm":
            mean, std = region_band_stats(region, mod.S2_NAMES)
        else:
            mean, std = mod.MEAN, mod.STD
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 1, -1, 1, 1))
        if name == "agrifm":
            with torch.no_grad():
                feats = self.encoder(torch.zeros(1, 12, len(self.bands), 128, 128))["features_list"]
            chans = [f.shape[1] for f in sorted(feats, key=lambda f: -f.shape[-1])]
            self.head = UPerHead(chans, decoder_ch)
        else:
            self.patch = mod.PATCH_PX
            self.ltae = LTAE(mod.DIM, month_positions(), n_head=16, d_k=8, d_model=256, mlp=(ltae_dim,), dropout=0.1)
            self.pyramid = ViTPyramid(ltae_dim, decoder_ch)
            self.head = UPerHead([decoder_ch] * 4, decoder_ch)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()                       # frozen encoder: no dropout / norm statistics updates
        return self

    def encode_months(self, x, meta):
        """x [B, T, C, H, W] normalised -> patch features [B, T, D, n, n]."""
        b, t, c, h, w = x.shape
        n = h // self.patch
        if self.name == "prithvi":
            frames = self.mod.FRAMES_PER_PASS
            out = []
            for f0 in range(0, t, frames):
                clip = x[:, f0:f0 + frames].transpose(1, 2)                 # [B, C, T', H, W]
                kw = {}
                if self.mod.USE_COORDS:
                    kw = {"temporal_coords": meta["year_doy"][:, f0:f0 + frames].to(x.device),
                          "location_coords": meta["latlon"].to(x.device)}
                tok = self.encoder.forward_features(clip, **kw)[-1][:, 1:, :]
                out.append(tok.reshape(b, clip.shape[2], n, n, -1))
            return torch.cat(out, 1).permute(0, 1, 4, 2, 3)
        if self.name == "terramind":
            tok = self.encoder({"S2L2A": x.reshape(b * t, c, h, w)})[-1]
            return tok.reshape(b, t, n, n, -1).permute(0, 1, 4, 2, 3)
        # clay
        week = meta["week"].to(x.device) * 2 * math.pi / 52
        hour = torch.full_like(week, 11 * 2 * math.pi / 24)
        times = torch.stack([week.sin(), week.cos(), hour.sin(), hour.cos()], -1).reshape(b * t, 4)
        lat, lon = torch.deg2rad(meta["latlon"][:, 0]), torch.deg2rad(meta["latlon"][:, 1])
        ll = torch.stack([lat.sin(), lat.cos(), lon.sin(), lon.cos()], -1).to(x.device)
        datacube = {"pixels": x.reshape(b * t, c, h, w), "time": times.float(),
                    "latlon": ll.repeat_interleave(t, 0).float(),
                    "gsd": torch.tensor(self.mod.GSD_M, device=x.device),
                    "waves": torch.tensor(self.mod.WAVES, device=x.device)}
        tokens, kept, _, _ = self.encoder(datacube)
        patches = tokens[:, 1:, :]
        grid = torch.zeros_like(patches).scatter(1, kept.long().unsqueeze(-1).expand_as(patches), patches)
        return grid.reshape(b, t, n, n, -1).permute(0, 1, 4, 2, 3)

    def encode_pixels(self, x, meta):
        """x [B, T, 12, H, W] raw reflectance with NaN -> pixel features [B, 128, H, W] (Presto / TESSERA)."""
        b, t, c, h, w = x.shape
        valid = ~torch.isnan(x).any(2)
        px = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        pv = valid.permute(0, 2, 3, 1).reshape(b * h * w, t)
        latlon = meta["latlon"].to(x.device).repeat_interleave(h * w, 0)
        doys = meta["year_doy"][0, :, 1].long()
        f = self.encoder(px, pv, latlon, self.start_month, doys)
        return f.reshape(b, h, w, -1).permute(0, 3, 1, 2)

    def forward(self, x, meta):
        if self.name in PIXEL_ENCODERS:
            with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
                f = self.encode_pixels(x, meta)
            return self.head(self.pyramid(f.float()))
        valid = ~torch.isnan(x).any(2)                                      # [B, T, H, W]
        x = torch.nan_to_num((x[:, :, self.bands] - self.mean) / self.std, nan=0.0)
        if self.name == "agrifm":
            with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
                feats = self.encoder(x)["features_list"]
            feats = sorted(feats, key=lambda f: -f.shape[-1])
            logits = self.head([f.float() for f in feats])
        else:
            with torch.set_grad_enabled(not self.frozen and torch.is_grad_enabled()):
                f = self.encode_months(x, meta)                             # [B, T, D, n, n]
            b, t, d, n, _ = f.shape
            pv = F.adaptive_max_pool2d(valid.float(), n) > 0                # [B, T, n, n] month observed in patch
            seq = f.float().permute(0, 3, 4, 1, 2).reshape(b * n * n, t, d)
            mask = pv.permute(0, 2, 3, 1).reshape(b * n * n, t)
            mask = mask | ~mask.any(1, keepdim=True)
            z, _ = self.ltae(seq, mask)
            z = z.reshape(b, n, n, -1).permute(0, 3, 1, 2)
            logits = self.head(self.pyramid(z))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x with W frozen, A random (Kaiming), B zero (Hu et al. 2022)."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / r

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale


def add_lora(module, r=8, alpha=16, targets=("qkv", "to_qkv")):
    """Replace every nn.Linear whose name is in targets by a LoRALinear; returns the number replaced."""
    n = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha))
                n += 1
    if n == 0:
        raise SystemExit(f"LoRA: no layer named {targets} found in the encoder")
    return n


# ---------------------------------------------------------------- U-TAE
class UTAE(nn.Module):
    """U-TAE (Garnot & Landrieu 2021): a shared convolutional encoder per month, L-TAE at the coarsest level giving
    one attention mask per head and pixel, the masks (upsampled) aggregate the skip features of every level over
    time (channel groups = heads), and a convolutional decoder. Input [B, T, 12, H, W] standardised."""

    def __init__(self, in_ch, enc=(64, 64, 64, 128), dec=(32, 32, 64, 128), n_head=16, d_k=4, d_model=256, n_classes=3,
                 mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 1, -1, 1, 1))
        self.n_head = n_head
        self.in_conv = nn.Sequential(conv_gn(in_ch, enc[0]), conv_gn(enc[0], enc[0]))
        self.downs = nn.ModuleList([nn.Sequential(nn.Conv2d(enc[i], enc[i], 4, 2, 1), nn.GroupNorm(16, enc[i]),
                                                  nn.ReLU(inplace=True), conv_gn(enc[i], enc[i + 1]),
                                                  conv_gn(enc[i + 1], enc[i + 1])) for i in range(3)])
        self.ltae = LTAE(enc[3], month_positions(), n_head=n_head, d_k=d_k, d_model=d_model, mlp=(d_model, dec[3]),
                         dropout=0.2)
        self.ups = nn.ModuleList()
        for i in (2, 1, 0):
            self.ups.append(nn.ModuleDict({
                "up": nn.Sequential(nn.ConvTranspose2d(dec[i + 1], dec[i], 4, 2, 1), nn.GroupNorm(8, dec[i]), nn.ReLU(inplace=True)),
                "skip": nn.Conv2d(enc[i], enc[i], 1),
                "conv": nn.Sequential(conv_gn(dec[i] + enc[i], dec[i]), conv_gn(dec[i], dec[i]))}))
        self.out = nn.Sequential(conv_gn(dec[0], 32), nn.Conv2d(32, n_classes, 1))

    def forward(self, x, meta=None):
        b, t = x.shape[:2]
        valid_t = ~torch.isnan(x).any(2).flatten(2).all(2)                 # [B, T] month with any clear pixel
        valid_t = valid_t | ~valid_t.any(1, keepdim=True)
        x = torch.nan_to_num((x - self.mean) / self.std, nan=0.0)
        h = self.in_conv(x.flatten(0, 1))
        feats = [h]
        for d in self.downs:
            h = d(h)
            feats.append(h)
        c, hh, ww = h.shape[1:]
        seq = h.view(b, t, c, hh, ww).permute(0, 3, 4, 1, 2).reshape(b * hh * ww, t, c)
        z, att = self.ltae(seq, valid_t.repeat_interleave(hh * ww, 0))
        out = z.view(b, hh, ww, -1).permute(0, 3, 1, 2)
        att = att.view(b, hh, ww, self.n_head, t).permute(0, 3, 4, 1, 2)    # [B, heads, T, h, w]
        for level, block in zip((2, 1, 0), self.ups):
            s = feats[level]
            cs, hs, ws = s.shape[1:]
            a = F.interpolate(att.flatten(1, 2), size=(hs, ws), mode="bilinear", align_corners=False).view(b, self.n_head, t, hs, ws)
            s = s.view(b, t, self.n_head, cs // self.n_head, hs, ws)
            agg = torch.einsum("bhtyx,bthcyx->bhcyx", a, s).reshape(b, cs, hs, ws)
            up = block["up"](out)
            out = block["conv"](torch.cat([up, block["skip"](agg)], 1))
        return self.out(out)


# ---------------------------------------------------------------- training
def collate(batch):
    x = torch.from_numpy(np.stack([b[0] for b in batch]))
    y = torch.from_numpy(np.stack([b[1] for b in batch]))
    meta = {k: torch.from_numpy(np.stack([b[2][k] for b in batch])) for k in batch[0][2]}
    return x, y, meta, [b[3] for b in batch]


def loader(ds, batch, shuffle, workers):
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                                       drop_last=shuffle and len(ds) > batch, pin_memory=DEVICE == "cuda",
                                       persistent_workers=workers > 0)


@torch.no_grad()
def predict_tiles(model, ds, pixel_folds, workers, batch):
    """Class probabilities of the labelled pixels of `pixel_folds` in the tiles of ds: (row numbers of df, [n, 3])."""
    model.eval()
    rows, probs = [], []
    table = ds.table
    for x, _, meta, ids in loader(ds, batch, False, workers):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            p = torch.softmax(model(x.to(DEVICE), meta).float(), 1).cpu().numpy()
        for j, i in enumerate(ids):
            idx = table.groups[ds.tiles[i]]
            idx = idx[np.isin(ds.fold[idx], pixel_folds)]
            rows.append(idx)
            probs.append(p[j][:, ds.rows[idx], ds.cols[idx]].T)
    return np.concatenate(rows), np.concatenate(probs)


def train_tile_model(model, train_ds, val_ds, val_fold, weights, seed, lr, weight_decay, batch, epochs, patience,
                     workers, log, encoder_lr=None, warmup_epochs=0, grad_clip=None):
    """Masked, class-weighted cross-entropy; early stopping on the validation pixels' 3-class macro-F1. Logged per
    epoch: training loss, validation loss (the same weighted cross-entropy), validation 3-class and cropland
    macro-F1. Trainable encoder weights (LoRA or full) get encoder_lr (default lr) and are kept in the best state."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    enc = [p for n, p in named if n.startswith("encoder.")]
    head = [p for n, p in named if not n.startswith("encoder.")]
    log.info(f"    trainable parameters: {sum(p.numel() for _, p in named) / 1e6:.2f} M "
             f"(encoder {sum(p.numel() for p in enc) / 1e6:.2f} M, head {sum(p.numel() for p in head) / 1e6:.2f} M)")
    groups = [{"params": head, "lr": lr}]
    if enc:                                       # LoRA adapters or the whole encoder (E05)
        groups.append({"params": enc, "lr": encoder_lr or lr})
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_ds) // batch)
    sched = None
    if warmup_epochs:                             # linear warm-up over the first epoch(s), then constant
        warm = warmup_epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / warm))
    params = [p for _, p in named]
    trainable = {n for n, _ in named}
    w = np.asarray(weights, dtype=np.float64)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE), ignore_index=-1)
    y_all = train_ds.labels
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y, meta, _ in loader(train_ds, batch, True, workers):
            if (y >= 0).sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                logits = model(x.to(DEVICE), meta)
            loss = loss_fn(logits.float(), y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            losses.append(loss.item())
        t_train = time.time() - t0
        rows, p = predict_tiles(model, val_ds, [val_fold], workers, batch)
        yv = y_all[rows] - 1
        pred = p.argmax(1)
        f1 = f1_score(yv, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        cm = np.zeros((3, 3))
        np.add.at(cm, (yv, pred), 1)
        crop_f1 = metrics_from_confusion(cm)["cropland_macro_f1"]
        nll = -np.log(np.clip(p[np.arange(len(yv)), yv], 1e-7, 1.0))      # same weighted loss as in training
        val_loss = float((w[yv] * nll).sum() / w[yv].sum())
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_loss": val_loss,
                        "val_macro_f1": float(f1), "val_cropland_f1": float(crop_f1),
                        "seconds": round(time.time() - t0), "seconds_train": round(t_train)})
        if f1 > best:
            best, wait = f1, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                          if not k.startswith("encoder.") or k in trainable}
        else:
            wait += 1
        log.info(f"    epoch {epoch:3d} loss {np.mean(losses):.4f} val loss {val_loss:.4f} val macro-F1 {f1:.4f} "
                 f"cropland {crop_f1:.4f} ({time.time() - t0:.0f}s){' *' if wait == 0 else ''}")
        if wait >= patience:
            break
    model.load_state_dict(best_state, strict=False)
    return model, history


def label_subset(df, train_mask, fraction, seed):
    """Rows of the training folds that keep their label when only `fraction` of the training parcels is used (E05).
    Every parcel gets one fixed random number (seed); per class (irrigated, rainfed) the training parcels with the
    smallest numbers are kept, so a smaller share is always a subset of a larger one and the class shares stay as in
    the full set. Not-cropland pixels are kept in the tiles that still hold a kept parcel. Validation and test folds
    are not touched. Returns (keep [N] bool, one-row summary dict)."""
    lab, keys = df["label"].to_numpy(), df["parcel_key"].to_numpy()
    if fraction >= 1:
        keep = train_mask.copy()
    else:
        all_keys = np.unique(keys[keys != ""])
        u = dict(zip(all_keys, np.random.default_rng(seed).random(len(all_keys))))
        keep = np.zeros(len(df), dtype=bool)
        for c in (1, 2):
            ks = np.unique(keys[train_mask & (lab == c)])
            n = max(1, int(round(fraction * len(ks))))
            chosen = ks[np.argsort([u[k] for k in ks])][:n]
            keep |= train_mask & np.isin(keys, chosen)
        tiles = np.unique(df["tile_id"].to_numpy()[keep])
        keep |= train_mask & (lab == 3) & np.isin(df["tile_id"].to_numpy(), tiles)
    summary = {"fraction": fraction}
    for c, name in ((1, "irrigated"), (2, "rainfed"), (3, "not_cropland")):
        if c < 3:
            summary[f"parcels_{name}"] = len(np.unique(keys[keep & (lab == c)]))
            summary[f"parcels_{name}_all"] = len(np.unique(keys[train_mask & (lab == c)]))
        summary[f"pixels_{name}"] = int((keep & (lab == c)).sum())
    summary["tiles"] = len(np.unique(df["tile_id"].to_numpy()[keep]))
    return keep, summary


def run_tile_experiment(exp_dir, exp_id, region, run_name, build_model, settings, lr, weight_decay=1e-4, batch=8,
                        epochs=50, patience=8, n_seeds=3, seed=42, workers=8, quick=False, quick_tiles=60,
                        label_fraction=1.0, subset_seed=7, encoder_lr=None, warmup_epochs=0, grad_clip=None):
    """PASTIS rotation with a tile model; build_model(train_rows, df, info) returns a new model for one round / seed.
    label_fraction < 1: only that share of the training-fold parcels is labelled (label_subset); the validation and
    test folds are unchanged, so every run is scored on the same out-of-fold pixels."""
    out_dir = results_dir(exp_dir, region, run_name, quick)
    log = get_logger(out_dir)
    df, X, info = load_samples(region)
    if quick:                                                          # a few tiles per fold
        rng = np.random.default_rng(seed)
        keep = []
        for f in range(5):
            t = df.loc[df["fold"] == f, "tile_id"].unique()
            keep += list(rng.choice(t, min(quick_tiles // 5, len(t)), replace=False))
        sel = df["tile_id"].isin(keep).to_numpy()
        df, X = df[sel].reset_index(drop=True), X[sel]
    table = TileTable(df, info["start_month"])
    y, folds = df["label"].to_numpy(), df["fold"].to_numpy()
    log.info(f"{exp_id} {region} {run_name}: {len(df):,} pixels in {len(table.groups):,} tiles, device {DEVICE}")
    proba = np.zeros((len(df), 3), dtype=np.float32)
    history, subsets = [], []
    for k, train_f, val_f, test_f in rounds():
        if label_fraction < 1:
            keep, summ = label_subset(df, np.isin(folds, train_f), label_fraction, subset_seed)
            subsets.append(dict(summ, round=k + 1))
            tr_rows = np.flatnonzero(keep)
            train_tiles = sorted(df["tile_id"].iloc[tr_rows].unique())
            train_ds = TileDataset(region, table, train_tiles, train_f, augment=True, keep=keep)
            log.info(f"  round {k + 1}: {label_fraction:.0%} of the training parcels: irrigated "
                     f"{summ['parcels_irrigated']:,}/{summ['parcels_irrigated_all']:,}, rainfed "
                     f"{summ['parcels_rainfed']:,}/{summ['parcels_rainfed_all']:,}, not-cropland pixels "
                     f"{summ['pixels_not_cropland']:,}")
        else:
            tr_rows = np.flatnonzero(np.isin(folds, train_f))
            train_ds = TileDataset(region, table, table.tiles_with(train_f), train_f, augment=True)
        val_ds = TileDataset(region, table, table.tiles_with([val_f]), [val_f])
        test_ds = TileDataset(region, table, table.tiles_with([test_f]), [test_f])
        w = class_weights(y[tr_rows])
        log.info(f"  round {k + 1}: tiles train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,}")
        acc = np.zeros((len(df), 3), dtype=np.float32)
        for s in range(n_seeds):
            sd = seed + 1000 * s
            torch.manual_seed(sd)                     # head (and, from scratch, encoder) initialisation per seed
            model = build_model(tr_rows, df, X, info)
            model, hist = train_tile_model(model, train_ds, val_ds, val_f, w, sd, lr, weight_decay, batch, epochs,
                                           patience, workers, log, encoder_lr, warmup_epochs, grad_clip)
            history += [dict(h, round=k + 1, seed=sd) for h in hist]
            rows, p = predict_tiles(model, test_ds, [test_f], workers, batch)
            acc[rows] += p
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        te = np.flatnonzero(folds == test_f)
        proba[te] = acc[te] / n_seeds
    settings = dict(settings, experiment=exp_id, region=region, lr=lr, weight_decay=weight_decay, batch=batch,
                    epochs=epochs, patience=patience, n_seeds=n_seeds, tiles=len(table.groups),
                    label_fraction=label_fraction, subset_seed=subset_seed, encoder_lr=encoder_lr,
                    warmup_epochs=warmup_epochs, grad_clip=grad_clip)
    tables = {"training_history.csv": pd.DataFrame(history)}
    if subsets:
        tables["label_subset.csv"] = pd.DataFrame(subsets)
    save_run(out_dir, df, proba, settings, log, tables, quick)


def s2_stats(df, X, info, rows):
    """Per-band mean / std of the training pixels' Sentinel-2 values (for U-TAE)."""
    return standardiser(s2_series(X[rows], info))
