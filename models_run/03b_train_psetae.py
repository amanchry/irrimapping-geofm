"""
03b_train_psetae.py  —  deep-learning baseline: PSE-TAE trained from scratch on the parcels' pixels
====================================================================================================

What it does
------------
Trains a Pixel-Set Encoder + Temporal Attention Encoder (PSE-TAE) to classify parcels as irrigated or
rainfed directly from their Sentinel-2 (+ Sentinel-1) pixel time series. It is the deep-learning
baseline of the study and answers: "is a pretrained foundation model better than a deep model trained
from scratch on our own labelled parcels?" It uses the same parcels, pixels, spatial folds and output
format as 03_train_evaluate.py, so 04_compare_models.py compares it with all other inputs.

The model
---------
PSE-TAE (Garnot, Landrieu, Giordano & Chehata 2020, "Satellite Image Time Series Classification with
Pixel-Set Encoders and Temporal Self-Attention", CVPR; https://arxiv.org/abs/1911.07757;
official code https://github.com/VSainteuf/pytorch-psetae). It was designed for parcel classification:
  1. Pixel-Set Encoder (PSE): for every month, a small network embeds a random SET of the parcel's
     pixels (no image grid needed), then mean + standard deviation over the set -> one vector per month.
  2. Temporal Attention Encoder (TAE): self-attention over the 12 monthly vectors (with a positional
     encoding of the month) -> one vector per parcel.
  3. Decoder: small MLP -> irrigated / rainfed.
Layer sizes follow the paper's default configuration (MLP1 32-64, mean+std pooling, MLP2 128,
4 attention heads with d_k = 32, MLP3 512-128-128, decoder 64-32-classes). The paper's extra
geometric features (parcel size, perimeter) are not used, so the model sees only the pixels.

How it is validated (same spatial folds as every other input)
------------------------------------------------------------
Outer loop: each of the 5 spatial folds (20 km blocks) is the TEST fold once.
Inside the remaining 4 folds: 3 folds train the network, the next fold (k + 1) is the VALIDATION fold
used only to stop training at the best epoch (early stopping on macro-F1). The architecture and
optimiser settings are fixed in advance (paper defaults), so there is no hyperparameter grid; the test
fold is never used for any decision. Every parcel is predicted once, by a model that never saw its block.

Inputs
------
outputs/dataset/03_tiles/agyear/views/pixelsets.npz   all pixels of every parcel (dataset/04_make_views.py, view B)
outputs/dataset/03_tiles/agyear/views/views_info.json Sentinel-1 orbit used for the chips (same orbit here)
outputs/dataset/02_folds/parcels_folds.geojson        fold and block of every parcel

Steps
-----
1. Load the pixel sets; keep the chosen bands; drop edge pixels (parcels with too few inner pixels keep all).
2. For each outer fold: compute band mean / std on the training pixels only (normalisation).
3. Train PSE-TAE (Adam, weighted cross-entropy for the class imbalance), random pixel sets each epoch;
   keep the epoch with the best validation macro-F1.
4. Predict the test fold, averaging the probabilities over several random pixel sets.
5. Metrics on all out-of-fold predictions with 95% block-bootstrap intervals (as in 03_train_evaluate.py).
6. Save predictions, metrics, training curves and the trained networks.

Output (outputs/study/03_models/psetae_<WINDOW>_<BANDS>/)
---------------------------------------------------------
oof_predictions.csv     every parcel's out-of-fold prediction and rainfed probability (read by 04_compare_models.py)
metrics_with_ci.csv     all metrics: estimate, ci_low, ci_high
metrics_summary.csv     headline metrics "estimate [ci_low-ci_high]"
metrics_per_fold.csv    metrics of each outer fold
training_history.csv    loss and validation macro-F1 per epoch and fold
model_fold<k>.pt        trained network of each outer fold (+ normalisation)
run_config.json, run.log

Run (from the project folder; a GPU is recommended, a CPU works but is slow)
---------------------------------------------------------------------------
    python models_run/03b_train_psetae.py
QUICK_TEST = True gives a fast check (few epochs, small bootstrap).
"""
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_score, recall_score)

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                         # project folder
WINDOW = "agyear"
VIEWS_DIR = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW / "views"
PIXELSET_FILE = VIEWS_DIR / "pixelsets.npz"
SPLIT_FILE = ROOT / "outputs" / "dataset" / "02_folds" / "parcels_folds.geojson"

BANDS = "s2s1"                 # "s2s1" = 10 Sentinel-2 bands + Sentinel-1 VV, VH | "s2" = Sentinel-2 only
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]   # the 10 bands used in the paper
USE_EDGE_PIXELS = False        # False = leave out pixels within 10 m of the outline (as the baseline features)
MIN_INNER_PIXELS = 4           # parcels with fewer inner pixels keep all their pixels

N_PIXELS = 64                  # pixels sampled per parcel and step (paper: 64)
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 15                  # stop when validation macro-F1 has not improved for this many epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EVAL_DRAWS = 5                 # random pixel sets averaged at prediction time
N_BOOT = 1000                  # block-bootstrap draws for the confidence intervals
SEED = 42
QUICK_TEST = False             # True = 3 epochs, 100 bootstrap draws (only to check that everything runs)

RUN_NAME = f"psetae_{WINDOW}_{BANDS}"
OUT_DIR = ROOT / "outputs" / "study" / "03_models" / RUN_NAME
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# ==========================================================================

if QUICK_TEST:
    MAX_EPOCHS, N_BOOT = 3, 100
NODATA = -32768
CLASSES = [1, 2]                                  # 1 = irrigated, 2 = rainfed (network classes 0 / 1)


# ------------------------------------------------------------ data
def load_data(log):
    """Pixel sets -> int16 pixels [N_px, months, channels], per-parcel pixel indices, labels, folds."""
    d = np.load(PIXELSET_FILE)
    bands = [str(b) for b in d["bands"]]
    channels = list(S2_BANDS)
    if BANDS == "s2s1":
        orbit = json.loads((VIEWS_DIR / "views_info.json").read_text())["s1_orbit_for_chips_and_pixelsets"]
        channels += [f"VV_{orbit}", f"VH_{orbit}"]
    pixels = d["pixels"][:, :, [bands.index(c) for c in channels]]   # keep the chosen bands only
    edge, offsets = d["edge"].astype(bool), d["offsets"]
    ids, labels = d["parcel_id"].astype(str), d["label"].astype(int)

    # Pixel indices of each parcel: inner pixels if there are enough, otherwise all of them.
    idx = []
    for i in range(len(ids)):
        own = np.arange(offsets[i], offsets[i + 1])
        inner = own[~edge[own]]
        idx.append(own if USE_EDGE_PIXELS or len(inner) < MIN_INNER_PIXELS else inner)

    # Folds and blocks (same file as 03_train_evaluate.py).
    gj = json.loads(SPLIT_FILE.read_text())
    meta = pd.DataFrame([f["properties"] for f in gj["features"]]).set_index("parcel_id")
    keep = np.isin(ids, meta.index)
    log.info(f"{len(ids):,} parcels with pixels ({keep.sum():,} with a fold), {len(pixels):,} pixels, "
             f"channels {channels}")
    ids, labels, idx = ids[keep], labels[keep], [x for x, k in zip(idx, keep) if k]
    folds = meta.loc[ids, "fold"].to_numpy().astype(int)
    blocks = meta.loc[ids, "block_id"].astype(str).to_numpy()
    return pixels, idx, ids, labels, folds, blocks, channels


def physical(px, channels):
    """int16 -> float: Sentinel-2 reflectance (0-1), Sentinel-1 dB; missing values -> NaN."""
    x = px.astype(np.float32)
    x[px == NODATA] = np.nan
    s2 = [i for i, c in enumerate(channels) if c in S2_BANDS]
    s1 = [i for i, c in enumerate(channels) if c not in S2_BANDS]
    x[..., s2] /= 10000.0
    x[..., s1] /= 100.0
    return x


def norm_stats(pixels, idx, parcels, channels, rng, n_max=200_000):
    """Mean and std per channel from (a sample of) the TRAINING parcels' pixels."""
    pool = np.concatenate([idx[i] for i in parcels])
    pick = rng.choice(pool, size=min(n_max, len(pool)), replace=False)
    x = physical(pixels[pick], channels)                         # [n, months, channels]
    return np.nanmean(x, axis=(0, 1)), np.nanstd(x, axis=(0, 1)) + 1e-6


def make_batch(pixels, idx, parcels, stats, channels, rng):
    """Random pixel set of each parcel -> x [B, months, channels, N_PIXELS], mask [B, N_PIXELS]."""
    mean, std = stats
    T, C = pixels.shape[1], pixels.shape[2]
    x = np.zeros((len(parcels), T, C, N_PIXELS), np.float32)
    mask = np.zeros((len(parcels), N_PIXELS), np.float32)
    for b, i in enumerate(parcels):
        own = idx[i]
        pick = rng.choice(own, size=N_PIXELS, replace=False) if len(own) > N_PIXELS else own   # small parcel: all pixels
        v = (physical(pixels[pick], channels) - mean) / std     # standardise
        x[b, :, :, :len(pick)] = np.nan_to_num(v, nan=0.0).transpose(1, 2, 0)   # missing -> 0 (= mean)
        mask[b, :len(pick)] = 1.0
    return torch.from_numpy(x), torch.from_numpy(mask)


# ------------------------------------------------------------ model (PSE-TAE, Garnot et al. 2020)
def mlp(sizes, last_relu=True):
    """Linear + BatchNorm + ReLU blocks."""
    layers = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers += [nn.Linear(a, b), nn.BatchNorm1d(b)]
        if last_relu or k < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PixelSetEncoder(nn.Module):
    """Per month: MLP1 on each pixel -> masked mean + std over the pixel set -> MLP2."""
    def __init__(self, n_channels, mlp1=(32, 64), mlp2=(128,)):
        super().__init__()
        self.mlp1 = mlp([n_channels, *mlp1])
        self.mlp2 = mlp([2 * mlp1[-1], *mlp2])

    def forward(self, x, mask):                       # x [B, T, C, S], mask [B, S]
        B, T, C, S = x.shape
        h = self.mlp1(x.permute(0, 1, 3, 2).reshape(B * T * S, C)).view(B, T, S, -1)   # embed every pixel
        m = mask[:, None, :, None]                                                       # [B, 1, S, 1]
        n = m.sum(2).clamp(min=1)
        mean = (h * m).sum(2) / n                                                        # mean over the set
        std = (((h - mean[:, :, None]) ** 2 * m).sum(2) / n).clamp(min=1e-8).sqrt()      # std over the set
        out = self.mlp2(torch.cat([mean, std], -1).view(B * T, -1))
        return out.view(B, T, -1)                                                        # [B, T, 128]


class TemporalAttentionEncoder(nn.Module):
    """Self-attention over the months with one 'master query' per head (TAE of the paper)."""
    def __init__(self, positions, d_in=128, n_head=4, d_k=32, mlp3=(512, 128, 128), dropout=0.2, T=1000):
        super().__init__()
        self.n_head, self.d_k = n_head, d_k
        self.in_norm = nn.LayerNorm(d_in)
        pe = torch.zeros(len(positions), d_in)                        # sinusoidal encoding of the day of the season
        pos = torch.tensor(positions, dtype=torch.float32)[:, None]
        div = torch.pow(T, torch.arange(0, d_in, 2, dtype=torch.float32) / d_in)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / div), torch.cos(pos / div)
        self.register_buffer("pe", pe)
        self.fc_q = nn.Linear(d_in, n_head * d_k)
        self.fc_k = nn.Linear(d_in, n_head * d_k)
        self.fc_q2 = nn.Sequential(nn.BatchNorm1d(n_head * d_k), nn.Linear(n_head * d_k, n_head * d_k))
        self.mlp3 = mlp([n_head * d_in, *mlp3[1:]])
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(mlp3[-1])

    def forward(self, x):                             # x [B, T, d_in]
        B, T, _ = x.shape
        x = self.in_norm(x) + self.pe[None]           # add the month position
        q = self.fc_q2(self.fc_q(x).mean(1)).view(B, self.n_head, self.d_k)   # master query (mean over time)
        k = self.fc_k(x).view(B, T, self.n_head, self.d_k)
        att = torch.softmax(torch.einsum("bhd,bthd->bht", q, k) / math.sqrt(self.d_k), dim=-1)   # weight per month
        out = torch.einsum("bht,btd->bhd", att, x).reshape(B, -1)              # weighted sum, heads side by side
        return self.out_norm(self.dropout(self.mlp3(out))), att


class PSETAE(nn.Module):
    def __init__(self, n_channels, positions, n_classes=2):
        super().__init__()
        self.pse = PixelSetEncoder(n_channels)
        self.tae = TemporalAttentionEncoder(positions)
        self.decoder = nn.Sequential(mlp([128, 64, 32]), nn.Linear(32, n_classes))

    def forward(self, x, mask):
        z, _ = self.tae(self.pse(x, mask))
        return self.decoder(z)


# ------------------------------------------------------------ training
def predict(model, pixels, idx, parcels, stats, channels, seed):
    """Rainfed probability per parcel, averaged over EVAL_DRAWS random pixel sets."""
    model.eval()
    rng = np.random.default_rng(seed)
    prob = np.zeros(len(parcels))
    with torch.no_grad():
        for _ in range(EVAL_DRAWS):
            for s in range(0, len(parcels), 512):
                x, m = make_batch(pixels, idx, parcels[s:s + 512], stats, channels, rng)
                prob[s:s + 512] += torch.softmax(model(x.to(DEVICE), m.to(DEVICE)), -1)[:, 1].cpu().numpy()
    return prob / EVAL_DRAWS


def train_fold(pixels, idx, labels, train, val, channels, positions, outer, log):
    """Train on `train` parcels, early stopping on macro-F1 of `val` parcels. Returns model, stats, history."""
    rng = np.random.default_rng(SEED + outer)
    torch.manual_seed(SEED + outer)
    stats = norm_stats(pixels, idx, train, channels, rng)              # normalisation from training pixels only
    y = (labels == 2).astype(np.int64)                                 # 0 irrigated, 1 rainfed
    w = torch.tensor(len(train) / (2 * np.bincount(y[train], minlength=2)), dtype=torch.float32, device=DEVICE)
    model = PSETAE(len(channels), positions).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order, losses, t0 = rng.permutation(train), [], time.time()
        for s in range(0, len(order), BATCH_SIZE):
            batch = order[s:s + BATCH_SIZE]
            if len(batch) < 2:                                         # BatchNorm needs > 1 sample
                continue
            x, m = make_batch(pixels, idx, batch, stats, channels, rng)    # new random pixel sets every epoch
            loss = F.cross_entropy(model(x.to(DEVICE), m.to(DEVICE)), torch.from_numpy(y[batch]).to(DEVICE), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        p_val = predict(model, pixels, idx, val, stats, channels, SEED)
        f1 = f1_score(y[val], (p_val >= 0.5).astype(int), average="macro")
        history.append({"outer_fold": outer, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": f1})
        log.info(f"  fold {outer} epoch {epoch:3d}  loss {np.mean(losses):.4f}  val macro-F1 {f1:.3f}  ({time.time() - t0:.0f}s)")
        if f1 > best:                                                  # keep the best epoch
            best, best_state, wait = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, stats, history, best


# ------------------------------------------------------------ metrics (same definitions as 03_train_evaluate.py)
def metrics(y_true, y_pred, p_rainfed):
    is_rf = (y_true == 2).astype(int)
    m = {"macro_f1": f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0),
         "overall_accuracy": accuracy_score(y_true, y_pred),
         "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
    for c, name in ((1, "irrigated"), (2, "rainfed")):
        m[f"f1_{name}"] = f1_score(y_true, y_pred, pos_label=c, zero_division=0)
        m[f"precision_{name}"] = precision_score(y_true, y_pred, pos_label=c, zero_division=0)
        m[f"recall_{name}"] = recall_score(y_true, y_pred, pos_label=c, zero_division=0)
    if len(np.unique(y_true)) == 2:
        m["pr_auc_rainfed"] = average_precision_score(is_rf, p_rainfed)
        m["brier"] = brier_score_loss(is_rf, p_rainfed)
    return m


def block_bootstrap(oof):
    """95% intervals: whole spatial blocks resampled with replacement (blocks are the independent units)."""
    y, yhat, p = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["score"].to_numpy()
    blocks = oof["block_id"].to_numpy()
    groups = {b: np.flatnonzero(blocks == b) for b in np.unique(blocks)}
    names = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        sel = np.concatenate([groups[b] for b in rng.choice(names, size=len(names), replace=True)])
        if len(np.unique(y[sel])) == 2:
            draws.append(metrics(y[sel], yhat[sel], p[sel]))
    draws = pd.DataFrame(draws)
    point = metrics(y, yhat, p)
    return pd.DataFrame([{"metric": k, "estimate": v, "ci_low": np.nanquantile(draws[k], 0.025),
                          "ci_high": np.nanquantile(draws[k], 0.975)} for k, v in point.items()]).set_index("metric")


# ------------------------------------------------------------ main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(OUT_DIR / "run.log", mode="w"), logging.StreamHandler()])
    log = logging.getLogger()
    log.info(f"Run {RUN_NAME} on {DEVICE}{' (QUICK_TEST)' if QUICK_TEST else ''}")

    # 1. Data.
    pixels, idx, ids, labels, folds, blocks, channels = load_data(log)
    n_months = pixels.shape[1]
    positions = [int(round(15 + 30.44 * k)) for k in range(n_months)]   # mid-month, days since 1 September

    # 2-4. Outer spatial folds.
    parts, fold_rows, history = [], [], []
    fold_ids = sorted(np.unique(folds))
    for outer in fold_ids:
        val_fold = fold_ids[(fold_ids.index(outer) + 1) % len(fold_ids)]          # next fold = early-stopping fold
        test = np.flatnonzero(folds == outer)
        val = np.flatnonzero(folds == val_fold)
        train = np.flatnonzero((folds != outer) & (folds != val_fold))
        log.info(f"Outer fold {outer}: train {len(train):,} | validation (fold {val_fold}) {len(val):,} | test {len(test):,}")
        model, stats, hist, best = train_fold(pixels, idx, labels, train, val, channels, positions, outer, log)
        history += hist
        p = predict(model, pixels, idx, test, stats, channels, SEED + 100 + outer)     # test fold, never seen before
        y_pred = np.where(p >= 0.5, 2, 1)
        parts.append(pd.DataFrame({"model": "psetae", "parcel_id": ids[test], "outer_fold": outer,
                                   "y_true": labels[test], "y_pred": y_pred, "score": p,
                                   "is_probability": True, "block_id": blocks[test]}))
        m = metrics(labels[test], y_pred, p)
        fold_rows.append({"outer_fold": outer, "best_val_macro_f1": best, "epochs": len(hist), **m})
        log.info(f"Outer fold {outer}: test macro-F1 {m['macro_f1']:.3f}")
        torch.save({"state_dict": model.state_dict(), "mean": stats[0], "std": stats[1], "channels": channels,
                    "positions": positions}, OUT_DIR / f"model_fold{outer}.pt")

    # 5. Metrics on all out-of-fold predictions.
    oof = pd.concat(parts, ignore_index=True)
    ci = block_bootstrap(oof)
    summary = ci.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}-{r.ci_high:.3f}]", axis=1).rename("psetae")
    log.info("Out-of-fold metrics (95% block-bootstrap CI):\n" + summary.to_string())

    # 6. Save.
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    pd.concat({"psetae": ci}, names=["model"]).to_csv(OUT_DIR / "metrics_with_ci.csv")
    summary.to_frame().to_csv(OUT_DIR / "metrics_summary.csv")
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "metrics_per_fold.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    (OUT_DIR / "run_config.json").write_text(json.dumps({
        "run": RUN_NAME, "model": "PSE-TAE (Garnot et al. 2020)", "channels": channels,
        "use_edge_pixels": USE_EDGE_PIXELS, "n_pixels": N_PIXELS, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "eval_draws": EVAL_DRAWS, "loss": "class-weighted cross-entropy",
        "validation": "outer 5 spatial folds; per outer fold 3 folds train, next fold early stopping",
        "positions_days": positions, "n_parcels": int(len(oof)), "n_boot": N_BOOT, "seed": SEED,
        "quick_test": QUICK_TEST, "device": DEVICE}, indent=2))
    log.info(f"Saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
