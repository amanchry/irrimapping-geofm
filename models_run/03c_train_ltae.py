"""
03c_train_ltae.py  —  neural heads on the frozen inputs: MLP on the PCA input (E03) and L-TAE + MLP (E04)
=========================================================================================================

What it does
------------
Trains a small neural head on one input (S2 baseline features or the embeddings of one foundation model) with
the same parcels, folds and PASTIS rotation as 03_train_evaluate.py, and writes its out-of-fold predictions in the
same format, so 04_compare_models.py compares it directly with the classical classifiers.

  HEAD = "mlp"   (E03)  exactly the E02 input: 12 months concatenated, months without a clear pixel filled with
                        the training median, standardised, PCA 128 (fitted on the training folds; the baseline's 180
                        features are only standardised), then an MLP 128 -> 256 -> 128 -> 2 (ReLU, batch
                        normalisation, dropout). E02 -> E03 changes only the head.
  HEAD = "ltae"  (E04)  the 12 months as a sequence: per-channel standardisation (training folds), months without
                        a clear pixel masked in the attention, L-TAE (Garnot & Landrieu 2020; 16 heads, d_k 8,
                        d_model 256, output MLP 256-128), then a linear layer with 2 outputs. E03 -> E04 changes only
                        how the months are combined. AgriFM has one vector per year and is not used with L-TAE.

Training (both heads): AdamW, class-weighted cross-entropy, early stopping on the validation fold's macro-F1
(PATIENCE epochs), the best epoch is kept. N_SEEDS independent trainings per round; the out-of-fold probability is
their mean and the spread over seeds is saved. Label efficiency: TRAIN_FRACTION < 1 trains on that share of the
training blocks (same block draws as scripts 03 and 03b, models_run/experiment_utils.py).

Steps
-----
1. Read the input (as 03_train_evaluate.py) and the folds of the region; keep parcels with data and a fold.
2. Five rounds (PASTIS): 3 folds train (all blocks or a draw), the next fold validates (early stopping), 1 fold tests;
   every fitted step (imputation, standardisation, PCA) uses the training parcels only.
3. Per round and seed: train the head, predict the test fold; average the seeds.
4. Metrics on all out-of-fold predictions with 95% block-bootstrap CI; per fold and per seed.
5. L-TAE: attention weight of every month, averaged per class on the test folds (which months matter).

Output: experiments/<EXPERIMENT>/results/<REGION>/<RUN_NAME>/ (or outputs/<REGION>/study/03_models/<RUN_NAME>/)
-------------------------------------------------------------------------------------------------------------
RUN_NAME = <input run>_<HEAD>[_f010_d1], e.g. clay_agyear_tiles_mlp, clay_agyear_tiles_2023-24_ltae,
s1s2-s2_agyear_tiles-inner_ltae
oof_predictions.csv     model (= HEAD), parcel_id, outer_fold, y_true, y_pred, score, is_probability, block_id
metrics_with_ci.csv, metrics_summary.csv, metrics_per_fold.csv, metrics_per_seed.csv, oof_per_seed.csv (the
probability of every seed, so learning curves can compare single-seed runs), training_history.csv,
attention_by_month.csv (L-TAE), run_config.json, run.log

Run (from the project folder)
-----------------------------
    python experiments/run_experiment.py     # EXPERIMENT = "E03_frozen_pca_mlp" or "E04_frozen_ltae"
    python models_run/03c_train_ltae.py      # development mode (settings below)
QUICK_TEST = True gives a fast check (3 epochs, 100 bootstrap draws).
"""
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

from experiment_utils import (describe_draw, draw_table, draw_training_blocks, env_override, fraction_tag,
                              region_paths, results_dir, start_month, views_name)

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                 # project folder
EXPERIMENT = None              # e.g. "E03_frozen_pca_mlp" / "E04_frozen_ltae": settings from its config.yaml (section "ltae")
REGION = "morocco"             # "morocco" | "utah"
VIEWS_NAME = None              # None = the region's default views folder (Morocco "views", Utah "views_2023-24")
INPUT = "clay"                 # "s1s2" | "clay" | "prithvi" | "terramind" | "agrifm" (agrifm: HEAD = "mlp" only)
WINDOW = "agyear"
HEAD = "mlp"                   # "mlp" (E03: MLP on the PCA input) | "ltae" (E04: L-TAE + MLP on the monthly sequence)

FEATURE_MODE = "concat"        # mlp head: "concat" (12 months + PCA) | "mean" (average over months)
PCA_DIM = 128                  # mlp head: PCA components (fitted on the training folds)
TILE_FEATURES = "inner"        # baseline: parcel features without edge pixels (as 03_train_evaluate.py)
VALIDATION = "pastis"          # only the PASTIS rotation (3 train / 1 validation / 1 test)

N_HEADS = 16                   # L-TAE (Garnot & Landrieu 2020 defaults)
D_K = 8
D_MODEL = 256
MLP = [256, 128]               # hidden layers (MLP head) / output MLP of the L-TAE
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 15                  # early stopping: epochs without a better validation macro-F1
N_SEEDS = 3                    # trainings per round (SEED, SEED+1000, ...); OOF = mean probability

TRAIN_FRACTION = 1.0           # label efficiency: share of the training blocks (1.0 = all)
FRACTION_DRAW = 1              # block draw (1, 2, 3); same draw as scripts 03 and 03b
SEED = 42
N_BOOT = 1000
QUICK_TEST = False             # True = 3 epochs, 100 bootstrap draws

EXPERIMENT = os.environ.get("EXPERIMENT", EXPERIMENT)
if EXPERIMENT:
    import yaml
    _cfg = yaml.safe_load((ROOT / "experiments" / EXPERIMENT / "config.yaml").read_text(encoding="utf-8"))
    for _key, _value in (_cfg.get("ltae") or {}).items():
        if _key not in globals():
            raise SystemExit(f"config.yaml: unknown setting '{_key}' for 03c_train_ltae.py")
        globals()[_key] = _value
INPUT = os.environ.get("EXPERIMENT_INPUT", INPUT)
env_override(globals())

VIEWS_NAME = views_name(REGION, VIEWS_NAME)
_P = region_paths(REGION, WINDOW, VIEWS_NAME)
SPLIT_FILE, VIEWS_DIR = _P["split_file"], _P["views_dir"]
EMB_TAG = _P["embedding_tag"]
if HEAD not in ("mlp", "ltae"):
    raise SystemExit(f"HEAD must be 'mlp' or 'ltae', not {HEAD!r}")
if HEAD == "ltae" and INPUT == "agrifm":
    raise SystemExit("AgriFM has one vector per year: L-TAE needs a sequence of months (use HEAD = 'mlp')")
if VALIDATION != "pastis":
    raise SystemExit("03c_train_ltae.py only supports VALIDATION = 'pastis'")
INPUT_RUN = f"s1s2-s2_{WINDOW}_tiles-{TILE_FEATURES}" if INPUT == "s1s2" else f"{INPUT}_{EMB_TAG}"
RUN_NAME = f"{INPUT_RUN}_{HEAD}" + fraction_tag(TRAIN_FRACTION, FRACTION_DRAW)
OUT_DIR = results_dir(EXPERIMENT, REGION, RUN_NAME, _P["study_dir"] / "03_models", QUICK_TEST)   # quick runs: results_quick/
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if QUICK_TEST:
    EPOCHS, N_BOOT = 3, 100
# ==========================================================================

CLASSES = [1, 2]
S1_VARIABLES = {"VV", "VH", "VV_ASC", "VH_ASC", "VV_DSC", "VH_DSC"}
NOT_FEATURES = {"n_px_used", "used_inner", "fold", "label", "ag_season"}
COUNTS = ("S2_NCLEAR", "S1_N_ASC", "S1_N_DSC")


# ------------------------------------------------------------ data
def load_folds():
    t = pyogrio.read_dataframe(SPLIT_FILE, columns=["parcel_id", "label", "block_id", "fold"], read_geometry=False)
    t["parcel_id"] = t["parcel_id"].astype(str)
    return t.drop_duplicates("parcel_id").set_index("parcel_id")


def load_baseline():
    """Sentinel-2 parcel features (view A) as a sequence [N, 12 months, 15 variables]; fully cloudy months are
    filled by linear interpolation in time per parcel (as 03_train_evaluate.py)."""
    feats = pd.read_csv(VIEWS_DIR / f"parcel_features_{TILE_FEATURES}.csv", dtype={"parcel_id": str})
    feats = feats.drop_duplicates("parcel_id").set_index("parcel_id")
    var = lambda c: c.rsplit("_", 1)[0]
    cols = [c for c in feats.columns if c not in NOT_FEATURES and "_m" in c
            and var(c) not in COUNTS and var(c) not in S1_VARIABLES]
    variables = list(dict.fromkeys(var(c) for c in cols))
    months = sorted({c.rsplit("_", 1)[1] for c in cols})
    seq = np.stack([feats[[f"{v}_{m}" for m in months]].to_numpy(np.float32) for v in variables], axis=2)
    keep = ~np.isnan(seq).all(axis=(1, 2))
    seq, ids = seq[keep], feats.index[keep]
    for j in range(seq.shape[2]):                                    # interpolate gaps over the months
        seq[:, :, j] = pd.DataFrame(seq[:, :, j]).interpolate(axis=1, limit_direction="both").to_numpy()
    valid = ~np.isnan(seq).any(axis=2)
    first = pd.Timestamp(year=2023, month=start_month(REGION), day=1)        # 'm01' -> first month of the year
    names = [(first + pd.DateOffset(months=int(m[1:]) - 1)).strftime("%b") for m in months]
    return ids.to_numpy().astype(str), seq, valid, names


def load_input():
    """ids [N], x [N, T, C] float32 (NaN where missing), valid [N, T], step names."""
    if INPUT == "s1s2":
        return load_baseline()
    d = np.load(_P["embeddings_dir"] / f"{INPUT}_{EMB_TAG}.npz", allow_pickle=False)
    x = d["embeddings"].astype(np.float32)
    valid = d["valid"].astype(bool)
    x[~valid] = np.nan
    return d["parcel_id"].astype(str), x, valid, [str(s) for s in d["steps"]]


# ------------------------------------------------------------ features of one round (fitted on training parcels)
def mlp_features(x, train):
    """E03 input: concat months -> training-median fill -> standardise -> PCA (embeddings only)."""
    n, t, c = x.shape
    flat = x.reshape(n, t * c) if FEATURE_MODE == "concat" else np.nanmean(x, axis=1)
    med = np.nanmedian(flat[train], axis=0)
    med = np.where(np.isnan(med), 0, med)
    flat = np.where(np.isnan(flat), med, flat)
    mu, sd = flat[train].mean(0), flat[train].std(0) + 1e-6
    flat = (flat - mu) / sd
    if INPUT != "s1s2" and FEATURE_MODE == "concat" and flat.shape[1] > PCA_DIM:
        pca = PCA(n_components=PCA_DIM, random_state=SEED).fit(flat[train])
        flat = pca.transform(flat)
        flat = (flat - flat[train].mean(0)) / (flat[train].std(0) + 1e-6)
    return flat.astype(np.float32)


def sequence_features(x, valid, train):
    """E04 input: per-channel standardisation over the training parcels' observed months; missing -> 0 (masked)."""
    obs = x[train][valid[train]]
    mu, sd = np.nanmean(obs, axis=0), np.nanstd(obs, axis=0) + 1e-6
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


# ------------------------------------------------------------ models
class MLPHead(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, 2)

    def forward(self, x, mask=None):
        return self.out(self.body(x))


class LTAE(nn.Module):
    """Lightweight temporal attention encoder (Garnot & Landrieu 2020): channels -> d_model, sinusoidal encoding of
    the acquisition day, one learned master query per head, values = the d_model channels split over the heads;
    the attended heads are concatenated and passed through an MLP. Masked months get no attention."""

    def __init__(self, d_in, positions, n_head, d_k, d_model, mlp, dropout, T=1000):
        super().__init__()
        self.n_head, self.d_k = n_head, d_k
        self.inconv = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model))
        pos = torch.tensor(positions, dtype=torch.float32)[:, None]
        div = torch.pow(T, 2 * (torch.arange(d_model // n_head) // 2) / (d_model // n_head))
        table = pos / div
        table[:, 0::2], table[:, 1::2] = torch.sin(table[:, 0::2]), torch.cos(table[:, 1::2])
        self.register_buffer("pe", table.repeat(1, n_head))          # [T, d_model]
        self.query = nn.Parameter(torch.randn(n_head, d_k) * d_k ** -0.5)
        self.fc_k = nn.Linear(d_model, n_head * d_k)
        layers, d = [], d_model
        for h in mlp:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2)
        self.last_attention = None

    def forward(self, x, mask):
        b, t, _ = x.shape
        h = self.inconv(x) + self.pe[:t]                                  # [B, T, d_model]
        k = self.fc_k(h).view(b, t, self.n_head, self.d_k)
        score = torch.einsum("bthd,hd->bht", k, self.query) / math.sqrt(self.d_k)
        score = score.masked_fill(~mask[:, None, :], -1e9)
        att = torch.softmax(score, dim=-1)                                # [B, heads, T]
        self.last_attention = att.detach()
        v = h.view(b, t, self.n_head, -1)                                 # [B, T, heads, d_model / heads]
        z = torch.einsum("bht,bthc->bhc", att, v).reshape(b, -1)          # [B, d_model]
        return self.out(self.out_norm(self.dropout(self.mlp(z))))


# ------------------------------------------------------------ training
def predict(model, x, mask, rows):
    model.eval()
    probs, atts = [], []
    with torch.no_grad():
        for s in range(0, len(rows), 1024):
            r = rows[s:s + 1024]
            logits = model(torch.from_numpy(x[r]).to(DEVICE), torch.from_numpy(mask[r]).to(DEVICE))
            probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
            if HEAD == "ltae":
                atts.append(model.last_attention.mean(1).cpu().numpy())   # mean over heads: [B, T]
    return np.concatenate(probs), (np.concatenate(atts) if atts else None)


def train_round(x, mask, y01, train, val, positions, seed, outer, log):
    torch.manual_seed(seed + outer)
    rng = np.random.default_rng(seed + outer)
    if HEAD == "mlp":
        model = MLPHead(x.shape[1], MLP, DROPOUT)
    else:
        model = LTAE(x.shape[2], positions, N_HEADS, D_K, D_MODEL, MLP, DROPOUT)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    w = torch.tensor(len(train) / (2 * np.bincount(y01[train], minlength=2)), dtype=torch.float32, device=DEVICE)
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order, losses = rng.permutation(train), []
        for s in range(0, len(order), BATCH_SIZE):
            batch = order[s:s + BATCH_SIZE]
            if len(batch) < 2:                                            # batch normalisation needs > 1 sample
                continue
            logits = model(torch.from_numpy(x[batch]).to(DEVICE), torch.from_numpy(mask[batch]).to(DEVICE))
            loss = F.cross_entropy(logits, torch.from_numpy(y01[batch]).to(DEVICE), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        p_val, _ = predict(model, x, mask, val)
        f1 = f1_score(y01[val], (p_val >= 0.5).astype(int), average="macro")
        history.append({"seed": seed, "outer_fold": outer, "epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_macro_f1": f1})
        if f1 > best:
            best, best_state, wait = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    log.info(f"  fold {outer} seed {seed}: {len(history)} epochs, best validation macro-F1 {best:.3f}")
    return model, history, best


# ------------------------------------------------------------ metrics (same definitions as 03_train_evaluate.py)
def metrics(y, yhat, p):
    m = {"overall_accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
         "macro_f1": f1_score(y, yhat, average="macro", labels=CLASSES, zero_division=0),
         "mcc": matthews_corrcoef(y, yhat)}
    for c, name in ((1, "irrigated"), (2, "rainfed")):
        m[f"precision_{name}"] = precision_score(y, yhat, pos_label=c, zero_division=0)
        m[f"recall_{name}"] = recall_score(y, yhat, pos_label=c, zero_division=0)
        m[f"f1_{name}"] = f1_score(y, yhat, pos_label=c, zero_division=0)
    if len(np.unique(y)) == 2:
        rf = (y == 2).astype(int)
        m["roc_auc"] = roc_auc_score(rf, p)
        m["pr_auc_rainfed"] = average_precision_score(rf, p)
        m["brier"] = brier_score_loss(rf, p)
        bins = np.minimum((p * 10).astype(int), 9)
        m["ece"] = sum((bins == b).mean() * abs(rf[bins == b].mean() - p[bins == b].mean())
                       for b in range(10) if (bins == b).any())
    return m


def block_bootstrap(oof):
    y, yhat, p, blocks = (oof[c].to_numpy() for c in ("y_true", "y_pred", "score", "block_id"))
    groups = {b: np.flatnonzero(blocks == b) for b in np.unique(blocks)}
    names = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        sel = np.concatenate([groups[b] for b in rng.choice(names, size=len(names), replace=True)])
        if len(np.unique(y[sel])) == 2:
            draws.append(metrics(y[sel], yhat[sel], p[sel]))
    draws, point = pd.DataFrame(draws), metrics(y, yhat, p)
    return pd.DataFrame([{"metric": k, "estimate": v, "ci_low": np.nanquantile(draws[k], 0.025),
                          "ci_high": np.nanquantile(draws[k], 0.975)} for k, v in point.items()]).set_index("metric")


# ------------------------------------------------------------ main
def main():
    if os.environ.get("EXPERIMENT_SKIP_DONE") and (OUT_DIR / "oof_predictions.csv").exists():
        print(f"{RUN_NAME}: already done in {OUT_DIR} (skipped; delete the folder to run it again)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", force=True,
                        handlers=[logging.FileHandler(OUT_DIR / "run.log", mode="w"), logging.StreamHandler()])
    log = logging.getLogger()
    t0 = time.time()

    ids, x, valid, steps = load_input()
    if HEAD == "ltae" and x.shape[1] < 2:
        raise SystemExit(f"{INPUT} has {x.shape[1]} time step: L-TAE needs a sequence of months (use HEAD = 'mlp')")
    folds_table = load_folds()
    keep = np.isin(ids, folds_table.index) & valid.any(axis=1)
    ids, x, valid = ids[keep], x[keep], valid[keep]
    meta = folds_table.loc[ids]
    labels, folds = meta["label"].to_numpy().astype(int), meta["fold"].to_numpy().astype(int)
    blocks = meta["block_id"].astype(str).to_numpy()
    y01 = (labels == 2).astype(np.int64)
    positions = [int(round(15 + 30.44 * k)) for k in range(x.shape[1])]   # mid-month, days since the season start
    log.info(f"Run {RUN_NAME} on {DEVICE}{' (QUICK_TEST)' if QUICK_TEST else ''} | region {REGION} ({VIEWS_NAME}) | "
             f"input {INPUT} {tuple(x.shape)} | head {HEAD} | training blocks {TRAIN_FRACTION:.0%} | seeds {N_SEEDS}")
    log.info(f"{len(ids):,} parcels ({(labels == 1).sum():,} irrigated, {(labels == 2).sum():,} rainfed); "
             f"empty months {1 - valid.mean():.1%}")

    split_draw = draw_table(SPLIT_FILE, VIEWS_DIR) if TRAIN_FRACTION < 1 else None
    seeds = [SEED + 1000 * s for s in range(N_SEEDS)]
    fold_ids = sorted(np.unique(folds))
    parts, fold_rows, history, seed_parts, draws, att_rows = [], [], [], [], [], []
    for k, outer in enumerate(fold_ids):
        val_fold = fold_ids[(k + 1) % len(fold_ids)]
        test = np.flatnonzero(folds == outer)
        val = np.flatnonzero(folds == val_fold)
        train = np.flatnonzero((folds != outer) & (folds != val_fold))
        if TRAIN_FRACTION < 1:
            train_folds = [f for f in fold_ids if f not in (outer, val_fold)]
            chosen = draw_training_blocks(split_draw, train_folds, TRAIN_FRACTION, FRACTION_DRAW, REGION, outer)
            train = train[np.isin(blocks[train], [str(b) for b in chosen])]
            draws.append({"round": k + 1, "test_fold": int(outer), **describe_draw(split_draw, chosen, train_folds)})
            log.info(f"  {TRAIN_FRACTION:.0%} of the training blocks (draw {FRACTION_DRAW}): {draws[-1]}")
        log.info(f"Round {k + 1}: train {len(train):,} | validation (fold {val_fold}) {len(val):,} | "
                 f"test (fold {outer}) {len(test):,}")
        if HEAD == "mlp":
            xr, mask = mlp_features(x, train), np.ones((len(x), 1), dtype=bool)
        else:
            xr, mask = sequence_features(x, valid, train), valid
        p, best = np.zeros(len(test)), []
        for seed in seeds:
            model, hist, b = train_round(xr, mask, y01, train, val, positions, seed, outer, log)
            history += hist
            best.append(b)
            p_seed, att = predict(model, xr, mask, test)
            p += p_seed / len(seeds)
            seed_parts.append(pd.DataFrame({"seed": seed, "parcel_id": ids[test], "outer_fold": outer,
                                            "y_true": labels[test], "score": p_seed}))
            if att is not None and seed == seeds[0]:
                for c in CLASSES:
                    a = att[labels[test] == c].mean(0)
                    att_rows += [{"outer_fold": outer, "class": c, "step": steps[i], "attention": float(a[i])}
                                 for i in range(len(steps))]
        y_pred = np.where(p >= 0.5, 2, 1)
        parts.append(pd.DataFrame({"model": HEAD, "parcel_id": ids[test], "outer_fold": outer, "y_true": labels[test],
                                   "y_pred": y_pred, "score": p, "is_probability": True, "block_id": blocks[test]}))
        m = metrics(labels[test], y_pred, p)
        fold_rows.append({"model": HEAD, "outer_fold": outer, "best_val_macro_f1": float(np.mean(best)), **m})
        log.info(f"Round {k + 1}: test macro-F1 {m['macro_f1']:.3f} (mean probability of {len(seeds)} seeds)")

    oof = pd.concat(parts, ignore_index=True)
    ci = block_bootstrap(oof)
    summary = ci.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}-{r.ci_high:.3f}]", axis=1).rename(HEAD)
    log.info("Out-of-fold metrics (95% block-bootstrap CI):\n" + summary.to_string())
    per_seed = pd.concat(seed_parts, ignore_index=True)
    seed_table = pd.DataFrame([{"seed": s, **metrics(g["y_true"].to_numpy(), np.where(g["score"] >= 0.5, 2, 1),
                                                     g["score"].to_numpy())} for s, g in per_seed.groupby("seed")])
    log.info("Per seed macro-F1: " + ", ".join(f"{r.macro_f1:.3f}" for r in seed_table.itertuples()))

    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    pd.concat({HEAD: ci}, names=["model"]).to_csv(OUT_DIR / "metrics_with_ci.csv")
    summary.to_frame().to_csv(OUT_DIR / "metrics_summary.csv")
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "metrics_per_fold.csv", index=False)
    seed_table.to_csv(OUT_DIR / "metrics_per_seed.csv", index=False)
    per_seed.assign(model=HEAD).to_csv(OUT_DIR / "oof_per_seed.csv", index=False)   # single-seed scores (learning curves)
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    if att_rows:
        pd.DataFrame(att_rows).to_csv(OUT_DIR / "attention_by_month.csv", index=False)
    (OUT_DIR / "run_config.json").write_text(json.dumps({
        "run": RUN_NAME, "experiment": EXPERIMENT, "region": REGION, "views": VIEWS_NAME, "input": INPUT,
        "head": HEAD, "input_shape": list(x.shape), "steps": steps, "start_month": start_month(REGION),
        "feature_mode": FEATURE_MODE if HEAD == "mlp" else "sequence", "pca_dim": PCA_DIM if HEAD == "mlp" else None,
        "ltae": {"n_heads": N_HEADS, "d_k": D_K, "d_model": D_MODEL} if HEAD == "ltae" else None,
        "mlp": MLP, "dropout": DROPOUT, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "patience": PATIENCE, "seeds": seeds,
        "loss": "class-weighted cross-entropy", "validation": "PASTIS: 3 folds train / next fold early stopping / 1 test",
        "train_fraction": TRAIN_FRACTION, "fraction_draw": FRACTION_DRAW, "block_draws": draws,
        "n_parcels": int(len(oof)), "n_boot": N_BOOT, "quick_test": QUICK_TEST, "device": DEVICE},
        indent=2, default=str))
    log.info(f"Done in {(time.time() - t0) / 60:.1f} min -> {OUT_DIR}")


if __name__ == "__main__":
    main()
