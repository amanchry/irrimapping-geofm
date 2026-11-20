"""
nn_common.py  —  neural building blocks and the pixel training loop of the pixel-wise experiments (not run on its own)
====================================================================================================================

LTAE        lightweight temporal attention encoder (Garnot & Landrieu 2020): channels -> d_model, sinusoidal
            encoding of the day in the agricultural year, one learned master query per head; masked months get
            no attention. Returns the aggregated feature vector and the attention per head (used by E02, E04, E03).
TempCNN     three 1-D convolutions over the months + dense layer (Pelletier et al. 2019) (E02).
train_pixel_model   weighted cross-entropy, AdamW, early stopping on the validation fold's 3-class macro-F1;
                    the best epoch is kept.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class LTAE(nn.Module):
    """x [B, T, d_in], mask [B, T] (True = observed) -> features [B, d_out], attention [B, heads, T]."""

    def __init__(self, d_in, positions, n_head=16, d_k=8, d_model=256, mlp=(256, 128), dropout=0.2, T=1000):
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
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.d_out = d

    def forward(self, x, mask=None):
        b, t, _ = x.shape
        h = self.inconv(x) + self.pe[:t]
        k = self.fc_k(h).view(b, t, self.n_head, self.d_k)
        s = torch.einsum("bthd,hd->bht", k, self.query) / math.sqrt(self.d_k)
        if mask is not None:
            s = s.masked_fill(~mask[:, None, :], -1e4)
        att = torch.softmax(s.float(), dim=-1).to(h.dtype)
        v = h.view(b, t, self.n_head, -1)
        z = torch.einsum("bht,bthc->bhc", att, v).reshape(b, -1)
        return self.dropout(self.mlp(z)), att


class LTAEClassifier(nn.Module):
    def __init__(self, d_in, positions, n_classes=3, **kw):
        super().__init__()
        self.ltae = LTAE(d_in, positions, **kw)
        self.out = nn.Linear(self.ltae.d_out, n_classes)
        self.last_attention = None

    def forward(self, x, mask):
        z, att = self.ltae(x, mask)
        self.last_attention = att.detach()
        return self.out(z)


class TempCNN(nn.Module):
    """Pelletier et al. (2019): 3 x (Conv1d k=5, BN, ReLU, dropout), flatten, dense + BN + ReLU + dropout, output."""

    def __init__(self, d_in, n_months=12, n_classes=3, filters=64, kernel=5, dense=256, dropout=0.2):
        super().__init__()
        layers, c = [], d_in
        for _ in range(3):
            layers += [nn.Conv1d(c, filters, kernel, padding=kernel // 2), nn.BatchNorm1d(filters), nn.ReLU(),
                       nn.Dropout(dropout)]
            c = filters
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(filters * n_months, dense), nn.BatchNorm1d(dense),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense, n_classes))

    def forward(self, x, mask=None):                    # x [B, T, C]
        return self.head(self.conv(x.transpose(1, 2)))


@torch.no_grad()
def predict_proba(model, x, mask=None, batch=4096):
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = torch.from_numpy(x[s:s + batch]).to(DEVICE)
        mb = torch.from_numpy(mask[s:s + batch]).to(DEVICE) if mask is not None else None
        out.append(torch.softmax(model(xb, mb).float(), -1).cpu().numpy())
    return np.concatenate(out)


def train_pixel_model(model, x, y, train, val, weights, seed, mask=None, lr=1e-3, weight_decay=1e-4,
                      batch_size=512, epochs=100, patience=10, log=None):
    """Train on rows `train`, early stopping on the 3-class macro-F1 of rows `val`. y holds classes 1-3.
    Returns the model with the best epoch's weights and the training history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    y0 = (y - 1).astype(np.int64)
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order, losses = rng.permutation(train), []
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            if len(b) < 2:
                continue
            xb = torch.from_numpy(x[b]).to(DEVICE)
            mb = torch.from_numpy(mask[b]).to(DEVICE) if mask is not None else None
            loss = loss_fn(model(xb, mb), torch.from_numpy(y0[b]).to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        pv = predict_proba(model, x[val], mask[val] if mask is not None else None)
        f1 = f1_score(y0[val], pv.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": float(f1)})
        if f1 > best:
            best, wait = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if log and (epoch % 10 == 0 or wait == 0):
            log.info(f"    epoch {epoch:3d} loss {np.mean(losses):.4f} val macro-F1 {f1:.4f}{' *' if wait == 0 else ''}")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, history
