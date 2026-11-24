"""
a9_experiment_explainers.py  —  one start-to-end schematic per experiment (E01-E05), for any region and year
===========================================================================================================

Each figure has a strip of numbered steps on top and one panel per step (3 x 3), in the order of the experiment's
run.py. The figures read no data and show no results: the pictures (parcels, tiles, curves, bars, histograms) are
made-up illustrations, so the same figure explains the experiment for Morocco, Utah or any other region. The
results of a region are on the experiment pages. Steps shared by all experiments (labelled pixels, rotation over the
folds, out-of-fold predictions, scores, block bootstrap, paired tests) are drawn by the same functions.

Writes docs/experiments/figures/e0X_explainer.png for every experiment in EXPERIMENTS; E04 and E05 get one
complete figure per encoder (e04_<encoder>_explainer.png, E04_ENCODERS) and per training strategy
(e05_<strategy>_explainer.png, E05_KEYS), since their steps differ.

Run (Mac or cluster; no data needed):
    python tools/a9_experiment_explainers.py
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

R = Path(__file__).resolve().parents[1]
OUT_DIR = R / "docs/experiments/figures"

# ================================ SETTINGS ================================
EXPERIMENTS = ["e01", "e02", "e03"]                # E04, E05: one figure per encoder / strategy (below)
E04_ENCODERS = ["prithvi", "clay", "terramind", "agrifm", "presto", "tessera"]   # one figure per E04 encoder
E05_KEYS = ["frozen", "lora", "finetune", "scratch"]                                # one figure per E05 strategy
N_FOLDS = 5
SEED = 7
# ==========================================================================

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "NDVI", "NDWI"]
MONTHS = [f"m{m}" for m in range(1, 13)]           # months of the agricultural year (start month set per region)
FOLD_COL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
C_IRR, C_RAIN = "#1f77b4", "#d62728"
C_TR, C_VA, C_TE = "#9ecae1", "#fdd49e", "#fc9272"
STEP_COL, LOOP_COL = "#34495e", "#2e86c1"
C_LOCK, C_TRAIN = "#95a5a6", "#e67e22"            # frozen part, trained part
LOOP = set()                                       # step numbers drawn in blue (repeated per round)
NOTE_W = 76                                        # characters per line of the notes under the panels


# ================================================================ layout helpers
def new_figure(title, steps, loop, footer, nrows=3, ncols=3, size=(18, 20), note_width=76, bottom=0.085):
    """Figure with the title, the strip of steps and a grid of panels (3 x 3 by default)."""
    global LOOP, NOTE_W
    LOOP, NOTE_W = set(loop), note_width
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.28, top=0.89, bottom=bottom)
    ax = fig.add_axes([0.11, 0.925, 0.80, 0.034])
    ax.axis("off")
    w = 1 / len(steps)
    for i, s in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i * w + 0.004, 0.0), w - 0.022, 1.0, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc=LOOP_COL if i + 1 in LOOP else STEP_COL, ec="none", transform=ax.transAxes))
        ax.text(i * w + w / 2 - 0.007, 0.5, f"{i + 1}\n{s}", ha="center", va="center", color="white", fontsize=9.5,
                transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate("", xy=((i + 1) * w + 0.003, 0.5), xytext=((i + 1) * w - 0.019, 0.5),
                        xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2))
    if LOOP:
        span = f"Step {min(LOOP)} (blue) repeats" if len(LOOP) == 1 else f"Steps {min(LOOP)}–{max(LOOP)} (blue) repeat"
        ax.text(0.5, 1.15, f"{span} in each of the {N_FOLDS} rounds",
                ha="center", fontsize=9.5, color=LOOP_COL, transform=ax.transAxes)
    fig.suptitle(title, fontsize=15, y=0.99)
    fig.text(0.5, 0.012, "Schematic: the pictures illustrate the steps and show no results. " + footer,
             ha="center", fontsize=9, color="0.3")
    return fig, gs


def step(ax, n, title, sub=""):
    ax.set_title(f"{n} · {title}" + (f"\n{sub}" if sub else ""), fontsize=11.5, loc="left",
                 color=LOOP_COL if n in LOOP else STEP_COL)


def note(ax, text, y=-0.04, width=None):
    """Explanation under a panel, wrapped to the panel width."""
    ax.text(0.0, y, textwrap.fill(" ".join(text.split()), width or NOTE_W), transform=ax.transAxes, fontsize=8.6, va="top",
            color="0.15")


def box(ax, x, y, w, h, text, fc, tc="white", fs=9, ec="none", lw=1, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else None)


def arrow(ax, a, b, color="0.4", lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, lw=lw))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


# ================================================================ shared panels
def panel_pixels(ax, n, rng, tiles=False):
    """Labelled pixels of the surveyed parcels in 20 km blocks coloured by fold (tiles=True: 128 px tiles drawn)."""
    nb = 4
    fold_of = rng.permutation(np.resize(np.arange(N_FOLDS), nb * nb)).reshape(nb, nb)
    for i in range(nb):
        for j in range(nb):
            ax.add_patch(Rectangle((j, i), 1, 1, fc=FOLD_COL[fold_of[i, j]], alpha=0.18, ec="white", lw=2))
            ax.text(j + 0.05, i + 0.93, f"fold {fold_of[i, j]}", fontsize=7, color=FOLD_COL[fold_of[i, j]], va="top")
    for _ in range(14):
        bi, bj = rng.integers(0, nb, 2)
        cx, cy = bj + rng.uniform(0.3, 0.7), bi + rng.uniform(0.3, 0.65)
        wx, wy = rng.uniform(0.14, 0.24), rng.uniform(0.1, 0.18)
        col = C_IRR if rng.random() < 0.75 else C_RAIN
        if tiles:
            ax.add_patch(Rectangle((cx - 0.3, cy - 0.3), 0.6, 0.6, fill=False, ec="0.35", lw=0.9, ls="--"))
        ax.add_patch(Polygon([(cx - wx, cy - wy), (cx + wx, cy - wy * 0.8), (cx + wx * 0.9, cy + wy),
                              (cx - wx, cy + wy * 0.9)], fc="none", ec=col, lw=1.4))
        xs, ys = np.meshgrid(np.linspace(cx - wx * 0.6, cx + wx * 0.6, 4), np.linspace(cy - wy * 0.5, cy + wy * 0.5, 3))
        ax.scatter(xs, ys, s=5, c=col)
    blank(ax, (0, nb), (0, nb))
    ax.set_aspect("equal")
    ax.plot([], [], "s", mfc="none", mec=C_IRR, label="irrigated parcel")
    ax.plot([], [], "s", mfc="none", mec=C_RAIN, label="rainfed parcel")
    ax.plot([], [], ".", c="0.3", label="labelled pixel")
    if tiles:
        ax.plot([], [], "s", mfc="none", mec="0.35", ls="--", label="128 px tile")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4 if tiles else 3, frameon=False)
    extra = (" The model sees the whole 128 × 128 px tiles (1.28 km) around the parcels, but only the labelled pixels "
             "count in training and scoring.") if tiles else ""
    note(ax, "Each labelled pixel inherits the survey label of its parcel and the fold of its 20 km block "
             "(squares), so all pixels of a parcel stay in one fold; the same pixels in every experiment (see the "
             "Pixel samples page)." + extra, y=-0.07)
    step(ax, n, "Input: labelled pixels of the surveyed parcels", "colour of a square = spatial fold of the block")


def panel_rotation(ax, n):
    for k in range(N_FOLDS):
        val = (k + 1) % N_FOLDS
        for f in range(N_FOLDS):
            c, lab = (C_TE, "test") if f == k else (C_VA, "val") if f == val else (C_TR, "train")
            ax.add_patch(Rectangle((f, N_FOLDS - 1 - k), 0.94, 0.86, fc=c, ec="white"))
            ax.text(f + 0.47, N_FOLDS - 1 - k + 0.43, lab, ha="center", va="center", fontsize=9)
        ax.text(-0.1, N_FOLDS - 1 - k + 0.43, f"round {k + 1}", ha="right", va="center", fontsize=9.5)
    for f in range(N_FOLDS):
        ax.text(f + 0.47, N_FOLDS + 0.05, f"fold {f}", ha="center", fontsize=9.5, color=FOLD_COL[f], fontweight="bold")
    blank(ax, (-1.3, N_FOLDS), (-0.2, N_FOLDS + 0.4))
    note(ax, "Round k: fold k is the test fold, fold k+1 the validation fold, the other three train the model "
             "(PASTIS scheme). After all rounds every pixel has been tested exactly once, by a model that never saw "
             "its block.", y=-0.02)
    step(ax, n, f"Rotate over the {N_FOLDS} spatial folds", "the same rounds in every experiment E01–E05")


def panel_oof(ax, n, what="model", extra=""):
    for k in range(N_FOLDS):
        ax.add_patch(Rectangle((0, N_FOLDS - 1 - k), 1.9, 0.86, fc=C_TE, ec="white"))
        ax.text(0.95, N_FOLDS - 1 - k + 0.43, f"round {k + 1}: {what}\npredicts fold {k}", ha="center", va="center",
                fontsize=8.5)
        arrow(ax, (1.95, N_FOLDS - 1 - k + 0.43), (2.65, N_FOLDS / 2), color="0.5", lw=1)
    for f in range(N_FOLDS):
        ax.add_patch(Rectangle((2.7, f), 0.7, 1.0, fc=FOLD_COL[N_FOLDS - 1 - f], alpha=0.6, ec="white"))
    ax.text(3.05, N_FOLDS + 0.15, "all pixels", ha="center", fontsize=9)
    ax.text(3.6, N_FOLDS / 2, "one out-of-fold\nprediction per pixel\n(probability of\nirrigated, rainfed;\n"
            "class = the larger)", va="center", fontsize=9)
    blank(ax, (0, 5.6), (-0.2, N_FOLDS + 0.5))
    note(ax, "Stacking the test folds of all rounds gives one prediction for every labelled pixel "
             "(oof_predictions.parquet). Every experiment is scored on exactly the same pixels, so experiments can "
             "be compared pixel by pixel. " + extra, y=-0.02)
    step(ax, n, "Predict the test fold, collect all rounds", "out-of-fold predictions")


def panel_scores(ax, n):
    """Confusion matrix, per-class scores and macro-F1."""
    cells = [["correct\nirrigated", "irrigated\ntaken for\nrainfed"], ["rainfed\ntaken for\nirrigated", "correct\nrainfed"]]
    cols = [["#08306b", "#deebf7"], ["#c6dbef", "#2171b5"]]
    cw, ch, y0 = 2.2, 1.0, 1.6
    for i in range(2):
        for k in range(2):
            ax.add_patch(Rectangle((k * cw, y0 + (1 - i) * ch), cw, ch, fc=cols[i][k], ec="white", lw=2))
            ax.text(k * cw + cw / 2, y0 + (1 - i) * ch + ch / 2, cells[i][k], ha="center", va="center", fontsize=9,
                    color="white" if cols[i][k] in ("#08306b", "#2171b5") else "black")
    ax.text(cw, y0 + 2 * ch + 0.42, "predicted", ha="center", fontsize=9.5)
    ax.text(cw / 2, y0 + 2 * ch + 0.1, "irrigated", ha="center", fontsize=9)
    ax.text(1.5 * cw, y0 + 2 * ch + 0.1, "rainfed", ha="center", fontsize=9)
    ax.text(-0.1, y0 + 1.5 * ch, "irrigated", ha="right", va="center", fontsize=8.5)
    ax.text(-0.1, y0 + 0.5 * ch, "rainfed", ha="right", va="center", fontsize=8.5)
    ax.text(-1.4, y0 + ch, "survey label", rotation=90, ha="center", va="center", fontsize=9.5)
    ax.text(2 * cw + 0.3, y0 + ch, "per class:\nprecision = correct / predicted\nrecall = correct / surveyed\n"
            "F1 = 2·P·R / (P + R)", va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((-0.2, 0.15), 8.6, 0.95, boxstyle="round,pad=0.05", fc="#eaf2f8", ec=LOOP_COL))
    ax.text(4.1, 0.62, "headline score:\nmacro-F1 = (F1 irrigated + F1 rainfed) / 2", ha="center", va="center",
            fontsize=9.5, color=STEP_COL, fontweight="bold")
    blank(ax, (-1.7, 8.6), (0, y0 + 2 * ch + 0.7))
    note(ax, "Macro-F1 weighs the rare rainfed class as much as the common irrigated class, so a model cannot score "
             "well by calling everything irrigated. Also reported: overall accuracy, and every score per year and per "
             "zone (metrics_by_group.csv).", y=-0.02)
    step(ax, n, "Score all out-of-fold pixels", "confusion matrix → precision, recall, F1")


def panel_bootstrap(ax, n, rng):
    draws = rng.normal(0.5, 0.1, 1000)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    ax.hist(draws, bins=35, color="#c8d6e5", ec="white")
    ax.axvline(0.5, color="k", lw=1.5)
    for q in (lo, hi):
        ax.axvline(q, color=LOOP_COL, lw=1.5, ls="--")
    top = ax.get_ylim()[1]
    ax.annotate("", xy=(hi, top * 0.92), xytext=(lo, top * 0.92), arrowprops=dict(arrowstyle="<->", color=LOOP_COL))
    ax.text(0.5, top * 0.95, "95% confidence interval", ha="center", color=LOOP_COL, fontsize=9.5)
    ax.text(lo - 0.01, top * 0.6, "2.5% of\nthe draws", ha="right", fontsize=8.5, color=LOOP_COL)
    ax.text(hi + 0.01, top * 0.6, "97.5% of\nthe draws", ha="left", fontsize=8.5, color=LOOP_COL)
    ax.text(0.505, top * 0.8, "score of\nall pixels", ha="left", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("macro-F1 of one bootstrap draw")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    note(ax, "Draw the 20 km blocks with replacement (as many as there are), recompute macro-F1 from their pixels; "
             "repeat 1,000 times. Blocks, not pixels, because neighbouring pixels are not independent. The same draws "
             "serve every experiment, so models are compared pair by pair.", y=-0.07)
    step(ax, n, "How certain is the score? Block bootstrap", "black = score of all pixels; dashed = the 95% interval")


def panel_paired(ax, n, rng, a, b, text, sub):
    """Paired difference of two runs on the same bootstrap draws."""
    d = rng.normal(0.025, 0.018, 1000)
    lo, hi = np.quantile(d, [0.025, 0.975])
    ax.hist(d, bins=35, color="#d5e8d4", ec="white")
    ax.axvline(0, color="#c0392b", lw=1.5)
    ax.axvline(np.mean(d), color="k", lw=1.5)
    for q in (lo, hi):
        ax.axvline(q, color=LOOP_COL, lw=1.5, ls="--")
    top = ax.get_ylim()[1]
    ax.text(-0.002, top * 0.7, "0 = no\ndifference", ha="right", fontsize=8.5, color="#c0392b")
    ax.annotate("", xy=(hi, top * 0.88), xytext=(lo, top * 0.88), arrowprops=dict(arrowstyle="<->", color=LOOP_COL))
    ax.text((lo + hi) / 2, top * 0.91, "95% CI of the difference", ha="center", color=LOOP_COL, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"macro-F1 of {a} − macro-F1 of {b}, per draw")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    note(ax, text, y=-0.07)
    step(ax, n, f"Is {a} better than {b}?", sub)


def panel_early_stopping(ax, n, rng, patience, max_epochs, seeds=True, extra=""):
    ep = np.arange(1, 41)
    val = 0.75 - 0.35 * np.exp(-ep / 4) + rng.normal(0, 0.012, len(ep)) - 0.0015 * np.clip(ep - 14, 0, None)
    tr = 0.9 * np.exp(-ep / 7) + 0.05
    best = int(np.argmax(val[:30]))
    stop = best + patience
    ax.plot(ep[:stop + 1], 0.33 + 0.47 * tr[:stop + 1] / tr[0], color="0.55", lw=1.5, label="training loss")
    ax.plot(ep[:stop + 1], val[:stop + 1], color=LOOP_COL, lw=2, label="validation macro-F1")
    ax.plot(ep[best], val[best], "o", color=C_TRAIN, ms=10, zorder=5, label="best epoch → kept")
    ax.axvspan(ep[best], ep[stop], color=C_TRAIN, alpha=0.1)
    ax.text((ep[best] + ep[stop]) / 2, 0.42, f"{patience} epochs\nwithout\nimprovement\n→ stop", ha="center",
            fontsize=8.5, color=C_TRAIN)
    ax.set_xlabel("epoch")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0.3, 0.85)
    ax.legend(fontsize=8, loc="lower right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    seed_txt = (" Repeated with 3 seeds (42, 1042, 2042); the out-of-fold probability is their mean." if seeds else "")
    note(ax, f"After each epoch the model predicts the validation fold; the epoch with the best macro-F1 is kept. "
             f"Training stops after {patience} epochs without improvement (at most {max_epochs})." + seed_txt + " " +
             extra, y=-0.07)
    step(ax, n, "Train with early stopping", "the validation fold decides when to stop")


def panel_tile_input(ax, n, bands_text, extra):
    """Stack of 12 monthly images of one tile."""
    for m in range(12):
        off = m * 0.18
        ax.add_patch(Rectangle((off, off), 3, 3, fc=plt.cm.YlGn(0.25 + 0.5 * np.sin(np.pi * m / 11)), ec="0.4", lw=0.8))
    ax.text(3 + 11 * 0.18 + 0.1, 11 * 0.18 + 3, "m12", fontsize=8, va="center")
    ax.text(-0.1, 0.05, "m1", fontsize=8, ha="right")
    ax.annotate("", xy=(3.2, 0), xytext=(3.2 + 11 * 0.18, 11 * 0.18), arrowprops=dict(arrowstyle="<->", color="0.4"))
    ax.text(4.2, 0.6, "12 months", fontsize=9, color="0.3")
    ax.annotate("", xy=(0, -0.25), xytext=(3, -0.25), arrowprops=dict(arrowstyle="<->", color="0.4"))
    ax.text(1.5, -0.55, "128 px = 1.28 km", ha="center", fontsize=9, color="0.3")
    for _ in range(3):                                              # labelled pixels on the front image
        x, y = np.random.default_rng(_).uniform(2.6, 4.3, 2)
        ax.add_patch(Rectangle((x, y), 0.5, 0.35, fc="none", ec=C_IRR if _ else C_RAIN, lw=1.4))
    blank(ax, (-0.6, 6.5), (-0.9, 5.4))
    ax.set_aspect("equal")
    note(ax, f"One sample = one tile: {bands_text} for each of the 12 months, 128 × 128 pixels. {extra}", y=-0.02)
    step(ax, n, "Input: the image time series of a tile", "every pixel of the tile, not only the labelled ones")


def panel_masked_loss(ax, n, rng, extra=""):
    g = 10
    lab = np.zeros((g, g), int)
    lab[1:4, 1:5], lab[6:9, 2:6], lab[2:5, 6:9] = 1, 2, 3          # 3 = parcel of another fold
    cols = {0: "#f4f4f4", 1: C_IRR, 2: C_RAIN, 3: "#bdbdbd"}
    for i in range(g):
        for j in range(g):
            ax.add_patch(Rectangle((j, g - 1 - i), 0.95, 0.95, fc=cols[lab[i, j]], ec="white"))
    blank(ax, (-0.2, g + 6.5), (-0.5, g + 0.3))
    ax.set_aspect("equal")
    for k, (c, t) in enumerate([(C_IRR, "irrigated pixel of a training\nparcel → counts in the loss"),
                                (C_RAIN, "rainfed pixel of a training\nparcel → counts (higher weight)"),
                                ("#bdbdbd", "pixel of a validation or test\nparcel → ignored"),
                                ("#f4f4f4", "unlabelled pixel → ignored")]):
        ax.add_patch(Rectangle((g + 0.5, g - 1.6 - k * 2.4), 0.9, 0.9, fc=c, ec="0.6"))
        ax.text(g + 1.6, g - 1.15 - k * 2.4, t, fontsize=8.5, va="center")
    note(ax, "The network predicts a class for every pixel of the tile, but the loss (class-weighted cross-entropy) "
             "is computed only at the labelled pixels of the training folds. A tile that also holds parcels of the "
             "validation or test fold contributes only its training labels. " + extra, y=-0.02)
    step(ax, n, "Loss only at labelled training pixels", "one tile, 10 × 10 pixels shown")


def panel_tile_to_pixels(ax, n):
    g = 8
    for i in range(g):
        for j in range(g):
            ax.add_patch(Rectangle((j, i), 0.95, 0.95, fc=C_IRR if (i + j) % 5 else C_RAIN, alpha=0.35, ec="white"))
    pts = [(1, 2), (2, 2), (5, 5), (6, 5), (5, 6)]
    for (x, y) in pts:
        ax.add_patch(Rectangle((x, y), 0.95, 0.95, fill=False, ec="k", lw=1.8))
    arrow(ax, (g + 0.3, g / 2), (g + 2.0, g / 2))
    ax.text(g + 2.2, g / 2, "read the prediction\nat the labelled pixels\nof the test fold\n(black squares)",
            va="center", fontsize=9)
    blank(ax, (-0.2, g + 7), (-0.5, g + 0.3))
    ax.set_aspect("equal")
    note(ax, "The model predicts every pixel of every test tile; the out-of-fold prediction is read at the labelled "
             "pixels of the test fold only, so the tile models are scored on exactly the same pixels as the pixel "
             "models of E01 and E02.", y=-0.02)
    step(ax, n, "Predict the test tiles, read the labelled pixels", "a map for the whole tile, scored where labels exist")


# ================================================================ E01
def draw_e01():
    rng = np.random.default_rng(SEED)
    fig, gs = new_figure(
        "E01 · Sentinel-2 pixel baseline: how the experiment works, from the labelled pixels to the scores",
        ["Labelled\npixels", "168 features\nper pixel", "Fill cloudy\nmonths", "Rotate folds:\ntrain · val · test",
         "Pick settings\non validation", "Predict the\ntest fold", "Score all\npixels", "Block bootstrap\n95% CI",
         "Which months\nand bands"], loop=(4, 5, 6),
        footer="Outputs per region and model (experiments/E01_s2_pixel_baseline/results/<region>/<rf|hgb>/): "
               "oof_predictions.parquet · metrics.csv · metrics_by_group.csv · confusion.png · chosen_params.csv · "
               "importance_*.csv · run_config.json · run.log")
    panel_pixels(fig.add_subplot(gs[0, 0]), 1, rng)

    ax = fig.add_subplot(gs[0, 1])
    vals = rng.normal(size=(len(BANDS), 12)).cumsum(1)
    vals = (vals - vals.mean(1, keepdims=True)) / (vals.std(1, keepdims=True) + 1e-9)
    ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.add_patch(Rectangle((6.5, -0.5), 1, len(BANDS), fill=False, ec="white", lw=2, ls="--"))
    ax.set_xticks(range(12), MONTHS, fontsize=8)
    ax.set_yticks(range(len(BANDS)), BANDS, fontsize=7.5)
    note(ax, "rows: 12 Sentinel-2 bands + NDVI + NDWI · columns: the 12 months of the agricultural year → flattened "
             "to 12 × 14 = 168 numbers = one row of the table the model learns from. The model sees 168 separate "
             "columns; it does not know they form a time series.", y=-0.08)
    step(ax, 2, "Features: one pixel = 12 months × 14 values", "dashed = a cloudy month (no clear observation)")

    ax = fig.add_subplot(gs[0, 2])
    t = np.arange(12)
    obs = 0.25 + 0.5 * np.exp(-((t - 5) / 2.2) ** 2) + rng.normal(0, 0.015, 12)
    filled = obs.copy()
    filled[7] = (obs[6] + obs[8]) / 2
    ax.plot(t, filled, "-", color=C_IRR, lw=1.5, label="after filling")
    keep = [m for m in t if m != 7]
    ax.plot(keep, obs[keep], "o", color=C_IRR, ms=7, label="clear month (observed)")
    ax.plot([7], [filled[7]], "o", mfc="white", mec=C_IRR, ms=9, mew=2, label="cloudy month: linear interpolation")
    ax.set_xticks(t, MONTHS, fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel("one band or index of one pixel")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    note(ax, "A cloudy month takes the straight line between its clear neighbours; a missing first or last month takes "
             "the nearest clear month; a band never observed takes the median of the training pixels. How often this "
             "happens depends on the region and year.", y=-0.08)
    step(ax, 3, "Fill the cloudy months", "the model needs all 168 values of every pixel")

    panel_rotation(fig.add_subplot(gs[1, 0]), 4)

    ax = fig.add_subplot(gs[1, 1])
    score = [0.62, 0.78, 0.55, 0.70]
    best = int(np.argmax(score))
    ax.barh(range(4), score, color=[LOOP_COL if i == best else "#c8d6e5" for i in range(4)])
    ax.text(score[best] + 0.02, best, "best → kept", va="center", fontsize=9.5, color=LOOP_COL)
    ax.set_yticks(range(4), ["setting A", "setting B", "setting C", "setting D"], fontsize=9)
    ax.set_xticks([])
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("macro-F1 on the validation fold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    note(ax, "Every setting of the grid is fitted on the 3 training folds and scored on the validation fold; the best "
             "one is kept. The test fold is never used for this choice. Random forest (500 trees): max_features ∈ "
             "{√p, 0.33} × min_samples_leaf ∈ {1, 5}. Gradient boosting: learning rate ∈ {0.05, 0.1} × "
             "max_leaf_nodes ∈ {31, 63}. Both with balanced class weights; the chosen setting of each round: "
             "chosen_params.csv.", y=-0.1)
    step(ax, 5, "In each round: choose the settings", "4 settings per model, judged on the validation fold")

    panel_oof(fig.add_subplot(gs[1, 2]), 6)
    panel_scores(fig.add_subplot(gs[2, 0]), 7)
    panel_bootstrap(fig.add_subplot(gs[2, 1]), 8, rng)

    ax = fig.add_subplot(gs[2, 2])
    for m in range(12):
        ax.add_patch(Rectangle((m, 1.2), 0.9, 0.6, fc="#3182bd" if m != 4 else "#fdae6b", ec="white"))
        ax.text(m + 0.45, 1.5, MONTHS[m], ha="center", va="center", fontsize=7.5, color="white" if m != 4 else "black")
    ax.annotate("shuffle this month's 14 values\nbetween the test pixels", (4.45, 1.85), (6.5, 2.35), fontsize=8.5,
                ha="left", arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.add_patch(Rectangle((0, 0.6), 9.0, 0.38, fc="#2171b5"))
    ax.text(0.2, 0.79, "macro-F1 with all months as they are", va="center", color="white", fontsize=9)
    ax.add_patch(Rectangle((0, 0.1), 6.2, 0.38, fc="#9ecae1"))
    ax.text(0.2, 0.29, "macro-F1 with m5 shuffled", va="center", fontsize=9)
    ax.plot([9.0, 9.0], [0.1, 0.98], color="0.5", lw=0.8, ls=":")
    ax.annotate("", (6.2, 0.29), (9.0, 0.29), arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.5))
    ax.text(9.2, 0.29, "drop =\nimportance\nof m5", va="center", fontsize=9, color="#d62728")
    blank(ax, (-0.2, 12.2), (0, 2.9))
    note(ax, "A month (or a band over all months) matters when shuffling it lowers the score. Computed for the random "
             "forest on up to 20,000 test pixels per round; mean and spread over the rounds (importance_month.csv, "
             "importance_band.csv).", y=-0.02)
    step(ax, 9, "What did the model use? Permutation importance", "per month and per band")
    return fig


# ================================================================ E02
def draw_e02():
    rng = np.random.default_rng(SEED)
    fig, gs = new_figure(
        "E02 · Temporal neural network (TempCNN): how the experiment works, from the labelled pixels to the scores",
        ["Labelled\npixels", "Sequence of\n12 months", "Fill and\nstandardise", "Rotate folds:\ntrain · val · test",
         "TempCNN reads\nthe sequence", "Train, stop\nearly, 3 seeds", "Predict the\ntest fold", "Score and\n95% CI",
         "Compare\nwith E01"], loop=(4, 5, 6, 7),
        footer="Outputs per region (experiments/E02_temporal_nn/results/<region>/tempcnn/): oof_predictions.parquet · "
               "metrics.csv · metrics_by_group.csv · confusion.png · training_history.csv · run_config.json · run.log")
    panel_pixels(fig.add_subplot(gs[0, 0]), 1, rng)

    ax = fig.add_subplot(gs[0, 1])
    vals = rng.normal(size=(len(BANDS), 12)).cumsum(1)
    vals = (vals - vals.mean(1, keepdims=True)) / (vals.std(1, keepdims=True) + 1e-9)
    ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.set_xticks(range(12), MONTHS, fontsize=8)
    ax.set_yticks(range(len(BANDS)), BANDS, fontsize=7.5)
    note(ax, "The same 14 values per month as E01 (12 Sentinel-2 bands + NDVI + NDWI), but kept as a sequence of 12 "
             "steps with 14 channels each, in their order: the network knows which month follows which, instead of "
             "seeing 168 unordered columns.", y=-0.08)
    step(ax, 2, "Input: one pixel = a sequence of 12 months", "same pixels and values as E01")

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    t = np.arange(12)
    b8 = 2000 + 2500 * np.exp(-((t - 5) / 2.2) ** 2)
    ndvi = 0.2 + 0.6 * np.exp(-((t - 5) / 2.2) ** 2)
    for k, (title, a1, a2) in enumerate([("before: very different ranges", b8, ndvi),
                                         ("after: both around 0, similar spread",
                                          (b8 - b8.mean()) / b8.std(), (ndvi - ndvi.mean()) / ndvi.std() * 0.85 - 0.3)]):
        sub = ax.inset_axes([0.0, 0.55 - k * 0.55, 1.0, 0.4])
        sub.plot(t, a1, "o-", color="#7f8c8d", ms=4, label="B8 (reflectance)")
        sub.plot(t, a2, "o-", color="#27ae60", ms=4, label="NDVI")
        sub.set_title(title, fontsize=9, loc="left")
        sub.set_xticks(t, MONTHS, fontsize=7)
        sub.tick_params(axis="y", labelsize=7)
        sub.legend(fontsize=7.5, loc="upper right")
    note(ax, "Cloudy months are filled as in E01 (linear interpolation in time). Each of the 14 channels is then "
             "standardised (minus mean, divided by standard deviation) with the training pixels of the round, so all "
             "channels have a similar range; a value still missing is set to 0 (= the training mean).", y=-0.08)
    step(ax, 3, "Fill the cloudy months and standardise", "statistics from the training folds of each round")

    panel_rotation(fig.add_subplot(gs[1, 0]), 4)

    ax = fig.add_subplot(gs[1, 1])
    x0 = 0
    box(ax, x0, 0.5, 1.1, 4, "12\nmonths\n×\n14\nchan-\nnels", "#7f8c8d", fs=8)
    for k in range(3):
        box(ax, 1.5 + k * 1.35, 0.5, 1.1, 4, f"conv\nalong\ntime\n\n64\nfilters\nk = 5", LOOP_COL, fs=7.5)
        arrow(ax, (1.15 + k * 1.35 + (0 if k == 0 else 0.25), 2.5), (1.5 + k * 1.35, 2.5))
    box(ax, 5.6, 1.0, 1.1, 3, "flatten\n→ dense\n256", "#5d6d7e", fs=8)
    arrow(ax, (5.3, 2.5), (5.6, 2.5))
    box(ax, 7.1, 1.8, 1.1, 1.4, "irrigated\nrainfed", C_TRAIN, fs=8)
    arrow(ax, (6.7, 2.5), (7.1, 2.5))
    for m in range(5):                                         # kernel window over 5 months
        ax.add_patch(Rectangle((1.55 + m * 0.2, 4.7), 0.18, 0.3, fc=C_TRAIN if m < 5 else "w", ec="0.5"))
    ax.text(2.6, 4.85, "a filter looks at 5 neighbouring months", fontsize=8, va="center")
    blank(ax, (-0.1, 8.4), (0, 5.4))
    note(ax, "Each filter slides along the 12 months and responds to a temporal pattern (e.g. a winter green-up "
             "followed by a dry summer); three such layers combine short patterns into longer ones. All weights are "
             "learnt from the labelled pixels (Pelletier et al. 2019); each convolution is followed by batch normalisation, "
             "ReLU and dropout 0.2.", y=-0.02)
    step(ax, 5, "TempCNN: convolutions along time", "0.24 M weights, trained from scratch")

    panel_early_stopping(fig.add_subplot(gs[1, 2]), 6, rng, patience=10, max_epochs=100,
                         extra="AdamW, learning rate 10⁻³, batch 512 pixels, class-weighted cross-entropy.")
    panel_oof(fig.add_subplot(gs[2, 0]), 7, what="TempCNN")

    ax = fig.add_subplot(gs[2, 1])
    panel_bootstrap(ax, 8, rng)
    ax.set_title("8 · Score and its 95% CI\nmacro-F1, per class, per year and zone; block bootstrap", fontsize=11.5,
                 loc="left", color=STEP_COL)
    panel_paired(fig.add_subplot(gs[2, 2]), 9, rng, "TempCNN", "E01 RF",
                 "For each bootstrap draw the two macro-F1 are computed on the same pixels and subtracted. TempCNN "
                 "counts as better only if the 95% interval of the difference excludes 0 and the Holm-corrected p < "
                 "0.05 (Holm over all runs of E01–E04). Run by pixelwise_run/05_compare_experiments.py.",
                 "paired difference on the same pixels and draws")
    return fig


# ================================================================ E03
def draw_e03():
    rng = np.random.default_rng(SEED)
    fig, gs = new_figure(
        "E03 · U-TAE from scratch: how the experiment works, from the labelled pixels to the scores",
        ["Labelled pixels\nand tiles", "Image time\nseries per tile", "Rotate folds:\ntrain · val · test",
         "U-TAE: space\nand time", "Loss at labelled\npixels only", "Train, stop\nearly, 3 seeds",
         "Predict test\ntiles", "Score and\n95% CI", "Compare with\nE01, E04"], loop=(3, 4, 5, 6, 7),
        footer="Outputs per region (experiments/E03_utae/results/<region>/utae/): oof_predictions.parquet · metrics.csv · "
               "metrics_by_group.csv · confusion.png · training_history.csv · run_config.json · run.log")
    panel_pixels(fig.add_subplot(gs[0, 0]), 1, rng, tiles=True)
    panel_tile_input(fig.add_subplot(gs[0, 1]), 2, "the 12 Sentinel-2 bands",
                     "Each band is standardised with the training pixels of the round; months without any clear pixel "
                     "are masked in the temporal attention. Unlike E01 and E02, the model sees the neighbours of every "
                     "pixel (spatial context).")
    panel_rotation(fig.add_subplot(gs[0, 2]), 3)

    ax = fig.add_subplot(gs[1, 0])
    for m in range(4):                                       # months entering the shared encoder
        ax.add_patch(Rectangle((0.1 + m * 0.12, 3.6 - m * 0.12), 1.0, 1.0, fc=plt.cm.YlGn(0.3 + 0.12 * m), ec="0.4"))
    ax.text(0.65, 2.9, "12 months", ha="center", fontsize=8)
    widths = [1.0, 0.8, 0.6, 0.45]
    for k, wd in enumerate(widths):                          # encoder levels
        box(ax, 1.9 + k * 1.05, 4.0 - wd / 2, 0.8, wd, f"1/{2 ** k}" if k else "full", "#5d6d7e", fs=7.5)
        if k:
            arrow(ax, (1.9 + (k - 1) * 1.05 + 0.82, 4.0), (1.9 + k * 1.05, 4.0), lw=1)
    arrow(ax, (1.2, 4.1), (1.9, 4.0))
    ax.text(3.4, 4.75, "encoder, same weights for every month", ha="center", fontsize=8)
    box(ax, 4.75, 2.2, 2.0, 0.9, "L-TAE: attention\nover the 12 months", C_TRAIN, fs=8)
    arrow(ax, (5.25, 3.75), (5.6, 3.1))
    for k, wd in enumerate(widths[::-1]):                    # decoder levels
        box(ax, 5.05 - k * 1.05, 0.6 - wd / 2 + 0.3, 0.8, wd, "", "#85929e", fs=7)
        if k:
            arrow(ax, (5.05 - (k - 1) * 1.05, 0.9), (5.05 - k * 1.05 + 0.82, 0.9), lw=1)
    arrow(ax, (5.4, 2.2), (5.4, 1.15))
    for k in range(4):                                       # attention masks applied to skips
        ax.add_patch(FancyArrowPatch((1.9 + k * 1.05 + 0.4, 4.0 - widths[k] / 2), (5.05 - (3 - k) * 1.05 + 0.4, 1.3),
                                     arrowstyle="-|>", mutation_scale=9, color=C_TRAIN, lw=0.8, ls="--"))
    box(ax, 0.0, 0.45, 1.2, 0.9, "class per\n10 m pixel", LOOP_COL, fs=8)
    arrow(ax, (1.9, 0.9), (1.2, 0.9))
    ax.text(3.4, 0.05, "decoder back to 10 m", ha="center", fontsize=8)
    blank(ax, (-0.1, 6.8), (-0.1, 5.0))
    note(ax, "A convolutional encoder (shared by all months) shrinks each image in 4 steps. At the smallest size the "
             "L-TAE decides, per pixel, how much each month matters; these attention weights (dashed) combine the "
             "months at every size, and a decoder brings the result back to 10 m. 1.15 M weights, all trained from "
             "random values (Garnot & Landrieu 2021).", y=-0.02)
    step(ax, 4, "U-TAE: an encoder–decoder with temporal attention", "the PASTIS baseline network")

    panel_masked_loss(fig.add_subplot(gs[1, 1]), 5, rng)
    panel_early_stopping(fig.add_subplot(gs[1, 2]), 6, rng, patience=10, max_epochs=100,
                         extra="AdamW, learning rate 10⁻³, batch 4 tiles, random flips and 90° rotations.")
    panel_tile_to_pixels(fig.add_subplot(gs[2, 0]), 7)
    ax = fig.add_subplot(gs[2, 1])
    panel_bootstrap(ax, 8, rng)
    ax.set_title("8 · Score and its 95% CI\nmacro-F1, per class, per year and zone; block bootstrap", fontsize=11.5,
                 loc="left", color=STEP_COL)
    panel_paired(fig.add_subplot(gs[2, 2]), 9, rng, "U-TAE", "E01 RF",
                 "Paired difference on the same pixels and bootstrap draws (Holm over all runs of E01–E04). U-TAE is "
                 "also the reference for E04: the frozen foundation models are compared with it, since both see the "
                 "same tiles and differ only in the encoder.", "and the reference for E04")
    return fig


# ================================================================ E04, one figure per encoder
# Settings of this study (pixelwise_run/encoders.py, tile_common.GeoFMSegmenter); shapes for one 128 x 128 px tile.
E04_MODELS = {
    "prithvi": dict(
        name="Prithvi-EO-2.0-300M-TL", family="image", colour="#1f77b4",
        bands="6 bands: B2 B3 B4 B8A B11 B12\n(the HLS set of its pretraining)",
        norm="z-score with Prithvi's pretraining\nmean and std",
        passes=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], pass_text="3 passes of 4 months",
        patch=16, tokens="16 × 16 px patches (160 m)\n8 × 8 = 64 tokens per month\n× 4 months = 256 tokens per pass\n(+ 1 class token, dropped)",
        encoder="ViT-Large\n24 transformer blocks\n1024 values per token\n304 M weights",
        extra_in=["date of each month\n(year, day of year)", "location\n(latitude, longitude)"],
        out="1024 × 8 × 8\nper month\n(12 months)", out_grid=8, out_dim="1024",
        time="ltae", grid_after="256 × 8 × 8",
        pyramid=[("×4", 32), ("×2", 16), ("×1", 8), ("×½", 4)], pyr_note="ViT feature pyramid\n(learnt up/down-sampling)",
        head_grid=32, head_m="40 m", trained="2.48 M"),
    "clay": dict(
        name="Clay v1.5", family="image", colour="#2ca02c",
        bands="10 bands: B2–B8A, B11, B12\n(no B1, B9) + their wavelengths",
        norm="z-score with Clay's Sentinel-2\nstatistics (metadata.yaml)",
        passes=[[m] for m in range(12)], pass_text="12 passes of 1 month",
        patch=8, tokens="8 × 8 px patches (80 m)\n16 × 16 = 256 tokens per month\n(+ 1 class token, dropped)\nband filters built from wavelengths",
        encoder="ViT-Large\n24 transformer blocks\n1024 values per token\n311 M weights",
        extra_in=["week of the month's mid-date,\nhour (sin, cos)", "location (sin, cos)\nand pixel size 10 m"],
        out="1024 × 16 × 16\nper month\n(12 months)", out_grid=16, out_dim="1024",
        time="ltae", grid_after="256 × 16 × 16",
        pyramid=[("×4", 64), ("×2", 32), ("×1", 16), ("×½", 8)], pyr_note="ViT feature pyramid\n(learnt up/down-sampling)",
        head_grid=64, head_m="20 m", trained="2.48 M"),
    "terramind": dict(
        name="TerraMind-1.0-base", family="image", colour="#9467bd",
        bands="12 bands: B1–B12 without B10\n(modality S2L2A only)",
        norm="z-score with TerraMind's\npretraining statistics",
        passes=[[m] for m in range(12)], pass_text="12 passes of 1 month",
        patch=16, tokens="16 × 16 px patches (160 m)\n8 × 8 = 64 tokens per month\n+ S2L2A modality embedding\n(no class token)",
        encoder="transformer\n12 blocks\n768 values per token\n87 M weights",
        extra_in=["no date input:\nmonths encoded separately", "no location input"],
        out="768 × 8 × 8\nper month\n(12 months)", out_grid=8, out_dim="768",
        time="ltae", grid_after="256 × 8 × 8",
        pyramid=[("×4", 32), ("×2", 16), ("×1", 8), ("×½", 4)], pyr_note="ViT feature pyramid\n(learnt up/down-sampling)",
        head_grid=32, head_m="40 m", trained="2.41 M"),
    "agrifm": dict(
        name="AgriFM", family="video", colour="#d62728",
        bands="10 bands: B2–B8A, B11, B12",
        norm="z-score with the region's own\nband statistics (D15)",
        passes=[list(range(12))], pass_text="1 pass of all 12 months",
        patch=4, tokens="3-D patches: 4 months × 4 × 4 px\n3 time steps × 32 × 32 tokens\n128 values per token",
        encoder="Video Swin Transformer\n4 stages (2, 2, 18, 2 blocks)\nwindowed 3-D attention\n88 M weights",
        extra_in=["each merge halves the map\nand the number of time steps", "no date or location input"],
        out="4 maps:\n384 × 32²  ·  512 × 16²\n512 × 8²  ·  1024 × 4²\n(time folded into channels)", out_grid=None,
        out_dim="", time="inside", grid_after="",
        pyramid=[("stage 1", 32), ("stage 2", 16), ("stage 3", 8), ("stage 4", 4)], pyr_note="AgriFM's own 4 maps\n(no extra pyramid)",
        head_grid=32, head_m="40 m", trained="3.23 M"),
    "presto": dict(
        name="Presto", family="pixel", colour="#ff7f0e",
        bands="10 bands + NDVI per month\n(Sentinel-1, weather, terrain masked)",
        norm="reflectance / 10,000;\ncloudy months interpolated",
        passes=[list(range(12))], pass_text="1 pass of the pixel's 12 months",
        patch=1, tokens="one pixel at a time (16,384 per tile)\n6 band groups × 12 months\n+ 1 location token = 73 tokens",
        encoder="transformer\n2 blocks\n128 values per token\n0.4 M weights",
        extra_in=["month of each token", "location (lat, lon\nas a point on the sphere)"],
        out="mean of the tokens\n→ 128 per pixel\n= 128 × 128 × 128 map", out_grid=None, out_dim="128",
        time="inside", grid_after="",
        pyramid=[("×1", 128), ("×½", 64), ("×¼", 32), ("×⅛", 16)], pyr_note="average pooling of\nthe 10 m map",
        head_grid=128, head_m="10 m", trained="1.72 M"),
    "tessera": dict(
        name="TESSERA v2 (Medium)", family="pixel", colour="#8c564b",
        bands="10 bands: B4 B2 B3 B8 B8A\nB5 B6 B7 B11 B12",
        norm="z-score with TESSERA's\nstatistics; clear months only",
        passes=[list(range(12))], pass_text="1 pass of the pixel's clear months",
        patch=1, tokens="one pixel at a time (16,384 per tile)\n1 token per clear month (dated\nby day of year), padded to 8 or 16",
        encoder="Sentinel-2 branch\n4 blocks, 440 values\nattention pooling over time\n21 M weights",
        extra_in=["day of year of each\nmonthly composite (15th)", "Sentinel-1 branch:\none empty observation"],
        out="MLP → 128 per pixel\n= 128 × 128 × 128 map", out_grid=None, out_dim="128",
        time="inside", grid_after="",
        pyramid=[("×1", 128), ("×½", 64), ("×¼", 32), ("×⅛", 16)], pyr_note="average pooling of\nthe 10 m map",
        head_grid=128, head_m="10 m", trained="1.72 M"),
}


def panel_e04_compare(ax, n, name):
    rows = [(f"{name}\nvs E01 random forest", "does a foundation model beat\nthe pixel baseline?"),
            (f"{name}\nvs E03 U-TAE", "does pretraining add more than\nspatial context? (same tiles)")]
    for k, (a, b) in enumerate(rows):
        y = 2.7 - k * 1.3
        box(ax, 0, y, 3.0, 0.8, a, LOOP_COL, fs=9)
        ax.text(3.25, y + 0.4, b, va="center", fontsize=9)
    ax.text(0, 0.35, "paired difference on the same pixels and bootstrap\ndraws; counts if its 95% CI excludes 0 and "
            "the\nHolm-corrected p < 0.05", fontsize=8.5, color="0.3")
    blank(ax, (-0.1, 7.2), (0, 3.9))
    note(ax, "Every run of E01–E04 is compared with two references; Holm corrects for the many comparisons. The "
             "answers to RQ1 are on the Comparison page.", y=-0.02)
    step(ax, n, "Compare: two reference runs", "pixelwise_run/05_compare_experiments.py")


def draw_e04_model(key):
    c = E04_MODELS[key]
    col = c["colour"]
    rng = np.random.default_rng(SEED)
    months_step = "L-TAE over\nthe months" if c["time"] == "ltae" else "Months merged\nin the encoder"
    pyr_step = {"image": "Feature\npyramid", "video": "AgriFM's\n4 maps", "pixel": "Pool to\n4 sizes"}[c["family"]]
    fig, gs = new_figure(
        f"E04 · {c['name']}, frozen, with a trained decoder: how the experiment works, from the tiles to the scores",
        ["Labelled pixels\nand tiles", "Rotate\nfolds", "Input:\nbands, months", "Cut into\ntokens", "Frozen\nencoder",
         "Encoder\noutput", months_step, pyr_step, "UPerNet\n→ 10 m", "Train the\nhead", "Predict,\nscore, CI",
         "Compare with\nE01, E03"], loop=(10,), nrows=3, ncols=4, size=(22, 18.5), note_width=68, bottom=0.11,
        footer=f"Grey = frozen (pretrained, never changed) · orange = trained in E04 · shapes for one 128 × 128 px tile. "
               f"Outputs: experiments/E04_frozen_geofm/results/<region>/{key}/ (oof_predictions.parquet · metrics.csv · "
               f"training_history.csv · run_config.json)")

    def title(ax, n, t):
        step(ax, n, t)

    panel_pixels(fig.add_subplot(gs[0, 0]), 1, rng, tiles=True)
    panel_rotation(fig.add_subplot(gs[0, 1]), 2)

    # 1 input
    ax = fig.add_subplot(gs[0, 2])
    for m in range(5):
        ax.add_patch(Rectangle((0.3 + m * 0.18, 2.4 - m * 0.18), 2.2, 2.2, fc=plt.cm.YlGn(0.25 + 0.1 * m), ec="0.4"))
    ax.text(1.75, 4.95, "12 monthly composites, 128 × 128 px", ha="center", fontsize=8.5)
    for g, grp in enumerate(c["passes"]):
        for mth in grp:
            ax.add_patch(Rectangle((0.1 + mth * 0.48, 1.25), 0.44, 0.45, fc=col, alpha=0.3 + 0.45 * (g % 2), ec="white"))
            ax.text(0.32 + mth * 0.48, 1.47, str(mth + 1), ha="center", va="center", fontsize=7)
    ax.text(0.1, 1.0, f"months: {c['pass_text']}", fontsize=9, va="top", color=col, fontweight="bold")
    ax.text(0.1, 0.45, c["bands"] + "\n" + c["norm"], fontsize=8.6, va="top")
    blank(ax, (0, 5.9), (-1.3, 5.2))
    ax.set_aspect("equal")
    ax.set_anchor("N")
    title(ax, 3, "What the encoder reads: bands, months")

    # 2 tokens
    ax = fig.add_subplot(gs[0, 3])
    ax.add_patch(Rectangle((0, 1.0), 3.2, 3.2, fc="#f2efe6", ec="0.3"))
    if c["patch"] > 1:
        n = 128 // c["patch"]
        step_ = 3.2 / n
        for k in range(n + 1):
            lw = 0.4 if n > 20 else 0.7
            ax.plot([k * step_, k * step_], [1.0, 4.2], color=col, lw=lw)
            ax.plot([0, 3.2], [1.0 + k * step_, 1.0 + k * step_], color=col, lw=lw)
        ax.add_patch(Rectangle((0, 4.2 - step_), step_, step_, fc=col, alpha=0.7))
        if c["family"] == "video":
            for d in (1, 2):
                ax.add_patch(Rectangle((0 + d * 0.12, 4.2 - step_ + d * 0.12), step_, step_, fc=col, alpha=0.35, ec=col))
    else:
        for k in range(0, 33, 4):
            ax.plot([k * 0.1, k * 0.1], [1.0, 4.2], color="0.85", lw=0.3)
            ax.plot([0, 3.2], [1.0 + k * 0.1, 1.0 + k * 0.1], color="0.85", lw=0.3)
        ax.add_patch(Rectangle((0, 4.12), 0.08, 0.08, fc=col))
        ax.annotate("one pixel (10 m)\nits own time series", (0.05, 4.15), (1.0, 3.3), fontsize=8.5, color=col,
                    arrowprops=dict(arrowstyle="->", color=col))
    ax.text(3.45, 2.6, c["tokens"], fontsize=8.8, va="center")
    blank(ax, (-0.1, 7.6), (0.2, 4.6))
    ax.set_aspect("equal")
    ax.set_anchor("N")
    title(ax, 4, "Cut the tile into tokens")

    # 3 encoder
    ax = fig.add_subplot(gs[1, 0])
    box(ax, 1.2, 1.4, 3.4, 2.8, c["encoder"], C_LOCK, fs=9.5)
    arrow(ax, (0.0, 2.8), (1.15, 2.8))
    ax.text(0.0, 3.0, "tokens", fontsize=8.5)
    ax.text(2.9, 4.45, "frozen: pretrained weights, no updates", ha="center", fontsize=8.5, color="0.4")
    extra = "also given to the encoder:\n" + "\n".join("• " + t.replace("\n", " ") for t in c["extra_in"])
    ax.text(2.9, 0.75, extra, fontsize=8.3, ha="center", va="top", color="0.25")
    arrow(ax, (2.9, 0.85), (2.9, 1.35), lw=1)
    blank(ax, (-0.1, 5.8), (-1.3, 4.8))
    ax.set_aspect("equal")
    ax.set_anchor("N")
    title(ax, 5, "Frozen encoder")

    # 4 output
    ax = fig.add_subplot(gs[1, 1])
    if c["out_grid"]:
        g = c["out_grid"]
        for m in range(3):
            off = m * 0.25
            ax.add_patch(Rectangle((0.3 + off, 1.4 + off), 2.2, 2.2, fc="#d6eaf8", ec="0.5"))
        for k in range(g + 1):
            ax.plot([0.8 + k * 2.2 / g, 0.8 + k * 2.2 / g], [1.9, 4.1], color="#5dade2", lw=0.4)
            ax.plot([0.8, 3.0], [1.9 + k * 2.2 / g, 1.9 + k * 2.2 / g], color="#5dade2", lw=0.4)
        ax.text(1.9, 1.2, f"{g} × {g} grid, one vector\nof {c['out_dim']} values per cell", ha="center", va="top",
                fontsize=8.5)
    elif c["family"] == "video":
        for k, (lab, size) in enumerate([("32²", 1.6), ("16²", 1.15), ("8²", 0.8), ("4²", 0.5)]):
            ax.add_patch(Rectangle((0.2 + k * 0.95, 2.6 - size / 2), size * 0.55, size, fc="#d6eaf8", ec="0.5"))
            ax.text(0.2 + k * 0.95 + size * 0.27, 2.6 - size / 2 - 0.25, lab, ha="center", fontsize=8.5)
    else:
        ax.add_patch(Rectangle((0.6, 1.6), 2.4, 2.4, fc="#d6eaf8", ec="0.5"))
        ax.text(1.8, 2.8, "128 values\nfor every\n10 m pixel", ha="center", va="center", fontsize=9)
    ax.text(4.0, 2.8, c["out"], fontsize=9, va="center")
    blank(ax, (0, 6.6), (0.3, 4.8))
    ax.set_aspect("equal")
    ax.set_anchor("N")
    title(ax, 6, "Encoder output")

    # 5 time
    ax = fig.add_subplot(gs[1, 2])
    if c["time"] == "ltae":
        for m in range(12):
            ax.add_patch(Rectangle((0.2 + m * 0.42, 3.4), 0.36, 0.36, fc="#d6eaf8", ec="0.5"))
            att = 0.15 + 0.85 * np.exp(-((m - 5) / 2.0) ** 2) * (0.6 + 0.4 * (m % 3 == 0))
            ax.add_patch(Rectangle((0.2 + m * 0.42, 2.0), 0.36, 1.1 * att, fc=C_TRAIN, alpha=0.85))
        ax.text(0.2, 4.0, "12 monthly vectors of one patch", fontsize=8.5)
        ax.text(0.2, 1.75, "learnt attention per month (L-TAE, 16 heads)", fontsize=8.5, color=C_TRAIN, va="top")
        box(ax, 1.4, 0.2, 2.8, 0.8, f"{c['grid_after']}\none vector per patch", C_TRAIN, fs=9)
        arrow(ax, (2.8, 1.35), (2.8, 1.05))
    else:
        box(ax, 0.6, 1.6, 4.0, 1.4, "months already merged\ninside the encoder\n→ no L-TAE", "#ecf0f1", tc="0.2", fs=10,
            ec="0.6")
    blank(ax, (0, 5.4), (0, 4.4))
    title(ax, 7, "Combine the 12 months")

    # 6 pyramid
    ax = fig.add_subplot(gs[1, 3])
    maxg = c["pyramid"][0][1]
    for k, (lab, gsz) in enumerate(c["pyramid"]):
        s_ = 2.4 * gsz / maxg
        y = 3.6 - k * 1.0
        ax.add_patch(Rectangle((0.3, y - min(s_, 0.8) / 2), s_, min(s_, 0.8), fc="#85c1e9" if k != 2 or c["time"] != "ltae"
                               else "#aed6f1", ec="white"))
        ax.text(2.9, y, f"{lab}: {gsz} × {gsz}", fontsize=9, va="center")
    ax.text(4.6, 2.1, c["pyr_note"], fontsize=8.8, va="center", color=C_TRAIN if c["family"] == "image" else "0.25")
    blank(ax, (0, 7.0), (0.0, 4.4))
    title(ax, 8, "Four map sizes for the decoder")

    # 7 head
    ax = fig.add_subplot(gs[2, 0])
    box(ax, 0.2, 1.6, 1.9, 1.4, "UPerNet\nhead\n(trained)", C_TRAIN, fs=9.5)
    arrow(ax, (2.15, 2.3), (2.75, 2.3))
    hg = c["head_grid"]
    ax.add_patch(Rectangle((2.8, 1.3), 2.0, 2.0, fc="#fdebd0", ec="0.5"))
    if hg <= 64:
        for k in range(0, hg + 1, max(1, hg // 16)):
            ax.plot([2.8 + k * 2 / hg, 2.8 + k * 2 / hg], [1.3, 3.3], color="0.7", lw=0.3)
            ax.plot([2.8, 4.8], [1.3 + k * 2 / hg, 1.3 + k * 2 / hg], color="0.7", lw=0.3)
    ax.text(3.8, 1.05, f"scores on\n{hg} × {hg}\n({c['head_m']} cells)", ha="center", va="top", fontsize=8.5)
    if hg < 128:
        arrow(ax, (4.85, 2.3), (5.35, 2.3))
        ax.text(5.1, 2.6, "bilinear", fontsize=7.5, ha="center")
        ax.add_patch(Rectangle((5.4, 1.3), 2.0, 2.0, fc="#d4e6f1", ec="0.5"))
        ax.text(6.4, 1.05, "class per\n10 m pixel\n(128 × 128)", ha="center", va="top", fontsize=8.5)
    else:
        ax.text(5.1, 2.3, "= already 10 m:\none class score\nper pixel", fontsize=8.5, va="center")
    ax.text(0.2, 4.1, "the 4 sizes are fused into\n2 class scores per cell", fontsize=8.5, va="top")
    blank(ax, (0, 7.6), (-0.2, 4.4))
    ax.set_aspect("equal")
    ax.set_anchor("N")
    title(ax, 9, "Decoder → irrigated / rainfed")

    # 10-12 training, scoring, comparison (the same for every encoder)
    ax = fig.add_subplot(gs[2, 1])
    trained = f"the L-TAE and the decoder ({c['trained']} weights)" if c["time"] == "ltae" else \
        f"the decoder ({c['trained']} weights)"
    panel_early_stopping(ax, 10, rng, patience=8, max_epochs=50,
                         extra=f"Only {trained} learn; the encoder stays frozen. Loss only at the labelled pixels of "
                               "the training folds; AdamW 10⁻⁴, batch 8 tiles, flips and rotations.")
    ax.set_title("10 · Train the head, stop early\nper round and seed; encoder frozen", fontsize=11.5, loc="left",
                 color=LOOP_COL)
    ax = fig.add_subplot(gs[2, 2])
    panel_bootstrap(ax, 11, rng)
    ax.set_title("11 · Predict the test tiles, score, 95% CI\nread at the labelled pixels, as E01–E03", fontsize=11.5,
                 loc="left", color=STEP_COL)
    short = {"prithvi": "Prithvi", "clay": "Clay", "terramind": "TerraMind", "agrifm": "AgriFM", "presto": "Presto",
             "tessera": "TESSERA"}[key]
    panel_e04_compare(fig.add_subplot(gs[2, 3]), 12, short)
    return fig

# ================================================================ E05, one figure per strategy
E05_STRATEGIES = {
    "frozen": dict(name="Frozen", start="pretrained", enc_state="locked: no gradient, evaluation mode",
                   trained="2.5 M (L-TAE + head)", lr="head 10⁻⁴", kept="the head",
                   measures="what the pretrained features already hold for irrigated vs rainfed"),
    "lora": dict(name="LoRA", start="pretrained", enc_state="locked; low-rank adapters (rank 16) in all 24 blocks trained",
                 trained="8.0 M (5.5 M adapters + 2.5 M head)", lr="head and adapters 10⁻⁴", kept="head + adapters",
                 measures="how much a small, low-rank change of the encoder adds"),
    "finetune": dict(name="Full fine-tuning", start="pretrained", enc_state="every weight trained (training mode)",
                     trained="306 M (encoder + head)", lr="head 10⁻⁴, encoder 10⁻⁵", kept="head + encoder",
                     measures="what adapting every pretrained feature adds"),
    "scratch": dict(name="From scratch", start="random", enc_state="every weight trained from a random start",
                    trained="306 M (encoder + head)", lr="head and encoder 10⁻⁴", kept="head + encoder",
                    measures="what the same network learns without pretraining"),
}
E05_PAIRS = [("finetune", "frozen", "effect of training the encoder"), ("finetune", "scratch", "effect of the pretrained weights"),
             ("frozen", "scratch", "frozen against no pretraining"), ("lora", "frozen", "effect of the adapters"),
             ("finetune", "lora", "full vs parameter-efficient"), ("lora", "scratch", "adapters against no pretraining")]
SHORT = {"frozen": "Frozen", "lora": "LoRA", "finetune": "Fine-tuned", "scratch": "Scratch"}


def draw_e05_strategy(key):
    c = E05_STRATEGIES[key]
    rng = np.random.default_rng(SEED)
    fig, gs = new_figure(
        f"E05 · Prithvi, strategy “{c['name']}”: how the experiment works, from the tiles to the comparisons (RQ2)",
        ["Labelled pixels\nand tiles", "Four strategies:\nthis one", "Label shares\n10–100%", "Rotate\nfolds",
         "Input and\ntokens", "Encoder:\n" + ("locked" if key == "frozen" else "adapters" if key == "lora" else "trained"),
         "L-TAE +\nUPerNet head", "Train, stop\nearly", "What is kept\nand its cost", "Learning\ncurves",
         "Score per\nshare", "Paired\ntests"], loop=(8,), nrows=3, ncols=4, size=(22, 18.5), note_width=68, bottom=0.11,
        footer=f"Grey = locked · orange = trained. Outputs: experiments/E05_prithvi_strategies/results/<region>/"
               f"{key}_p<share>/ (oof_predictions.parquet · metrics.csv · training_history.csv · label_subset.csv)")

    panel_pixels(fig.add_subplot(gs[0, 0]), 1, rng, tiles=True)

    # 2 the four strategies, this one highlighted
    ax = fig.add_subplot(gs[0, 1])
    for k, (sk, sc) in enumerate(E05_STRATEGIES.items()):
        y = 3.4 - k * 1.0
        on = sk == key
        box(ax, 0, y, 2.4, 0.75, sc["name"], STEP_COL if on else "#d5d8dc", tc="white" if on else "0.35", fs=9.5,
            bold=on, ec=C_TRAIN if on else "none", lw=2.5)
        ax.text(2.6, y + 0.37, f"{sc['start']} · {sc['trained'].split(' (')[0]} trained", va="center", fontsize=9,
                color="black" if on else "0.55")
    blank(ax, (-0.1, 7.2), (-0.2, 4.3))
    note(ax, f"E05 trains the same Prithvi network in four ways. This figure follows “{c['name']}”: {c['measures']}. "
             "The other three strategies run the same steps on the same pixels, so all four can be compared pair by "
             "pair.", y=-0.02)
    step(ax, 2, f"This run: {c['name']}", "one of four ways to train the same network")

    # 3 label shares
    ax = fig.add_subplot(gs[0, 2])
    for k, sh in enumerate([0.1, 0.25, 0.5, 1.0]):
        y = 3.3 - k * 0.95
        ax.add_patch(Rectangle((1.4, y), 4.0, 0.6, fc="#ecf0f1", ec="0.7"))
        ax.add_patch(Rectangle((1.4, y), 4.0 * sh, 0.6, fc=C_TR, ec="none"))
        ax.text(1.3, y + 0.3, f"{int(sh * 100)}%", ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(3.4, 4.15, "training-fold parcels kept (per class)", ha="center", fontsize=8.5)
    box(ax, 5.7, 2.2, 1.0, 0.8, "val\n100%", C_VA, tc="black", fs=8.5)
    box(ax, 5.7, 1.1, 1.0, 0.8, "test\n100%", C_TE, tc="black", fs=8.5)
    blank(ax, (0, 6.9), (0.3, 4.5))
    note(ax, f"“{c['name']}” is run four times: with 10, 25, 50 and 100% of the training parcels (one GPU job each). "
             "Parcels are dropped per class in a fixed random order, so the smaller sets lie inside the larger ones "
             "and the four strategies see the same parcels. Validation and test folds stay complete.", y=-0.02)
    step(ax, 3, "Four label shares, one job each", "only the training folds are thinned")

    panel_rotation(fig.add_subplot(gs[0, 3]), 4)

    # 5 input and tokens
    ax = fig.add_subplot(gs[1, 0])
    for m in range(4):
        ax.add_patch(Rectangle((0.2 + m * 0.18, 2.3 - m * 0.18), 2.0, 2.0, fc=plt.cm.YlGn(0.25 + 0.12 * m), ec="0.4"))
    n = 8
    for k in range(n + 1):
        ax.plot([0.74 + k * 2.0 / n] * 2, [1.76, 3.76], color="#1f77b4", lw=0.5)
        ax.plot([0.74, 2.74], [1.76 + k * 2.0 / n] * 2, color="#1f77b4", lw=0.5)
    for mth in range(12):
        g = mth // 4
        ax.add_patch(Rectangle((0.1 + mth * 0.44, 0.9), 0.4, 0.42, fc="#1f77b4", alpha=0.3 + 0.45 * (g % 2), ec="white"))
        ax.text(0.3 + mth * 0.44, 1.11, str(mth + 1), ha="center", va="center", fontsize=7)
    ax.text(3.1, 3.2, "6 bands: B2 B3 B4 B8A\nB11 B12, z-scored\n16 px patches →\n8 × 8 tokens per month", fontsize=8.8,
            va="center")
    ax.text(0.1, 0.65, "3 passes of 4 months, with dates and location", fontsize=8.8, va="top", color="#1f77b4")
    blank(ax, (0, 5.6), (0.1, 4.4))
    note(ax, "The same input as E04 Prithvi for every strategy: the tile's 12 monthly composites in Prithvi's six "
             "bands and normalisation, read in three passes of four months.", y=-0.02)
    step(ax, 5, "Input and tokens (as E04 Prithvi)", "the same for all four strategies")

    # 6 encoder: what is trained
    ax = fig.add_subplot(gs[1, 1])
    enc_fc = C_LOCK if key in ("frozen", "lora") else C_TRAIN
    for b in range(6):
        hatch = {"finetune": "////", "scratch": ".."}.get(key)
        ax.add_patch(Rectangle((0.6, 0.6 + b * 0.55), 2.6, 0.45, fc=enc_fc, ec="white" if not hatch else "#7f8c8d",
                               hatch=hatch, lw=0.5))
        if key == "lora":
            ax.add_patch(Rectangle((3.3, 0.66 + b * 0.55), 0.18, 0.33, fc=C_TRAIN))
            ax.add_patch(Rectangle((3.55, 0.66 + b * 0.55), 0.18, 0.33, fc=C_TRAIN))
    ax.text(1.9, 4.05, "24 transformer blocks (6 drawn)", ha="center", fontsize=8.5)
    start_txt = "starts from the pretrained weights" if c["start"] == "pretrained" else "starts from random weights"
    ax.text(4.1, 2.9, start_txt, fontsize=9, color="#5d6d7e" if c["start"] == "pretrained" else "#c0392b",
            fontweight="bold")
    ax.text(4.1, 2.45, textwrap.fill(c["enc_state"], 30), fontsize=9, va="top")
    if key == "lora":
        ax.text(4.1, 1.15, "orange bars: adapters B·A\nnext to the qkv, fc1, fc2\nweights of each block", fontsize=8.5,
                color=C_TRAIN, va="top")
    if key in ("finetune", "scratch"):
        ax.text(4.1, 1.15, "hatched: pretrained weights,\nadapted" if key == "finetune" else
                "dotted: random weights,\nlearnt from the labels only", fontsize=8.5, color=C_TRAIN, va="top")
    blank(ax, (0, 7.6), (0.2, 4.4))
    note_txt = {"frozen": "The pretrained encoder is used as it is: no gradient reaches it and it stays in evaluation "
                          "mode, exactly as in E04.",
                "lora": "The pretrained weights stay locked; next to each weight matrix W of the attention and MLP layers "
                        "two thin matrices B (1024 × 16) and A (16 × 1024) are trained: output = W·x + (α/r)·B·A·x. B "
                        "starts at 0, so training starts exactly from the frozen network.",
                "finetune": "Every weight of the pretrained encoder is updated, at a learning rate ten times smaller than "
                            "the head's, so the pretrained features are adapted rather than overwritten.",
                "scratch": "The same architecture with random initial weights (drawn from the seed): every weight is "
                           "learnt from the labelled pixels only, at the head's learning rate."}[key]
    note(ax, note_txt, y=-0.02)
    step(ax, 6, "The encoder in this strategy", "grey = locked · orange = trained")

    # 7 head
    ax = fig.add_subplot(gs[1, 2])
    chain = [("encoder output\n1024 × 8 × 8 per month", enc_fc), ("L-TAE over\nthe 12 months", C_TRAIN),
             ("feature pyramid\n+ UPerNet", C_TRAIN), ("class per\n10 m pixel", LOOP_COL)]
    for k, (t, cc) in enumerate(chain):
        box(ax, 0.5, 3.4 - k * 1.05, 3.4, 0.8, t, cc, fs=9)
        if k:
            arrow(ax, (2.2, 3.4 - (k - 1) * 1.05), (2.2, 4.2 - k * 1.05))
    ax.text(4.2, 2.3, "the same head\nas E04 Prithvi\n(2.5 M weights),\nalways trained", fontsize=9, va="center",
            color=C_TRAIN)
    blank(ax, (0, 6.6), (0.1, 4.4))
    note(ax, "Behind the encoder: an L-TAE merges the 12 monthly outputs of each 160 m patch, a feature pyramid and a "
             "UPerNet head give class scores on a 40 m grid, upsampled to 10 m. Identical in all four strategies, so "
             "only the encoder treatment differs.", y=-0.02)
    step(ax, 7, "Head: L-TAE + pyramid + UPerNet", "trained in every strategy")

    # 8 training
    ax = fig.add_subplot(gs[1, 3])
    grad = {"frozen": "Gradients reach the head only.", "lora": "Gradients flow through the locked encoder to the "
            "adapters.", "finetune": "Gradients update the whole network.", "scratch": "Gradients update the whole "
            "network."}[key]
    panel_early_stopping(ax, 8, rng, patience=15, max_epochs=60,
                         extra=f"Learning rate: {c['lr']}; it rises over the first epoch (warm-up), gradients are "
                               f"clipped at norm 1; AdamW, batch 8 tiles, loss only at the kept training pixels. {grad}")
    ax.set_title("8 · Train, stop early\nper round and seed, at each label share", fontsize=11.5, loc="left",
                 color=LOOP_COL)

    # 9 kept and cost
    ax = fig.add_subplot(gs[2, 0])
    rows = [("Trained weights", c["trained"]), ("Learning rate", c["lr"]), ("Best epoch keeps", c["kept"]),
            ("Backward pass", "head only" if key == "frozen" else "through the whole encoder")]
    for k, (a, b) in enumerate(rows):
        y = 3.4 - k * 0.95
        box(ax, 0, y, 2.6, 0.72, a, STEP_COL, fs=9)
        ax.text(2.8, y + 0.36, b, va="center", fontsize=9)
    blank(ax, (-0.1, 7.4), (0.2, 4.3))
    note(ax, "Seconds per epoch and GPU hours of each run are summed by report.py (summary.csv): training the encoder "
             "(LoRA, fine-tuning, scratch) needs the backward pass through 300 M weights and more GPU memory than the "
             "frozen encoder.", y=-0.02)
    step(ax, 9, "What the run trains, keeps and costs", c["name"])

    # 10 learning curves
    ax = fig.add_subplot(gs[2, 1])
    ep = np.arange(1, 41)
    tr_loss = 0.08 + 0.55 * np.exp(-ep / 6) + rng.normal(0, 0.006, len(ep))
    va_loss = 0.32 + 0.25 * np.exp(-ep / 4) + 0.006 * np.clip(ep - 12, 0, None) + rng.normal(0, 0.012, len(ep))
    va_f1 = 0.8 - 0.3 * np.exp(-ep / 4) + rng.normal(0, 0.01, len(ep))
    ax.plot(ep, tr_loss, color="0.45", lw=1.8, label="training loss")
    ax.plot(ep, va_loss, color="#c0392b", lw=1.8, label="validation loss")
    ax.plot(ep, va_f1, color=LOOP_COL, lw=2, label="validation macro-F1")
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("epoch")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.82))
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    note(ax, "Every epoch logs training loss, validation loss and validation macro-F1 (training_history.csv). Averaged "
             "over rounds and seeds, they show how fast this strategy learns and when it overfits; report.py draws "
             "them next to the other strategies.", y=-0.07)
    step(ax, 10, "Learning curves", "illustration of what is logged")

    # 11 score per share
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(4)
    y = 0.62 + 0.12 * (1 - np.exp(-(x + 1) / 1.4))
    ax.errorbar(x, y, yerr=0.035, fmt="o-", color=C_TRAIN, lw=2, capsize=3, label=c["name"])
    ax.axhline(0.70, color="k", ls=":", lw=1)
    ax.text(-0.15, 0.705, "references (E01, E03) with all labels", fontsize=8, va="bottom")
    ax.set_xticks(x, ["10%", "25%", "50%", "100%"])
    ax.set_yticks([])
    ax.set_xlabel("share of the training parcels")
    ax.set_ylabel("out-of-fold macro-F1 (95% CI)")
    ax.legend(fontsize=8.5, loc="lower right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    note(ax, "Each of the four runs predicts every labelled pixel out of fold and is scored with its block-bootstrap "
             "CI, as in E01. Drawn against its share, the four points show how much the strategy loses with fewer "
             "labels.", y=-0.12)
    step(ax, 11, "Score at each label share", "illustration: one point per run")

    # 12 paired tests involving this strategy
    ax = fig.add_subplot(gs[2, 3])
    for k, (a, b, txt) in enumerate(E05_PAIRS):
        y = 5.0 - k * 0.85
        on = key in (a, b)
        box(ax, 0, y, 2.9, 0.65, f"{SHORT[a]} − {SHORT[b]}", LOOP_COL if on else "#d5d8dc",
            tc="white" if on else "0.45", fs=9)
        ax.text(3.1, y + 0.32, txt, va="center", fontsize=8.8, color="black" if on else "0.6")
    blank(ax, (-0.1, 6.8), (-0.2, 5.9))
    note(ax, f"The pairs that involve “{c['name']}” (blue) are compared at each share on the same pixels and bootstrap "
             "draws; Holm corrects over the 24 comparisons per metric; a difference counts when its 95% CI excludes 0 "
             "and the Holm-corrected p < 0.05 (report.py).", y=-0.02)
    step(ax, 12, "Paired tests at each share", "the comparisons this strategy enters")
    return fig


if __name__ == "__main__":
    DRAW = {"e01": draw_e01, "e02": draw_e02, "e03": draw_e03}
    for e in EXPERIMENTS:
        f = DRAW[e]()
        out = OUT_DIR / f"{e}_explainer.png"
        f.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(f)
        print(out)
    jobs = [(f"e04_{k}", draw_e04_model, k) for k in E04_ENCODERS] + [(f"e05_{k}", draw_e05_strategy, k) for k in E05_KEYS]
    for name, fn, key in jobs:
        f = fn(key)
        out = OUT_DIR / f"{name}_explainer.png"
        f.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(f)
        print(out)
