"""
make_fine_tuning_figures.py  —  schematic figures of the fine-tuning strategies (RQ2 page)
==========================================================================================

Writes, next to this script:
  fine_tuning_strategies.png   the six ways to adapt a pretrained encoder: which weights are frozen, trained or added
  fine_tuning_strategies.gif   the same, one strategy per frame
  lora.png                     how a LoRA update sits next to a frozen weight matrix

Run (Mac):  python docs/phase2_models/figures/make_fine_tuning_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Rectangle

# ================================ SETTINGS ================================
N_BLOCKS = 6                     # transformer blocks drawn per encoder (Prithvi-EO-2.0-300M has 24)
FROZEN, TRAINED, ADDED, SSL = "#c6c6c6", "#e6550d", "#3182bd", "#31a354"
SECONDS_PER_FRAME = 3
# ==========================================================================

OUT = Path(__file__).resolve().parent
W, H, GAP = 1.6, 0.42, 0.1       # block size and spacing


def block(ax, x, y, color, text="", w=W, h=H, hatch=None, fs=7, tc="black"):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="#444", lw=0.8, hatch=hatch))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def encoder(ax, x, y0, colors, side=None, head=TRAINED, head_text="head (UPerNet)"):
    """Input, patch embedding, N blocks, head from bottom to top; returns the top y."""
    block(ax, x, y0, "white", "Sentinel-2 tile")
    y = y0 + H + GAP
    block(ax, x, y, colors[0], "patch embedding")
    for i, c in enumerate(colors[1:], start=1):
        y += H + GAP
        block(ax, x, y, c, f"block {i}" if i < len(colors) - 1 else "block N")
        if side:
            block(ax, x + W + 0.04, y + 0.06, side, "", w=0.22, h=H - 0.12)
    y += H + GAP
    block(ax, x, y, head, head_text, tc="white" if head in (TRAINED, SSL) else "black")
    return y + H


def arrow(ax, x0, y0, x1, y1, text=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", va="bottom", fontsize=6.5)


def nb(c, n=N_BLOCKS):
    return [c] * (n + 1)


def panel_frozen(ax):
    encoder(ax, 0.3, 0, nb(FROZEN))


def panel_full(ax):
    encoder(ax, 0.3, 0, nb(TRAINED))


def panel_partial(ax):
    k = N_BLOCKS // 2
    encoder(ax, 0.3, 0, [FROZEN] * (k + 1) + [TRAINED] * (N_BLOCKS - k))


def panel_peft(ax):
    encoder(ax, 0.3, 0, nb(FROZEN), side=ADDED)
    ax.text(0.3 + W + 0.15, (N_BLOCKS + 3) * (H + GAP) + 0.05, "LoRA /\nadapter", fontsize=6.5, ha="center", color=ADDED)


def panel_lpft(ax):
    top = encoder(ax, 0.0, 0, nb(FROZEN))
    encoder(ax, 2.2, 0, nb(TRAINED))
    y = top / 2
    arrow(ax, 0.0 + W + 0.05, y, 2.2 - 0.05, y, "head\ncopied")
    ax.text(0.0 + W / 2, top + 0.12, "stage 1", ha="center", fontsize=7, weight="bold")
    ax.text(2.2 + W / 2, top + 0.12, "stage 2", ha="center", fontsize=7, weight="bold")


def panel_ssl(ax):
    top = encoder(ax, 0.0, 0, nb(SSL), head=SSL, head_text="reconstruct\nmasked patches")
    ax.texts[0].set_text("unlabelled tiles\nof the region")
    encoder(ax, 2.2, 0, nb(FROZEN), head=TRAINED)
    y = top / 2
    arrow(ax, 0.0 + W + 0.05, y, 2.2 - 0.05, y, "then any\nstrategy 1–5")
    ax.text(0.0 + W / 2, top + 0.12, "stage 1 (no labels)", ha="center", fontsize=7, weight="bold")
    ax.text(2.2 + W / 2, top + 0.12, "stage 2 (labels)", ha="center", fontsize=7, weight="bold")


PANELS = [
    ("1 · Frozen encoder + head", "only the head is trained", panel_frozen, 2.2),
    ("2 · Full fine-tuning", "every weight trained\n(encoder with a smaller learning rate)", panel_full, 2.2),
    ("3 · Partial fine-tuning", "only some blocks trained\n(here the upper ones)", panel_partial, 2.2),
    ("4 · Parameter-efficient (PEFT)", "encoder frozen; small added\nmodules + head trained", panel_peft, 2.6),
    ("5 · Linear probe, then fine-tune", "train the head first,\nthen every weight", panel_lpft, 4.0),
    ("6 · Continued self-supervised\npretraining", "adapt the encoder to the region\nwithout labels, then use 1–5", panel_ssl, 4.0),
]


def finish(ax, title, sub, width):
    top = (N_BLOCKS + 3) * (H + GAP)
    ax.set_xlim(-0.35, width)
    ax.set_ylim(-0.9, top + 0.55)
    ax.set_title(title, fontsize=9, weight="bold")
    ax.text(width / 2 - 0.2, -0.45, sub, ha="center", va="top", fontsize=7.5)
    ax.axis("off")


def legend(fig):
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="#444") for c in (FROZEN, TRAINED, ADDED, SSL)]
    fig.legend(handles, ["frozen (pretrained, not changed)", "trained with labels", "added, trained with labels",
                         "trained without labels (self-supervised)"], loc="lower center", ncol=4, fontsize=8,
               frameon=False)


def strategies_png():
    widths = [p[3] for p in PANELS]
    fig, axes = plt.subplots(1, 6, figsize=(17, 5.2), gridspec_kw={"width_ratios": widths})
    for ax, (title, sub, draw, w) in zip(axes, PANELS):
        draw(ax)
        finish(ax, title, sub, w)
    legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fine_tuning_strategies.png", dpi=140)
    plt.close(fig)


def strategies_gif():
    fig = plt.figure(figsize=(5.2, 5.4))

    def frame(i):
        fig.clf()
        title, sub, draw, w = PANELS[i]
        ax = fig.add_axes((0.05, 0.12, 0.9, 0.78))
        draw(ax)
        finish(ax, title, sub, w)
        handles = [Rectangle((0, 0), 1, 1, fc=c, ec="#444") for c in (FROZEN, TRAINED, ADDED, SSL)]
        fig.legend(handles, ["frozen", "trained", "added + trained", "self-supervised"], loc="lower center",
                   ncol=4, fontsize=7, frameon=False)

    anim = FuncAnimation(fig, frame, frames=len(PANELS))
    anim.save(OUT / "fine_tuning_strategies.gif", writer=PillowWriter(fps=1 / SECONDS_PER_FRAME), dpi=110)
    plt.close(fig)


def lora_png():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.text(0.5, 2.0, "input\nx\n(1024)", ha="center", va="center", fontsize=9)
    arrow(ax, 0.9, 2.0, 1.6, 3.0)
    arrow(ax, 0.9, 2.0, 1.6, 1.0)
    block(ax, 1.6, 2.35, FROZEN, "W  (pretrained, frozen)\n1024 × 3072 = 3.1 M weights", w=3.2, h=1.3, fs=8.5)
    block(ax, 1.6, 0.55, ADDED, "A\nr × 1024", w=1.0, h=0.9, fs=8.5, tc="white")
    block(ax, 3.1, 0.55, ADDED, "B\n3072 × r", w=1.0, h=0.9, fs=8.5, tc="white")
    arrow(ax, 2.6, 1.0, 3.1, 1.0)
    ax.text(3.85, 0.25, "r = 8:  8 × (1024 + 3072) = 32,768 weights (1% of W)", ha="center", fontsize=8,
            color=ADDED)
    arrow(ax, 4.85, 3.0, 5.9, 2.15)
    arrow(ax, 4.15, 1.0, 5.9, 1.85, "× α / r")
    ax.text(6.15, 2.0, "+", ha="center", va="center", fontsize=18)
    arrow(ax, 6.4, 2.0, 7.0, 2.0)
    ax.text(8.4, 2.0, "output  W x + (α/r) B A x\n\nB starts at zero, so training\nstarts from the pretrained model;"
            "\nafter training, B A can be added\ninto W (no extra cost at inference)", ha="center", va="center",
            fontsize=8.5)
    ax.set_title("LoRA on one attention projection (qkv) of a Prithvi block", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "lora.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    strategies_png()
    strategies_gif()
    lora_png()
    print(f"Figures in {OUT}")
