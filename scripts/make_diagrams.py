"""Generate the presentation diagrams used by the Streamlit app, the README and the capstone
brief: the master reference-architecture diagram and six compact "you are here" layer strips.

Unlike src/neatnest/figures.py (analysis charts for the brief, rebuilt by the data pipeline into
reports/figures/), these are brand-facing navigation assets, so they are written straight to
assets/diagrams/ and committed like any other image in assets/.

    PYTHONPATH=src python scripts/make_diagrams.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neatnest.config import ASSET_DIR, INK, INK_MUTED, SERIES  # noqa: E402
from neatnest.figures import BG, _style, draw_architecture  # noqa: E402

OUT = ASSET_DIR / "diagrams"

LAYERS = [
    ("1", "Brand & Product", "STP · JTBD · Kano · CBBE"),
    ("2", "Customer Factors", "Factor analysis · Van Westendorp"),
    ("3", "Acquisition", "Logistic propensity · decile lift"),
    ("4", "Retention & Value", "Lapse model · CLV · segments"),
    ("5", "Nest Copilot", "Guarded agent + RAG"),
]


def master():
    fig, ax = plt.subplots(figsize=(13, 11.1))
    draw_architecture(ax)
    fig.tight_layout()
    fig.savefig(OUT / "architecture_master.png", dpi=170)
    plt.close(fig)


def layer_strip(highlight: int | None, name: str):
    """A compact single-row strip of the five layers, with `highlight` (1-5, or None) emphasised.
    Meant to sit at the top of each Streamlit page as a small "you are here" cue."""
    fig, ax = plt.subplots(figsize=(11, 1.55))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 1.55)
    w, gap = 2.08, 0.12
    for i, (num, title, sub) in enumerate(LAYERS):
        x = 0.05 + i * (w + gap)
        active = highlight is not None and int(num) == highlight
        alpha = 0.22 if active else 0.08
        ax.add_patch(plt.Rectangle((x, 0.18), w, 1.2, fc=SERIES[i % len(SERIES)], ec=SERIES[i % len(SERIES)] if active else "none",
                                    lw=1.8 if active else 0, alpha=alpha))
        ax.text(x + 0.1, 1.16, f"{num}  {title}", fontsize=8.4, fontweight="bold",
                color=INK if active else INK_MUTED, va="top")
        ax.text(x + 0.1, 0.86, sub, fontsize=7, color=INK_MUTED, va="top")
        if i < len(LAYERS) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 0.78), xytext=(x + w + 0.02, 0.78),
                        arrowprops={"arrowstyle": "->", "color": INK_MUTED, "lw": 1.1})
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=170, facecolor=BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    master()
    layer_strip(None, "layer_strip_0")
    for num, _, _ in LAYERS:
        layer_strip(int(num), f"layer_strip_{num}")
    print(f"Wrote {1 + 1 + len(LAYERS)} diagrams to {OUT}")


if __name__ == "__main__":
    main()
