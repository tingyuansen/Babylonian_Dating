"""Shared plotting style, so every figure renders with the same typography.

The principle
carried over is the important part: a figure is scaled by LaTeX to the width of
its slot, so if the figure is drawn wider than that slot its text shrinks by the
same factor and the fonts stop matching between figures. Drawing at exactly the
slot width means every figure is placed at unit scale, and a 9 pt label is 9 pt
everywhere. Hence the fixed widths below and the deliberate refusal of
bbox="tight".

One thing does change from the source style. That paper is AASTeX and sets a
serif body font; this one loads helvet and makes sans the default family, so the
figures are drawn in sans here. The rule is that figure labels match the words
around them, and following the rule means changing the font, not keeping it.

Widths are computed from this document's geometry: a4paper is 8.27 in wide with
0.65 in margins on each side, leaving a 6.97 in text block.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

FULL_WIDTH = 6.97        # inches, the full text block
COLUMN_WIDTH = 3.38      # inches, half the block with a gutter

# A restrained palette. INK for structure and primary marks, BLUE and ORANGE as
# the two contrasting data colours, GREY for context that must recede. The
# sequence in SUPPORT is for the one figure that genuinely has many categories,
# ordered light to dark so that rank reads off the shade rather than the hue.
INK = "#1A1A1A"
BLUE = "#3B75AF"
ORANGE = "#C4552B"
GREEN = "#4A7A57"
GREY = "#7A7F87"
FAINT = "#B9BFC7"
PALE = "#E8EAED"
MARKER_FACE = "white"

# Light to dark, for nested cumulative bands (Top 1 through Top 20).
DEPTH = ["#DCE4EE", "#B6C6DC", "#8AA5C6", "#5E82AF", "#3B75AF", "#28517A"]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.7,
    "figure.constrained_layout.use": False,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
}


def use() -> None:
    mpl.rcParams.update(RC)


def save(fig, stem: Path) -> Path:
    """Write the figure as a PDF at its drawn size.

    The PDF carries no creation date, so a redraw from unchanged results gives
    the same bytes.
    """
    out = Path(stem).with_suffix(".pdf")
    fig.savefig(out, metadata={"CreationDate": None})
    return out


def panel_label(ax, text, x=0.015, y=0.985, color=INK, **kw):
    """Bold panel letter, placed inside the axes in the top-left by default."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color=color, **kw)
