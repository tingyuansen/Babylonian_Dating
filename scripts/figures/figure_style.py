"""Set the fonts, sizes, colours and saving of the figures of the paper.

Each figure script in scripts/figures/ calls use() before it draws and save()
to write figures/<name>.pdf. A figure is drawn at a fixed width in inches,
FULL_WIDTH, and saved at its drawn size without bbox="tight". At that width on
the page, its text keeps its sizes in points, so all figures share one type
size. Text is set in a sans-serif face.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

FULL_WIDTH = 6.97        # inches, the width of a full-width figure

# INK draws axes and text. BLUE, ORANGE and GREEN are the data colours, and
# GREY, FAINT and PALE draw the context.
INK = "#1A1A1A"
BLUE = "#3B75AF"
ORANGE = "#C4552B"
GREEN = "#4A7A57"
GREY = "#7A7F87"
FAINT = "#B9BFC7"
PALE = "#E8EAED"

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
    """Apply the style to matplotlib."""
    mpl.rcParams.update(RC)


def save(fig, stem: Path) -> Path:
    """Write the figure as a PDF at its drawn size.

    The PDF carries no creation date, so a redraw from unchanged results gives
    the same bytes.
    """
    out = Path(stem).with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None})
    return out


def panel_label(ax, text, x=0.015, y=0.985, color=INK, **kw):
    """Bold panel letter, placed inside the axes in the top-left by default."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color=color, **kw)
