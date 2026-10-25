#!/usr/bin/env python3
"""What a diary is, and what its lines are worth.

Panel (a) is a public-domain photograph of a Babylonian astronomical diary.
Panel (b) is the text of a different tablet against the number of candidate
years each line leaves standing, computed the same way as everything else in
the paper. The tablet in panel (b) is X301322, whose opening carries an era
formula, an outer planet, weather, and a night the scribe records as unwatched,
within seven lines.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as fs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "paper" / "Figures"
sys.path.insert(0, str(ROOT))

from tools import cal, clauses, statements as S  # noqa: E402

TABLET = "X301322"
PHOTO = ROOT / "data" / "source" / "photos" / "alexander_diary_BM45962.jpg"
LAST_LINE = 6
WRAP = 88

def tidy(text: str) -> list[str]:
    """Undo the edition's spaced punctuation, for display only."""
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = text.replace("( ", "(").replace(" )", ")")
    text = text.replace("[ ", "[").replace(" ]", "]")
    return textwrap.wrap(text, WRAP) or [""]


def panel_diary(ax, bar_ax, years) -> None:
    text = pd.read_parquet(
        ROOT / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "text"]).set_index("tablet").loc[TABLET, "text"]
    sts = clauses.load(clauses.REVIEWED)[TABLET]
    scores = S.candidate_scores(sts, S.months_in(text), years)
    allowed = (scores.outcomes == 1).sum(axis=0)

    bounds, cursor = [], 0
    for line in text.split("\n"):
        bounds.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    on_line: dict[int, list[int]] = {}
    for index, statement in enumerate(sts):
        for number, (start, end) in enumerate(bounds):
            if start <= statement.source_start < end:
                on_line.setdefault(number, []).append(index)
                break

    y, source = 0.55, text.split("\n")
    for number in range(min(LAST_LINE, len(bounds))):
        found = on_line.get(number, [])
        best = min(found, key=lambda i: allowed[i]) if found else None
        count = int(allowed[best]) if best is not None else len(years)
        colour = (fs.ORANGE if best is not None and scores.decisive[best]
                  and scores.weights[best] > 0
                  else (fs.BLUE if best is not None else fs.GREY))
        top = y
        for part in tidy(source[number]):
            ax.text(0.0, y, part, fontsize=7.0, va="top", ha="left",
                    color=colour, family="monospace")
            y += 1.0
        centre = (top + y - 1.0) / 2.0
        bar_ax.barh(centre, max(np.log10(count), 0.03), height=0.66,
                    color=colour, alpha=0.9, zorder=3)
        # The label carries both the count and the information it represents,
        # since the score is built from the second and the first is what a
        # reader can picture.
        bits = np.log2(len(years) / count)
        label = (f"all {len(years)}, 0.0 bits" if count >= len(years)
                 else f"{count}, {bits:.1f} bits")
        # Labels sit in their own column past the full-calendar rule, so a
        # short bar and a long one carry their numbers at the same place.
        bar_ax.text(np.log10(len(years)) + 0.12, centre, label,
                    fontsize=7.1, va="center", ha="left", color=colour)
        y += 0.55

    for panel in (ax, bar_ax):
        panel.set_ylim(y - 0.35, -1.15)
        panel.axis("off")
    ax.set_xlim(0, 1)
    # Reserve enough room after the bars for the full-calendar count label.
    bar_ax.set_xlim(0, np.log10(len(years)) * 2.25)
    bar_ax.plot([np.log10(len(years))] * 2, [y - 0.2, -0.3], color=fs.INK,
                linewidth=0.7, linestyle=(0, (3.0, 2.0)), zorder=1)
    ax.text(0.022, -1.05, f"{TABLET}, the opening of the published translation",
            fontsize=7.3, color=fs.GREY, va="top", ha="left")
    bar_ax.text(0.0, -1.05, "agreeing years, and bits",
                fontsize=7.3, color=fs.GREY, va="top", ha="left")


def main() -> Path:
    fs.use()
    years = np.array(cal.candidate_years(), dtype=int)

    fig = plt.figure(figsize=(fs.FULL_WIDTH, 5.05))

    photo = fig.add_axes([0.010, 0.535, 0.470, 0.455])
    image = plt.imread(PHOTO)
    photo.imshow(image)
    photo.axis("off")

    diary = fig.add_axes([0.010, 0.070, 0.740, 0.425])
    bars = fig.add_axes([0.762, 0.070, 0.232, 0.425])
    panel_diary(diary, bars, years)

    fig.text(0.006, 0.996, "a", fontsize=9.5, fontweight="bold", color=fs.INK,
             ha="left", va="top")
    fig.text(0.006, 0.512, "b", fontsize=9.5, fontweight="bold", color=fs.INK,
             ha="left", va="top")
    return fs.save(fig, OUT / "fig2_diary")


if __name__ == "__main__":
    print(main())
