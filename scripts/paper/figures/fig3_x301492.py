#!/usr/bin/env python3
"""One tablet carried the whole way, from clauses to a single year.

X301492 is the clearest case in the corpus rather than the typical one, and it
is drawn here because it shows the mechanism the rest of the paper measures:
each slow planet names a family of years, and the families intersect. Panel (b)
is the same tablet's score over the full candidate calendar, which is what the
method actually returns before any published date is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as fs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "paper" / "Figures"
sys.path.insert(0, str(ROOT))

from tools import cal, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402

TABLET = "X301492"


def short(statement: S.Statement) -> str:
    """A readable label for a clause, without the surrounding damage."""
    body = statement.planet.title()
    if statement.kind == "in_sign":
        return f"{body} in {statement.sign}"
    if statement.kind == "station":
        return f"{body} stations in {statement.sign}"
    if statement.kind == "entry":
        return f"{body} reaches {statement.sign}"
    if statement.kind == "last_app":
        return f"{body} last seen in {statement.sign}"
    return f"{body} {statement.kind.replace('_', ' ')}"


def panel_intersection(ax, years, sts, scores) -> None:
    # Counts occupy a separate margin, not calendar-year coordinates.
    count_transform = ax.get_yaxis_transform()
    alone_column, combined_column = 1.09, 1.28
    agree = scores.outcomes == 1
    decisive = sorted(
        (int(i) for i in np.flatnonzero(scores.decisive & (scores.weights > 0))),
        key=lambda i: -int(agree[:, i].sum()))

    labels, running = [], np.ones(len(years), dtype=bool)
    for row, index in enumerate(decisive):
        running = running & agree[:, index]
        ax.vlines(years[agree[:, index]], row - 0.28, row + 0.28,
                  color=fs.FAINT, linewidth=0.6, zorder=2)
        ax.vlines(years[running], row - 0.28, row + 0.28, color=fs.BLUE,
                  linewidth=1.0, zorder=3)
        ax.text(alone_column, row, f"{int(agree[:, index].sum())}", fontsize=7.4,
                va="center", ha="right", color=fs.GREY,
                transform=count_transform)
        ax.text(combined_column, row, f"{int(running.sum())}", fontsize=7.4,
                va="center", ha="right", color=fs.BLUE, fontweight="bold",
                transform=count_transform)
        labels.append(short(sts[index]))

    strict = years[scores.strict]
    row = len(decisive)
    ax.vlines(strict, row - 0.28, row + 0.28, color=fs.ORANGE, linewidth=1.6,
              zorder=4)
    ax.text(combined_column, row, f"{len(strict)}", fontsize=7.4, va="center",
            ha="right", color=fs.ORANGE, fontweight="bold",
            transform=count_transform)
    labels.append("strict set")

    ax.set_xlim(years[0] - 15, years[-1] + 15)
    ax.set_ylim(len(labels) - 0.35, -1.15)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.get_yticklabels()[-1].set_color(fs.ORANGE)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ticks = [-599, -499, -399, -299, -199, -99, 0, int(years[-1])]
    ax.set_xticks(ticks)
    ax.set_xticklabels([yr.year_label(t) for t in ticks])
    ax.set_xlabel("Babylonian year")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0, right=False)
    ax.tick_params(axis="x", which="both", top=False)
    ax.text(alone_column, -0.78, "alone", fontsize=7.2, color=fs.GREY, ha="right",
            va="bottom", transform=count_transform)
    ax.text(combined_column, -0.78, "combined", fontsize=7.2, color=fs.BLUE,
            ha="right", va="bottom", transform=count_transform)


NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine"}


def panel_score(ax, years, scores) -> None:
    ax.vlines(years, 0, scores.score, color=fs.FAINT, linewidth=0.6)
    best = int(np.argmax(scores.score))
    # Every year tied at the next-highest decisive bits, so no tied year is
    # left out, and the legend counts them and their clocks from the scores.
    rest = np.flatnonzero(np.arange(len(years)) != best)
    second = scores.score[rest].max()
    order = [int(i) for i in rest if scores.score[i] == second]
    ax.vlines(years[order], 0, scores.score[order], color=fs.BLUE,
              linewidth=1.2)
    ax.vlines(years[best], 0, scores.score[best], color=fs.ORANGE,
              linewidth=1.8)

    top = scores.score[best]
    clocks = {int(scores.priority[i]) for i in order}
    each = (f"{NUMBER[clocks.pop()]} clocks each" if len(clocks) == 1
            else "two or more clocks")
    ax.plot([], [], color=fs.ORANGE, linewidth=1.8,
            label=f"{yr.year_label(int(years[best]))}, "
                  f"{NUMBER[int(scores.priority[best])]} clocks, {top:.1f} bits")
    ax.plot([], [], color=fs.BLUE, linewidth=1.2,
            label=f"next {NUMBER[len(order)]} candidates, {each}")
    ax.plot([], [], color=fs.FAINT, linewidth=1.0,
            label="all other candidate years")
    ax.legend(loc="upper left", handlelength=1.4, handletextpad=0.5,
              borderpad=0.3, labelspacing=0.3)

    ax.set_xlim(years[0] - 15, years[-1] + 15)
    ax.set_ylim(0, top * 1.58)
    ticks = [-599, -499, -399, -299, -199, -99, 0, int(years[-1])]
    ax.set_xticks(ticks)
    ax.set_xticklabels([yr.year_label(t) for t in ticks])
    ax.set_xlabel("Babylonian year")
    ax.set_ylabel("Decisive bits")
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", which="both", right=False)


def main() -> Path:
    fs.use()
    text = pd.read_parquet(
        ROOT / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "text"]).set_index("tablet").loc[TABLET, "text"]
    years = np.array(cal.candidate_years(), dtype=int)
    sts = S.extract(text)
    scores = S.candidate_scores(sts, S.months_in(text), years)

    fig, axes = plt.subplots(
        2, 1, figsize=(fs.FULL_WIDTH, 3.90),
        gridspec_kw={"height_ratios": [1.06, 1.0], "hspace": 0.50})
    panel_intersection(axes[0], years, sts, scores)
    panel_score(axes[1], years, scores)
    fig.text(0.006, 0.988, "a", fontsize=9.5, fontweight="bold", color=fs.INK,
             ha="left", va="top")
    fig.text(0.006, 0.470, "b", fontsize=9.5, fontweight="bold", color=fs.INK,
             ha="left", va="top")
    fig.subplots_adjust(left=0.230, right=0.800, top=0.972, bottom=0.098)
    return fs.save(fig, OUT / "fig3_x301492")


if __name__ == "__main__":
    print(main())
