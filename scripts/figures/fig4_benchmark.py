#!/usr/bin/env python3
"""How often the published year survives, and how large a set that costs.

Every number here is retrospective and every one is paired with the size of the
answer it comes with, because a recall figure quoted without the set it refers
to is the easiest way to make broad ambiguity look like precise dating.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as fs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT))

from tools import cal  # noqa: E402


ORDER = ["tier", "decisive_families", "primary_bits", "support_bits",
         "contradict"]


def positions(candidates: pd.DataFrame) -> np.ndarray:
    """How many candidate years must be accepted to include the published one.

    Candidates are ordered by the same quantities the method reports, and ties
    are counted pessimistically, so a year tied with four others counts as five
    years accepted rather than one. Nothing here breaks a tie arbitrarily.
    """
    out = []
    for _, group in candidates.groupby("tablet", sort=False):
        keys = list(zip(group.tier,
                        -group.decisive_families,
                        -group.primary_bits.round(12),
                        -group.support_bits.round(12),
                        group.contradict))
        truth = keys[int(np.flatnonzero(group.correct.to_numpy())[0])]
        better = sum(1 for key in keys if key < truth)
        tied = sum(1 for key in keys if key == truth)
        out.append(better + tied)
    return np.array(out)


def curve(ax, place: np.ndarray, colour: str, label: str,
          anchors: bool, candidate_count: int) -> None:
    grid = np.arange(1, candidate_count + 1)
    recall = np.array([(place <= n).mean() * 100 for n in grid])
    ax.plot(grid, recall, color=colour, linewidth=1.6, zorder=3, label=label)
    if not anchors:
        return
    # One, three and ten. Beyond ten the accepted years are members of the
    # same recurrence lattice rather than neighbouring years, so a wider count
    # measures periodicity rather than dating.
    for n, (dx, dy) in zip((1, 3, 10), ((7, -4), (7, -4), (7, -4))):
        value = (place <= n).mean() * 100
        ax.plot([n], [value], "o", markersize=4.6, color=fs.ORANGE,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.annotate(f"{value:.0f}% at {n}", (n, value),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.4, color=fs.ORANGE, ha="left", va="top")


def dress(ax, candidate_count: int) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.85, candidate_count * 1.20)
    ax.set_ylim(0, 104)
    ticks = sorted({n for n in (1, 3, 5, 10, 50, candidate_count)
                    if n <= candidate_count})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(n) for n in ticks])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Candidate years accepted, in ranked order")
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", which="both", right=False)


def panel_groups(ax, sample) -> None:
    """How many distinct physical explanations must be considered."""
    groups = [
        (f"decisive clauses, {int(sample.decisive.sum())}",
         sample.decisive, fs.BLUE),
        (f"all, {len(sample)}",
         pd.Series(True, index=sample.index), fs.GREY),
        (f"supporting only, {int((~sample.decisive).sum())}",
         ~sample.decisive, fs.FAINT),
    ]
    # Later groups add no record, so the panel stops at two.
    ranks, width = [1, 2], 0.26
    for offset, (label, mask, colour) in enumerate(groups):
        part = sample[mask]
        value = [float(part.truth_evidence_group_rank.le(r).mean()) * 100
                 for r in ranks]
        x = np.arange(len(ranks)) + (offset - 1) * width
        ax.bar(x, value, width=width * 0.92, color=colour, label=label,
               zorder=3)
        for position, height in zip(x, value):
            ax.text(position, height + 1.4, f"{height:.0f}", ha="center",
                    va="bottom", fontsize=6.9, color=colour)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(["1", "1-2"])
    ax.set_xlim(-0.6, len(ranks) - 0.4)
    ax.set_xlabel("Groups accepted")
    ax.set_ylim(0, 142)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.055), handlelength=1.1,
              handletextpad=0.45, borderpad=0.25, labelspacing=0.32)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", which="both", top=False, length=0)
    ax.tick_params(axis="y", which="both", right=False)


def main() -> Path:
    fs.use()
    candidate_count = len(cal.candidate_years())
    sample = pd.read_parquet(
        ROOT / "data" / "benchmark" / "benchmark_results.parquet")
    candidates = pd.read_parquet(
        ROOT / "data" / "benchmark" / "ranking_candidates.parquet")

    active = positions(candidates[candidates.decisive])
    eligible = positions(candidates)
    supporting = positions(candidates[~candidates.decisive])

    fig, axes = plt.subplots(
        1, 2, figsize=(fs.FULL_WIDTH, 2.85),
        gridspec_kw={"width_ratios": [1.65, 1.0], "wspace": 0.215})
    for place, colour, label in (
            (active, fs.BLUE, f"decisive clauses, {len(active)}"),
            (eligible, fs.GREY, f"all, {len(eligible)}"),
            (supporting, fs.FAINT, f"supporting only, {len(supporting)}")):
        curve(axes[0], place, colour, label,
              anchors=label.startswith("decisive"),
              candidate_count=candidate_count)
    # Bottom right, where the three curves have already separated and the
    # annotations along them are not competing for space.
    axes[0].legend(loc="lower right", handlelength=1.4, handletextpad=0.5,
                   borderpad=0.3, labelspacing=0.3)
    axes[0].set_ylabel("Published year included (%)")

    panel_groups(axes[1], sample)
    dress(axes[0], candidate_count)
    for ax, letter in zip(axes, "ab"):
        fs.panel_label(ax, letter, x=-0.005, y=1.09)
    fig.subplots_adjust(left=0.078, right=0.995, top=0.912, bottom=0.152)
    return fs.save(fig, OUT / "fig4_benchmark")


if __name__ == "__main__":
    print(main())
