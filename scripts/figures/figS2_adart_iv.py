#!/usr/bin/env python3
"""Draw SI Fig. S2, the strict sets of the ranked ADART IV entries.

Each row is one ADART IV entry that step 3 ranks. A dark mark is a year of its
strict set among the 701 candidate years. The rows run from the narrowest
strict set at the top to the widest at the bottom. Many sets repeat at a
regular spacing, as an entry with one clock gives (SI Appendix, section S9).

The script reads data/results/adart_iv_dates.parquet and writes
figures/figS2_adart_iv.pdf.
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

from tools import cal, years as yr  # noqa: E402


def main() -> Path:
    fs.use()

    prospective = pd.read_parquet(
        ROOT / "data" / "results" / "adart_iv_dates.parquet"
    )
    ranked = prospective[prospective.claim.eq("ranked")].copy()

    # The 701 candidate years, 626 BCE to AD 75.
    years = np.array(cal.candidate_years(), dtype=int)
    year_to_position = {year: position for position, year in enumerate(years)}

    # The strict sets, from the narrowest to the widest.
    sets = sorted(
        (
            sorted(int(y) for y in row.strict_years_astro)
            for row in ranked.itertuples()
        ),
        key=lambda years: (len(years), years)
    )

    grid = np.zeros((len(sets), len(years)))
    for row_index, entry_years in enumerate(sets):
        for year in entry_years:
            position = year_to_position.get(year)
            if position is not None:
                grid[row_index, position] = 1.0

    fig, ax = plt.subplots(
        1, 1,
        figsize=(fs.FULL_WIDTH, 2.8)
    )

    ax.imshow(
        grid,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1.4,
        zorder=3,
        extent=[
            years[0] - 0.5,
            years[-1] + 0.5,
            len(sets) - 0.5,
            -0.5,
        ],
    )

    ax.set_ylabel(f"{len(sets)} ranked ADART IV entries,\none per row")
    ax.set_xlabel("Babylonian year")

    ax.set_ylim(len(sets) - 0.5, -0.5)
    ax.set_yticks([0, len(sets) - 1])
    ax.set_yticklabels(["narrowest", "widest"], fontsize=7.2)

    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ticks = [-599, -499, -399, -299, -199, -99, 0, int(years[-1])]
    ax.set_xticks(ticks)
    ax.set_xticklabels([yr.year_label(t) for t in ticks])
    # Keep the endpoint label inside the figure.
    ax.get_xticklabels()[-1].set_horizontalalignment("right")

    for side in ("right", "top"):
        ax.spines[side].set_visible(False)

    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", which="both", right=False)

    fig.subplots_adjust(
        left=0.128,
        right=0.995,
        top=0.96,
        bottom=0.20,
    )

    return fs.save(fig, OUT / "figS2_adart_iv")


if __name__ == "__main__":
    print(main())
