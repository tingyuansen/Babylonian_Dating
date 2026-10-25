#!/usr/bin/env python3
"""Write the ranked candidate years of ADART IV, as a table file and as SI Table S7.

One row per entry that keeps a decisive clause, in the order of Fig. S2: the
narrowest strict set first, and equal sizes by their years. Everything comes
from the committed results. The clocks are the families of each entry's
decisive clauses that carry bits, and a dagger marks an entry whose decisive
clause has a month. The others are tested over the whole year.

data/results/adart_iv_ranked.csv holds every year of each set. The LaTeX rows
of Table S7 list a group year by year when it holds at most ten years, and
give its size otherwise. They are written only when paper/ is present.

Usage:
    python scripts/paper/build_adart_iv_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

RESULTS = HERE / "data" / "results" / "adart_iv_dates.parquet"
CSV = HERE / "data" / "results" / "adart_iv_ranked.csv"
OUT = HERE / "paper" / "tableS7_adart_iv.tex"
# A group of more than ten years repeats the same few clock cycles, so the table
# gives its size, and the results file keeps every year.
LISTED = 10
CLOCKS = {"mars": "Mars", "jupiter": "Jupiter", "saturn": "Saturn",
          "lunar_eclipse": "lunar eclipse", "seasonal_sirius": "solar year"}


def runs(years: list[int]) -> str:
    """Years as runs of consecutive values, so a group of 100 stays short."""
    parts, start = [], None
    for i, year in enumerate(sorted(years)):
        if start is None:
            start = previous = year
        elif year == previous + 1:
            previous = year
            continue
        else:
            parts.append((start, previous))
            start = previous = year
        if i == len(years) - 1:
            parts.append((start, previous))
    if start is not None and (not parts or parts[-1][1] != previous):
        parts.append((start, previous))
    return ",\\,".join(f"${a}$" if a == b else f"${a}$--${b}$" for a, b in parts)


def listing(years: list[int]) -> str:
    """A group's years when it holds at most LISTED of them, else its size."""
    return runs(years) if len(years) <= LISTED else f"{len(years)} years"


def active_decisive(row) -> list[dict]:
    """The decisive clauses of an entry that carry bits."""
    return [st for st in json.loads(row.statement_verdicts_json)
            if st["decisive"] and st["weight_bits"] > 0]


def clock_names(row) -> list[str]:
    names = {family for st in active_decisive(row) for family in st["evidence_families"]}
    return [CLOCKS[name] for name in CLOCKS if name in names]


def has_month(row) -> bool:
    return any(st["month"] for st in active_decisive(row))


def main() -> int:
    results = pd.read_parquet(RESULTS)
    ranked = results[results.claim.eq("ranked")].copy()
    ranked["strict"] = ranked.strict_years_astro.map(lambda v: sorted(int(y) for y in v))
    ranked = ranked.iloc[sorted(range(len(ranked)), key=lambda i: (
        len(ranked.strict.iloc[i]), ranked.strict.iloc[i]))]
    rows, lines = [], []
    for row in ranked.itertuples():
        levels = json.loads(row.evidence_levels_json)[:3]
        entry = int(row.tablet.split("_")[3])
        names, dated = clock_names(row), has_month(row)
        rows.append({
            "entry": entry, "tablet": row.tablet, "printed_page": int(row.printed_page_start),
            "museum_no": row.museum_no, "clocks": ", ".join(names), "month_known": dated,
            "strict_set_size": int(row.strict_count), "first_group_size": int(row.top_evidence_group_size),
            "strict_set_years_astro": " ".join(map(str, row.strict)),
            **{f"group_{level['rank']}_years_astro": " ".join(map(str, sorted(level["years_astro"])))
               for level in levels},
        })
        groups = ",\\ ".join(f"$L_{level['rank']}$: {listing(level['years_astro'])}"
                             for level in levels)
        lines.append(" & ".join([
            str(entry), str(row.printed_page_start), row.museum_no,
            ", ".join(names) + ("$^\\dagger$" if dated else ""),
            str(row.strict_count), str(row.top_evidence_group_size), groups,
        ]) + r" \\")
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"{len(rows)} entries -> {CSV.relative_to(HERE)}")
    if OUT.parent.is_dir():
        OUT.write_text("% Written by scripts/paper/build_adart_iv_table.py; do not edit.\n"
                       + "\n".join(lines) + "\n", encoding="utf-8")
        print(f"{len(lines)} entries -> {OUT.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
