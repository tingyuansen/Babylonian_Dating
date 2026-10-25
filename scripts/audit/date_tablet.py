#!/usr/bin/env python3
"""Infer astronomy-only candidates, or check text against a supplied year.

    --year N     forward. Given a historical year (BCE by default), does every statement hold?
    (default)    backward. Given no year, which years hold?

Both modes use the record's reviewed clauses, the clauses that the dates use,
or with --parser the parser's clauses alone.

Backward scores every Babylonian year the calendar covers, weighting each
agreement by how few candidate years agree with that statement. It never reads
the published date, and it takes month headings from the tablet's own text
rather than from the catalogue, because the catalogue's reading of which months
survive is part of what an editor supplied.

Reported alongside the result are a strict Pareto set and a safety set. The
strict set contains years not dominated on independent decisive bodies and
discrimination-weighted information. The safety set also retains years
that lead after any one decisive extraction is removed. Their sizes are part of
the result; neither may be hidden behind recall alone.

Usage:
    python3 date_tablet.py X301492
    python3 date_tablet.py X301492 --year 150
    python3 date_tablet.py X301492 --parser
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, clauses, eras, statements as S   # noqa: E402
from tools import years as yr  # noqa: E402


def forward(text: str, sts: list[S.Statement], year_astro: int) -> int:
    months = S.months_in(text)
    verdicts = [S.check(st, year_astro, months) for st in sts]
    res = {result: sum(v.result == result for v in verdicts)
           for result in ("agree", "contradict", "untestable")}
    print(f"months the text itself preserves: {months or 'none found'}")
    print(f"\nchecking {len(sts)} statements against {yr.year_label(year_astro)}\n")
    for v in verdicts:
        mark = {"agree": "agree", "contradict": "CONTRADICT",
                "untestable": "untestable"}[v.result]
        print(f"  {mark:<11} {v.statement.text[:54]:<56} {v.detail}")
    print(f"\nagreed {res['agree']}, contradicted {res['contradict']}, "
          f"untestable {res['untestable']}")
    return 0 if res["contradict"] == 0 else 1


def backward(text: str, sts: list[S.Statement], truth: int | None, top: int,
             weighting: str) -> int:
    months = S.months_in(text)
    if not sts:
        print("no testable statement on this tablet, declining to guess")
        return 1
    print(f"{len(sts)} testable statements, month headings {months or 'none'}")
    for s in sts:
        print(f"  {s.kind:<9} {s.text}")

    years = cal.candidate_years()
    scores = S.candidate_scores(sts, months, years, weighting)
    order = scores.order
    tied = scores.years[scores.top]
    strict = scores.years[scores.strict]
    levels = S.evidence_levels(scores, max_levels=3)

    print(f"\nstatement discrimination ({weighting} weighting)")
    print("  tier kind       agrees equiv    weight  statement")
    for s, agreeing, testable, equivalent, weight, decisive in zip(
            sts, scores.agreeing, scores.testable, scores.equivalent,
            scores.weights, scores.decisive):
        share = f"{agreeing}/{testable}" if testable else "untestable"
        tier = "year" if decisive else "rank"
        print(f"  {tier:<4} {s.kind:<9} {share:>10} {equivalent:5d}  "
              f"{weight:8.3f}  {s.text[:54]}")

    print(f"\nscanned {len(years)} candidate years")
    print(f"{len(strict)} year{'s' if len(strict) != 1 else ''} on the strict "
          "body/information Pareto frontier")
    if weighting == "skill":
        print(f"{len(tied)} year{'s' if len(tied) != 1 else ''} retained after "
              "leave-one-decisive-statement-out checks")
    print("\nfirst three scientific score levels")
    for level in levels:
        if level["predicate_exact"]:
            years_text = ", ".join(
                str(year) for year in level["years_astro"])
            print(f"  evidence group {level['rank']}: "
                  f"{len(level['years_astro'])} year(s) {{{years_text}}}")
        else:
            print(f"  co-ranked level {level['rank']}: "
                  f"{len(level['physical_groups_years_astro'])} distinct "
                  "predicate-equivalent subgroups")
            for subgroup in level["physical_groups_years_astro"]:
                years_text = ", ".join(str(year) for year in subgroup)
                print(f"    {len(subgroup)} year(s) {{{years_text}}}")
    print("\n  year      bodies  primary  support  agreed  contradicted")
    for i in order[:top]:
        year_astro = int(scores.years[i])
        star = "   <-- published" if truth is not None and year_astro == truth else ""
        print(f"  {yr.year_label(year_astro):>8}    {scores.priority[i]:1d}    "
              f"{scores.score[i]:8.3f} "
              f"{scores.support[i]:8.3f}    {scores.agree[i]:2d}       "
              f"{scores.contradict[i]:2d}{star}")

    if truth is not None:
        hit = np.flatnonzero(scores.years == truth)
        if not len(hit):
            print(f"\npublished year {yr.year_label(truth)} is outside the calendar")
            return 1
        i = int(hit[0])
        outranks = ((scores.priority > scores.priority[i])
                    | ((scores.priority == scores.priority[i])
                       & (scores.score > scores.score[i] + 1e-12)))
        pos = int(outranks.sum()) + 1
        print(f"\npublished year {yr.year_label(truth)} explains {scores.priority[i]} bodies, "
              f"scores {scores.score[i]:.3f} primary, "
              f"{scores.support[i]:.3f} support, "
              f"{scores.agree[i]} agreed, {scores.contradict[i]} contradicted, "
              f"ranked {pos} of {len(scores.years)}")
        if len(strict) == 1 and int(strict[0]) == truth:
            print("strictly recovered uniquely")
        elif truth in set(strict):
            print(f"strictly recovered, tied with {len(strict)-1} other year(s)")
        elif truth in set(tied):
            print("retained in the safety set")
        else:
            print("NOT recovered")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tablet")
    ap.add_argument("--year", type=int, help="forward mode historical year number")
    ap.add_argument("--era", choices=["BCE", "AD"], default="BCE",
                    help="era for --year (default: BCE)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--weighting",
                    choices=["count", "information", "inverse", "skill"],
                    default="skill")
    ap.add_argument("--show-chronology", action="store_true",
                    help="display a text-internal dual-era lookup separately")
    ap.add_argument("--show-reference", action="store_true",
                    help="after inference, annotate output with catalogue truth")
    ap.add_argument("--parser", action="store_true",
                    help="use the parser's clauses alone, without the model review")
    args = ap.parse_args()

    # The default inference path loads only identifier and surviving text. A
    # catalogue reference is an explicit post-inference diagnostic option.
    tb = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet", columns=["tablet", "text"])
    r = tb[tb.tablet == args.tablet]
    if r.empty:
        print(f"no tablet {args.tablet}")
        return 2
    r = r.iloc[0]
    if not isinstance(r.text, str) or not r.text.strip():
        print(f"{args.tablet} has no text")
        return 2
    truth = None
    if args.show_reference:
        reference = pd.read_parquet(
            HERE / "data" / "corpus" / "tablets.parquet",
            columns=["tablet", "bce_first"]).set_index("tablet")
        value = reference.at[args.tablet, "bce_first"]
        truth = yr.bce_to_astro(int(value)) if pd.notna(value) else None

    print(args.tablet)
    print("zodiac signs: Britton (2010) Babylonian sidereal zodiac\n")

    if args.show_chronology:
        era_readings = eras.extract(r.text)
        print("chronology annotation (not astronomy-only evidence):")
        if not era_readings:
            print("  no secure absolute chronology statement")
        for reading in era_readings:
            if reading.system == "dual_ae_se":
                label = f"AE {reading.ae_year} = SE {reading.se_year}"
            else:
                label = f"SE {reading.se_year} (named Seleucid ruler)"
            print(f"  {label} = {yr.year_label(reading.astro_year)}")
        print()

    sts = (S.extract(r.text) if args.parser
           else clauses.load(clauses.REVIEWED).get(args.tablet, []))
    if args.year is not None:
        year_astro = (yr.bce_to_astro(args.year)
                      if args.era == "BCE" else int(args.year))
        return forward(r.text, sts, year_astro)
    return backward(r.text, sts, truth, args.top, args.weighting)


if __name__ == "__main__":
    raise SystemExit(main())
