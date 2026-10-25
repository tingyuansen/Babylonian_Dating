#!/usr/bin/env python3
"""Explain why a leading candidate outranks a tablet's published year.

This is a diagnostic, never an inference path: it reads the published year only
after ranking has finished, then runs the same statements forward at the truth
and at a selected leading alternative.  The side-by-side verdicts identify
which reading or tool decision changed the rank so it can be worked through
the ten causes in Dating Skill section 3b.

Usage:
    python3 scripts/diagnose_backward.py X103725
    python3 scripts/diagnose_backward.py X103725 --candidate 616
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tablet")
    ap.add_argument("--candidate", type=int,
                    help="candidate to compare; defaults to the ranked leader")
    ap.add_argument("--weighting",
                    choices=["count", "information", "inverse", "skill"],
                    default="skill")
    ap.add_argument("--all", action="store_true",
                    help="show statements with identical verdicts too")
    args = ap.parse_args()

    tb = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet")
    hit = tb[tb.tablet == args.tablet]
    if hit.empty:
        print(f"no tablet {args.tablet}")
        return 2
    row = hit.iloc[0]
    if pd.isna(row.bce_first) or not isinstance(row.text, str):
        print(f"{args.tablet} lacks a published year or readable text")
        return 2

    truth = yr.bce_to_astro(int(row.bce_first))
    sts = S.extract(row.text)
    months = S.months_in(row.text)
    years = cal.candidate_years()
    scores = S.candidate_scores(sts, months, years,
                                args.weighting)
    truth_i = int(np.flatnonzero(years == truth)[0])
    leader_i = int(scores.order[0])
    if args.candidate is not None:
        candidate_astro = yr.bce_to_astro(args.candidate)
        chosen = np.flatnonzero(years == candidate_astro)
        if not len(chosen):
            print(f"candidate {args.candidate} BCE is outside the calendar")
            return 2
        leader_i = int(chosen[0])
    leader = int(years[leader_i])
    outranks = ((scores.priority > scores.priority[truth_i])
                | ((scores.priority == scores.priority[truth_i])
                   & (scores.score > scores.score[truth_i] + 1e-12)))
    rank = int(outranks.sum()) + 1

    print(f"{args.tablet}  published {yr.year_label(truth)}; "
          f"compare {yr.year_label(leader)}")
    print(f"{args.weighting} score: published {scores.priority[truth_i]} bodies, "
          f"{scores.score[truth_i]:.3f} "
          f"+ {scores.support[truth_i]:.3f} support, candidate "
          f"{scores.priority[leader_i]} bodies, {scores.score[leader_i]:.3f} "
          f"+ {scores.support[leader_i]:.3f} "
          f"support; published rank {rank}")
    print(f"month headings from text: {months or 'none'}")
    print("\nstatement-by-statement")
    print("  kind/body          agrees equiv weight  published verdict   candidate verdict")
    for st, agreeing, testable, equivalent, weight in zip(
            sts, scores.agreeing, scores.testable, scores.equivalent,
            scores.weights):
        true_v = S.check(st, truth, months)
        lead_v = S.check(st, leader, months)
        if not args.all and true_v.result == lead_v.result:
            continue
        share = f"{agreeing}/{testable}" if testable else "untestable"
        print(f"\n  {st.kind}/{st.planet:<8} {share:>8} {equivalent:5d} "
              f"{weight:6.3f}  "
              f"{true_v.result:<10}          {lead_v.result:<10}")
        print(f"    {st.text}")
        print(f"    published: {true_v.detail}")
        print(f"    candidate: {lead_v.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
