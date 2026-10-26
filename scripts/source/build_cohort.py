#!/usr/bin/env python3
"""Group the dated diaries into cohorts by the evidence that their text keeps.

Reads data/corpus/evidence.parquet and writes data/corpus/cohorts.parquet. The
benchmark ledger (scripts/audit/build_benchmark.py) and the sensitivity audit
(SI Appendix, section S7) carry these cohorts as strata. Benchmark membership
does not depend on them.

The cohorts separate three kinds of record. Some state their own era year, so
the year follows from the text without astronomy. Some join several fragments,
and the published date may rest on the join, which the text does not show. In
the rest, the editors dated the tablet from its astronomy.

The cohorts use only the regular-expression marks of classify_evidence.py. They
read no model label and no dating result. Two marks define them, whether the
text has an outer planet or an eclipse, and whether it states an era year.

    testable_checked   outer planet or eclipse, and an era year. The stated
                       year gives a check on the published date.
    testable           outer planet or eclipse, and no era year. The published
                       date is the only reference.
    weak_evidence      other astronomy only, mostly lunar distances and inner
                       planets. These records stay in the tests.
    lookup_only        an era year, and no outer planet or eclipse.
    no_text            none of these marks, mostly records without text.

The dating does not read a stated year in any cohort. It compares the result
with that year only after dating (SI Appendix, section S1). This check is
useful because many published dates were themselves derived from astronomy.

The column joined marks tablets of two or more fragments. They stay in the
tests, so results can be reported with and without them.

Usage:
    python3 scripts/source/build_cohort.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, default=HERE / "data" / "corpus" / "evidence.parquet")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "corpus" / "cohorts.parquet")
    args = ap.parse_args()

    e = pd.read_parquet(args.evidence)

    era = e.era_regex.astype(bool)
    outer = e.outer_regex.astype(bool)
    joined = e.n_joins > 1

    astro = outer | e.eclipse_regex
    e = e.assign(era=era, outer=outer, astro=astro, joined=joined,
                 cohort_source="regex")

    # Any mark of astronomy or of a stated year.
    any_content = (e.lunar_regex | e.inner_regex | e.outer_regex
                   | e.eclipse_regex | era)

    e["cohort"] = "no_text"
    e.loc[any_content, "cohort"] = "weak_evidence"
    e.loc[any_content & era & ~astro, "cohort"] = "lookup_only"
    e.loc[astro & era, "cohort"] = "testable_checked"
    e.loc[astro & ~era, "cohort"] = "testable"

    e.to_parquet(args.out, index=False)
    n = len(e)
    print(f"{n} diaries\n")
    order = ["testable_checked", "testable", "weak_evidence", "lookup_only",
             "no_text"]
    for c in order:
        g = e[e.cohort == c]
        print(f"  {c:<18} {len(g):>3}  ({100*len(g)/n:>2.0f}%)  "
              f"of which joins {int((g.n_joins > 1).sum()):>3}")
    test = e[e.cohort.str.startswith("testable")]
    print(f"\ntestable in total: {len(test)} tablets")
    print("  by volume:", dict(test.project.str[-1].value_counts().sort_index()))
    print(f"  excluding joins  : {int((test.n_joins <= 1).sum())}")
    print(f"  carrying an eclipse: {int(test.eclipse_regex.sum())}")
    print(f"  carrying lunar too : {int(test.lunar_regex.sum())}")

    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
