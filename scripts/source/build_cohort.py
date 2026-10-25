#!/usr/bin/env python3
"""Assign reproducible coarse text-property cohorts to dated diaries.

An accuracy figure over the whole corpus answers no useful question, because the
population mixes three incompatible cases. Tablets that state their own era year,
where dating is a lookup and the astronomy contributes nothing. Tablets composed
by joining fragments, where the published date may rest on the physical join and
no text-only method can reach it. And tablets where the editor reasoned from the
astronomy in the surviving text, which is the only case where a comparison means
anything.

The cohorts use deterministic regex properties only. Model labels in
``evidence.parquet`` do not enter them. Nothing here reads success.

The split is a two by two. Whether the surviving text carries evidence capable
of fixing a year, and whether it states a regnal or era year.

    testable_checked   strong astronomy, era year also stated. Testable, and the
                       stated year gives an independent check on the answer. The
                       era statement is WITHHELD from the agent.
    testable           strong astronomy, no era year stated. Testable, with the
                       published date as the only reference.
    weak_evidence      astronomical content but no outer planet and no eclipse,
                       typically lunar intervals and inner planets. Probably
                       hard, possibly not impossible, and NOT excluded. A run
                       recorded lunar intervals dating tablets on occasion, so
                       these are tested and only reclassified if the diagnostic
                       shows the text cannot support the published date.
    lookup_only        era year stated but no usable astronomy.
    no_text            nothing readable survives. The only honest exclusion.

The era statement is never given to the agent in either testable cohort. On the
first it exists only as a second opinion on the published date, which is worth
having because the published dates are themselves astronomical in origin.

Joins are flagged rather than excluded. The edition's introduction confirms that
joining fragments was part of how the corpus was dated, so a published date may
rest on a join that no text-only method can reach. Report performance with and
without them rather than deciding in advance.

Usage:
    python3 build_cohort.py
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

    # Any astronomical or chronological content at all, however weak.
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
