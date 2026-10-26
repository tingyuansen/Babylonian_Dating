#!/usr/bin/env python3
"""Write all 701 candidate years of each benchmark record, with their scores and rank.

Phase 1 scores the candidate years of each record from its reviewed clauses,
before any published year is read. Phase 2 then adds the published year. Fig.
4a uses the output to find how far down the ranking the published year lies.

Output:
    data/benchmark/ranking_candidates.parquet   one row per record and candidate year

Usage:
    python3 scripts/audit/rank_candidates.py [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, clauses, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402


# The entries of the ranking r(y) of Eq. 5 in the main text, in order. Years
# that tie on all of them form one group. The year itself is not a key.
EVIDENCE_KEY_COLUMNS = [
    "tier", "decisive_families", "primary_bits", "support_bits",
    "contradict",
]


def infer_candidates() -> pd.DataFrame:
    """Score all candidate years of each benchmark record, before any published year is read."""
    texts = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet", columns=["tablet", "text"])
    frozen = pd.read_parquet(
        HERE / "data" / "benchmark" / "benchmark_records.parquet",
        columns=["tablet", "tool_benchmark_eligible"])
    eligible = set(frozen.loc[frozen.tool_benchmark_eligible, "tablet"])
    # A record has decisive clauses when its result keeps at least one.
    decisive = (pd.read_parquet(HERE / "data" / "results" / "tablet_dates.parquet",
                                columns=["tablet", "n_decisive_statements"])
                .set_index("tablet").n_decisive_statements.fillna(0).gt(0).to_dict())
    years = cal.candidate_years()
    reviewed = clauses.load(clauses.REVIEWED)
    rows: list[dict] = []
    for tablet in texts.itertuples(index=False):
        if tablet.tablet not in eligible or not isinstance(tablet.text, str):
            continue
        statements = reviewed.get(tablet.tablet, [])
        usable = [st for st in statements
                  if not (st.kind == "eclipse" and st.omitted)]
        if len(usable) < 3:
            continue
        scores = S.candidate_scores(
            statements, S.months_in(tablet.text), years, weighting="skill")
        # Supporting-only records stay. Their I_D is zero in all years, and
        # their I_S still orders the years. The benchmark scores them as a
        # separate group.
        active = int((scores.decisive & (scores.weights > 0)).sum())
        order_position = np.empty(len(years), dtype=int)
        order_position[scores.order] = np.arange(1, len(years) + 1)
        # The column tier is 0 in the strict set, 1 in the rest of the safety
        # set and 2 elsewhere. It is t(y) of Eq. 5, counted the other way. The
        # columns decisive_families, primary_bits and support_bits hold K, I_D
        # and I_S.
        tier = np.where(scores.strict, 0, np.where(scores.safety, 1, 2))
        for i, year in enumerate(years):
            rows.append({
                "tablet": tablet.tablet,
                "candidate_year_astro": int(year),
                "baseline_rank": int(order_position[i]),
                "tier": int(tier[i]),
                "strict": bool(scores.strict[i]),
                "safety": bool(scores.safety[i]),
                "strict_count": int(scores.strict.sum()),
                "safety_count": int(scores.safety.sum()),
                "n_statements": len(statements),
                "active_decisive": active,
                "decisive": bool(decisive[tablet.tablet]),
                "decisive_families": int(scores.priority[i]),
                "primary_bits": float(scores.score[i]),
                "support_bits": float(scores.support[i]),
                "agree": int(scores.agree[i]),
                "contradict": int(scores.contradict[i]),
                "untestable": int(scores.untestable[i]),
                # A hash of the verdicts of all clauses in this year. Two years
                # with equal hashes have the same verdicts, so no score can
                # separate them.
                "verdict_signature_sha256": hashlib.sha256(
                    scores.outcomes[i].tobytes()).hexdigest(),
            })
    return pd.DataFrame(rows)


def add_reference(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add the published year of each record, after all candidate years are scored."""
    reference = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "bce_first"])
    reference = reference[reference.bce_first.notna()].copy()
    reference["truth"] = reference.bce_first.astype(int).map(yr.bce_to_astro)
    result = candidates.merge(
        reference[["tablet", "truth"]], on="tablet", validate="many_to_one")
    result["correct"] = result.candidate_year_astro.eq(result.truth)
    return result


def evidence_levels(group: pd.DataFrame) -> pd.DataFrame:
    """Number the groups of tied years of one record, from 1 at the top of the ranking."""
    keys = group[EVIDENCE_KEY_COLUMNS].drop_duplicates().sort_values(
        EVIDENCE_KEY_COLUMNS,
        ascending=[True, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    keys["evidence_group_rank"] = np.arange(1, len(keys) + 1)
    return group.merge(keys, on=EVIDENCE_KEY_COLUMNS, validate="many_to_one")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path,
        default=HERE / "data" / "benchmark" / "ranking_candidates.parquet")
    args = parser.parse_args()
    print("phase 1/2: astronomy-only candidate scores", flush=True)
    candidates = infer_candidates()
    print(f"  {candidates.tablet.nunique()} records; {len(candidates)} candidate rows",
          flush=True)
    print("phase 2/2: attach the published years", flush=True)
    candidates = add_reference(candidates)
    candidates.to_parquet(args.out, index=False)
    print(f"candidates -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
