#!/usr/bin/env python3
"""Score the reviewed clauses of one run of the model review, from its caches alone.

The run's final clause list is rebuilt from the review and search caches in its
run directory. It holds the parser clauses as reviewed, plus the search's
additions. The model client is replaced by one that raises, so a cache miss
stops the rebuild and spends no money. Each record with a final clause is then
scored over the 701 candidate years, and each clause gets its verdict at the
published year.

    python scripts/agent/rescore_model_review.py --model gpt-6-luna-2026-09-22

Inputs and outputs are in the run directory (default
data/clauses/model_review_<model>).
The two outputs are the input of scripts/paper/build_paper_counts.py:

    reviewed_clause_bits.parquet       per clause: tier, agreeing and testable
                                       years, and bits
    reviewed_clause_verdicts.parquet   per clause: verdict at the published year
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_review as review  # noqa: E402
from tools import cal  # noqa: E402

S = review.S
WEIGHTING = "skill"


class APICallBlocked(RuntimeError):
    pass


def _no_api_client(*_args, **_kwargs):
    raise APICallBlocked(
        "A model call would be required, which means the run's caches are "
        "missing or do not match this code, model and prompt version. "
        "The cache-only rebuild will not make model calls."
    )


review.get_client = _no_api_client


def check_canonical_signs(combined: dict[str, list[Any]]) -> None:
    """The review canonicalizes signs. Confirm that nothing slipped past it."""
    for tablet, sts in combined.items():
        for st in sts:
            sign = st.sign or ""
            if st.kind in review.SIGN_KINDS:
                if review._canonical_sign_name(sign) != sign:
                    raise RuntimeError(f"{tablet}: non-canonical sign {sign!r} on {st.kind}")
            elif sign:
                raise RuntimeError(f"{tablet}: {st.kind} carries sign {sign!r}")


def reconstruct_final_statements(model: str, out_dir: Path, scope: str):
    """The run's final clause list, from its caches only."""
    tablets = review.load_tablets(scope)
    extracted = review.extract_all(tablets)
    total = sum(len(sts) for _, sts in extracted.values())
    print(f"parser: {total:,} clauses from {len(tablets):,} records")

    reviewed, _records, dropped = review.review_all(
        extracted=extracted, model=model, cache_path=out_dir / "review_cache.jsonl",
        workers=1, force=False, pilot=None)
    additions, _recall_rows = review.run_raw_recall(
        tablets=tablets, base_statements=reviewed, model=model,
        cache_path=out_dir / "recall_cache.jsonl", workers=1, force=False,
        core_chars=2400, context_chars=700, overlap_chars=350,
        dropped_statements=dropped)
    combined = review.merge_recall_additions(
        reviewed, additions, dict(zip(tablets.tablet, tablets.text)))

    before = sum(len(v) for v in reviewed.values())
    added = sum(len(v) for v in additions.values())
    after = sum(len(v) for v in combined.values())
    if after != before + added:
        raise RuntimeError(f"merge lost clauses: {before} + {added} != {after}")
    print(f"final clauses: {before:,} reviewed parser clauses + {added:,} from "
          f"the search = {after:,}")
    check_canonical_signs(combined)
    return tablets, combined


def clause_bits(tablets: pd.DataFrame, combined: dict[str, list[Any]]) -> pd.DataFrame:
    """Score every record with a final clause over the candidate years."""
    text_by_tablet = dict(zip(tablets.tablet.astype(str), tablets.text))
    years = [int(y) for y in cal.candidate_years()]
    selected = sorted(t for t, sts in combined.items() if sts)
    rows: list[dict[str, Any]] = []
    for number, tablet in enumerate(selected, start=1):
        text = text_by_tablet.get(tablet)
        if not isinstance(text, str):
            raise RuntimeError(f"{tablet}: missing source text")
        sts = combined[tablet]
        scores = S.candidate_scores(sts, S.months_in(text), years, WEIGHTING)
        for index, st in enumerate(sts):
            rows.append({
                "tablet": tablet,
                "statement_index": index,
                "kind": st.kind,
                "planet": st.planet,
                "target_planet": st.target_planet,
                "parser_rule": st.parser_rule,
                "restored": bool(st.restored),
                "omitted": bool(st.omitted),
                "decisive_tier": bool(scores.decisive[index]),
                "agreeing_years": int(scores.agreeing[index]),
                "testable_years": int(scores.testable[index]),
                "information_bits": float(scores.weights[index]),
                "equivalent_predicates": int(scores.equivalent[index]),
                "candidate_first_astro": int(years[0]),
                "candidate_last_astro": int(years[-1]),
                "candidate_year_count": len(years),
                "scoring_fingerprint": review.scoring_fingerprint(),
                "statement_fingerprint": review.statement_fingerprint(st),
                "scoring_unit": int(scores.scoring_units[index]),
                "joint_positions_json": json.dumps(st.joint_positions),
            })
        if number % 50 == 0 or number == len(selected):
            print(f"  scored records: {number:,}/{len(selected):,}")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=review.DEFAULT_MODEL)
    parser.add_argument("--scope", choices=sorted(review.SCOPES), default="adart-i-iii")
    parser.add_argument("--out-dir", type=Path,
                        help="run directory; default data/clauses/model_review_<model>")
    args = parser.parse_args(argv)
    out_dir = (args.out_dir or review.default_out_dir(args.model)).resolve()
    print(f"run directory: {out_dir} (cache only, no model calls)")

    tablets, combined = reconstruct_final_statements(args.model, out_dir, args.scope)
    bits = clause_bits(tablets, combined)
    _, verdicts = review.build_table5(tablets, combined)
    expected = sum(len(v) for v in combined.values())
    if len(bits) != expected or len(verdicts) != expected:
        raise RuntimeError(f"{len(bits):,} and {len(verdicts):,} rows for {expected:,} clauses")

    bits.to_parquet(out_dir / "reviewed_clause_bits.parquet", index=False)
    verdicts.to_parquet(out_dir / "reviewed_clause_verdicts.parquet", index=False)
    print(f"{expected:,} clauses -> reviewed_clause_bits.parquet, "
          "reviewed_clause_verdicts.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
