#!/usr/bin/env python3
"""Rebuild the final clauses of one run of the language-model review from its caches.

The review is step 1 of the paper (SI Appendix, section S2). The script reads the
review and search caches in the run directory and rebuilds the run's final
clause list. The list holds the parser clauses as reviewed and the clauses
that the search added. The model client is replaced by one that raises an
error, so a cache miss stops the rebuild before any model call. The dates use
these final clauses, and tools/clauses.py reads them back.

    python scripts/agent/rescore_model_review.py --model gpt-6-luna-2026-09-22
    python scripts/agent/rescore_model_review.py --scope adart-iv \
        --out-dir data/clauses/model_review_gpt-6-luna-2026-09-22_adart_iv

For ADART I to III, the inputs are in the run directory (default
data/clauses/model_review_<model>). The rebuild writes the search audit
search_proposals.csv there again, and its other outputs are

    data/clauses/reviewed_clauses.parquet   the final clauses, one row per
                                            clause, with their origin
    reviewed_clause_bits.parquet            per clause: tier, agreeing and
                                            testable years, and bits
    reviewed_clause_verdicts.parquet        per clause: verdict at the
                                            published year

The last two are in the run directory. scripts/audit/build_paper_counts.py
reads them for SI Appendix, Tables S3 and S4. For ADART IV, the output is
data/clauses/adart_iv_reviewed_clauses.parquet. Its run directory is not in the
repository, because its caches quote the copyrighted volume.
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
from tools import cal, clauses  # noqa: E402

S = review.S
WEIGHTING = "skill"


class APICallBlocked(RuntimeError):
    """A model call was needed, so the caches do not cover the run."""


def _no_api_client(*_args, **_kwargs):
    raise APICallBlocked(
        "A model call would be required, which means the run's caches are "
        "missing or do not match this code, model and prompt version. "
        "The cache-only rebuild will not make model calls."
    )


review.get_client = _no_api_client


def check_canonical_signs(combined: dict[str, list[Any]]) -> None:
    """Raise an error for a sign in a form other than the review's canonical one.

    A clause of a kind without a sign must carry no sign.
    """
    for tablet, sts in combined.items():
        for st in sts:
            sign = st.sign or ""
            if st.kind in review.SIGN_KINDS:
                if review._canonical_sign_name(sign) != sign:
                    raise RuntimeError(f"{tablet}: non-canonical sign {sign!r} on {st.kind}")
            elif sign:
                raise RuntimeError(f"{tablet}: {st.kind} carries sign {sign!r}")


def clause_provenance(extracted, records, reviewed, additions, combined):
    """Origin, parser index and review action of each final clause, in its order.

    The order is that of merge_recall_additions. A record without additions
    keeps the order of its reviewed parser clauses. In a record with additions,
    all clauses are sorted by their passage.
    """
    by_index = {(r["tablet"], int(r["statement_index"])): r for r in records}
    out = {}
    for tablet, final in combined.items():
        text, parsed = extracted[tablet]
        kept = []
        for index, st in enumerate(parsed):
            record = by_index[(tablet, index)]
            if review.record_to_statement(st, record, text) is not None:
                kept.append((index, record["action"].lower()))
        items = [(st, ("parser", index, action))
                 for st, (index, action) in zip(reviewed.get(tablet, []), kept, strict=True)]
        if tablet in additions:
            items += [(st, ("search", -1, "add")) for st in additions[tablet]]
            items.sort(key=lambda item: (item[0].source_start, item[0].source_end,
                                         item[0].kind, item[0].planet))
        if [(st.source_start, st.source_end, st.kind, st.planet, st.text) for st, _ in items] != \
                [(st.source_start, st.source_end, st.kind, st.planet, st.text) for st in final]:
            raise RuntimeError(f"{tablet}: provenance does not follow the merged clauses")
        out[tablet] = [origin for _, origin in items]
    return out


def reconstruct_final_statements(model: str, out_dir: Path, scope: str):
    """The run's final clause list and its provenance, from its caches only."""
    tablets = review.load_tablets(scope)
    extracted = review.extract_all(tablets)
    total = sum(len(sts) for _, sts in extracted.values())
    print(f"parser: {total:,} clauses from {len(tablets):,} records")

    reviewed, records, dropped = review.review_all(
        extracted=extracted, model=model, cache_path=out_dir / "review_cache.jsonl",
        workers=1, force=False, pilot=None)
    additions, recall_rows = review.run_raw_recall(
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
    provenance = clause_provenance(extracted, records, reviewed, additions, combined)
    # The search audit, written again so that each row gives the final clause
    # that code makes from the cached responses.
    review.write_recall_audit(recall_rows, out_dir / "search_proposals.csv")
    return tablets, combined, provenance


def write_clauses(combined, provenance, path: Path) -> None:
    """The final clauses as one table, checked to read back unchanged."""
    rows = [row for tablet in sorted(combined)
            for row in clauses.to_rows(tablet, combined[tablet], provenance[tablet])]
    pd.DataFrame(rows).to_parquet(path, index=False)
    back = clauses.load(path)
    if any(back.get(tablet, []) != sts for tablet, sts in combined.items()):
        raise RuntimeError(f"{path} does not read back to the final clauses")
    print(f"{len(rows):,} clauses -> {path.relative_to(clauses.HERE)}")


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

    tablets, combined, provenance = reconstruct_final_statements(args.model, out_dir, args.scope)
    if args.scope == "adart-iv":
        write_clauses(combined, provenance, clauses.ADART_IV_REVIEWED)
        return 0
    write_clauses(combined, provenance, clauses.REVIEWED)
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
