#!/usr/bin/env python3
"""Date every record twice, with the parser clauses alone and with the reviewed
clauses, and compare the two at the published years (SI Appendix, section S10).

The reviewed clauses are the parser clauses as the model review leaves them,
plus the observations that its search added. They are the clauses that the
dates use. Both runs call the decision of scripts/inference/date_all.py with
nothing changed but the clauses. The published years are loaded only after
both runs have dated every record.

ADART IV holds no published year, so for its undated entries the comparison
counts the entries with a decisive clause and the size of their strict sets.
That part needs the local ADART IV text, and it keeps its last committed
values when the text is absent.

Usage:
    python scripts/audit/compare_clause_sets.py

Output:
    data/benchmark/clause_set_comparison.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.inference.date_all import astronomy_one  # noqa: E402
from tools import cal, clauses, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402

OUT = HERE / "data" / "benchmark" / "clause_set_comparison.json"
ADART_IV = HERE / "data" / "corpus" / "adart_iv_tablets.parquet"
SOURCES = ("parser", "reviewed")


def date_records(tablets: pd.DataFrame, text_column: str, reviewed: dict) -> pd.DataFrame:
    """The decision of each record under each clause set."""
    years = [int(year) for year in cal.candidate_years()]
    rows = []
    for row in tablets.itertuples(index=False):
        text = getattr(row, text_column)
        for source in SOURCES:
            statements = (None if not isinstance(text, str) or not text.strip()
                          else S.extract(text) if source == "parser"
                          else reviewed.get(row.tablet, []))
            result = astronomy_one(row.tablet, text, years, statements=statements)
            levels = json.loads(result["evidence_levels_json"])
            rows.append({
                "source": source, "tablet": row.tablet, "claim": result["claim"],
                "basis": result["basis"],
                "n_clauses": int(result.get("n_statements", 0)),
                "n_decisive": int(result.get("n_decisive_statements", 0)),
                "strict": [int(y) for y in result["strict_years_astro"]],
                "levels": [level["years_astro"] for level in levels],
            })
    return pd.DataFrame(rows)


def median(values) -> float | None:
    values = pd.Series(list(values), dtype=float)
    return float(values.median()) if len(values) else None


def scores(frame: pd.DataFrame) -> dict:
    """The rates of one clause set at the published years."""
    bench = frame[frame.eligible]
    active = bench[bench.year_bearing]
    support = bench[~bench.year_bearing]
    single = active[active.first_size.eq(1)]
    current = active[active.n_decisive.gt(0)]
    outside = frame[~frame.eligible]
    outside_decisive = outside[outside.n_decisive.gt(0)]
    committed = frame[frame.claim.eq("committed")]
    return {
        "clauses": int(frame.n_clauses.sum()),
        "records_with_clauses": int(frame.n_clauses.gt(0).sum()),
        "records_with_decisive_clauses": len(active),
        "first_group": int(active.rank_of_published.eq(1).sum()),
        "first_three_groups": int(active.rank_of_published.le(3).sum()),
        "strict_set": int(active.in_strict.sum()),
        "median_strict_set": median(active.strict_size),
        "single_year_first_groups": len(single),
        "single_year_first_groups_correct": int(single.rank_of_published.eq(1).sum()),
        "with_decisive_clause_now": len(current),
        "median_strict_set_now": median(current.strict_size),
        "first_group_now": int(current.rank_of_published.eq(1).sum()),
        "supporting_only_records": len(support),
        "supporting_only_first_group": int(support.rank_of_published.eq(1).sum()),
        "supporting_only_with_decisive_clause_now": int(support.n_decisive.gt(0).sum()),
        "outside_benchmark": len(outside),
        "outside_benchmark_with_decisive_clause": len(outside_decisive),
        "outside_benchmark_median_strict_set": median(outside_decisive.strict_size),
        "commitments": len(committed),
        "commitments_correct": int((committed.in_strict & committed.strict_size.eq(1)).sum()),
        "commitments_in_benchmark": int(committed.eligible.sum()),
        "commitment_bases": {str(k): int(v) for k, v in committed.basis.value_counts().items()},
    }


def adart_iv(frame: pd.DataFrame) -> dict:
    ranked = frame[frame.claim.eq("ranked")]
    return {
        "entries": len(frame),
        "entries_with_clauses": int(frame.n_clauses.gt(0).sum()),
        "clauses": int(frame.n_clauses.sum()),
        "entries_with_decisive_clause": int(frame.n_decisive.gt(0).sum()),
        "ranked_entries": len(ranked),
        "median_strict_set": median(ranked.strict.map(len)),
    }


def main() -> int:
    # Phase 1: identifiers and texts only.
    texts = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet",
                            columns=["tablet", "text"])
    dated = date_records(texts, "text", clauses.load(clauses.REVIEWED))
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    if ADART_IV.exists():
        entries = pd.read_parquet(ADART_IV, columns=["tablet", "astronomy_text"])
        undated = date_records(entries, "astronomy_text", clauses.load(clauses.ADART_IV_REVIEWED))
        iv = {source: adart_iv(undated[undated.source.eq(source)]) for source in SOURCES}
    else:
        iv = previous["adart_iv"]

    # Phase 2: the frozen benchmark labels and the published years.
    frozen = pd.read_parquet(HERE / "data" / "benchmark" / "benchmark_records.parquet",
                             columns=["tablet", "tool_benchmark_eligible", "tool_year_bearing"])
    published = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet",
                                columns=["tablet", "bce_first"])
    dated = dated.merge(frozen, on="tablet", validate="many_to_one").merge(
        published, on="tablet", validate="many_to_one")
    dated = dated.rename(columns={"tool_benchmark_eligible": "eligible",
                                  "tool_year_bearing": "year_bearing"})
    dated["year_bearing"] = dated.year_bearing & dated.eligible
    truth = dated.bce_first.astype(int).map(yr.bce_to_astro)
    dated["in_strict"] = [t in set(s) for t, s in zip(truth, dated.strict)]
    dated["strict_size"] = dated.strict.map(len)
    dated["first_size"] = dated.levels.map(lambda levels: len(levels[0]) if levels else 701)
    dated["rank_of_published"] = [
        next((rank for rank, years in enumerate(levels, start=1) if t in years), 99)
        for t, levels in zip(truth, dated.levels)]

    parser, reviewed = (dated[dated.source.eq(source)].set_index("tablet") for source in SOURCES)
    bench = parser.eligible & parser.year_bearing
    changed = bench & ((parser.rank_of_published != reviewed.rank_of_published)
                       | (parser.strict_size != reviewed.strict_size))
    summary = {
        "schema": "clause-set-comparison-v1",
        "adart_i_iii": {source: scores(dated[dated.source.eq(source)]) for source in SOURCES},
        "benchmark_records_that_change": {
            "records": int(changed.sum()),
            "strict_set_narrower": int((changed & (reviewed.strict_size < parser.strict_size)).sum()),
            "published_year_ranked_higher": int(
                (changed & (reviewed.rank_of_published < parser.rank_of_published)).sum()),
            "published_year_ranked_lower": int(
                (changed & (reviewed.rank_of_published > parser.rank_of_published)).sum()),
            "published_year_leaves_strict_set": int(
                (bench & parser.in_strict & ~reviewed.in_strict).sum()),
        },
        "adart_iv": iv,
        "inputs": {str(path.relative_to(HERE)): sha256(path)
                   for path in (clauses.REVIEWED, clauses.ADART_IV_REVIEWED,
                                HERE / "tools" / "statements.py",
                                HERE / "scripts" / "inference" / "date_all.py")},
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("adart_i_iii", "benchmark_records_that_change",
                                              "adart_iv")}, indent=2))
    return 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
