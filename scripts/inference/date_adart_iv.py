#!/usr/bin/env python3
"""Run prospective, answer-hidden dating over the ADART IV undated corpus.

The astronomy pass receives only ``astronomy_text`` produced by
``build_adart_iv_corpus.py``. Museum numbers, page order, editorial dating
discussion, and source-internal ruler/year language are joined or considered
only after the astronomical ordering is frozen. The output therefore reports
candidate support and abstention coverage, not accuracy: the section has no
independent published truth year for most entries.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.inference.date_all import add_chronology, astronomy_one  # noqa: E402
from tools import cal  # noqa: E402
from tools import years as yr  # noqa: E402

SOURCE = HERE / "data" / "corpus" / "adart_iv_tablets.parquet"
OUT = HERE / "data" / "results" / "adart_iv_dates.parquet"
SUMMARY = HERE / "data" / "results" / "adart_iv_summary.json"


def sha256(path: Path) -> str:
    """Return a SHA-256 for provenance attached after inference."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(series: pd.Series) -> dict[str, int]:
    """Stable JSON-ready value counts."""
    return {str(key): int(value)
            for key, value in series.value_counts(dropna=False).sort_index().items()}


def main() -> int:
    # The inference view deliberately omits museum number, pages, translation,
    # chronology, and every possible reference-year field.
    inference = pd.read_parquet(SOURCE, columns=["tablet", "astronomy_text"])
    years = [int(year) for year in cal.candidate_years()]
    rows = []
    for number, row in enumerate(inference.itertuples(index=False), start=1):
        rows.append(astronomy_one(
            row.tablet, row.astronomy_text, years, has_transliteration=True))
        if number % 100 == 0:
            print(f"  astronomy {number}/{len(inference)}")
    result = pd.DataFrame(rows)

    # Only now load source-internal chronology. Editorial Date/Comments blocks
    # were already excluded during corpus construction. This post-pass cannot
    # change an astronomy-only claim, score, candidate tier, or ordering.
    post = pd.read_parquet(
        SOURCE,
        columns=["tablet", "translation", "museum_no", "pdf_page_start",
                 "pdf_page_end", "printed_page_start", "printed_page_end",
                 "n_translation_lines", "chronology_text"])
    translation = post.set_index("tablet").translation.to_dict()
    result = pd.DataFrame([
        add_chronology(record, translation[record["tablet"]])
        for record in result.to_dict("records")
    ])
    # The date lines of the volume stay out of the published results.
    post = post.drop(columns=["translation", "chronology_text"])
    post["n_physical_fragments"] = post.museum_no.fillna("").apply(
        lambda value: len(re.split(r"\s*\+\s*", value)) if value.strip() else 0)
    result = result.merge(post, on="tablet", how="left", validate="one_to_one")

    result["method_version"] = "adart-iv-prospective-v3-local-eclipses"
    provenance = {
        "corpus_sha256": SOURCE,
        "date_adart_iv_sha256": Path(__file__),
        "date_all_sha256": HERE / "scripts" / "inference" / "date_all.py",
        "statements_sha256": HERE / "tools" / "statements.py",
        "skill_sha256": HERE / "skills" / "dating_protocol.md",
        "cal_sha256": HERE / "tools" / "cal.py",
        "sky_sha256": HERE / "tools" / "sky.py",
        "eclipse_tool_sha256": HERE / "tools" / "eclipses.py",
        "eras_sha256": HERE / "tools" / "eras.py",
        "calendar_sha256": HERE / "data" / "physical" / "calendar.parquet",
        "ephemeris_sha256": HERE / "data" / "physical" / "ephemeris.parquet",
        "eclipse_data_sha256": HERE / "data" / "physical" / "eclipses.parquet",
        "normal_stars_sha256": HERE / "data" / "physical" / "normal_stars.parquet",
    }
    for column, path in provenance.items():
        result[column] = sha256(path)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT, index=False)

    active = result[result.n_decisive_statements.gt(0)]
    recognized = result[result.n_statements.gt(0)]
    summary = {
        "schema": "adart-iv-prospective-summary-v1",
        "interpretation": (
            "Prospective candidate support and coverage; no accuracy is implied "
            "without independent reference years"),
        "candidate_babylonian_years_astro": [min(years), max(years)],
        "candidate_babylonian_year_labels": [
            yr.year_label(min(years)), yr.year_label(max(years))],
        "candidate_year_count": len(years),
        "entries": len(result),
        "entries_with_translation": int(result.n_translation_lines.gt(0).sum()),
        "entries_with_recognized_astronomical_facts": len(recognized),
        "entries_with_at_least_three_facts": int(result.n_statements.ge(3).sum()),
        "entries_with_decisive_astronomy": len(active),
        "recognized_facts": int(result.n_statements.sum()),
        "active_decisive_facts": int(result.n_decisive_statements.sum()),
        "astronomy_claims": counts(result.claim),
        "astronomy_bases": counts(result.basis),
        "unique_strict_year": int(result.strict_count.eq(1).sum()),
        "active_singleton_first_physical_group": int(
            active.top_evidence_group_size.eq(1).sum()),
        "all_rows_singleton_first_group_diagnostic": int(
            result.top_evidence_group_size.eq(1).sum()),
        "candidate_range_limit": (
            "Candidate inference spans the complete implemented "
            "Parker--Dubberstein 1956 interval, 626 BCE through AD 75. "
            "The dated diary series begins earlier, so claims do not cover "
            "652--627 BCE."),
        "combined_claims": counts(result.combined_claim),
        "chronology_relations": counts(result.chronology_relation),
        "active_median_strict_set_size": (
            float(active.strict_count.median()) if len(active) else None),
        "active_median_safety_set_size": (
            float(active.safety_count.median()) if len(active) else None),
        "active_median_first_group_size": (
            float(active.top_evidence_group_size.median()) if len(active) else None),
        "active_median_first_three_group_union": (
            float(active.top_three_evidence_groups_size.median())
            if len(active) else None),
        "inputs": {str(SOURCE.relative_to(HERE)): sha256(SOURCE)},
        "outputs": {str(OUT.relative_to(HERE)): sha256(OUT)},
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
