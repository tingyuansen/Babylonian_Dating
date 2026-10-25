#!/usr/bin/env python3
"""Score the frozen test on the dated addenda of ADART IV.

The test is registered in data/benchmark/addenda_registration.json. This
script reads the candidate sets that date_adart_iv_addenda.py wrote, and only
then the published years that build_adart_iv_addenda.py wrote. It computes the
scores that the registration names, for the texts that ADART IV does not link
to a text of ADART I to III, and in a separate row for the texts that it
links. It changes no candidate set.

A text counts as included when its first group, or its strict set, holds the
published year, or one of the two years of a heading such as No. -212/1. The
benchmark rates beside them are those of the 112 benchmark records with
decisive clauses, from data/paper_counts/counts.json.

Output:
    data/benchmark/addenda_results.json
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.source.build_adart_iv_corpus import CHRONOLOGY  # noqa: E402

REGISTRATION = HERE / "data" / "benchmark" / "addenda_registration.json"
YEARS = HERE / "data" / "benchmark" / "addenda_published_years.parquet"
DATES = HERE / "data" / "results" / "adart_iv_addenda_dates.parquet"
DATES_SUMMARY = HERE / "data" / "results" / "adart_iv_addenda_summary.json"
CORPUS = HERE / "data" / "corpus" / "adart_iv_addenda_tablets.parquet"
CORPUS_SUMMARY = HERE / "data" / "corpus" / "adart_iv_addenda_summary.json"
COUNTS = HERE / "data" / "paper_counts" / "counts.json"
OUT = HERE / "data" / "benchmark" / "addenda_results.json"
CODE = [
    "scripts/source/build_adart_iv_addenda.py",
    "scripts/inference/date_adart_iv_addenda.py",
    "scripts/audit/score_adart_iv_addenda.py",
    "scripts/source/build_adart_iv_corpus.py",
    "scripts/inference/date_adart_iv.py",
    "scripts/inference/date_all.py",
    "tools/statements.py", "tools/cal.py", "tools/sky.py", "tools/eclipses.py",
    "tools/eras.py", "tools/years.py",
    "skills/dating_protocol.md",
]
# What the registration freezes: the rules, the tools, the corpus filters and
# the physical tables.
FROZEN_PATHS = ["tools", "skills", "scripts/inference/date_all.py",
                "scripts/inference/date_adart_iv.py",
                "scripts/source/build_adart_iv_corpus.py", "data/physical",
                "data/source"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(part: int, whole: int) -> dict:
    return {"count": int(part), "of": int(whole),
            "percent": round(100.0 * part / whole, 1) if whole else None}


def median(series: pd.Series):
    return float(series.median()) if len(series) else None


def frozen_files_match(commit: str) -> bool | None:
    """Whether the frozen files equal those of the registered commit."""
    try:
        done = subprocess.run(["git", "diff", "--quiet", commit, "--", *FROZEN_PATHS],
                              cwd=HERE, capture_output=True)
    except OSError:
        return None
    return {0: True, 1: False}.get(done.returncode)


def scores(part: pd.DataFrame) -> dict:
    """The registered scores of one group of texts."""
    translated = part[part.n_translation_lines.gt(0)]
    active = translated[translated.n_decisive_statements.gt(0)]
    committed = translated[translated.claim.eq("committed")]
    declined = translated[translated.claim.eq("decline")]
    return {
        "texts": len(part),
        "diary_numbers": int(part.entry_no.nunique()),
        "texts_with_translation": len(translated),
        "texts_with_a_decisive_clause": len(active),
        "published_year_in_first_group": rate(active.in_first_group.sum(), len(active)),
        "published_year_in_strict_set": rate(active.in_strict_set.sum(), len(active)),
        "median_first_group_size": median(active.top_evidence_group_size),
        "median_strict_set_size": median(active.strict_count),
        "commitments": len(committed),
        "commitments_matching_the_published_year": int(committed.in_strict_set.sum()),
        "ranked": int(translated.claim.eq("ranked").sum()),
        "declined": len(declined),
        "declined_by_basis": {str(key): int(value) for key, value
                              in declined.basis.value_counts().sort_index().items()},
    }


def main() -> int:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    # The candidate sets are read first, and are complete before any
    # published year is read.
    dates = pd.read_parquet(DATES)
    years = pd.read_parquet(YEARS)
    if set(dates.tablet) != set(years.tablet) or dates.tablet.duplicated().any():
        raise RuntimeError("the dated texts differ from the texts with a published year")
    table = dates.merge(years.drop(columns=["museum_no"]), on="tablet",
                        validate="one_to_one")
    published = table.published_years_astro.apply(lambda value: {int(y) for y in value})
    table["in_first_group"] = [bool(p & {int(y) for y in group}) for p, group
                               in zip(published, table.top_evidence_group_years_astro)]
    table["in_strict_set"] = [bool(p & {int(y) for y in strict}) for p, strict
                              in zip(published, table.strict_years_astro)]
    table = table.sort_values(["pdf_page_start", "tablet"]).reset_index(drop=True)

    new = table[~table.overlaps_adart_i_iii]
    linked = table[table.overlaps_adart_i_iii]
    decisive = []
    for row in table[table.n_decisive_statements.gt(0)].itertuples():
        decisive.append({
            "tablet": row.tablet,
            "diary_number": row.entry_no,
            "letter": row.letter,
            "museum_no": row.museum_no,
            "published_years_astro": [int(y) for y in row.published_years_astro],
            "linked_to_adart_i_iii": bool(row.overlaps_adart_i_iii),
            "decisive_clocks": int(row.decisive_families),
            "decisive_statements": int(row.n_decisive_statements),
            "strict_set_size": int(row.strict_count),
            "first_group_size": int(row.top_evidence_group_size),
            "first_group_years_astro": ([int(y) for y in row.top_evidence_group_years_astro]
                                        if row.top_evidence_group_size <= 10 else None),
            "published_year_in_first_group": bool(row.in_first_group),
            "published_year_in_strict_set": bool(row.in_strict_set),
            "claim": row.claim,
        })
    links = [{
        "tablet": row.tablet,
        "diary_number": row.entry_no,
        "letter": row.letter,
        "museum_no": row.museum_no,
        "adsd_museum_matches": list(row.adsd_museum_matches),
        "corrigenda_evidence": list(row.corrigenda_evidence),
        "note_evidence": list(row.note_evidence),
    } for row in linked.itertuples()]

    # Checks of the blind input. A year, king or ruler whose word a bracket
    # splits, such as [Y]ear, is not matched by the corpus filter. Each such
    # passage is listed with the clauses that the parser read from it.
    corpus = pd.read_parquet(CORPUS, columns=["tablet", "astronomy_text"])
    split = []
    for row in corpus.itertuples():
        plain, where = [], []
        for index, char in enumerate(row.astronomy_text):
            if char not in "[]":
                plain.append(char)
                where.append(index)
        plain = "".join(plain)
        if CHRONOLOGY.search(row.astronomy_text):
            raise RuntimeError(f"{row.tablet}: chronology in the astronomy text")
        verdicts = json.loads(table.set_index("tablet").statement_verdicts_json[row.tablet])
        for match in CHRONOLOGY.finditer(plain):
            start, end = where[match.start()], where[match.end() - 1] + 1
            split.append({
                "tablet": row.tablet,
                "passage": row.astronomy_text[max(0, start - 10):end + 10],
                "clauses_read_from_it": sum(
                    1 for verdict in verdicts
                    if verdict["source_start"] < end and verdict["source_end"] > start),
            })

    benchmark = json.loads(COUNTS.read_text(encoding="utf-8"))[
        "existing_benchmark"]["frozen_year_bearing"]
    corpus_summary = json.loads(CORPUS_SUMMARY.read_text(encoding="utf-8"))
    numbers = years.drop_duplicates("entry_no").entry_no.tolist()
    result = {
        "schema": "adart-iv-addenda-frozen-test-results-v1",
        "registration": str(REGISTRATION.relative_to(HERE)),
        "frozen_commit": registration["frozen_commit"],
        "frozen_files_match_commit": frozen_files_match(registration["frozen_commit"]),
        "segmentation": {
            "pdf_pages": corpus_summary["pdf_pages"],
            "printed_pages": corpus_summary["printed_pages"],
            "diary_number_headings": int(years.starts_new_number.sum()),
            "diary_numbers": len(numbers),
            "first_and_last_number": [numbers[0], numbers[-1]],
            "texts": len(years),
            "texts_with_translation": corpus_summary["texts_with_translation"],
            "texts_without_a_number_heading": [
                f"{row.entry_no} {row.letter}: {row.museum_no}"
                for row in years[~years.starts_new_number].itertuples()],
            "numbers_with_several_texts": {
                number: group.letter.tolist()
                for number, group in years.groupby("entry_no", sort=False)
                if len(group) > 1},
            "numbers_with_two_years": years.loc[
                years.published_years_astro.apply(len).gt(1), "entry_no"].tolist(),
        },
        "input_checks": {
            "texts_with_the_published_year_number_in_input": years.loc[
                years.year_number_in_input.apply(len).gt(0), "tablet"].tolist(),
            "bracket_split_chronology_words": split,
        },
        "scores": {
            "not_linked_to_adart_i_iii": scores(new),
            "linked_to_adart_i_iii": scores(linked),
            "all": scores(table),
        },
        "benchmark_for_comparison": {
            "records": int(benchmark["records"]),
            "published_year_in_first_group": rate(
                benchmark["truth_in_top_evidence_group"]["count"], benchmark["records"]),
            "published_year_in_strict_set": rate(
                benchmark["truth_in_strict"]["count"], benchmark["records"]),
            "median_first_group_size": benchmark["median_top_evidence_group_size"],
            "median_strict_set_size": benchmark["median_strict_count"],
        },
        "texts_with_a_decisive_clause": decisive,
        "texts_linked_to_adart_i_iii": links,
        "inputs": {str(path.relative_to(HERE)): sha256(path) for path in
                   (REGISTRATION, YEARS, DATES, DATES_SUMMARY, CORPUS,
                    CORPUS_SUMMARY, COUNTS)},
        "code": {path: sha256(HERE / path) for path in CODE},
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(result["scores"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
