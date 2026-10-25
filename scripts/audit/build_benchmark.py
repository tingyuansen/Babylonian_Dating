#!/usr/bin/env python3
"""Check that the dating code is blind, and write the frozen benchmark population.

The population ledger separates raw text properties, reproducible coarse
labels and tool-extractable eligibility. None of its inclusion fields depends
on a dating result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.inference import date_all  # noqa: E402
from tools import statements as S  # noqa: E402


def check_blind() -> None:
    """Fail if the dating code reads a published date or a catalogue field."""
    inference_source = "\n".join([
        inspect.getsource(date_all.astronomy_one),
        inspect.getsource(S.extract),
        inspect.getsource(S.candidate_scores),
    ])
    forbidden = [
        field for field in (
            "bce_first", "bce_second", "ancient_year", "period", "museum_no")
        if field in inference_source
    ]
    if forbidden:
        raise RuntimeError(
            "astronomy inference references answer/catalogue fields: "
            + ", ".join(forbidden))


def regex_cohort(row) -> str:
    astronomy = bool(row.outer_regex or row.eclipse_regex)
    any_content = bool(
        row.lunar_regex or row.inner_regex or row.outer_regex
        or row.eclipse_regex or row.era_regex)
    if astronomy and row.era_regex:
        return "testable_checked"
    if astronomy:
        return "testable"
    if any_content and row.era_regex:
        return "lookup_only"
    if any_content:
        return "weak_evidence"
    return "no_text"


def population_ledger() -> pd.DataFrame:
    evidence = pd.read_parquet(HERE / "data" / "corpus" / "evidence.parquet")
    cohorts = pd.read_parquet(
        HERE / "data" / "corpus" / "cohorts.parquet",
        columns=["tablet", "cohort", "cohort_source", "joined"])
    facts = pd.read_parquet(HERE / "data" / "clauses" / "parser_clauses.parquet")
    dates = pd.read_parquet(
        HERE / "data" / "results" / "tablet_dates.parquet",
        columns=["tablet", "n_decisive_statements"])

    # The evaluation population was frozen before the station-to-star
    # extension. New supporting grammars remain visible in the fact ledger but
    # cannot silently admit extra benchmark tablets merely by raising an
    # extraction count. A future prospective cohort needs expert text labels,
    # not a moving parser-defined threshold.
    post_freeze_rules = {
        "measured_station_normal_star",
        "measured_station_normal_star_compound",
        "measured_planet_planet",
        "measured_planet_planet_compound",
    }
    fact_counts = (facts.assign(
        eligible=~(facts.kind.eq("eclipse") & facts.omitted),
        frozen_eligible=(~(facts.kind.eq("eclipse") & facts.omitted)
                         & ~facts.parser_rule.isin(post_freeze_rules)))
        .groupby("tablet")
        .agg(n_extracted_facts=("statement_index", "size"),
             n_extractable_facts=("eligible", "sum"),
             n_frozen_extractable_facts=("frozen_eligible", "sum"),
             n_semantic_decisive=("semantic_tier",
                                  lambda values: int(values.eq("decisive").sum())))
        .reset_index())
    keep = [
        "tablet", "project", "lang", "n_lines", "n_joins", "era_regex",
        "outer_regex", "inner_regex", "lunar_regex", "eclipse_regex",
        "era_llm", "outer_llm",
    ]
    result = (evidence[keep]
              .merge(cohorts, on="tablet", validate="one_to_one")
              .merge(fact_counts, on="tablet", how="left", validate="one_to_one")
              .merge(dates, on="tablet", how="left", validate="one_to_one"))
    for column in ("n_extracted_facts", "n_extractable_facts",
                   "n_frozen_extractable_facts",
                   "n_semantic_decisive", "n_decisive_statements"):
        result[column] = result[column].fillna(0).astype(int)
    result["regex_cohort"] = result.apply(regex_cohort, axis=1)
    result = result.rename(columns={
        "cohort": "coarse_cohort",
        "n_decisive_statements": "n_active_decisive",
    })
    result["tool_benchmark_eligible"] = result.n_frozen_extractable_facts.ge(3)
    result["tool_year_bearing"] = (
        result.tool_benchmark_eligible & result.n_active_decisive.gt(0))
    result["selection_reason"] = "fewer_than_three_extractable_statements"
    result.loc[result.tool_benchmark_eligible, "selection_reason"] = (
        "three_or_more_extractable_statements")
    result["human_comparison_label"] = "not_independently_expert_labeled"
    result["selection_uses_backward_success"] = False
    result["selection_version"] = "pre_station_extension_2026-08-14"
    return result


def write_with_hash(frame: pd.DataFrame, path: Path) -> None:
    frame = frame.copy()
    frame["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population-out", type=Path,
        default=HERE / "data" / "benchmark" / "benchmark_records.parquet")
    parser.add_argument(
        "--rebuild-population", action="store_true",
        help=("also rebuild the population ledger; omitted by default because "
              "benchmark membership is frozen"))
    args = parser.parse_args()
    check_blind()
    if args.rebuild_population:
        population = population_ledger()
        write_with_hash(population, args.population_out)
        print(f"{len(population)} dated diary population rows -> {args.population_out}")
        print("  coarse cohorts", population.coarse_cohort.value_counts().to_dict())
        print("  tool benchmark eligible",
              int(population.tool_benchmark_eligible.sum()),
              "year-bearing", int(population.tool_year_bearing.sum()))
    else:
        print(f"preserved frozen population ledger -> {args.population_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
