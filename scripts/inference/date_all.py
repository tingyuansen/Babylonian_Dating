#!/usr/bin/env python3
"""Write one date-blind committed, ranked, or decline result per tablet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, clauses, eras, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402


def encoded(values) -> str:
    """Stable JSON for evidence lists retained inside a flat parquet table."""
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def spans(years: list[int]) -> list[str]:
    """Human-readable Julian-year span of each Babylonian candidate year."""
    return [yr.babylonian_span_label(year) for year in years]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_result(tablet: str, claim: str, basis: str, reason: str,
                strict: list[int], safety: list[int]) -> dict:
    # With no active predicate all candidate years are physically equivalent.
    # Retaining that explicit group is more informative than an empty ranking.
    empty_level = {
        "rank": 1,
        "tier": "strict",
        "decisive_families": 0,
        "primary_bits": 0.0,
        "support_bits": 0.0,
        "contradictions": 0,
        "years_astro": strict,
        "physical_groups_years_astro": [strict],
        "predicate_exact": True,
    }
    return {
        "tablet": tablet,
        "claim": claim,
        "basis": basis,
        "reason": reason,
        "strict_years_astro": strict,
        "safety_years_astro": safety,
        "strict_spans": spans(strict),
        "safety_spans": spans(safety),
        "strict_count": len(strict),
        "safety_count": len(safety),
        "top_evidence_group_years_astro": strict,
        "top_evidence_group_size": len(strict),
        "top_evidence_group_predicate_exact": True,
        "top_three_evidence_groups_years_astro": strict,
        "top_three_evidence_groups_size": len(strict),
        "top_three_evidence_groups_predicate_exact": True,
        "evidence_group_count": 1,
        "evidence_levels_json": encoded([empty_level]),
        "n_statements": 0,
        "n_testable_statements": 0,
        "n_decisive_statements": 0,
        "n_restored_statements": 0,
        "n_scoring_units": 0,
        "n_joint_positions": 0,
        "decisive_families": 0,
        "primary_bits": 0.0,
        "support_bits": 0.0,
        "chronology_status": "absent",
        "chronology_years_astro": [],
        "chronology_relation": "absent",
        "chronology_evidence_json": "[]",
        "combined_claim": claim,
        "combined_basis": basis,
        "combined_years_astro": strict,
        "decisive_evidence_json": "[]",
        "supporting_evidence_json": "[]",
        "statement_verdicts_json": "[]",
    }


def statement_records(statements, scores, strict_mask, safety_mask) -> list[dict]:
    """Public per-statement results over both reported candidate sets."""
    labels = {-1: "contradict", 0: "untestable", 1: "agree"}
    records = []
    for i, statement in enumerate(statements):
        record = {
            "index": i,
            "kind": statement.kind,
            "planet": statement.planet,
            "target_planet": statement.target_planet,
            "star": statement.star,
            "sign": statement.sign,
            "month": statement.month,
            "day": statement.day,
            "year_offset": statement.year_offset,
            "restored": statement.restored,
            "omitted": statement.omitted,
            "source_start": statement.source_start,
            "source_end": statement.source_end,
            "parser_rule": statement.parser_rule,
            "amount": statement.amount,
            "unit": statement.unit,
            "direction": statement.direction,
            "station_phase": statement.station_phase,
            "evidence_families": list(S.evidence_families(statement)),
            "joint_positions": [list(pair) for pair in statement.joint_positions],
            "joint_source_start": statement.joint_source_start,
            "joint_source_end": statement.joint_source_end,
            "joint_day": statement.joint_day,
            "joint_restored": statement.joint_restored,
            "joint_omitted": statement.joint_omitted,
            "scoring_unit": int(scores.scoring_units[i]),
            "decisive": bool(scores.decisive[i] and scores.weights[i] > 0),
            "weight_bits": round(float(scores.weights[i]), 9),
            "agreeing_candidates": int(scores.agreeing[i]),
            "testable_candidates": int(scores.testable[i]),
            "equivalent_predicates": int(scores.equivalent[i]),
            "text": statement.text,
        }
        for name, mask in (("strict", strict_mask), ("safety", safety_mask)):
            record[f"{name}_verdict_years_astro"] = {
                label: [int(year) for year in scores.years[
                    mask & (scores.outcomes[:, i] == value)]]
                for value, label in labels.items()
            }
        records.append(record)
    return records


def astronomy_one(tablet: str, text: str, years: list[int],
                  has_transliteration: bool = False,
                  statements: list[S.Statement] | None = None) -> dict:
    """Infer from astronomy without chronology or catalogue metadata.

    The clauses are the reviewed clauses of the record when given, and the
    parser's clauses of the text otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        if has_transliteration:
            return base_result(
                tablet, "decline", "unsupported_transliteration",
                "transliteration survives but the statement reader does not "
                "yet parse this untranslated text", years, years)
        return base_result(
            tablet, "decline", "no_text",
            "no readable text survives", years, years)

    if statements is None:
        statements = S.extract(text)
    if not statements:
        return base_result(
            tablet, "decline", "no_recognized_evidence",
            "text survives but no supported astronomical statement is recognized",
            years, years)

    months = S.months_in(text)
    scores = S.candidate_scores(statements, months, years, weighting="skill")
    levels = S.evidence_levels(scores)
    reported_levels = levels[:3]
    top_evidence_group = levels[0]["years_astro"]
    top_three_evidence_groups = sorted({
        year for level in reported_levels for year in level["years_astro"]
    })
    strict = [int(year) for year in scores.years[scores.strict]]
    safety = [int(year) for year in scores.years[scores.safety]]
    active_decisive = scores.decisive & (scores.weights > 0)
    decisive_indices = [int(i) for i in active_decisive.nonzero()[0]]
    supporting_indices = [
        i for i in range(len(statements))
        if scores.weights[i] > 0 and not active_decisive[i]
    ]

    if not decisive_indices:
        claim = "decline"
        basis = "supporting_only"
        reason = (
            "recognized evidence is supporting only; no secure outer-planet, "
            "locally visible eclipse, or equivalent decisive family survives")
    elif not strict or len(strict) == len(years):
        claim = "decline"
        basis = "non_discriminating"
        reason = "decisive-looking evidence does not reduce the candidate calendar"
    else:
        strict_idx = scores.strict.nonzero()[0]
        best_families = int(scores.priority[strict_idx].max())
        # E(y*) of the commitment rule: an observed lunar eclipse must agree
        # with the one strict year itself, not merely exist on the tablet.
        secure_lunar = len(strict) == 1 and any(
            statements[i].kind == "eclipse"
            and statements[i].planet == "lunar"
            and not statements[i].omitted
            and not statements[i].restored
            and scores.outcomes[strict_idx[0], i] == 1
            for i in decisive_indices)
        can_commit = (len(strict) == 1
                      and (secure_lunar
                           or (best_families >= 2 and len(safety) == 1)))
        if can_commit:
            claim = "committed"
            basis = ("secure_lunar_eclipse" if secure_lunar
                     else "independent_astronomy")
            reason = (
                "secure decisive astronomy leaves one candidate year"
                if secure_lunar else
                "independent astronomy leaves one candidate even after any "
                "one decisive statement is removed")
        else:
            claim = "ranked"
            basis = "astronomical_frontier"
            if len(strict) == 1 and best_families < 2:
                reason = (
                    "one candidate leads, but only one decisive family survives; "
                    "a recurrence cannot support commitment")
            elif len(strict) == 1:
                reason = (
                    f"one candidate leads with {best_families} decisive families, "
                    f"but removing one decisive statement leaves {len(safety)} "
                    "candidates")
            else:
                reason = (
                    f"{len(strict)} candidates remain on the independent-body/"
                    "information Pareto frontier")

    result = base_result(tablet, claim, basis, reason, strict, safety)
    result.update({
        "n_statements": len(statements),
        "n_testable_statements": int((scores.testable > 0).sum()),
        "n_decisive_statements": len(set(scores.scoring_units[active_decisive])),
        "n_scoring_units": len(set(scores.scoring_units)),
        "n_joint_positions": len({S.scoring_unit_key(st, i)
                                  for i, st in enumerate(statements)
                                  if st.joint_positions}),
        "n_restored_statements": sum(st.restored for st in statements),
        "top_evidence_group_years_astro": top_evidence_group,
        "top_evidence_group_size": len(top_evidence_group),
        "top_evidence_group_predicate_exact": bool(
            levels[0]["predicate_exact"]),
        "top_three_evidence_groups_years_astro": top_three_evidence_groups,
        "top_three_evidence_groups_size": len(top_three_evidence_groups),
        "top_three_evidence_groups_predicate_exact": bool(
            all(level["predicate_exact"] for level in reported_levels)),
        "evidence_group_count": len(levels),
        "evidence_levels_json": encoded(reported_levels),
        "decisive_families": (int(scores.priority[scores.strict].max())
                              if scores.strict.any() else 0),
        "primary_bits": (float(scores.score[scores.strict].max())
                         if scores.strict.any() else 0.0),
        "support_bits": (float(scores.support[scores.strict].max())
                         if scores.strict.any() else 0.0),
        "decisive_evidence_json": encoded(
            statements[i].text for i in decisive_indices),
        "supporting_evidence_json": encoded(
            statements[i].text for i in supporting_indices),
        "statement_verdicts_json": encoded(
            statement_records(statements, scores, scores.strict, scores.safety)),
    })
    return result


def add_chronology(result: dict, text: str) -> dict:
    """Annotate astronomy after inference; never manufacture astronomy success."""
    readings = eras.extract(text) if isinstance(text, str) else []
    years = sorted({int(reading.astro_year) for reading in readings})
    result["chronology_years_astro"] = years
    result["chronology_evidence_json"] = encoded(
        reading.text for reading in readings)
    if not years:
        return result
    if len(years) > 1:
        result["chronology_status"] = "multiple"
        result["chronology_relation"] = "multiple_absolute_years"
        result["combined_basis"] = "chronology_conflict"
        return result

    year = years[0]
    result["chronology_status"] = "single"
    strict = set(result["strict_years_astro"])
    safety = set(result["safety_years_astro"])
    if result["claim"] == "committed":
        if year in strict:
            result["chronology_relation"] = "confirms_astronomy"
        else:
            result["chronology_relation"] = "conflicts_with_astronomy"
            result["combined_claim"] = "ranked"
            result["combined_basis"] = "chronology_conflict"
            result["combined_years_astro"] = sorted(strict | {year})
    elif result["n_decisive_statements"] > 0 and year in strict:
        result["chronology_relation"] = "selects_strict_candidate"
        result["combined_claim"] = "committed"
        result["combined_basis"] = "chronology_adjudicated"
        result["combined_years_astro"] = [year]
    elif result["n_decisive_statements"] > 0 and year in safety:
        result["chronology_relation"] = "selects_safety_candidate"
        result["combined_claim"] = "committed"
        result["combined_basis"] = "chronology_adjudicated"
        result["combined_years_astro"] = [year]
    elif result["n_decisive_statements"] > 0:
        result["chronology_relation"] = "outside_astronomy"
        result["combined_basis"] = "chronology_conflict"
        result["combined_years_astro"] = sorted(strict | {year})
    else:
        result["chronology_relation"] = "lookup_only"
        result["combined_basis"] = "chronology_only_not_astronomy"
    return result


def date_one(tablet: str, text: str, years: list[int],
             has_transliteration: bool = False,
             statements: list[S.Statement] | None = None) -> dict:
    """Run answer-blind astronomy first, then add text-internal chronology."""
    result = astronomy_one(tablet, text, years, has_transliteration, statements)
    return add_chronology(result, text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=HERE / "data" / "results" / "tablet_dates.parquet")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    # Deliberately select only ID and text. Reference dates and catalogue
    # chronology never enter this process.
    tablets = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet", columns=["tablet", "text"])
    lines = pd.read_parquet(
        HERE / "data" / "corpus" / "lines.parquet", columns=["tablet", "translit"])
    has_transliteration = (lines.assign(
        present=lines.translit.fillna("").str.strip().ne(""))
        .groupby("tablet").present.any())
    tablets["has_transliteration"] = (
        tablets.tablet.map(has_transliteration).eq(True))
    if args.limit:
        tablets = tablets.head(args.limit)
    years = [int(year) for year in cal.candidate_years()]
    reviewed = clauses.load(clauses.REVIEWED)
    rows = []
    for number, tablet in enumerate(tablets.itertuples(index=False), start=1):
        rows.append(date_one(
            tablet.tablet, tablet.text, years, tablet.has_transliteration,
            reviewed.get(tablet.tablet, [])))
        if number % 100 == 0:
            print(f"  {number}/{len(tablets)} tablets")

    result = pd.DataFrame(rows)
    # This metadata is appended only after inference. It documents a known
    # physical-join limitation but cannot influence a candidate score.
    catalogue = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet", columns=["tablet", "museum_no"])
    catalogue["n_physical_fragments"] = catalogue.museum_no.fillna("").apply(
        lambda value: len(re.split(r"\s*\+\s*", value))
        if value.strip() else 0)
    result = result.merge(
        catalogue[["tablet", "n_physical_fragments"]], on="tablet", how="left")
    result["method_version"] = "date-all-v7-reviewed-clauses"
    result["date_all_sha256"] = sha256(Path(__file__))
    result["statements_sha256"] = sha256(HERE / "tools" / "statements.py")
    result["skill_sha256"] = sha256(HERE / "skills" / "dating_protocol.md")
    # Hash every scientific dependency that can alter a candidate verdict.
    # These hashes are attached only after inference and therefore document
    # reproducibility without becoming ranking features.
    provenance = {
        "cal_sha256": HERE / "tools" / "cal.py",
        "sky_sha256": HERE / "tools" / "sky.py",
        "eclipse_tool_sha256": HERE / "tools" / "eclipses.py",
        "eras_sha256": HERE / "tools" / "eras.py",
        "clauses_sha256": clauses.REVIEWED,
        "calendar_sha256": HERE / "data" / "physical" / "calendar.parquet",
        "ephemeris_sha256": HERE / "data" / "physical" / "ephemeris.parquet",
        "eclipse_data_sha256": HERE / "data" / "physical" / "eclipses.parquet",
        "normal_stars_sha256": HERE / "data" / "physical" / "normal_stars.parquet",
    }
    for column, path in provenance.items():
        result[column] = sha256(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    print(f"\n{len(result)} tablet results -> {args.out}")
    print("  astronomy-only claims")
    for claim, count in result.claim.value_counts().items():
        print(f"  {claim:<10} {count:>3}")
    print("  astronomy committed basis", result[result.claim == "committed"]
          .basis.value_counts().to_dict())
    print("  astronomy decline basis  ", result[result.claim == "decline"]
          .basis.value_counts().to_dict())
    print("  chronology relations", result.chronology_relation.value_counts()
          .to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
