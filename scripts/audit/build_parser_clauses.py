#!/usr/bin/env python3
"""Write each parser clause of ADsD with its tablet lines and its verdict at the published year.

The parser reads only the ADsD translation of each record (step 1). Each row
gives one parser clause with its fields, its passage, the tablet lines it
comes from, its parser rule and its tier. The published years are read after
all clauses are built. They give each clause its verdict at the published
year, as counted for the parser clauses in SI Appendix, Table S4.

build_benchmark.py, build_paper_counts.py and scripts/site/build_site_data.py
read this file. The dating of steps 2 and 3 does not.

Output:
    data/clauses/parser_clauses.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import statements as S
from tools import years as yr  # noqa: E402


# The origin of each parser rule, for the column rule_origin. A label that
# names the agent marks a rule that the coding agent of step 1 proposed or
# checked.
RULE_ORIGIN = {
    "typed_eclipse": "source_semantics_local_parser",
    "measured_normal_star": "source_semantics_local_parser",
    "measured_normal_star_compound": "agent_audited_recurring_source_form",
    "measured_station_normal_star": "answer_blind_agent_audited_source_form",
    "measured_station_normal_star_compound": "answer_blind_agent_audited_source_form",
    "measured_planet_planet": "answer_blind_agent_audited_source_form",
    "measured_planet_planet_compound": "answer_blind_agent_audited_source_form",
    "measured_moon_planet": "agent_suggested_X300963_generalized",
    "measured_moon_planet_compound": "shared_grammar_not_yet_observed",
    "monthly_path": "agent_suggested_X103000_generalized",
    "monthly_path_after": "agent_suggested_X301413_generalized",
    "coordinated_sign": "agent_suggested_X300730_generalized",
    "triple_coordinated_sign": "local_corpus_generalization",
    "in_sign": "local_deterministic_parser",
    "station": "source_semantics_local_parser",
    "entry": "source_semantics_local_parser",
    "last_app": "source_semantics_local_parser",
    "first_app": "source_semantics_local_parser",
    "acronychal_rising": "source_semantics_local_parser",
    "season_solstice": "source_semantics_local_parser",
    "season_equinox": "source_semantics_local_parser",
    "sirius_rising": "published_scheme_sachs_1952",
    "sirius_setting": "published_scheme_sachs_1952",
    "sirius_acronychal": "published_scheme_sachs_1952",
}


def encoded(value) -> str:
    """Return compact JSON for a value kept in one parquet cell."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def source_lines(group: pd.DataFrame) -> list[dict]:
    """Give each translated line of a record its span in the record text.

    The spans follow fetch_texts.py, which joins the lines with newlines.
    """
    rows = []
    cursor = 0
    for line in group.itertuples(index=False):
        translation = line.translation if isinstance(line.translation, str) else ""
        if not translation:
            continue
        start = cursor
        end = start + len(translation)
        rows.append({
            "start": start,
            "end": end,
            "label": line.label,
            "translation": translation,
            "translit": line.translit if isinstance(line.translit, str) else "",
        })
        cursor = end + 1
    return rows


def overlapping(lines: list[dict], start: int, end: int) -> list[dict]:
    """Return the lines that overlap the span from start to end."""
    return [line for line in lines
            if line["end"] > start and line["start"] < end]


def context_source(text: str, statement: S.Statement) -> str:
    """Label where the month of a clause comes from."""
    if statement.month is None:
        return "missing_or_lost"
    if (statement.kind in {"in_sign", "station", "entry", "last_app"}
            and S.preceding_summary_month(text, statement.source_start) is not None):
        return "inferred_immediately_preceding_summary"
    return "explicit_surviving_heading"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=HERE / "data" / "clauses" / "parser_clauses.parquet")
    args = parser.parse_args()

    # The parser reads the text alone. The published years are read only
    # after all clauses are built.
    tablets = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet", columns=["tablet", "text"])
    raw_lines = pd.read_parquet(
        HERE / "data" / "corpus" / "lines.parquet",
        columns=["tablet", "label", "translation", "translit"])
    by_tablet = {tablet: source_lines(group)
                 for tablet, group in raw_lines.groupby("tablet", sort=False)}

    rows = []
    statement_cache = {}
    for tablet in tablets.itertuples(index=False):
        text = tablet.text if isinstance(tablet.text, str) else ""
        lines = by_tablet.get(tablet.tablet, [])
        rebuilt = "\n".join(line["translation"] for line in lines)
        if rebuilt != text:
            raise RuntimeError(
                f"{tablet.tablet}: source-line join does not reproduce tablet text")
        months = S.months_in(text)
        for index, statement in enumerate(S.extract(text)):
            statement_cache[(tablet.tablet, index)] = statement
            if statement.source_start < 0 or statement.source_end <= statement.source_start:
                raise RuntimeError(
                    f"{tablet.tablet} statement {index} lacks source offsets")
            touched = overlapping(
                lines, statement.source_start, statement.source_end)
            if not touched:
                raise RuntimeError(
                    f"{tablet.tablet} statement {index} has no source line")
            rows.append({
                "tablet": tablet.tablet,
                "statement_index": index,
                "source_char_start": statement.source_start,
                "source_char_end": statement.source_end,
                "source_labels_json": encoded([line["label"] for line in touched]),
                "source_translation": text[
                    statement.source_start:statement.source_end],
                "source_line_translation": "\n".join(
                    line["translation"] for line in touched),
                "source_line_translit": "\n".join(
                    line["translit"] for line in touched),
                "tablet_text_sha256": hashlib.sha256(
                    text.encode("utf-8")).hexdigest(),
                "parser_rule": statement.parser_rule,
                # A clause read across the brackets of a restoration takes the
                # origin of its rule without the suffix _bracket. The column
                # restored records the restoration.
                "rule_origin": RULE_ORIGIN.get(
                    statement.parser_rule.removesuffix("_bracket"),
                    "unclassified_rule_origin"),
                "kind": statement.kind,
                "planet": statement.planet,
                "target_planet": statement.target_planet,
                "star": statement.star,
                "sign": statement.sign,
                "month": statement.month,
                "day": statement.day,
                "year_offset": statement.year_offset,
                "amount": statement.amount,
                "unit": statement.unit,
                "direction": statement.direction,
                "station_phase": statement.station_phase,
                "evidence_families_json": encoded(S.evidence_families(statement)),
                "joint_positions_json": encoded(statement.joint_positions),
                "joint_source_char_start": statement.joint_source_start,
                "joint_source_char_end": statement.joint_source_end,
                "joint_day": statement.joint_day,
                "scoring_unit_key_json": encoded(S.scoring_unit_key(statement, index)),
                "offset_degrees": statement.offset,
                "latitudinal": statement.latitudinal,
                "restored": statement.restored,
                "omitted": statement.omitted,
                "context_source": context_source(text, statement),
                "semantic_tier": (
                    "decisive" if S.is_decisive(statement) else "supporting"),
                "normalized_text": statement.text,
                "months_in_tablet_json": encoded(months),
            })

    facts = pd.DataFrame(rows)
    references = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "bce_first", "bce_second"])
    facts = facts.merge(references, on="tablet", how="left", validate="many_to_one")
    verdict, detail = [], []
    for fact in facts.itertuples(index=False):
        if pd.isna(fact.bce_first):
            verdict.append("not_evaluated")
            detail.append("tablet has no independent reference year")
            continue
        statement = statement_cache[(fact.tablet, fact.statement_index)]
        checked = S.check(
            statement, yr.bce_to_astro(int(fact.bce_first)),
            json.loads(fact.months_in_tablet_json))
        verdict.append(checked.result)
        detail.append(checked.detail)
    facts["forward_verdict"] = verdict
    facts["forward_detail"] = detail
    facts["extractor_sha256"] = hashlib.sha256(
        (HERE / "tools" / "statements.py").read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    facts.to_parquet(args.out, index=False)
    print(f"{len(facts):,} extracted facts from "
          f"{facts.tablet.nunique():,} tablets -> {args.out}")
    print("  parser rules", facts.parser_rule.value_counts().to_dict())
    print("  forward verdicts", facts.forward_verdict.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
