#!/usr/bin/env python3
"""Read-only MCP server for the astronomical tools of the paper's three steps.

Agents call these tools through the Model Context Protocol (MCP). Each tool
wraps code the pipeline already runs. The parser and the verdicts come from
tools/statements.py, the decision from scripts/inference/date_all.py, and the
calendar, sky and eclipse tables from tools/cal.py, sky.py and eclipses.py.

The server is blind in the same way as blind_case_tools.py. A tool takes text
or explicit parameters from the caller and reads only the physical tables in
data/physical. It never opens the corpus, the catalogue, the results or the
audit ledgers, so it can return no published year and no record identifier.

    extract_clauses       the parser's clauses in a passage, with their fields
    clause_verdict        one clause in one candidate year, or in all 701
    score_text            strict set, safety set, first groups and decision
    babylonian_to_julian  a Babylonian date as a Julian day and date
    julian_to_babylonian  a Julian date as a Babylonian year, month and day
    body_position         Babylonian longitude and sign of a body
    eclipses_near         eclipses near a Julian date, with Babylon visibility

Years are astronomical numbers, so 626 BCE is -625 and AD 1 is 1. A year
argument takes era BCE (the default), AD or ASTRO. Verdicts use the paper's
words agree, contradict and untestable.

It speaks MCP over stdio and finds the repository from its own path. A
client starts it with this command, given an absolute path if need be:

    python scripts/agent/mcp_server.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.agent import blind_case_tools as blind  # noqa: E402
from scripts.inference import date_all  # noqa: E402
from tools import cal, eclipses, sky, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402

Era = Literal["BCE", "AD", "ASTRO"]
Body = Literal["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]


def astro_year(year: int, era: str) -> int:
    if era.upper() not in {"BCE", "AD", "ASTRO"}:
        raise ValueError("era must be BCE, AD or ASTRO")
    return blind.input_year(year, era)


def candidate_year(year: int, era: str) -> int:
    value = astro_year(year, era)
    if not yr.FIRST_CANDIDATE_ASTRO <= value <= yr.LAST_CANDIDATE_ASTRO:
        raise ValueError(f"{yr.year_label(value)} is outside the 701 "
                         "candidate years, 626 BCE to AD 75")
    return value


def physical_jd(year: int, month: int, day: int, era: str,
                hour: float = 0.0) -> float:
    """Julian day of a valid Julian date inside the ephemeris."""
    value = astro_year(year, era)
    if not 0.0 <= hour < 24.0:
        raise ValueError("hour must be at least 0 and less than 24")
    if not 1 <= month <= 12 or cal.from_jd(cal.to_jd(value, month, day)) != (
            value, month, day):
        raise ValueError(f"{yr.year_label(value)}-{month}-{day} is not a "
                         "Julian calendar date")
    jd = cal.to_jd(value, month, day, hour)
    if not (cal.to_jd(yr.FIRST_PHYSICAL_ASTRO, 1, 1) <= jd
            <= cal.to_jd(yr.LAST_PHYSICAL_ASTRO, 12, 31)):
        raise ValueError("the tables cover 700 BCE to AD 76")
    return jd


def julian_date(jd: float) -> dict:
    y, m, d = cal.from_jd(jd)
    return {"jd": round(float(jd), 6), "julian_year_astro": y,
            "julian_year": yr.year_label(y), "julian_month": m,
            "julian_day": d}


def require_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is empty")
    return text


def parse(text: str) -> tuple[list[S.Statement], list[str]]:
    require_text(text)
    return S.extract(text), S.months_in(text)


def clause_record(index: int, st: S.Statement) -> dict:
    out = blind.statement_record(index, st)
    out["decisive_type"] = S.is_decisive(st)
    out["evidence_families"] = list(S.evidence_families(st))
    return out


def number(value) -> float | None:
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def words(value) -> str | None:
    return value if isinstance(value, str) else None


def extract_clauses(text: str) -> dict:
    """Read a passage of translated text into the parser's clauses (step 1).

    Each clause has its fields (kind, planet, sign, star, month, day and so
    on), its passage and offsets, the parser rule that read it, its restored
    and omitted marks, and whether its type is decisive. Month and day
    headings in the passage govern the clauses after them. The index of a
    clause is its position in this list. clause_verdict takes that index.
    """
    statements, months = parse(text)
    return {"months": months, "clause_count": len(statements),
            "clauses": [clause_record(i, st)
                        for i, st in enumerate(statements)]}


def clause_verdict(text: str, index: int, year: int | None = None,
                   era: Era = "BCE") -> dict:
    """Test one clause of a passage in one candidate year, or in all 701 (step 2).

    The clause is clause number index of extract_clauses on the same text,
    so the headings of the passage still govern it. With a year, the result
    is the verdict and the reason the tools give. Without a year, the result
    counts agree, contradict and untestable over the 701 candidate years,
    626 BCE to AD 75, and lists the agreeing years. bits is
    -log2(agreeing years / testable years) for this clause alone (Eq. 1).
    score_text can give a clause fewer bits, when other clauses agree in
    exactly the same years.
    """
    statements, months = parse(text)
    if not 0 <= index < len(statements):
        raise ValueError(f"index must be 0 to {len(statements) - 1}. The "
                         f"parser found {len(statements)} clauses")
    st = statements[index]
    out = {"clause": clause_record(index, st)}
    if year is not None:
        value = candidate_year(year, era)
        verdict = S.check(st, value, months)
        out.update(year_astro=value, year=yr.year_label(value),
                   verdict=verdict.result, detail=verdict.detail)
        return out
    years = [int(y) for y in cal.candidate_years()]
    results = [S.check(st, y, months).result for y in years]
    agreeing = [y for y, result in zip(years, results) if result == "agree"]
    testable = len(agreeing) + results.count("contradict")
    out.update(
        candidate_years=len(years), agree=len(agreeing),
        contradict=results.count("contradict"),
        untestable=results.count("untestable"),
        bits=round(float(np.log2(testable / len(agreeing))), 9)
        if agreeing else 0.0,
        agreeing_years_astro=agreeing,
        agreeing_years=[yr.year_label(y) for y in agreeing])
    return out


def score_text(text: str) -> dict:
    """Score a passage in all 701 candidate years and decide (step 3).

    This runs the scoring and the decision of scripts/inference/date_all.py
    unchanged. The strict set holds the years no other year beats on decisive
    clocks and decisive bits. The safety set adds each year that one misread
    decisive clause could have pushed out. first_groups are the first three
    groups of tied years in rank order. decision is committed, ranked or
    decline, with its basis and reason. Each clause lists the years of each
    set in which it agrees, contradicts or is untestable.
    """
    years = [int(y) for y in cal.candidate_years()]
    result = date_all.astronomy_one("passage", require_text(text), years)
    groups = json.loads(result["evidence_levels_json"])
    for group in groups:
        group["years"] = [yr.year_label(y) for y in group["years_astro"]]
    clauses = json.loads(result["statement_verdicts_json"])
    strict, safety = result["strict_years_astro"], result["safety_years_astro"]
    return {
        "decision": result["claim"], "basis": result["basis"],
        "reason": result["reason"],
        "strict_set_astro": strict,
        "strict_set": [yr.year_label(y) for y in strict],
        "safety_set_astro": safety,
        "safety_set": [yr.year_label(y) for y in safety],
        "first_groups": groups, "group_count": result["evidence_group_count"],
        "clause_count": result["n_statements"],
        "testable_clauses": result["n_testable_statements"],
        "decisive_clauses": result["n_decisive_statements"],
        "decisive_clocks": result["decisive_families"],
        "decisive_bits": result["primary_bits"],
        "supporting_bits": result["support_bits"],
        "clauses": clauses,
    }


def babylonian_to_julian(year: int, month: str, day: int,
                         era: Era = "BCE") -> dict:
    """Convert a Babylonian date to a Julian day and a Julian calendar date.

    year is the year in which the Babylonian year begins. month is a Roman
    numeral I to XII, or VI2 or XII2 for a doubled month. The Julian date is
    the daytime of the Babylonian day, which began at the sunset before. The
    calendar is Parker and Dubberstein (1956), 626 BCE to AD 75.
    """
    value = astro_year(year, era)
    name = S.canonical_month(month.strip())
    if name not in cal.MONTHS:
        raise ValueError(f"month must be one of {', '.join(cal.MONTHS)}")
    if not 1 <= day <= 30:
        raise ValueError("day must be 1 to 30")
    jd = cal.bab_to_jd(value, name, day)
    if jd is None:
        raise ValueError(f"the calendar has no month {name} in the "
                         f"Babylonian year {yr.babylonian_span_label(value)}")
    return {"babylonian_year_astro": value,
            "babylonian_year": yr.babylonian_span_label(value),
            "month": name, "day": day, **julian_date(jd)}


def julian_to_babylonian(year: int, month: int, day: int,
                         era: Era = "BCE") -> dict:
    """Convert a Julian calendar date to a Babylonian year, month and day.

    year, month and day are a date in the proleptic Julian calendar. The
    Babylonian day is the one whose daytime falls on that date. The calendar
    is Parker and Dubberstein (1956), 626 BCE to AD 75.
    """
    jd = physical_jd(year, month, day, era)
    value = astro_year(year, era)
    # A Babylonian year begins in spring, so the date falls in this one or
    # in the one before.
    for babylonian_year in (value, value - 1):
        starts = sorted((start, name) for name in cal.MONTHS
                        if (start := cal.month_start_jd(babylonian_year, name))
                        is not None)
        if not starts:
            continue
        following = cal.month_start_jd(babylonian_year + 1, "I")
        ends = [start for start, _ in starts[1:]]
        ends.append(following if following is not None
                    else starts[-1][0] + 30.0)
        for (start, name), end in zip(starts, ends):
            if start <= jd < end:
                return {"babylonian_year_astro": babylonian_year,
                        "babylonian_year":
                            yr.babylonian_span_label(babylonian_year),
                        "month": name, "day": int(jd - start) + 1,
                        **julian_date(jd)}
    raise ValueError("the date is outside the calendar, which covers the "
                     "Babylonian years 626 BCE to AD 75")


def body_position(body: Body, year: int, month: int, day: int,
                  era: Era = "BCE", hour: float = 0.0) -> dict:
    """Give the Babylonian longitude and zodiacal sign of a body on a date.

    year, month and day are a proleptic Julian date, and hour counts from
    its midnight, the instant the verdicts use for a calendar day. The
    longitude is the tropical longitude of date plus Britton's offset, and a
    sign is a 30-degree bin of it, Aries first. The ephemeris is geocentric
    and covers 700 BCE to AD 76.
    """
    if body not in sky.BODIES:
        raise ValueError(f"body must be one of {', '.join(sky.BODIES)}")
    jd = physical_jd(year, month, day, era, hour)
    longitude = float(sky.babylonian_lon(body, jd))
    return {"body": body, **julian_date(jd), "hour": hour,
            "babylonian_longitude_deg": round(longitude, 4),
            "sign": sky.sign(body, jd),
            "degrees_in_sign": round(longitude % 30.0, 4),
            "tropical_longitude_of_date_deg":
                round(float(sky.lon_date(body, jd)), 4),
            "babylonian_offset_deg":
                round(float(sky.babylonian_offset_deg(jd)), 4)}


def eclipses_near(year: int, month: int, day: int, era: Era = "BCE",
                  days: float = 15.0,
                  kind: Literal["lunar", "solar"] | None = None) -> dict:
    """List the eclipses within some days of a Julian date, with Babylon status.

    local_status is robust_visible, robust_not_visible or delta_t_sensitive
    over the three Delta T trials. agreement_eligible is true when an eclipse
    of that kind can agree with a clause. Times are Universal Time. The
    catalogue is NASA's Five Millennium Canon.
    """
    if not 0 < days <= 200:
        raise ValueError("days must be more than 0 and at most 200")
    jd = physical_jd(year, month, day, era)
    rows = eclipses.events(jd - days, jd + days, kind)
    found = []
    for row in rows.sort_values("jd_ut").itertuples(index=False):
        found.append({
            "kind": row.kind, "type": row.eclipse_type,
            "nasa_number": int(row.catalog_num),
            **julian_date(row.jd_ut),
            "time_ut": f"{int(row.hour):02d}:{int(row.minute):02d}",
            "days_from_date": round(float(row.jd_ut - jd), 2),
            "magnitude": number(row.global_magnitude),
            "local_status": row.local_status,
            "local_phase": words(row.local_phase),
            "local_magnitude": number(row.local_magnitude),
            "local_altitude_deg": number(row.local_apparent_altitude),
            "visible_babylon": bool(row.visible_babylon),
            "agreement_eligible": bool(row.agreement_eligible),
            "local_reason": words(row.local_decision_reason)})
    return {**julian_date(jd), "days": days, "eclipses": found}


TOOLS = [extract_clauses, clause_verdict, score_text, babylonian_to_julian,
         julian_to_babylonian, body_position, eclipses_near]


def build_server():
    """The FastMCP server, with every tool marked read-only."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP("babylonian-dating", instructions=(
        "Deterministic astronomy for dating the Babylonian Astronomical "
        "Diaries. Years are astronomical numbers, so 626 BCE is -625. No "
        "tool knows a published date."), log_level="WARNING")
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                idempotentHint=True, openWorldHint=False)
    for tool in TOOLS:
        server.add_tool(tool, annotations=read_only)
    return server


def main() -> int:
    build_server().run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
