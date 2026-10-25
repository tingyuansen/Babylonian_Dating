#!/usr/bin/env python3
"""Held-out test by volume, SI Appendix section S8.

The rules were developed on the dated records that the benchmark tests. This
audit splits ADART I to III by volume and dates the benchmark records of each
volume with choices made again from the other two. A fourth round makes the
same choices from all three volumes; it must reproduce the main text, and it
is checked record by record against the unmodified pipeline.

Three kinds of choice were made at the published dates, and each round makes
them again from all dated records of its training volumes, benchmark or not,
and from nothing else.

    tolerances      Moon and planet distance tolerances, the smallest value on
                    a fixed grid that covers a fixed share of the residuals of
                    measured relations on the stated night
    parser rules    each rule that entered the code after a check of its
                    clauses at the published dates is kept only if it passes
                    the same gate as a decisive type
    decisive types  a type is decisive if it is physically eligible and its
                    rule agrees at the published years in at least
                    GATE_MIN_AGREEMENT of at least GATE_MIN_TESTABLE cases

Physical eligibility, the clause conditions of the decisive tier, the clocks,
the calendar, the ephemeris, the zodiac, the time windows, the context windows
and the formulas of step 3 stay fixed. The reasons are in ELIGIBILITY_REASONS
and in the design note that accompanies this script.

The order is blind in the same way as the other audits. Choices read only the
training volumes' published years. Candidate sets are computed before any
published year of the test records is loaded, and the published years enter
only when the rows of Table S6 are formed.

The held-out rounds are the test, and the reading of the test must be fixed
before anyone sees them. They are therefore not scored unless --run-held-out
is given. Without it the script computes every round's choices and scores only
the round that uses all three volumes.

Usage:
    python3 scripts/audit/audit_held_out.py                  # choices + all volumes
    python3 scripts/audit/audit_held_out.py --choices-only   # no scoring at all
    python3 scripts/audit/audit_held_out.py --run-held-out   # the test itself
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, clauses, sky, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402
from scripts.audit.audit_sensitivity import retained  # noqa: E402

# The repository's own tier rule, captured before any round replaces it.
REPO_IS_DECISIVE = S.is_decisive

VOLUMES = {"ADART I": "adsd/adart1", "ADART II": "adsd/adart2",
           "ADART III": "adsd/adart3"}
ROUNDS = [
    {"round": "ADART I", "train": ["ADART II", "ADART III"], "test": ["ADART I"]},
    {"round": "ADART II", "train": ["ADART I", "ADART III"], "test": ["ADART II"]},
    {"round": "ADART III", "train": ["ADART I", "ADART II"], "test": ["ADART III"]},
    {"round": "all volumes", "train": list(VOLUMES), "test": list(VOLUMES)},
]
HELD_OUT = [spec["round"] for spec in ROUNDS if spec["round"] != "all volumes"]
REPO_JOB = "unmodified pipeline"
CHUNK_RECORDS = 6

# The main-text values for the committed code. They move when tools/ changes,
# so the criterion for the all-volume round is the unmodified pipeline rerun on
# the same code, not these numbers; they are reported for orientation only.
MAIN_TEXT = {"records": 112, "first_group": 108, "first_three_groups": 110,
             "strict_set": 112, "median_strict_set": 16.5}

# --- The reading of the test, fixed before any held-out round was scored. ---
# The pooled row of the three held-out rounds is close to the all-volume round
# if its inclusion rate drops by at most this many percentage points, for the
# first group and for the strict set. The rows of single volumes are
# descriptive only: one record of ADART II is 6.7 points.
CLOSE_POINTS = 5.0

# --- The gate of SI section S5, shared by decisive types and parser rules. ---
# A type is decisive, and a parser rule is kept, if its clauses agree at the
# published years of the training volumes in at least this share of at least
# this many testable cases. Testable means agree or contradict at the published
# year; restored clauses count, as in Table S4. On all three volumes these two
# values reproduce the current decisive types and parser rules exactly, and the
# choices of every round are the same for any agreement threshold between the
# values that `gate_interval` reports.
GATE_MIN_AGREEMENT = 0.80
GATE_MIN_TESTABLE = 3

# --- The tolerance rule of SI section S4. ---
# Each tolerance is the smallest value on this grid such that at least
# TOL_QUANTILE of the residuals of its body class lie within it. The residual
# is the difference between the stated and the computed distance at the best
# moment of darkness on the stated night or on the night either side. The
# check itself sweeps the Moon over two and a half days, about 33 degrees, so
# agreement in the check hardly depends on the Moon's tolerance and cannot fix
# it; restricting the residual to the night measures the stated distance.
# On all three volumes the share must lie in (88.2%, 89.7%] to give 3 and 2
# degrees, and 89% is the one whole percentage there. The planet class is
# small, so a round can jump a grid step or more; the summary reports the
# interval of shares that leaves each round's choice unchanged.
TOL_GRID = tuple(round(0.5 * k, 1) for k in range(1, 13))   # 0.5 ... 6.0
TOL_QUANTILE = 0.89
# The night of Babylonian day n, relative to the civil day of its daylight
# that cal.bab_to_jd returns (0h UT): 15h to 3h UT, about 18h to 6h at Babylon.
NIGHT = (-0.375, 0.125)
NIGHT_SLACK_DAYS = (-1, 0, 1)   # the check's one-day calendar slack
NIGHT_STEP_DAYS = 0.01

OUTER = {"mars", "jupiter", "saturn"}
INNER = {"venus", "mercury"}
MEASURED = {"appulse", "planet_appulse", "planet_planet_appulse"}
PHENOMENON = {
    "in_sign": "planet in sign", "station": "station", "entry": "sign entry",
    "first_app": "first appearance", "last_app": "last appearance",
    "acronychal": "acronychal rising", "appulse": "distance to Normal Star",
    "planet_appulse": "Moon distance to planet",
    "planet_planet_appulse": "planet distance to planet",
    "station_appulse": "station distance to Normal Star",
    "eclipse": "eclipse", "season": "solstice or equinox", "sirius": "Sirius date",
}

# Physical eligibility is fixed in every round. Agreement cannot set it: at the
# published years of all three volumes, Moon positions against a Normal Star
# agree in 92%, inner-planet types in 89 to 97% and last appearances of outer
# planets in 97% of testable cases, and each would pass the gate.
ELIGIBILITY_REASONS = {
    "lunar eclipse": (
        "fixed_decisive",
        "the verdict comes from catalogued eclipses and computed visibility "
        "at Babylon in three Delta T trials, with no distance tolerance; the "
        "dated corpus has too few testable reports (two parser clauses) for "
        "any count gate"),
    "moon": (
        "fixed_supporting",
        "the Moon's place on a given day of a lunar month is set mostly by the "
        "month itself, so it measures the calendar rather than the year"),
    "inner": (
        "fixed_supporting",
        "Venus and Mercury stay within 47 and 28 degrees of the Sun, so their "
        "position in a given month repeats on short cycles"),
    "last appearance": (
        "fixed_supporting",
        "a last appearance is a visibility event near the Sun, and a "
        "visibility model based on elongation alone failed its held-out check"),
}

PARSER_UNITS = [
    # (unit, method_provenance item)
    ("Moon-to-planet distances", "measured Moon-to-planet"),
    ("compound units", "compound cubit-plus-finger distances"),
    ("monthly paths and qualified signs", "qualified signs and monthly paths"),
    ("coordinated signs", "coordinated planet signs"),
    ("month headings and Scorpius alias",
     "scoped month headings and Scorpius alias"),
    ("summary months", "preceding-summary windows"),
]


# ----------------------------------------------------------------- taxonomy

def base_rule(st: S.Statement) -> str:
    """The grammar that read a clause, whether or not through a bracket."""
    return st.parser_rule.removesuffix("_bracket")


def body_class(body: str | None) -> str:
    if body in OUTER:
        return "outer"
    if body in INNER:
        return "inner"
    return str(body)


def type_key(st: S.Statement) -> str:
    """Observation type of a clause: phenomenon, body class and unit.

    The keys are fine enough that the repository's tier rule gives one status
    per key, which `repo_decisive_types` checks on the whole dated corpus.
    """
    unit = "compound" if base_rule(st).endswith("_compound") else "single"
    if st.joint_positions:
        bodies = {p for p, _ in st.joint_positions}
        return ("coordinated signs|"
                + ("outer" if bodies <= OUTER else "with inner") + "|-")
    phenomenon = PHENOMENON.get(st.kind, st.kind)
    if st.kind == "planet_appulse":
        return f"{phenomenon}|{body_class(st.target_planet)}|{unit}"
    if st.kind == "planet_planet_appulse":
        both_outer = st.planet in OUTER and st.target_planet in OUTER
        return f"{phenomenon}|{'outer' if both_outer else 'with inner'}|{unit}"
    if st.kind in {"appulse", "station_appulse"}:
        return f"{phenomenon}|{body_class(st.planet)}|{unit}"
    if st.kind == "eclipse":
        return f"eclipse|{st.planet}|-"
    if st.kind in {"season", "sirius"}:
        return f"{phenomenon}|scheme|-"
    return f"{phenomenon}|{body_class(st.planet)}|-"


def eligibility(key: str) -> str:
    """fixed_decisive, fixed_supporting or gated."""
    phenomenon, body, _ = key.split("|")
    if phenomenon == "eclipse" and body == "lunar":
        return ELIGIBILITY_REASONS["lunar eclipse"][0]
    if body == "moon":
        return ELIGIBILITY_REASONS["moon"][0]
    if body in {"inner", "with inner"}:
        return ELIGIBILITY_REASONS["inner"][0]
    if phenomenon == "last appearance":
        return ELIGIBILITY_REASONS["last appearance"][0]
    return "gated"


def repo_type_decisive(st: S.Statement) -> bool:
    """The repository's tier rule with the clause conditions satisfied.

    Clearing the restored and unobserved marks and supplying a plain month and
    day leaves only what the rule says about the type. Calling the rule itself
    keeps this in step with any later change to it.
    """
    probe = replace(
        st, parser_rule=base_rule(st), restored=False, omitted=False,
        joint_restored=False, joint_omitted=False, month="I", day=1,
        year_offset=0, joint_day=1 if st.joint_positions else None)
    return bool(REPO_IS_DECISIVE(probe))


def repo_decisive_types(statements: list[S.Statement]) -> set[str]:
    status: dict[str, set[bool]] = {}
    for st in statements:
        status.setdefault(type_key(st), set()).add(repo_type_decisive(st))
    mixed = sorted(key for key, values in status.items() if len(values) > 1)
    if mixed:
        raise RuntimeError(
            "type keys too coarse for the repository's tier rule: "
            + ", ".join(mixed))
    return {key for key, values in status.items() if True in values}


def clause_is_secure(st: S.Statement) -> bool:
    """Clause conditions for a type the repository never makes decisive.

    Only reached if a round promotes such a type. It mirrors the clause
    conditions of the repository's tier rule: unrestored text and an observed
    event. The rule does not require a month, so neither does this.
    """
    return (not st.restored and not st.omitted and not st.joint_restored
            and not st.joint_omitted)


# ------------------------------------------------------------- parser units

_REVIEWED: dict | None = None


def reviewed(tablet: str) -> tuple[list[S.Statement], list[str]]:
    """The reviewed clauses of a record, and the origin of each."""
    global _REVIEWED
    if _REVIEWED is None:
        _REVIEWED = clauses.load(clauses.REVIEWED, provenance=True)
    items = _REVIEWED.get(tablet, [])
    return [item[0] for item in items], [item[1] for item in items]


def unit_members(text: str, statements: list[S.Statement],
                 origins: list[str]) -> dict[str, set[int]]:
    """Indices of the clauses each gated parser rule contributes.

    Rejecting a unit drops exactly these clauses. A reading rule contributes
    only parser clauses, because the search finds what the parser misses. The
    three context rules give every clause its month, and they use the removal
    mechanisms of the sensitivity audit.
    """
    def dropped(ablation: str) -> set[int]:
        kept = {id(st) for st in retained(text, statements, statements, ablation, origins)}
        return {i for i, st in enumerate(statements) if id(st) not in kept}

    def chosen(predicate) -> set[int]:
        return {i for i, (st, origin) in enumerate(zip(statements, origins, strict=True))
                if origin == "parser" and predicate(st)}

    qualified = ("beginning of", "middle of", "end of")
    return {
        "Moon-to-planet distances": chosen(
            lambda st: base_rule(st) == "measured_moon_planet"),
        "compound units": chosen(lambda st: base_rule(st).endswith("_compound")),
        "monthly paths and qualified signs": chosen(
            lambda st: base_rule(st) in {"monthly_path", "monthly_path_after"}
            or (base_rule(st) == "in_sign"
                and any(word in " ".join(st.text.split()) for word in qualified))),
        "coordinated signs": chosen(
            lambda st: base_rule(st) in {"coordinated_sign",
                                         "triple_coordinated_sign"}),
        "month headings and Scorpius alias": (
            dropped("extended_month_heading") | dropped("scorpius_alias")),
        "summary months": dropped("summary_month"),
    }


def kept_statements(text: str, statements: list[S.Statement],
                    rejected: list[str], origins: list[str]) -> list[S.Statement]:
    if not rejected:
        return statements
    members = unit_members(text, statements, origins)
    drop = set().union(*(members[unit] for unit in rejected))
    return [st for i, st in enumerate(statements) if i not in drop]


# --------------------------------------------------------------- tolerances

def tolerance_class(st: S.Statement) -> str | None:
    if st.kind not in MEASURED:
        return None
    return "moon" if st.planet == "moon" else "planet"


def night_residual(st: S.Statement, year_astro: int) -> float | None:
    """|computed - stated| distance at the best moment of the stated night.

    The separation is the one the check tests: longitude relative to the star
    or the target, in the stated direction, and longitude proximity alone for
    above and below. Only darkness on the stated night and the night either
    side is searched, so the Moon's own motion cannot absorb the residual.
    """
    if tolerance_class(st) is None or st.offset is None:
        return None
    if st.day is None or not st.month or "|" in st.month:
        return None
    civil = cal.bab_to_jd(year_astro + st.year_offset, st.month, st.day)
    if civil is None:
        return None
    jds = np.concatenate([
        np.arange(civil + k + NIGHT[0], civil + k + NIGHT[1] + 1e-9,
                  NIGHT_STEP_DAYS)
        for k in NIGHT_SLACK_DAYS])
    if st.kind == "appulse":
        star = sky.star(st.star)
        if star is None:
            return None
        want = (star[0] + st.offset) % 360.0
        gap = np.abs((sky.lon(st.planet, jds) - want + 180.0) % 360.0 - 180.0)
    else:
        relative = ((sky.lon(st.planet, jds) - sky.lon(st.target_planet, jds)
                     + 180.0) % 360.0 - 180.0)
        gap = np.abs(relative - st.offset)
    return float(gap.min())


def grid_tolerance(residuals: list[float]) -> float:
    values = np.asarray(residuals, float)
    for value in TOL_GRID:
        if (values <= value + 1e-12).mean() >= TOL_QUANTILE - 1e-12:
            return value
    return TOL_GRID[-1]


def quantile_interval(residuals: list[float], chosen: float) -> list[float]:
    """Quantiles in (lo, hi] that give the same grid value."""
    values = np.asarray(residuals, float)
    share = {value: float((values <= value + 1e-12).mean()) for value in TOL_GRID}
    below = [share[v] for v in TOL_GRID if v < chosen]
    return [round(max(below) if below else 0.0, 4), round(share[chosen], 4)]


# ------------------------------------------------------------------ records

def load_texts() -> pd.DataFrame:
    """Identifier, volume and text of every ADART I-III record. No dates."""
    texts = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "project", "text"])
    volume = {project: name for name, project in VOLUMES.items()}
    texts = texts[texts.project.isin(volume)].copy()
    texts["volume"] = texts.project.map(volume)
    texts["text"] = texts.text.fillna("")
    return texts.drop(columns="project").reset_index(drop=True)


def published_years(tablets: list[str]) -> dict[str, int]:
    """Published years, in astronomical numbering, of the named records only."""
    wanted = set(tablets)
    dates = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "bce_first"])
    dates = dates[dates.tablet.isin(wanted) & dates.bce_first.notna()]
    return {row.tablet: yr.bce_to_astro(int(row.bce_first))
            for row in dates.itertuples(index=False)}


def benchmark() -> pd.DataFrame:
    """The frozen 179 records and their 112 / 67 labels."""
    population = pd.read_parquet(
        HERE / "data" / "benchmark" / "benchmark_records.parquet",
        columns=["tablet", "project", "tool_benchmark_eligible",
                 "tool_year_bearing"])
    return population[population.tool_benchmark_eligible].reset_index(drop=True)


# ---------------------------------------------------------------- settings

@contextmanager
def round_settings(choices: dict | None):
    """Apply one round's tolerances and decisive types to the tools.

    ``None`` leaves the repository exactly as it is. Every module value is put
    back afterwards, as in the sensitivity audit.
    """
    saved = (S.TOL_MOON, S.TOL_PLANET, S.TOL_PLANET_PAIR, S.is_decisive)
    try:
        if choices is not None:
            S.TOL_MOON = choices["tolerance"]["moon"]
            # One planet tolerance covers every measured planet relation,
            # including stations against a star and planet pairs.
            S.TOL_PLANET = S.TOL_PLANET_PAIR = choices["tolerance"]["planet"]
            decisive = frozenset(choices["decisive_types"])
            repo = frozenset(choices["repo_decisive_types"])

            def round_is_decisive(st: S.Statement) -> bool:
                key = type_key(st)
                if key not in decisive:
                    return False
                if key in repo:
                    return REPO_IS_DECISIVE(st)
                return clause_is_secure(st)

            S.is_decisive = round_is_decisive
        yield
    finally:
        S.TOL_MOON, S.TOL_PLANET, S.TOL_PLANET_PAIR, S.is_decisive = saved


# ---------------------------------------------------------------------- fit

def gate(agree: int, testable: int) -> bool:
    return (testable >= GATE_MIN_TESTABLE
            and agree >= GATE_MIN_AGREEMENT * testable - 1e-9)


def gate_interval(rows: list[dict]) -> list[float]:
    """Agreement thresholds in (lo, hi] that leave every decision unchanged."""
    counted = [row for row in rows if row["testable"] >= GATE_MIN_TESTABLE]
    passing = [row["agreement"] for row in counted if row["passes_gate"]]
    failing = [row["agreement"] for row in counted if not row["passes_gate"]]
    return [round(max(failing), 4) if failing else 0.0,
            round(min(passing), 4) if passing else 1.0]


def count_interval(rows: list[dict]) -> list[int]:
    """Minimum counts in [lo, hi] that leave every decision unchanged."""
    passing = [row["testable"] for row in rows if row["passes_gate"]]
    short = [row["testable"] for row in rows
             if 0 < row["testable"] < GATE_MIN_TESTABLE
             and row["agree"] >= GATE_MIN_AGREEMENT * row["testable"] - 1e-9]
    return [max(short) + 1 if short else 1, min(passing) if passing else 10**6]


def fit_round(spec: dict, texts: pd.DataFrame,
              repo_types: set[str]) -> dict:
    """Make the round's choices from its training volumes alone."""
    started = time.time()
    train = texts[texts.volume.isin(spec["train"])]
    leaked = set(train.volume) & (set(spec["test"]) - set(spec["train"]))
    if leaked:
        raise RuntimeError(f"test volume in training data: {leaked}")
    years = published_years(list(train.tablet))
    train = train[train.tablet.isin(years) & train.text.str.strip().ne("")]
    parsed = [(row.tablet, row.text, years[row.tablet], S.months_in(row.text),
               *reviewed(row.tablet)) for row in train.itertuples(index=False)]

    # 1. Tolerances, from every measured relation of the reviewed clauses.
    residuals: dict[str, list[float]] = {"moon": [], "planet": []}
    for _, _, year, _, statements, _ in parsed:
        for st in statements:
            value = night_residual(st, year)
            if value is not None:
                residuals[tolerance_class(st)].append(value)
    tolerance = {cls: grid_tolerance(values) for cls, values in residuals.items()}
    tolerance_detail = {
        cls: {"residuals": len(values), "tolerance": tolerance[cls],
              "share_within": {str(v): round(float(
                  (np.asarray(values) <= v + 1e-12).mean()), 4) for v in TOL_GRID},
              "quantile_interval_same_choice": quantile_interval(
                  values, tolerance[cls])}
        for cls, values in residuals.items()}
    choices = {"tolerance": tolerance, "decisive_types": [],
               "repo_decisive_types": sorted(repo_types)}

    # 2. and 3. Forward verdicts at the published years, at these tolerances.
    unit_counts = {unit: [0, 0] for unit, _ in PARSER_UNITS}
    verdicts = []
    with round_settings({**choices, "decisive_types": sorted(repo_types)}):
        for _, text, year, months, statements, origins in parsed:
            results = [S.check(st, year, months).result for st in statements]
            members = unit_members(text, statements, origins)
            for unit, indices in members.items():
                for i in indices:
                    if results[i] != "untestable":
                        unit_counts[unit][0] += results[i] == "agree"
                        unit_counts[unit][1] += 1
            verdicts.append((members, statements, results))

    unit_rows = []
    for unit, item in PARSER_UNITS:
        agree, testable = unit_counts[unit]
        unit_rows.append({
            "unit": unit, "provenance_item": item, "testable": testable,
            "agree": agree,
            "agreement": round(agree / testable, 4) if testable else None,
            "passes_gate": gate(agree, testable)})
    rejected = [row["unit"] for row in unit_rows if not row["passes_gate"]]

    type_counts: dict[str, list[int]] = {}
    for members, statements, results in verdicts:
        drop = set().union(*(members[unit] for unit in rejected)) if rejected else set()
        for i, st in enumerate(statements):
            if i in drop:
                continue
            counts = type_counts.setdefault(type_key(st), [0, 0])
            if results[i] != "untestable":
                counts[0] += results[i] == "agree"
                counts[1] += 1
    keys = sorted(set(type_counts) | repo_types)
    type_rows = []
    for key in keys:
        agree, testable = type_counts.get(key, [0, 0])
        status = eligibility(key)
        passes = gate(agree, testable)
        decisive = (status == "fixed_decisive"
                    or (status == "gated" and passes))
        type_rows.append({
            "type": key, "eligibility": status, "testable": testable,
            "agree": agree,
            "agreement": round(agree / testable, 4) if testable else None,
            "passes_gate": passes, "decisive": decisive,
            "repo_decisive": key in repo_types})
    decisive_types = sorted(row["type"] for row in type_rows if row["decisive"])
    choices.update({
        "decisive_types": decisive_types,
        "rejected_units": rejected,
        "demoted_types": sorted(repo_types - set(decisive_types)),
        "promoted_types": sorted(set(decisive_types) - repo_types),
        "training_records": len(parsed),
        "tolerance_detail": tolerance_detail,
        "parser_units": unit_rows,
        "types": type_rows,
        "gate_interval_units": gate_interval(unit_rows),
        "gate_interval_types": gate_interval(
            [row for row in type_rows if row["eligibility"] == "gated"]),
        "count_interval_units": count_interval(unit_rows),
        "count_interval_types": count_interval(
            [row for row in type_rows if row["eligibility"] == "gated"]),
        "seconds": round(time.time() - started, 1),
    })
    return choices


# ------------------------------------------------------------------ scoring

def score_chunk(choices: dict | None, rejected: list[str],
                records: list[tuple[str, str]]) -> list[dict]:
    """Candidate sets for some records, with no published year in sight."""
    years = [int(year) for year in cal.candidate_years()]
    rows = []
    with round_settings(choices):
        for tablet, text in records:
            statements, origins = reviewed(tablet)
            statements = kept_statements(text, statements, rejected, origins)
            scores = S.candidate_scores(statements, S.months_in(text), years,
                                        weighting="skill")
            levels = S.evidence_levels(scores)
            rows.append({
                "tablet": tablet,
                "n_statements": len(statements),
                "active_decisive": int(
                    (scores.decisive & (scores.weights > 0)).sum()),
                "strict_years_astro": [int(y) for y in scores.years[scores.strict]],
                "safety_years_astro": [int(y) for y in scores.years[scores.safety]],
                "strict_count": int(scores.strict.sum()),
                "safety_count": int(scores.safety.sum()),
                "first_group_size": len(levels[0]["years_astro"]),
                "evidence_group_count": len(levels),
                "level_years_json": json.dumps(
                    [level["years_astro"] for level in levels]),
            })
    return rows


def timed_score_chunk(choices: dict | None, rejected: list[str],
                      records: list[tuple[str, str]]) -> tuple[list[dict], float]:
    started = time.time()
    return score_chunk(choices, rejected, records), time.time() - started


def scoring_choices(choices: dict | None) -> dict | None:
    """Only what scoring reads, so each task ships a small payload."""
    if choices is None:
        return None
    return {key: choices[key] for key in
            ("tolerance", "decisive_types", "repo_decisive_types",
             "rejected_units")}


def pmap(executor, fn, *iterables) -> list:
    """Map in the shared pool when there is one, in this process otherwise."""
    if executor is None:
        return list(map(fn, *iterables))
    return list(executor.map(fn, *iterables))


def score_jobs(executor, jobs: list[tuple[str, dict | None, list]]) -> dict:
    """Score several rounds at once in one pool.

    The rounds are independent, so their records are cut into small chunks
    and interleaved; every round then finishes in about the time of one.
    """
    tasks = []
    for label, choices, records in jobs:
        for index, start in enumerate(range(0, len(records), CHUNK_RECORDS)):
            tasks.append((index, label, scoring_choices(choices),
                          choices["rejected_units"] if choices else [],
                          records[start:start + CHUNK_RECORDS]))
    tasks.sort(key=lambda task: task[0])
    results = pmap(executor, timed_score_chunk, [t[2] for t in tasks],
                   [t[3] for t in tasks], [t[4] for t in tasks])
    out = {label: {"rows": [], "cpu_seconds": 0.0} for label, _, _ in jobs}
    for task, (rows, seconds) in zip(tasks, results):
        out[task[1]]["rows"].extend(rows)
        out[task[1]]["cpu_seconds"] += seconds
    for label, _, records in jobs:
        order = {tablet: i for i, (tablet, _) in enumerate(records)}
        out[label]["rows"].sort(key=lambda row: order[row["tablet"]])
    return out


def exposure(choices: dict, records: pd.DataFrame) -> dict:
    """How much of the test records a round's choices touch, without scoring.

    Counts clauses only: which lose or gain the decisive tier, which a rejected
    rule removes, and which measured relations a changed tolerance can move.
    No candidate year and no published year enters.
    """
    counts = {"records": len(records),
              "records_with_decisive_clauses": int(records.tool_year_bearing.sum()),
              "clauses": 0, "clauses_kept": 0, "decisive_repo": 0,
              "decisive_round": 0, "records_with_changed_decisive_clauses": 0,
              "moon_relations": 0, "planet_relations": 0}
    for tablet, text in zip(records.tablet, records.text):
        statements, origins = reviewed(tablet)
        kept = kept_statements(text, statements, choices["rejected_units"], origins)
        repo = {(st.kind, st.source_start, st.planet) for st in statements
                if REPO_IS_DECISIVE(st)}
        with round_settings(choices):
            ours = {(st.kind, st.source_start, st.planet) for st in kept
                    if S.is_decisive(st)}
        counts["clauses"] += len(statements)
        counts["clauses_kept"] += len(kept)
        counts["decisive_repo"] += len(repo)
        counts["decisive_round"] += len(ours)
        counts["records_with_changed_decisive_clauses"] += repo != ours
        for st in kept:
            if st.kind == "station_appulse" or tolerance_class(st) == "planet":
                counts["planet_relations"] += 1
            elif tolerance_class(st) == "moon":
                counts["moon_relations"] += 1
    return counts


# --------------------------------------------------------------- evaluation

def evaluate(rows: pd.DataFrame) -> pd.DataFrame:
    """Join the published years, after every candidate set is frozen."""
    truth = published_years(list(rows.tablet))
    rows = rows.copy()
    rows["truth_astro"] = rows.tablet.map(truth)
    if rows.truth_astro.isna().any():
        raise RuntimeError("benchmark record without a published year")
    rows["truth_astro"] = rows.truth_astro.astype(int)
    rows["truth_in_strict"] = [
        t in set(s) for t, s in zip(rows.truth_astro, rows.strict_years_astro)]
    rows["truth_in_safety"] = [
        t in set(s) for t, s in zip(rows.truth_astro, rows.safety_years_astro)]
    rows["truth_group_rank"] = [
        next(rank for rank, years in enumerate(json.loads(levels), start=1)
             if t in years)
        for t, levels in zip(rows.truth_astro, rows.level_years_json)]
    return rows


def table_row(label: str, rows: pd.DataFrame) -> dict:
    """One row of Table S6, for the records with decisive clauses."""
    active = rows[rows.frozen_active]
    n = len(active)

    def rate(mask) -> dict:
        count = int(mask.sum())
        return {"n": count, "percent": round(100 * count / n, 1) if n else None}

    return {
        "test_volume": label, "records": n,
        "first_group": rate(active.truth_group_rank.eq(1)),
        "first_three_groups": rate(active.truth_group_rank.le(3)),
        "strict_set": rate(active.truth_in_strict),
        "median_strict_set": (float(active.strict_count.median()) if n else None),
    }


def compare_with_repo(ours: pd.DataFrame, repo: pd.DataFrame) -> dict:
    """Record-by-record identity of the all-volume round and the pipeline."""
    columns = ["strict_years_astro", "safety_years_astro", "level_years_json",
               "active_decisive", "n_statements"]

    def canon(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [int(item) for item in value]
        return value.item() if isinstance(value, np.generic) else value

    merged = ours.merge(repo, on="tablet", suffixes=("", "_repo"),
                        validate="one_to_one")
    differ = []
    for column in columns:
        same = np.array([canon(a) == canon(b) for a, b in
                         zip(merged[column], merged[f"{column}_repo"])], bool)
        differ.extend(merged.tablet[~same].tolist())
    return {"records": len(merged), "identical": not differ,
            "differing_records": sorted(set(differ))}


# --------------------------------------------------------------------- main

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path,
        default=HERE / "data" / "benchmark" / "held_out_runs.parquet")
    parser.add_argument(
        "--summary-out", type=Path,
        default=HERE / "data" / "benchmark" / "held_out_summary.json")
    # One pool serves every stage and every round. Each worker holds its own
    # ephemeris, about 0.8 GB.
    parser.add_argument("--workers", type=int,
                        default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument(
        "--run-held-out", action="store_true",
        help="score the held-out rounds; off by default until the reading "
             "of the test is fixed")
    parser.add_argument("--choices-only", action="store_true",
                        help="compute every round's choices and score nothing")
    parser.add_argument("--skip-repo-check", action="store_true",
                        help="do not rerun the unmodified pipeline")
    args = parser.parse_args()
    if args.run_held_out and args.choices_only:
        parser.error("--run-held-out and --choices-only exclude each other")
    started = time.time()

    texts = load_texts()
    population = benchmark()
    population["volume"] = population.project.map(
        {project: name for name, project in VOLUMES.items()})
    population = population.merge(texts[["tablet", "volume", "text"]],
                                  on=["tablet", "volume"], validate="one_to_one")
    population = population[population.text.str.strip().ne("")]

    corpus = [st for tablet in texts.tablet for st in reviewed(tablet)[0]]
    repo_types = repo_decisive_types(corpus)

    summary = {
        "settings": {
            "gate_min_agreement": GATE_MIN_AGREEMENT,
            "gate_min_testable": GATE_MIN_TESTABLE,
            "tolerance_grid_deg": list(TOL_GRID),
            "tolerance_quantile": TOL_QUANTILE,
            "night_jd_relative_to_civil_day": list(NIGHT),
            "night_slack_days": list(NIGHT_SLACK_DAYS),
            "repo_tolerances": {"moon": S.TOL_MOON, "planet": S.TOL_PLANET,
                                "planet_pair": S.TOL_PLANET_PAIR},
            "eligibility_reasons": {k: {"status": v[0], "reason": v[1]}
                                    for k, v in ELIGIBILITY_REASONS.items()},
            "workers": args.workers,
            "close_points": CLOSE_POINTS,
        },
        "held_out_scored": bool(args.run_held_out),
        "rounds": [],
        "hashes": {name: sha256(HERE / "tools" / f"{name}.py")
                   for name in ("statements", "cal", "sky", "eclipses")}
        | {"audit_held_out": sha256(Path(__file__)),
           "reviewed_clauses": sha256(clauses.REVIEWED)},
    }
    tests = {spec["round"]: population[population.volume.isin(spec["test"])]
             for spec in ROUNDS}

    executor = (ProcessPoolExecutor(max_workers=args.workers)
                if args.workers > 1 else None)
    try:
        print("choices of each round, from its training volumes only")
        fitted = pmap(executor, fit_round, ROUNDS, [texts] * len(ROUNDS),
                      [repo_types] * len(ROUNDS))
        fits = {spec["round"]: choices for spec, choices in zip(ROUNDS, fitted)}
        for spec in ROUNDS:
            choices = fits[spec["round"]]
            units = ", ".join(choices["rejected_units"]) or "none"
            print(f"  {spec['round']:<12} trained on "
                  f"{'+'.join(spec['train']):<22}"
                  f" {choices['training_records']:>3} records  "
                  f"tolerances {choices['tolerance']['moon']}/"
                  f"{choices['tolerance']['planet']} deg  "
                  f"rejected rules: {units}  "
                  f"demoted: {', '.join(choices['demoted_types']) or 'none'}  "
                  f"promoted: {', '.join(choices['promoted_types']) or 'none'}  "
                  f"({choices['seconds']} s)")

        every = fits["all volumes"]
        for name in HELD_OUT:
            mine = fits[name]
            mine["same_choices_as_all_volumes"] = (
                mine["tolerance"] == every["tolerance"]
                and mine["rejected_units"] == every["rejected_units"]
                and mine["decisive_types"] == every["decisive_types"])
        reproduces = {
            "tolerances": (every["tolerance"]["moon"] == S.TOL_MOON
                           and every["tolerance"]["planet"] == S.TOL_PLANET
                           and S.TOL_PLANET_PAIR == S.TOL_PLANET),
            "parser_rules": not every["rejected_units"],
            "decisive_types": set(every["decisive_types"]) == repo_types,
        }
        summary["all_volumes_reproduces_repo_choices"] = reproduces
        print(f"  all-volume choices reproduce the repository: {reproduces}")

        exposures = pmap(executor, exposure,
                         [fits[spec["round"]] for spec in ROUNDS],
                         [tests[spec["round"]] for spec in ROUNDS])
        print("\nwhat each held-out round's choices touch in its benchmark "
              "records (clauses only, nothing scored)")
        for spec, touched in zip(ROUNDS, exposures):
            entry = {"round": spec["round"], "train": spec["train"],
                     "test": spec["test"], "choices": fits[spec["round"]],
                     "exposure": touched, "scored": False, "table_row": None}
            summary["rounds"].append(entry)
            if spec["round"] in HELD_OUT:
                print(f"  {spec['round']:<12} test records "
                      f"{touched['records']:>3} "
                      f"({touched['records_with_decisive_clauses']} with "
                      f"decisive clauses); decisive clauses "
                      f"{touched['decisive_repo']} -> {touched['decisive_round']}"
                      f" in {touched['records_with_changed_decisive_clauses']} "
                      f"records; clauses removed by rejected rules "
                      f"{touched['clauses'] - touched['clauses_kept']}")

        # The held-out rounds are scored only on request. All rounds that are
        # scored, and the unmodified pipeline, share the pool at once.
        jobs = []
        if not args.choices_only:
            for spec in ROUNDS:
                if spec["round"] in HELD_OUT and not args.run_held_out:
                    continue
                records = tests[spec["round"]]
                jobs.append((spec["round"], fits[spec["round"]],
                             list(zip(records.tablet, records.text))))
            if not args.skip_repo_check:
                records = tests["all volumes"]
                jobs.append((REPO_JOB, None,
                             list(zip(records.tablet, records.text))))
        scoring_started = time.time()
        scored = score_jobs(executor, jobs) if jobs else {}
        summary["scoring_wall_seconds"] = round(time.time() - scoring_started, 1)
    finally:
        if executor is not None:
            executor.shutdown()

    # Published years of the test records enter only from here on.
    runs = []
    for entry in summary["rounds"]:
        if entry["round"] not in scored:
            continue
        records = tests[entry["round"]]
        rows = pd.DataFrame(scored[entry["round"]]["rows"])
        entry["scoring_cpu_seconds"] = round(
            scored[entry["round"]]["cpu_seconds"], 1)
        rows = rows.merge(records[["tablet", "volume", "tool_year_bearing"]],
                          on="tablet", validate="one_to_one").rename(
            columns={"tool_year_bearing": "frozen_active"})
        rows.insert(0, "round", entry["round"])
        if entry["round"] == "all volumes" and REPO_JOB in scored:
            repo = pd.DataFrame(scored[REPO_JOB]["rows"])
            entry["repo_check"] = compare_with_repo(rows, repo)
            entry["repo_check"]["cpu_seconds"] = round(
                scored[REPO_JOB]["cpu_seconds"], 1)
            repo = evaluate(repo.merge(
                records[["tablet", "tool_year_bearing"]], on="tablet").rename(
                columns={"tool_year_bearing": "frozen_active"}))
            entry["repo_table_row"] = table_row(REPO_JOB, repo)
        rows = evaluate(rows)
        entry["scored"] = True
        # The all-volume round is not broken down by volume. A round whose
        # choices equal the all-volume choices dates its records identically,
        # so a per-volume row of this round can be its held-out row.
        entry["table_row"] = table_row(entry["round"], rows)
        runs.append(rows)

    if args.run_held_out:
        pooled = pd.concat([run for run in runs
                            if run["round"].iloc[0] in HELD_OUT])
        summary["all_three_rounds"] = table_row("all three rounds", pooled)
        reference = next(entry["table_row"] for entry in summary["rounds"]
                         if entry["round"] == "all volumes")
        summary["reading"] = {
            measure: {
                "all_volumes_percent": reference[measure]["percent"],
                "held_out_percent": summary["all_three_rounds"][measure]["percent"],
                "drop_points": round(reference[measure]["percent"]
                                     - summary["all_three_rounds"][measure]["percent"], 1),
                "close": (reference[measure]["percent"]
                          - summary["all_three_rounds"][measure]["percent"]) <= CLOSE_POINTS,
            }
            for measure in ("first_group", "strict_set")}
    summary["main_text"] = MAIN_TEXT
    summary["seconds"] = round(time.time() - started, 1)

    if runs:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(runs, ignore_index=True).to_parquet(args.out, index=False)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print("\nTable S6 rows that were run")
    for entry in summary["rounds"]:
        row = entry["table_row"]
        if row is None:
            print(f"  {entry['round']:<12} not scored")
            continue
        print(f"  {entry['round']:<12} records {row['records']:>3}  first group "
              f"{row['first_group']['n']:>3} ({row['first_group']['percent']}%)  "
              f"first three {row['first_three_groups']['n']:>3} "
              f"({row['first_three_groups']['percent']}%)  strict "
              f"{row['strict_set']['n']:>3} ({row['strict_set']['percent']}%)  "
              f"median strict {row['median_strict_set']}")
        if "repo_check" in entry:
            check = entry["repo_check"]
            print(f"  {'':<12} identical to the unmodified pipeline on "
                  f"{check['records']} records: {check['identical']}")
    if "all_three_rounds" in summary:
        row = summary["all_three_rounds"]
        print(f"  all three rounds: {json.dumps(row)}")
        for measure, reading in summary["reading"].items():
            print(f"  {measure}: {reading['all_volumes_percent']}% with all volumes, "
                  f"{reading['held_out_percent']}% held out, a drop of "
                  f"{reading['drop_points']} points; close: {reading['close']}")
    print(f"\n{summary['seconds']} s in all, "
          f"{summary['scoring_wall_seconds']} s scoring on {args.workers} "
          f"workers; summary -> {args.summary_out}")
    if runs:
        print(f"runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
