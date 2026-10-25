#!/usr/bin/env python3
"""Retrospective sensitivity and ablation audit of backward inference.

All configurations use the same text-defined population and its reviewed
clauses. Candidate sets are computed before reference dates or stratification
metadata are loaded. These are robustness checks on already-seen data, not
fresh test-set estimates.
"""

from __future__ import annotations

import argparse
import re
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import json

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, clauses, sky, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402


LEGACY_MONTH_RE = re.compile(
    r"(?:\bMonth|\bMon\s*\]\s*th)\s*\]?\s*"
    r"(XII(?:2|₂)?|VI(?:2|₂)?|XI|IX|VIII|VII|VI|IV|V|"
    r"III|II|I|X)(?:\s*\[\s*(I{1,3})(?=\s*[\],.]))?(?=\W|$)")

AGENT_RULES = {
    "measured_moon_planet", "measured_moon_planet_compound",
    "measured_normal_star_compound", "monthly_path", "monthly_path_after",
    "coordinated_sign",
    "measured_station_normal_star", "measured_station_normal_star_compound",
    "measured_planet_planet", "measured_planet_planet_compound",
}

CONFIGS = [
    {"config": "baseline", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "none"},
    {"config": "tolerance_narrow", "moon_tol": 2.0,
     "planet_tol": 1.5, "weighting": "skill", "ablation": "none"},
    {"config": "tolerance_wide", "moon_tol": 4.0,
     "planet_tol": 3.0, "weighting": "skill", "ablation": "none"},
    {"config": "no_agent_suggested_rules", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "agent_rules"},
    {"config": "no_station_relative_to_star", "moon_tol": 3.0, "planet_tol": 2.0, "weighting": "skill",
     "ablation": "station_relation"},
    {"config": "no_planet_to_planet_distance", "moon_tol": 3.0, "planet_tol": 2.0, "weighting": "skill",
     "ablation": "planet_pair"},
    {"config": "no_restored_month_context", "moon_tol": 3.0, "planet_tol": 2.0, "weighting": "skill",
     "ablation": "restored_month_context"},
    {"config": "no_extended_month_heading", "moon_tol": 3.0, "planet_tol": 2.0, "weighting": "skill",
     "ablation": "extended_month_heading"},
    # Refusing to read through the editors' restoration brackets, which is what
    # a grammar written for clean text does by default.
    {"config": "no_first_appearance", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "first_appearance"},
    {"config": "no_acronychal_rising", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "acronychal"},
    {"config": "no_restoration_reading", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "restoration_reading"},
    {"config": "no_scorpius_alias", "moon_tol": 3.0, "planet_tol": 2.0, "weighting": "skill",
     "ablation": "scorpius_alias"},
    {"config": "no_inferred_summary_month", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "summary_month"},
    # One cubit to two degrees is the convention the whole distance grammar
    # rests on, and the value measured from observed appulses in the modern
    # literature is nearer 2.2. Rescaling every converted distance asks whether
    # the result depends on the convention.
    # The ephemeris is evaluated at civil-date Julian days while the kernel
    # expects dynamical time. The difference is Delta T, four hours or so here,
    # and this asks whether any reported class depends on it.
    {"config": "delta_t_applied", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "delta_t"},
    {"config": "cubit_at_2_2_degrees", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "cubit_scale"},
    # Britton gives the zodiac's constant term to about 0.1 degree. Moving it
    # by half a degree either way asks whether any result rests on a planet
    # that sits close to a sign boundary.
    {"config": "britton_offset_minus_half", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "zodiac_minus"},
    {"config": "britton_offset_plus_half", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "zodiac_plus"},
    {"config": "no_mars_family", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "mars"},
    {"config": "no_jupiter_family", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "jupiter"},
    {"config": "no_saturn_family", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "saturn"},
    {"config": "no_sirius_scheme", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "sirius"},
    {"config": "no_sun_family", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "sun"},
    {"config": "no_lunar_eclipse_family", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "lunar_eclipse"},
    # Not a defensive check but a bridge to ADART IV, where 68 of 867 entries
    # preserve a month heading against 106 of the 112 active I--III tablets.
    # Stripping the calendar context from I--III asks what those undated
    # fragments would look like if they were otherwise as rich.
    {"config": "no_month_context", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "month_context"},
    # Not a capability but an ordering choice, and the one that decides whether
    # the method commits against the published chronology on X102732.
    {"config": "contradictions_before_information", "moon_tol": 3.0, "planet_tol": 2.0, "weighting": "skill",
     "ablation": "order_contradictions_first"},
    {"config": "contradictions_bound_the_leading_set", "moon_tol": 3.0, "planet_tol": 2.0, "weighting": "skill",
     "ablation": "frontier_penalises_contradictions"},
    {"config": "count_weighting", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "count", "ablation": "none"},
    {"config": "information_weighting", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "information", "ablation": "none"},
    {"config": "independent_joint_atoms", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "independent_joint_atoms"},
    # "Above" and "below" state a latitude that no rule tests, yet such a
    # distance of Mars, Jupiter or Saturn to a star, or of the Moon to one of
    # them, can be decisive. This keeps every such distance supporting.
    {"config": "longitude_only_distances_supporting", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "latitudinal_supporting"},
    # The tiers, clocks, strict and safety sets and ranking of the baseline,
    # with one unit per agreeing clause in place of its bits. It isolates the
    # weighting by bits, which count_weighting changes together with the tiers.
    {"config": "equal_weighting_with_tiers", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "equal", "ablation": "none"},
    # A measured relation of the Moon tested only over the night that opens its
    # Babylonian day, or the night after, in place of two and a half days.
    {"config": "moon_night_window", "moon_tol": 3.0,
     "planet_tol": 2.0, "weighting": "skill", "ablation": "moon_night"},
]


MEASURED_KINDS = {"appulse", "planet_appulse", "station_appulse", "planet_planet_appulse"}
DAY_KINDS = {"eclipse", "season", "sirius", "acronychal"}


def context_restored(text: str, st: S.Statement) -> bool:
    """The restored mark that the headings give a clause at its passage."""
    if st.kind in MEASURED_KINDS:
        return S.appulse_context(text, st.source_start)[3]
    if st.kind in DAY_KINDS:
        return S.eclipse_context(text, st.source_start)[3]
    return S.positional_context(text, st.source_start)[2]


def without_restored_month_context(text: str, items: list[tuple]) -> list[S.Statement]:
    """The reviewed clauses as they would be without the restored month context.

    A parser clause takes the restored mark of the parser without this rule,
    plus any mark that the review added. An added clause loses the mark that
    its month heading gave it, unless its own passage holds a bracket.
    """
    parsed = S.extract(text)
    old = S.PROPAGATE_RESTORED_MONTH_CONTEXT
    out = []
    try:
        S.PROPAGATE_RESTORED_MONTH_CONTEXT = False
        legacy = S.extract(text)
        for st, origin, index, _review in items:
            if origin == "parser":
                restored = (legacy[index].restored
                            or (st.restored and not parsed[index].restored))
            else:
                S.PROPAGATE_RESTORED_MONTH_CONTEXT = True
                now = context_restored(text, st)
                S.PROPAGATE_RESTORED_MONTH_CONTEXT = False
                then = context_restored(text, st)
                passage = text[st.source_start:st.source_end]
                restored = (st.restored if now == then
                            else then or "[" in passage or "]" in passage)
            out.append(replace(st, restored=bool(restored)))
    finally:
        S.PROPAGATE_RESTORED_MONTH_CONTEXT = old
    return S.bind_joint_positions(text, out)


def retained(text: str, statements: list[S.Statement],
             legacy_month_statements: list[S.Statement], ablation: str,
             origins: list[str] | None = None):
    if ablation == "restoration_reading":
        return [st for st in statements
                if not st.parser_rule.endswith("_bracket")]
    if ablation in ("none", "order_contradictions_first",
                    "frontier_penalises_contradictions",
                    "zodiac_minus", "zodiac_plus",
                    "latitudinal_supporting", "moon_night"):
        return statements
    if ablation == "restored_month_context":
        return legacy_month_statements
    if ablation == "scorpius_alias":
        return [st for st in statements if "Scorpius" not in
                text[st.source_start:st.source_end]]
    if ablation == "extended_month_heading":
        def legacy_governs(st):
            current = [m for m in S.month_matches(text, 0, st.source_start)]
            legacy = [m for m in LEGACY_MONTH_RE.finditer(
                text, 0, st.source_start)]
            return (not current or
                    (legacy and legacy[-1].start() == current[-1].start()))
        return [st for st in statements if legacy_governs(st)]
    if ablation == "month_context":
        # Keep every clause but forget which month and day governed it, so a
        # statement is tested against the tablet's whole span instead of one
        # month. Nothing is deleted; only the calendar context is withdrawn.
        return [replace(st, month=None, day=None) for st in statements]
    if ablation == "cubit_scale":
        # offset is the recorded distance already converted, so rescaling it is
        # exactly a change in how many degrees a cubit is worth.
        return [st if st.offset is None
                else replace(st, offset=st.offset * (2.2 / 2.0))
                for st in statements]
    if ablation == "agent_rules":
        # The rules are parser rules. The search finds what the parser misses,
        # so its additions stay.
        origins = origins or ["parser"] * len(statements)
        return [st for st, origin in zip(statements, origins, strict=True)
                if origin == "search" or st.parser_rule not in AGENT_RULES]
    if ablation == "summary_month":
        return [st for st in statements
                if not (st.source_start >= 0
                        and S.preceding_summary_month(text, st.source_start)
                        is not None)]
    if ablation == "station_relation":
        return [st for st in statements
                if not st.parser_rule.startswith(
                    "measured_station_normal_star")]
    if ablation == "planet_pair":
        return [st for st in statements
                if not st.parser_rule.startswith("measured_planet_planet")]
    if ablation == "first_appearance":
        return [st for st in statements if st.kind != "first_app"]
    if ablation == "acronychal":
        return [st for st in statements if st.kind != "acronychal"]
    if ablation == "independent_joint_atoms":
        return [replace(st, joint_positions=(), joint_source_start=-1,
                        joint_source_end=-1, joint_day=None,
                        joint_restored=False, joint_omitted=False)
                for st in statements]
    if ablation == "sirius":
        return [st for st in statements if st.kind != "sirius"]
    if ablation == "sun":
        return [st for st in statements
                if st.kind not in ("season", "sirius")]
    if ablation == "lunar_eclipse":
        return [st for st in statements
                if not (st.kind == "eclipse" and st.planet == "lunar")]
    return [st for st in statements
            if st.planet != ablation and st.target_planet != ablation]


def infer(config: dict, population: list[tuple]) -> list[dict]:
    old_moon, old_planet = S.TOL_MOON, S.TOL_PLANET
    old_offset = sky.TIME_OFFSET_DAYS
    old_zodiac = sky.BABYLONIAN_OFFSET_YEAR0_DEG
    sky.BABYLONIAN_OFFSET_YEAR0_DEG = old_zodiac + {
        "zodiac_minus": -0.5, "zodiac_plus": 0.5}.get(config["ablation"], 0.0)
    sky.TIME_OFFSET_DAYS = (sky.delta_t_days(350.0)
                            if config["ablation"] == "delta_t" else 0.0)
    old_order = S.ORDER_CONTRADICTIONS_FIRST
    S.TOL_MOON = config["moon_tol"]
    S.TOL_PLANET = config["planet_tol"]
    old_frontier = S.FRONTIER_PENALISES_CONTRADICTIONS
    old_latitudinal = S.LATITUDINAL_DISTANCES_DECISIVE
    old_night = S.MOON_NIGHT_WINDOWS
    S.ORDER_CONTRADICTIONS_FIRST = (
        config["ablation"] == "order_contradictions_first")
    S.FRONTIER_PENALISES_CONTRADICTIONS = (
        config["ablation"] == "frontier_penalises_contradictions")
    S.LATITUDINAL_DISTANCES_DECISIVE = (
        config["ablation"] != "latitudinal_supporting")
    S.MOON_NIGHT_WINDOWS = config["ablation"] == "moon_night"
    years = [int(year) for year in cal.candidate_years()]
    rows = []
    try:
        for (tablet, text, baseline_statements, origins,
             legacy_month_statements, frozen_active) in population:
            statements = retained(
                text, baseline_statements, legacy_month_statements,
                config["ablation"], origins)
            scores = S.candidate_scores(
                statements, S.months_in(text), years,
                weighting=config["weighting"])
            strict = [int(year) for year in scores.years[scores.strict]]
            safety = [int(year) for year in scores.years[scores.safety]]
            # Freeze the public inspection order before reference dates enter.
            # The strict and safety sets remain the scientific outputs; this
            # short list only permits an honest diagnostic of a forced top-k
            # display without recomputing a truth-aware ordering later.
            display_top10 = [
                int(scores.years[index]) for index in scores.order[:10]
            ]
            # Report the same unit as the headline result. A lattice of years
            # the evidence cannot separate is one physical explanation, so
            # counting raw candidate years overstates the ambiguity by exactly
            # the periodicity the method is trying to describe.
            levels = S.evidence_levels(scores, max_levels=4)
            leading = levels[0]["years_astro"] if levels else []
            level_years = [level["years_astro"] for level in levels]
            rows.append({
                **config,
                "tablet": tablet,
                "leading_group_size": len(leading),
                "level_years_json": json.dumps(level_years),
                "n_statements": len(statements),
                "active_decisive": int(
                    (scores.decisive & (scores.weights > 0)).sum()),
                "frozen_active": bool(frozen_active),
                "strict_years_astro": strict,
                "safety_years_astro": safety,
                "display_top10_years_astro": display_top10,
                "strict_count": len(strict),
                "safety_count": len(safety),
            })
    finally:
        S.TOL_MOON, S.TOL_PLANET = old_moon, old_planet
        sky.TIME_OFFSET_DAYS = old_offset
        sky.BABYLONIAN_OFFSET_YEAR0_DEG = old_zodiac
        S.ORDER_CONTRADICTIONS_FIRST = old_order
        S.FRONTIER_PENALISES_CONTRADICTIONS = old_frontier
        S.LATITUDINAL_DISTANCES_DECISIVE = old_latitudinal
        S.MOON_NIGHT_WINDOWS = old_night
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, group in results.groupby("config", sort=False):
        active = group.frozen_active
        strong = group[active]
        rows.append({
            "config": config,
            "dated_tablets": len(group),
            "active_year_bearing": len(strong),
            "current_active_year_bearing": int(
                strong.active_decisive.gt(0).sum()),
            "truth_strict": int(strong.truth_strict.sum()),
            "strict_recall": (float(strong.truth_strict.mean())
                              if len(strong) else None),
            "median_strict_count": (float(strong.strict_count.median())
                                    if len(strong) else None),
            "median_leading_group": (float(strong.leading_group_size.median())
                                     if len(strong) else None),
            "truth_in_leading_group": int(strong.truth_group_rank.eq(1).sum()),
            "leading_group_rate": (float(strong.truth_group_rank.eq(1).mean())
                                   if len(strong) else None),
            "truth_safety": int(strong.truth_safety.sum()),
            "safety_recall": (float(strong.truth_safety.mean())
                              if len(strong) else None),
            "median_safety_count": (float(strong.safety_count.median())
                                    if len(strong) else None),
            "unique_correct": int((strong.truth_strict
                                   & strong.strict_count.eq(1)).sum()),
            "display_top1_correct": int(strong.display_top1_correct.sum()),
            "display_top1_rate": (float(strong.display_top1_correct.mean())
                                  if len(strong) else None),
            "display_top3_correct": int(strong.display_top3_correct.sum()),
            "display_top3_rate": (float(strong.display_top3_correct.mean())
                                  if len(strong) else None),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=HERE / "data" / "benchmark" / "sensitivity_runs.parquet")
    parser.add_argument(
        "--summary-out", type=Path,
        default=HERE / "data" / "benchmark" / "sensitivity_summary.parquet")
    # One worker per core. Each configuration is an independent rerun of the
    # whole evaluation, so this scales until the machine runs out of cores.
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    texts = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet", columns=["tablet", "text"])
    frozen_population = pd.read_parquet(
        HERE / "data" / "benchmark" / "benchmark_records.parquet",
        columns=[
            "tablet", "tool_benchmark_eligible", "tool_year_bearing",
            "n_frozen_extractable_facts",
        ])
    frozen_population = frozen_population.set_index("tablet")
    eligible_ids = set(frozen_population.loc[
        frozen_population.tool_benchmark_eligible].index)
    reviewed = clauses.load(clauses.REVIEWED, provenance=True)
    population = []
    for tablet in texts.itertuples(index=False):
        if not isinstance(tablet.text, str):
            continue
        if tablet.tablet in eligible_ids:
            items = reviewed.get(tablet.tablet, [])
            population.append((
                tablet.tablet, tablet.text, [item[0] for item in items],
                [item[1] for item in items],
                without_restored_month_context(tablet.text, items),
                bool(frozen_population.loc[
                    tablet.tablet, "tool_year_bearing"]),
            ))
    print(f"fixed text-defined population: {len(population)} tablets")

    rows_by_config = {}
    if args.workers <= 1:
        for number, config in enumerate(CONFIGS, start=1):
            print(f"  {number:>2}/{len(CONFIGS)} {config['config']}", flush=True)
            rows_by_config[config["config"]] = infer(config, population)
    else:
        print(f"running {len(CONFIGS)} configurations with "
              f"{args.workers} worker processes", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {
                executor.submit(infer, config, population): config
                for config in CONFIGS
            }
            for completed, future in enumerate(as_completed(pending), start=1):
                config = pending[future]
                rows_by_config[config["config"]] = future.result()
                print(f"  {completed:>2}/{len(CONFIGS)} finished "
                      f"{config['config']}", flush=True)
    rows = [row for config in CONFIGS
            for row in rows_by_config[config["config"]]]
    result = pd.DataFrame(rows)

    # Reference and strata enter only after every configuration is inferred.
    reference = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "project", "bce_first", "bce_second"])
    strata = pd.read_parquet(
        HERE / "data" / "corpus" / "cohorts.parquet",
        columns=["tablet", "cohort", "joined"])
    result = (result.merge(reference, on="tablet", validate="many_to_one")
              .merge(strata, on="tablet", validate="many_to_one"))
    result = result[result.bce_first.notna()].copy()
    result["truth_bce"] = result.bce_first.astype(int)
    result["truth_astro"] = result.truth_bce.map(yr.bce_to_astro)
    result["truth_group_rank"] = result.apply(
        lambda row: next(
            (rank for rank, years in enumerate(
                json.loads(row.level_years_json), start=1)
             if row.truth_astro in years), 99), axis=1)
    result = result.drop(columns=["level_years_json"])
    result["truth_strict"] = result.apply(
        lambda row: row.truth_astro in set(row.strict_years_astro), axis=1)
    result["truth_safety"] = result.apply(
        lambda row: row.truth_astro in set(row.safety_years_astro), axis=1)
    result["display_top1_correct"] = result.apply(
        lambda row: row.truth_astro == row.display_top10_years_astro[0], axis=1)
    result["display_top3_correct"] = result.apply(
        lambda row: row.truth_astro in set(row.display_top10_years_astro[:3]), axis=1)
    result["display_top1_abs_error"] = result.apply(
        lambda row: abs(row.truth_astro - row.display_top10_years_astro[0]), axis=1)
    result["display_top3_min_abs_error"] = result.apply(
        lambda row: min(abs(row.truth_astro - year)
                        for year in row.display_top10_years_astro[:3]), axis=1)
    result["chronology_band"] = pd.cut(
        result.truth_bce, bins=[0, 100, 200, 300, 400, 500, 700],
        labels=["1-100 BCE", "101-200 BCE", "201-300 BCE", "301-400 BCE", "401-500 BCE", "501-700 BCE"])
    summary = summarize(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    summary.to_parquet(args.summary_out, index=False)
    print(summary.to_string(index=False))
    print(f"results -> {args.out}\nsummary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
