#!/usr/bin/env python3
"""Run backward inference over a sample of dated tablets, and score it honestly.

The number that matters is not how often a statement agrees at the published
year. A window wide enough to agree with everything discriminates between
nothing, so forward agreement can be improved by making the method worse. What
matters is whether the published year outscores the other six hundred.

Four figures are reported and they mean different things.

    unique           the strict set holds the published year alone
    truth_in_strict  the strict set holds it, possibly with other years
    truth_in_safety  the safety set holds it
    strict_count     how many years the strict set holds, which is the
                     tablet's own ceiling

A large strict set is not a failure. It is a statement about how much the
surviving text can determine, and it is knowable in advance of any method.

Usage:
    python3 backward_sample.py [--n 40] [--min-statements 3]
    python3 backward_sample.py --weighting count  # historical baseline

The default evaluates the complete independently dated population. ``--n`` is
an explicit, fixed-seed diagnostic subsample and must not overwrite a reported
full-population result unless the user intentionally supplies that option.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, clauses, statements as S   # noqa: E402
from tools import years as yr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--min-statements", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weighting",
                    choices=["count", "information", "inverse", "skill"],
                    default="skill")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "benchmark" / "benchmark_results.parquet")
    ap.add_argument("--statement-out", type=Path,
                    default=HERE / "data" / "benchmark" / "benchmark_clause_bits.parquet")
    args = ap.parse_args()

    # Phase 1 is deliberately limited to identifier and surviving text. The
    # reference date and stratification metadata are not loaded until every
    # candidate set has been frozen below.
    tb = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet", columns=["tablet", "text"])
    tb = tb[tb.text.notna()]
    years = [int(y) for y in cal.candidate_years()]

    frozen_population = pd.read_parquet(
        HERE / "data" / "benchmark" / "benchmark_records.parquet",
        columns=[
            "tablet", "tool_benchmark_eligible", "tool_year_bearing",
            "n_frozen_extractable_facts",
        ])
    frozen_population = frozen_population.set_index("tablet")
    eligible_ids = set(frozen_population.loc[
        frozen_population.tool_benchmark_eligible].index)
    frozen_active = frozen_population.tool_year_bearing.to_dict()
    reviewed = clauses.load(clauses.REVIEWED)
    cand = []
    for _, r in tb.iterrows():
        sts = reviewed.get(r.tablet, [])
        usable = [s for s in sts
                  if not (s.kind == "eclipse" and s.omitted)]
        # Population membership was frozen before this harnessing cycle.  Use
        # the pre-registered extractable-fact count for the optional threshold
        # rather than allowing a parser edit to add or remove evaluation cases.
        frozen_n = (int(frozen_population.loc[
            r.tablet, "n_frozen_extractable_facts"])
            if r.tablet in eligible_ids else 0)
        if frozen_n >= args.min_statements and r.tablet in eligible_ids:
            cand.append((r.tablet, r.text, sts))
    print(f"{len(cand)} tablets carry {args.min_statements} or more "
          f"extractable statements before date availability is consulted")

    inferred = []
    score_cache = {}
    for tablet, text, sts in cand:
        months = S.months_in(text)
        scores = S.candidate_scores(sts, months, years,
                                    args.weighting)
        strict = scores.years[scores.strict]
        safety = scores.years[scores.safety]
        display_top10 = [
            int(scores.years[index]) for index in scores.order[:10]
        ]
        levels = S.evidence_levels(scores)
        reported_levels = levels[:3]
        # A reported level is normally one exact predicate equivalence class.
        # It need not be: two candidate years can explain equally many decisive
        # families with equal information and still disagree about which
        # statements they satisfy. That is a real tie between two physical
        # explanations, so it is disclosed as co-ranked subgroups rather than
        # broken by an arbitrary rule. The count is reported, not suppressed.
        co_ranked = sum(not level["predicate_exact"]
                        for level in reported_levels)
        top_evidence_group = levels[0]["years_astro"]
        top_three_evidence_groups = sorted({
            year for level in reported_levels for year in level["years_astro"]
        })
        score_cache[tablet] = scores
        omitted_eclipses = sum(
            s.kind == "eclipse" and s.omitted for s in sts)
        inferred.append({
            "tablet": tablet,
            "n": len(sts) - omitted_eclipses,
            "omitted_eclipses": omitted_eclipses,
            "semantic_decisive": int(scores.decisive.sum()),
            "active_decisive": int(
                (scores.decisive & (scores.weights > 0)).sum()),
            "frozen_active": bool(frozen_active[tablet]),
            "weighting": args.weighting,
            "best": float(scores.score.max()),
            "best_bodies": int(scores.priority.max()),
            "strict_years_astro": [int(year) for year in strict],
            "safety_years_astro": [int(year) for year in safety],
            "display_top10_years_astro": display_top10,
            "top_evidence_group_years_astro": top_evidence_group,
            "top_evidence_group_size": len(top_evidence_group),
            "top_evidence_group_predicate_exact": bool(
                levels[0]["predicate_exact"]),
            "top_three_evidence_groups_years_astro": top_three_evidence_groups,
            "top_three_evidence_groups_size": len(top_three_evidence_groups),
            "top_three_evidence_groups_predicate_exact": bool(
                all(level["predicate_exact"] for level in reported_levels)),
            "evidence_group_count": len(levels),
            "co_ranked_reported_levels": int(co_ranked),
            "strict_count": len(strict),
            "safety_count": len(safety),
        })

    # Phase 2 evaluates already-frozen inference. Truth and cohort fields can
    # change metrics and strata, but cannot change a candidate year or score.
    d = pd.DataFrame(inferred)
    references = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "bce_first", "bce_second"])
    cohorts = pd.read_parquet(
        HERE / "data" / "corpus" / "cohorts.parquet",
        columns=["tablet", "cohort", "joined"])
    d = d.merge(references, on="tablet", validate="one_to_one")
    d = d[d.bce_first.notna()].merge(
        cohorts, on="tablet", validate="one_to_one")
    if args.n is not None and len(d) > args.n:
        random.seed(args.seed)
        selected = set(random.sample(list(d.tablet), args.n))
        d = d[d.tablet.isin(selected)].copy()
    d["truth_astro"] = d.bce_first.astype(int).map(yr.bce_to_astro)
    in_domain = d.truth_astro.isin(set(years))
    if (~in_domain).any():
        outside = d.loc[~in_domain, ["tablet", "bce_first"]]
        print(f"{len(outside)} dated tablet(s) lie outside the candidate calendar "
              "and are excluded from accuracy metrics:")
        for row in outside.itertuples(index=False):
            print(f"  {row.tablet}: {int(row.bce_first)} BCE")
    d = d.loc[in_domain].copy()
    print(f"{len(d)} independently dated tablets enter evaluation")
    evaluated = []
    for row in d.itertuples(index=False):
        truth = int(row.truth_astro)
        scores = score_cache[row.tablet]
        strict = set(row.strict_years_astro)
        safety = set(row.safety_years_astro)
        truth_i = int((scores.years == truth).nonzero()[0][0])
        levels = S.evidence_levels(scores)
        truth_group_rank = next(
            level["rank"] for level in levels
            if truth in level["years_astro"])
        truth_signature = scores.outcomes[truth_i].tobytes()
        predicate_equivalent = sum(
            scores.outcomes[i].tobytes() == truth_signature
            for i in range(len(scores.years))
        )
        outranks = ((scores.priority > scores.priority[truth_i])
                    | ((scores.priority == scores.priority[truth_i])
                       & (scores.score > scores.score[truth_i] + 1e-12)))
        rank = int(outranks.sum()) + 1
        evaluated.append({
            "rank": rank,
            "unique": truth in strict and len(strict) == 1,
            "truth_in_strict": truth in strict,
            "truth_in_safety": truth in safety,
            "truth_evidence_group_rank": truth_group_rank,
            "truth_in_top_evidence_group": truth_group_rank == 1,
            "truth_predicate_equivalent_years": predicate_equivalent,
            "display_top1_correct": truth == row.display_top10_years_astro[0],
            "display_top3_correct": truth in row.display_top10_years_astro[:3],
            "display_top1_abs_error": abs(
                truth - row.display_top10_years_astro[0]),
            "display_top3_min_abs_error": min(
                abs(truth - year) for year in row.display_top10_years_astro[:3]),
        })
    d = pd.concat(
        [d.reset_index(drop=True), pd.DataFrame(evaluated)], axis=1)
    d.to_parquet(args.out, index=False)

    # How far each surviving clause narrows the calendar, recorded per
    # statement rather than per tablet. This is the quantity that separates a
    # clue which is common from one which is informative, and it is measured
    # over the same configured candidate calendar as everything else. It carries no
    # reference year: ``agreeing_years`` counts years the clause agrees with,
    # not years anyone believes.
    statement_rows = []
    for tablet, text, sts in cand:
        scores = score_cache[tablet]
        for index, statement in enumerate(sts):
            statement_rows.append({
                "tablet": tablet,
                "statement_index": index,
                "kind": statement.kind,
                "planet": statement.planet,
                "target_planet": statement.target_planet,
                "parser_rule": statement.parser_rule,
                "restored": bool(statement.restored),
                "omitted": bool(statement.omitted),
                "decisive_tier": bool(scores.decisive[index]),
                "agreeing_years": int(scores.agreeing[index]),
                "testable_years": int(scores.testable[index]),
                "information_bits": float(scores.weights[index]),
                "equivalent_predicates": int(scores.equivalent[index]),
            })
    pd.DataFrame(statement_rows).to_parquet(
        args.statement_out, index=False)

    n = len(d)
    print(f"\n{n} tablets, Britton (2010) Babylonian sidereal zodiac, "
          f"{args.weighting} weighting")
    print(f"  published year uniquely Pareto: {int(d.unique.sum()):3d}/{n} "
          f"({d.unique.mean()*100:.0f}%)")
    print(f"  published year on Pareto set : {int(d.truth_in_strict.sum()):3d}/{n} "
          f"({d.truth_in_strict.mean()*100:.0f}%)")
    print(f"  published year in safety set : {int(d.truth_in_safety.sum()):3d}/{n} "
          f"({d.truth_in_safety.mean()*100:.0f}%)")
    active = d.frozen_active
    print(f"  forced display top 1, active : "
          f"{int(d.loc[active, 'display_top1_correct'].sum()):3d}/"
          f"{int(active.sum())} "
          f"({d.loc[active, 'display_top1_correct'].mean()*100:.1f}%)")
    print(f"  forced display top 3, active : "
          f"{int(d.loc[active, 'display_top3_correct'].sum()):3d}/"
          f"{int(active.sum())} "
          f"({d.loc[active, 'display_top3_correct'].mean()*100:.1f}%)")
    print(f"  best evidence group, active : "
          f"{int(d.loc[active, 'truth_in_top_evidence_group'].sum()):3d}/"
          f"{int(active.sum())} "
          f"({d.loc[active, 'truth_in_top_evidence_group'].mean()*100:.1f}%)")
    print(f"  first three evidence groups : "
          f"{int(d.loc[active, 'truth_evidence_group_rank'].le(3).sum()):3d}/"
          f"{int(active.sum())} "
          f"({d.loc[active, 'truth_evidence_group_rank'].le(3).mean()*100:.1f}%)")
    print(f"  median best evidence-group size: "
          f"{d.loc[active, 'top_evidence_group_size'].median():.0f}")
    print(f"  median first-three-group union: "
          f"{d.loc[active, 'top_three_evidence_groups_size'].median():.0f}")
    print(f"  median Pareto / safety set   : {d.strict_count.median():.0f} / "
          f"{d.safety_count.median():.0f}")
    print(f"  safety 75th / 90th percentile: {d.safety_count.quantile(.75):.0f} / "
          f"{d.safety_count.quantile(.90):.0f}")
    print(f"  median rank of published year: {d['rank'].median():.0f}")
    print(f"  published year outside safety: {int((~d.truth_in_safety).sum())}, "
          f"which is where evidence is wrongly outranking the right year")
    if args.weighting == "skill":
        weak = ~d.frozen_active
        strong = d.frozen_active
        print(f"\n  frozen year-bearing tablets  : {int(strong.sum())}")
        print(f"  currently year-bearing       : "
              f"{int(d.active_decisive.gt(0).sum())} (diagnostic only)")
        if strong.any():
            print(f"    published year on Pareto set: "
                  f"{int(d.loc[strong, 'truth_in_strict'].sum())}/"
                  f"{int(strong.sum())} "
                  f"({d.loc[strong, 'truth_in_strict'].mean()*100:.0f}%)")
            print(f"    published year in safety set: "
                  f"{int(d.loc[strong, 'truth_in_safety'].sum())}/{int(strong.sum())} "
                  f"({d.loc[strong, 'truth_in_safety'].mean()*100:.0f}%)")
            print(f"    median Pareto-set size      : "
                  f"{d.loc[strong, 'strict_count'].median():.0f}")
            print(f"    median safety-set size     : "
                  f"{d.loc[strong, 'safety_count'].median():.0f}")
        print(f"  supporting-only tablets      : {int(weak.sum())} "
              "(declined; full candidate set retained)")

        print("\n  year-bearing results by join status")
        for joined, group in d.loc[strong].groupby("joined"):
            label = "physical joins" if joined else "not joins"
            print(f"    {label:<14} n={len(group):3d}  strict "
                  f"{group.truth_in_strict.mean()*100:5.1f}%  safety "
                  f"{group.truth_in_safety.mean()*100:5.1f}%  median sets "
                  f"{group.strict_count.median():.0f}/{group.safety_count.median():.0f}")

        print("\n  year-bearing results by cohort")
        for cohort, group in d.loc[strong].groupby("cohort"):
            print(f"    {cohort:<18} n={len(group):3d}  strict "
                  f"{group.truth_in_strict.mean()*100:5.1f}%  safety "
                  f"{group.truth_in_safety.mean()*100:5.1f}%  median sets "
                  f"{group.strict_count.median():.0f}/{group.safety_count.median():.0f}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
