#!/usr/bin/env python3
"""Date and score the benchmark of 179 dated records, for Fig. 4 and the benchmark results.

Phase 1 dates each benchmark record from its reviewed clauses and its text
alone. Phase 2 then reads the published years and the cohorts, and scores the
candidate sets. Among the columns of each record:

    truth_in_top_evidence_group   the first group holds the published year
    truth_evidence_group_rank     the group that holds it, from 1 at the top
    truth_in_strict               the strict set holds it
    truth_in_safety               the safety set holds it
    unique                        the strict set holds it alone
    strict_count, safety_count    the sizes of the two sets

The size of the strict set shows how far the surviving astronomy limits the
year of the record.

Outputs:
    data/benchmark/benchmark_results.parquet       one row per record
    data/benchmark/benchmark_clause_bits.parquet   one row per clause, with its bits

Usage:
    python3 scripts/audit/score_benchmark.py [--weighting count] [--n N]

The default weighting, skill, is the scoring of steps 2 and 3. --weighting
selects another, as some runs of the sensitivity audit do. With --n N, the
script scores a random sample of N records, with a fixed seed. Both options
write to the default outputs unless --out and --statement-out are given.
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

    # Phase 1 reads only the identifier and the text of each record. The
    # published years and the cohorts are read after all candidate sets are
    # computed.
    tb = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet", columns=["tablet", "text"])
    tb = tb[tb.text.notna()]
    years = [int(y) for y in cal.candidate_years()]

    frozen_population = pd.read_parquet(
        HERE / "data" / "benchmark" / "benchmark_records.parquet",
        columns=["tablet", "tool_benchmark_eligible", "n_frozen_extractable_facts"])
    frozen_population = frozen_population.set_index("tablet")
    eligible_ids = set(frozen_population.loc[
        frozen_population.tool_benchmark_eligible].index)
    # A record has decisive clauses when its result keeps at least one, under
    # the final rules. The results hold no published year.
    decisive = (pd.read_parquet(HERE / "data" / "results" / "tablet_dates.parquet",
                                columns=["tablet", "n_decisive_statements"])
                .set_index("tablet").n_decisive_statements.fillna(0).gt(0).to_dict())
    reviewed = clauses.load(clauses.REVIEWED)
    cand = []
    for _, r in tb.iterrows():
        sts = reviewed.get(r.tablet, [])
        usable = [s for s in sts
                  if not (s.kind == "eclipse" and s.omitted)]
        # The threshold uses the parser clause counts of the fixed benchmark,
        # so that a change to the parser cannot add or remove a record.
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
        # The years of a group usually share all their verdicts. Two years can
        # also tie on all scores and differ in their verdicts. Such a group
        # stays whole, and co_ranked counts these groups among the first three.
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
            "decisive": bool(decisive[tablet]),
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

    # Phase 2 reads the published years and the cohorts, and scores the
    # candidate sets of phase 1.
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
        # The rank of the published year by the clocks K, then the decisive
        # bits I_D, alone.
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

    # One row per clause, with the numbers of agreeing and testable years and
    # the bits u_j of Eq. 1 in the main text. These rows hold no published
    # year.
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
    active = d.decisive
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
        weak = ~d.decisive
        strong = d.decisive
        print(f"\n  records with decisive clauses: {int(strong.sum())}")
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

        print("\n  records with decisive clauses, by join status")
        for joined, group in d.loc[strong].groupby("joined"):
            label = "physical joins" if joined else "not joins"
            print(f"    {label:<14} n={len(group):3d}  strict "
                  f"{group.truth_in_strict.mean()*100:5.1f}%  safety "
                  f"{group.truth_in_safety.mean()*100:5.1f}%  median sets "
                  f"{group.strict_count.median():.0f}/{group.safety_count.median():.0f}")

        print("\n  records with decisive clauses, by cohort")
        for cohort, group in d.loc[strong].groupby("cohort"):
            print(f"    {cohort:<18} n={len(group):3d}  strict "
                  f"{group.truth_in_strict.mean()*100:5.1f}%  safety "
                  f"{group.truth_in_safety.mean()*100:5.1f}%  median sets "
                  f"{group.strict_count.median():.0f}/{group.safety_count.median():.0f}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
