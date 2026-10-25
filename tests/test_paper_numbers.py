#!/usr/bin/env python3
"""Every number the PNAS paper reports, recomputed from the committed data.

Each check pairs a phrase that must appear in paper/PNAS_Paper.tex with the
value computed from data/. A check fails when either drifts: a rerun that moves
a result, or an edit to the paper that no longer matches the data. Run it after
any change to tools/, scripts/ or the paper.

Tables S3 and S4 are compared cell by cell with the scope builder's CSVs.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "paper" / "figures"))

from tools import cal  # noqa: E402
import fig4_benchmark  # noqa: E402

DATA = ROOT / "data"
PAPER = ROOT / "paper" / "PNAS_Paper.tex"
if not PAPER.is_file():
    if __name__ == "__main__":
        print("skipped: paper/PNAS_Paper.tex is kept outside the repository")
        raise SystemExit(0)
    raise unittest.SkipTest("paper/PNAS_Paper.tex is kept outside the repository")
TEX = re.sub(r"\s+", " ", PAPER.read_text(encoding="utf-8"))
VOLUMES = {"adsd/adart1": "I", "adsd/adart2": "II", "adsd/adart3": "III"}


def pct(part: int, whole: int) -> str:
    """Percent as the paper prints it, rounded to a whole number."""
    return f"{round(100 * part / whole)}"


class PaperNumbersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.population = pd.read_parquet(DATA / "benchmark" / "benchmark_records.parquet")
        cls.dated = set(cls.population.tablet)
        cls.tablets = pd.read_parquet(DATA / "corpus" / "tablets.parquet").set_index("tablet")
        cls.scores = pd.read_parquet(DATA / "benchmark" / "benchmark_results.parquet")
        cls.active = cls.scores[cls.scores.frozen_active]
        cls.results = pd.read_parquet(DATA / "results" / "tablet_dates.parquet").set_index("tablet")
        cls.facts = pd.read_parquet(DATA / "clauses" / "parser_clauses.parquet")
        cls.scope = json.loads((DATA / "paper_counts" / "counts.json").read_text())
        cls.information = pd.read_parquet(DATA / "benchmark" / "benchmark_clause_bits.parquet")
        cls.adart_iv = pd.read_parquet(DATA / "results" / "adart_iv_dates.parquet")
        cls.sensitivity = pd.read_parquet(DATA / "benchmark" / "sensitivity_summary.parquet").set_index("config")

    def says(self, phrase: str) -> None:
        self.assertIn(re.sub(r"\s+", " ", phrase), TEX, "phrase no longer in PNAS_Paper.tex")

    def test_candidate_years(self):
        years = cal.candidate_years()
        self.assertEqual((len(years), min(years), max(years)), (701, -625, 75))
        self.says("in all 701 years from 626 BCE to AD 75")

    def test_corpus(self):
        by_volume = self.population.project.map(VOLUMES).value_counts()
        self.assertEqual(len(self.dated), 417)
        self.assertEqual([by_volume[v] for v in ("I", "II", "III")], [94, 161, 162])
        with_text = int(self.tablets.loc[sorted(self.dated)].n_lines.gt(0).sum())
        self.assertEqual(with_text, 393)
        self.says("Of these, 393 have edition text")
        before = self.tablets.loc[sorted(self.dated)].bce_first.max()
        self.assertEqual(int(before), 652)

    def test_parser_and_reviewed_clauses(self):
        parser = self.facts[self.facts.tablet.isin(self.dated)]
        self.assertEqual((len(parser), parser.tablet.nunique()), (3193, 316))
        self.says("it finds 3,193 clauses in 316 of the 417 records with a published date")
        deterministic, reviewed = self.scope["deterministic_forward"], self.scope["reviewed_forward"]
        self.assertEqual((deterministic["clauses"], deterministic["records"]), (3193, 316))
        self.assertEqual((reviewed["clauses"], reviewed["records"]), (4010, 321))
        self.says("raises the 3,193 parser clauses to 4,010 clauses from 321 records")
        verdicts = parser.forward_verdict.value_counts()
        testable = int(verdicts.get("agree", 0) + verdicts.get("contradict", 0))
        self.assertEqual((int(verdicts["agree"]), testable), (2169, 2313))
        self.says("2,169 of 2,313 testable clauses agree (93.8\\%)")
        self.assertEqual(f"{100 * 2732 / 2920:.1f}", f"{reviewed['agreement_percent']:.1f}")
        self.says("the reviewed clauses agree in 93.6\\% of the cases that the tools can test and the parser clauses in 93.8\\%")

    def test_benchmark_population(self):
        self.assertEqual((len(self.scores), len(self.active)), (179, 112))
        self.says("Of these, 112 then had at least one decisive clause")
        self.says("The other 67 are supporting-only records")
        self.assertEqual(len(self.dated - set(self.scores.tablet)), 238)
        self.says("The same rules date the 238 dated records outside the benchmark")
        outside = self.results.loc[sorted(self.dated - set(self.scores.tablet))]
        ranked = outside[outside.claim.eq("ranked")]
        self.assertEqual((len(ranked), ranked.strict_years_astro.map(len).median()), (25, 87))
        self.assertFalse(outside.claim.eq("committed").any())
        self.says("and 25 of them have a decisive clause. The median strict set of these 25 holds 87 years")

    def test_first_groups(self):
        first = int(self.active.truth_in_top_evidence_group.sum())
        three = int(self.active.truth_evidence_group_rank.le(3).sum())
        self.assertEqual((first, pct(first, 112)), (104, "93"))
        self.says("it contains the published year in 104 (93\\%")
        self.assertEqual(int(self.active.top_evidence_group_size.eq(1).sum()), 78)
        self.says("For 78 of the 112, it pins down a single year, and in 73 this is the published year")
        self.says("For 78 of them, the ranking pins down one year, the published year in 73")
        now = self.active.active_decisive.astype(float).gt(0)
        self.assertEqual((int((~now).sum()), self.active[now].strict_count.median()), (13, 14))
        self.says("13 of these records no longer have a decisive clause, and their strict set holds all 701 years")
        first_now = int(self.active[now].truth_in_top_evidence_group.sum())
        self.assertEqual((first_now, pct(first_now, int(now.sum()))), (93, "94"))
        self.says("and the first group holds the published year in 93 (94\\%)")
        self.assertEqual((three, pct(three, 112)), (109, "97"))
        self.says("contain it in 109 (97\\%)")
        supporting = self.scores[~self.scores.frozen_active]
        self.assertEqual(pct(int(supporting.truth_in_top_evidence_group.sum()), len(supporting)), "90")
        self.says("The supporting-only records also reach 90\\% for the first group")

    def test_candidate_years_in_ranked_order(self):
        candidates = pd.read_parquet(DATA / "benchmark" / "ranking_candidates.parquet")
        places = fig4_benchmark.positions(candidates[candidates.frozen_active])
        self.assertEqual(pct(sum(1 for p in places if p <= 1), 112), "65")
        self.says("one year suffices in 65\\%")

    def test_strict_sets(self):
        strict = int(self.active.truth_in_strict.sum())
        self.assertEqual((strict, pct(strict, 112), self.active.strict_count.median()), (112, "100", 17.5))
        self.says("It contains the published year in all 112 records with decisive clauses, "
                  "with a median size of 17.5 years")

    def test_commitments_and_stated_years(self):
        dated = self.results.loc[sorted(self.dated)]
        committed = dated[dated.claim.eq("committed")]
        truth = 1 - self.tablets.loc[committed.index].bce_first
        years = committed.strict_years_astro.map(lambda y: int(y[0]))
        self.assertEqual(len(committed), 7)
        self.assertTrue((years == truth).all())
        self.assertEqual(committed.basis.value_counts().to_dict(),
                         {"independent_astronomy": 6, "secure_lunar_eclipse": 1})
        self.says("commits to one year for 7. All 7 match the published year")
        self.says("Six rest on two or more clocks and one on a lunar eclipse")
        stated = dated[~dated.chronology_relation.eq("absent")]
        narrowed = stated[~stated.chronology_relation.eq("lookup_only")]
        self.assertEqual((len(stated), len(narrowed)), (23, 11))
        self.assertTrue(narrowed.chronology_relation.isin(
            ["selects_strict_candidate", "confirms_astronomy"]).all())
        self.says("For 23 dated records, the system extracts a year")
        self.says("In all 11 records where the astronomy narrows the year, the stated year lies in the strict set")

    def test_worked_examples(self):
        def years(tablet, decisive=None):
            rows = self.information[self.information.tablet.eq(tablet)]
            if decisive is not None:
                rows = rows[rows.decisive_tier.eq(decisive)]
            return sorted(rows.agreeing_years.astype(int), reverse=True)

        self.assertEqual(years("X301492", decisive=True), [138, 116, 54, 50, 49])
        self.says("Five of its decisive clauses agree with 138, 116, 54, 50 and 49 years")
        record = self.results.loc["X301492"]
        self.assertEqual(list(record.strict_years_astro), [-149])
        self.assertEqual(round(float(record.primary_bits), 1), 16.3)
        self.assertEqual(int(record.decisive_families), 3)
        self.says("the decisive bits $I_D$ total 16.3")
        rows = years("X301322")
        for count in (656, 635, 11):
            self.assertIn(count, rows)
        self.says("so rows 2 and 4 agree with 656 and 635 years")
        self.says("The Mars row thus agrees with 11 years, which Fig.~\\ref{fig:diary}b shows as 6.0 bits")

    def test_adart_iv(self):
        ranked = self.adart_iv[self.adart_iv.claim.eq("ranked")]
        summary = json.loads((DATA / "corpus" / "adart_iv_corpus_summary.json").read_text())
        self.assertEqual((summary["entries"], summary["entries_with_translation"]), (867, 860))
        self.says("which gives 867 entries. Of these, 860 have an English translation")
        self.assertEqual((int(self.adart_iv.n_statements.gt(0).sum()), int(self.adart_iv.n_statements.sum())),
                         (321, 539))
        self.assertEqual((len(ranked), int(ranked.n_decisive_statements.eq(1).sum())), (52, 44))
        self.says("Of the 52 ranked entries, 44 hold a single decisive clause")
        self.assertEqual((ranked.strict_count.median(), ranked.top_evidence_group_size.median()), (101, 100))
        self.says("cut the 701 candidate years to a median of 101")
        self.says("the first groups of all 52 have a median of 100 years")
        self.assertFalse(self.adart_iv.claim.eq("committed").any())

    def test_clauses_without_a_month(self):
        # Decisive positions with no month are tested over their whole year.
        free = self.facts[self.facts.semantic_tier.eq("decisive") & self.facts.month.isna()
                          & self.facts.kind.isin(["in_sign", "entry", "station", "first_app", "last_app"])]
        tested = free[free.forward_verdict.isin(["agree", "contradict"])]
        joint = free.joint_positions_json.fillna("[]").ne("[]")
        self.assertEqual((len(free), free.tablet.nunique(), len(tested),
                          int(tested.forward_verdict.eq("agree").sum()), int(joint.sum())), (26, 21, 22, 22, 4))
        self.assertTrue(free[joint].forward_verdict.eq("untestable").all())
        self.says("All 22 such positions in ADART I to III that are testable at the published year agree with it.")
        sys.path.insert(0, str(ROOT / "scripts" / "paper"))
        import build_adart_iv_table as table
        ranked = self.adart_iv[self.adart_iv.claim.eq("ranked")]
        dated = sum(table.has_month(row) for row in ranked.itertuples())
        self.assertEqual((len(ranked) - dated, dated), (49, 3))
        self.says("In 49 of the 52, the decisive clauses lack a month")
        self.says("a dagger marks the three entries with a decisive clause that has a month")

    def test_model_pilot(self):
        pilot = json.loads((DATA / "clauses" / "model_review_pilot" / "comparison.json").read_text())
        small, large = pilot["gpt-6-luna-2026-09-22"], pilot["gpt-5.6-terra-2026-07-09"]
        self.assertEqual((small["cost_usd"], large["cost_usd"]), (0.25, 4.79))
        self.assertEqual(round(large["cost_usd"] / small["cost_usd"]), 19)
        self.says("the larger GPT-5.6-Terra cost an order of magnitude more")
        self.assertEqual((large["modified"], small["modified"], large["added"], small["added"]), (2, 6, 80, 69))
        tested = lambda row: row["added_at_published_year"]["agree"] + row["added_at_published_year"]["contradict"]
        share = lambda row: round(100 * row["added_at_published_year"]["agree"] / tested(row))
        earlier = pilot["gpt-5.6-luna-2026-07-09"]
        self.assertEqual((share(large), share(earlier), share(small)), (89, 89, 88))
        self.says("The added clauses of both agree at the published year in 89\\% of testable cases, against 88\\% for gpt-6-luna")
        self.assertEqual((round(large["cost_usd"] * 10, -1), small["cost_usd"] * 10), (50.0, 2.5))
        self.says("Scaled to 100 records, the larger model would cost about \\$50, against about \\$2.50 for gpt-6-luna")
        run = json.loads((DATA / "clauses" / "model_review_gpt-6-luna-2026-09-22" / "run_summary.json").read_text())
        self.assertEqual(round(run["all_cached_record_calls"]["cost_usd"]), 4)
        self.says("All review and search calls for ADART I to III with gpt-6-luna, superseded ones included, cost about \\$4")

    def test_sensitivity(self):
        self.assertEqual(len(self.sensitivity), 32)
        self.says("most of the 31 alternative settings of the method")
        self.says("The sensitivity audit tests 32 fixed configurations, including the baseline")
        baseline = self.sensitivity.loc["baseline"]
        self.assertEqual((pct(baseline.truth_in_leading_group, 112), pct(baseline.truth_strict, 112)), ("93", "100"))
        rate = lambda config, column: pct(self.sensitivity.loc[config, column], 112)
        self.assertEqual((rate("no_sun_family", "truth_in_leading_group"),
                          self.sensitivity.loc["no_sun_family", "median_leading_group"],
                          rate("no_sun_family", "truth_strict")), ("93", 1, "100"))
        self.says("Without this clock, the benchmark values stay as in the baseline")
        self.assertEqual(rate("contradictions_before_information", "truth_in_leading_group"), "95")
        self.assertEqual(rate("equal_weighting_with_tiers", "truth_in_leading_group"), "96")
        self.says("raise first-group inclusion by a few points")
        self.assertEqual((rate("contradictions_bound_the_leading_set", "truth_strict"),
                          rate("information_weighting", "truth_strict")), ("96", "92"))
        self.assertEqual((self.sensitivity.loc["contradictions_bound_the_leading_set", "median_strict_count"],
                          self.sensitivity.loc["information_weighting", "median_strict_count"]), (1, 1))
        self.says("The strict set then shrinks to about one year, which is more often wrong")

    def test_held_out(self):
        summary = json.loads((DATA / "benchmark" / "held_out_summary.json").read_text())
        self.assertTrue(summary["held_out_scored"])
        self.assertEqual(summary["settings"]["close_points"], 5.0)
        pooled = summary["all_three_rounds"]
        self.assertEqual((pooled["first_group"]["n"], pooled["strict_set"]["n"]), (103, 112))
        self.assertTrue(all(reading["close"] for reading in summary["reading"].values()))
        self.says("Over the three rounds, the first group holds the published year in 92\\% and the strict set in 100\\%")
        self.says("All three rounds & 112 & 103 (92\\%) & 109 (97\\%) & 112 (100\\%) & 19 years")

    def test_corrigenda(self):
        corrigenda = json.loads((DATA / "benchmark" / "adart_iv_corrigenda_summary.json").read_text())
        self.assertEqual((corrigenda["corrections"], corrigenda["reference_labels_already_incorporated"]), (27, 3))
        self.says("ADART IV makes 27 corrections")

    def test_tables_s3_and_s4(self):
        def tex_rows(label: str) -> set[tuple]:
            body = TEX.split(f"\\label{{{label}}}", 1)[1].split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
            rows = set()
            for line in body.split("\\\\"):
                cells = [c.strip() for c in line.replace("\\addlinespace", "").split("&")]
                if len(cells) > 1:
                    rows.add(tuple(float(c) for c in cells[1:]))
            return rows

        def csv_rows(name: str, columns: list[str]) -> set[tuple]:
            table = pd.read_csv(DATA / "paper_counts" / name)
            return {tuple(float(v) for v in row) for row in table[columns].itertuples(index=False)}

        self.assertEqual(tex_rows("tab:information"), csv_rows(
            "table_s3.csv",
            ["Clauses", "Testable across candidate years", "Median years allowed", "Bits",
             "Decisive %"]))
        forward = csv_rows("table_s4.csv",
                           ["Clauses", "Testable", "Agree", "Contradict", "Untestable", "Agreement %"])
        total = self.scope["reviewed_forward"]
        self.assertEqual(tex_rows("tab:forward") - {(total["clauses"], total["testable"], total["agree"], total["contradict"],
                                                     total["untestable"], round(total["agreement_percent"], 1))}, forward)
        self.says("The station against a star of the main text agrees in about half of its testable cases")


if __name__ == "__main__":
    unittest.main(verbosity=2)
