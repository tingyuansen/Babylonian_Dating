"""Offline tests of the Sirius dates, the month windows and the ADART IV text filter.

The Sirius scheme must give each phase a date in each candidate year, except the
last appearance in the first year. A time offset must reach the cached solstice
and Sirius dates. A month window holds the night that opens the month, and a
clause without a month heading is tested over its whole year. The ADART IV input
loses its year numbers and rulers and keeps the other text exactly.

    python3 tests/test_rules.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts" / "source"))
from tools import cal, sky, statements as S  # noqa: E402
import build_adart_iv_corpus as corpus  # noqa: E402

YEARS = [int(y) for y in cal.candidate_years()]


class SiriusSchemeTest(unittest.TestCase):
    def test_every_phase_has_a_date(self):
        for phase in ("rising", "acronychal"):
            self.assertTrue(all(S.sirius_scheme_date(y, phase) for y in YEARS), phase)
        # The last appearance is counted from the solstice of the year before,
        # so only the first candidate year has no date for it.
        missing = [y for y in YEARS if S.sirius_scheme_date(y, "setting") is None]
        self.assertEqual(missing, [YEARS[0]])

    def test_last_appearance_falls_early_in_the_year(self):
        slots = {S.sirius_scheme_date(y, "setting")[0] for y in YEARS[1:]}
        self.assertLessEqual(slots, {1, 2, 3})
        # Twelve- and thirteen-month years both give a date.
        shapes = {S._year_shape(y)[0] for y in YEARS[1:]}
        self.assertEqual(shapes, {12, 13})


class TimeOffsetCacheTest(unittest.TestCase):
    def tearDown(self):
        sky.TIME_OFFSET_DAYS = 0.0

    def test_shift_reaches_cached_dates(self):
        start = cal.month_start_jd(-300, "I")
        base = sky.summer_solstice(start, start + 300)
        sirius = S.sirius_scheme_date(-300, "rising")
        shift = sky.delta_t_days(350.0)
        sky.TIME_OFFSET_DAYS = shift
        moved = sky.summer_solstice(start, start + 300)
        self.assertAlmostEqual(moved - base, -shift, places=3)
        self.assertIsNotNone(S.sirius_scheme_date(-300, "rising"))
        sky.TIME_OFFSET_DAYS = 0.0
        self.assertEqual(sky.summer_solstice(start, start + 300), base)
        self.assertEqual(S.sirius_scheme_date(-300, "rising"), sirius)


class MonthWindowTest(unittest.TestCase):
    def test_window_holds_the_opening_night_and_the_whole_month(self):
        start = cal.month_start_jd(-300, "I")
        after = cal.month_start_jd(-300, "II")
        w = S.window(-300, ["I"])
        self.assertEqual(w[0], start - 1.0)
        self.assertEqual(w[-1], after)

    def test_opening_night_of_month_vii(self):
        # X102613, "Jupiter was in the end of Virgo": in 333 BCE Jupiter is in
        # Virgo only on the night that opens month VII.
        text = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet").set_index(
            "tablet").at["X102613", "text"]
        clause = next(st for st in S.extract(text)
                      if st.planet == "jupiter" and st.sign == "Virgo")
        self.assertEqual(S.check(clause, -332, S.months_in(text)).result, "agree")

    def test_clause_without_a_month_is_decisive_over_its_year(self):
        st = S.extract("Jupiter was in Leo.")[0]
        self.assertIsNone(st.month)
        self.assertTrue(S.is_decisive(st))
        year = S.window(-300, [])
        self.assertEqual(year[0], cal.month_start_jd(-300, 1) - 1.0)
        self.assertEqual(year[-1], cal.month_start_jd(-299, 1))


class AdartIvChronologyTest(unittest.TestCase):
    def clean(self, line):
        return corpus.astronomy_only(corpus.normalize_translation(line))

    def test_year_numbers_and_rulers_are_cut(self):
        self.assertEqual(self.clean("Year 141?, Month II, night of the 9th, the moon was 1 cubit behind Jupiter."),
                         "Month II, night of the 9th, the moon was 1 cubit behind Jupiter.")
        self.assertNotIn("12", self.clean("Month VII, year 12 [... ] Jupiter was in Leo."))
        self.assertEqual(self.clean("to Ezida because of... which before in year 136 6'[...of Es]angil"), "")
        self.assertNotIn("Arsaces", self.clean("Month V, King Arsaces and Mitratu, the chief [of troops...]"))

    def test_other_text_is_kept_exactly(self):
        line = "Month IV, the 1st (of which followed the 30th), 1,21+x [...] Mars was 2.5 cubits above α Leonis."
        self.assertEqual(corpus.without_chronology(line), line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
