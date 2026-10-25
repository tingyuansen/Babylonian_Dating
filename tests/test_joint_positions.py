"""Offline regression checks for simultaneous planetary sign statements."""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tools import statements as S, sky


class JointPositionsTest(unittest.TestCase):
    def setUp(self):
        self.text = "Month III. Jupiter and Mars were in Leo."
        self.atoms = S.extract(self.text)

    def test_atomic_provenance_and_joint_link(self):
        self.assertEqual(len(self.atoms), 2)
        self.assertEqual({s.planet for s in self.atoms}, {"jupiter", "mars"})
        self.assertEqual(self.atoms[0].joint_positions, self.atoms[1].joint_positions)
        self.assertEqual(self.atoms[0].source_start, self.atoms[1].source_start)
        self.assertTrue(all(S.is_decisive(s) for s in self.atoms))

    def test_three_planets(self):
        atoms = S.extract("Month III. Jupiter, Saturn and Mars were in Leo.")
        self.assertEqual(len(atoms), 3)
        self.assertTrue(all(len(s.joint_positions) == 3 for s in atoms))

    def test_separate_source_clauses_are_not_paired(self):
        atoms = S.extract("Month III. Jupiter was in Leo. Mars was in Leo.")
        self.assertTrue(all(not s.joint_positions for s in atoms))

    def test_restored_joint_is_supporting(self):
        atoms = S.extract("Month III. [Jupiter and Mars were in Leo].")
        self.assertEqual(len(atoms), 2)
        self.assertTrue(all(s.joint_positions and not S.is_decisive(s) for s in atoms))

    def test_inner_planet_does_not_acquire_decisive_power(self):
        atoms = S.extract("Month III. Jupiter and Mercury were in Leo.")
        self.assertTrue(all(s.joint_positions and not S.is_decisive(s) for s in atoms))

    def test_incomplete_review_does_not_invent_conjunct(self):
        atoms = S.bind_joint_positions(self.text, self.atoms[:1])
        self.assertFalse(atoms[0].joint_positions)

    def test_day_heading_is_secure_and_not_a_monthly_summary(self):
        atoms = S.extract("Month III, the 15th, Jupiter and Mars were in Leo.")
        self.assertTrue(all(s.joint_day == 15 for s in atoms))
        summary = S.extract("Month III, the 15th, clouds. Jupiter and Mars were in Leo.")
        self.assertTrue(all(s.joint_day is None for s in summary))

    def test_missing_month_is_untestable(self):
        atoms = S.extract("Jupiter and Mars were in Leo.")
        self.assertEqual(S.check(atoms[0], -149, []).result, "untestable")

    def test_month_alternatives_are_not_joined_into_one_span(self):
        st = replace(self.atoms[0], month="II|IV")
        windows = S.joint_windows(st, -149)
        self.assertEqual(len(windows), 2)
        self.assertLess(windows[0][1], windows[1][0])

    def test_same_month_but_no_simultaneous_overlap(self):
        def occupancy(body, sign, lo, hi):
            return [(lo, lo + 5)] if body == "jupiter" else [(lo + 10, hi)]
        with patch.object(sky, "sign_occupancy_intervals", side_effect=occupancy):
            self.assertEqual(S.check(self.atoms[0], -149, ["III"]).result, "contradict")

    def test_true_overlap_agrees_and_reports_interval(self):
        def occupancy(body, sign, lo, hi):
            return [(lo, lo + 15)] if body == "jupiter" else [(lo + 10, hi)]
        with patch.object(sky, "sign_occupancy_intervals", side_effect=occupancy):
            v = S.check(self.atoms[0], -149, ["III"])
            self.assertEqual(v.result, "agree")
            self.assertIn("JD", v.detail)

    def test_boundary_contact_is_not_positive_overlap(self):
        def occupancy(body, sign, lo, hi):
            return [(lo, lo + 5)] if body == "jupiter" else [(lo + 5, hi)]
        with patch.object(sky, "sign_occupancy_intervals", side_effect=occupancy):
            self.assertEqual(sky.joint_sign_intervals(self.atoms[0].joint_positions, 0, 20), [])

    def test_shared_score_and_whole_source_removal(self):
        def verdict(st, year, months):
            return S.Verdict(st, "agree" if year == -149 else "contradict", "fixture")
        with patch.object(S, "check", side_effect=verdict):
            scores = S.candidate_scores(self.atoms, ["III"], [-149, -148])
        np.testing.assert_allclose(scores.weights, [0.5, 0.5])
        np.testing.assert_array_equal(scores.priority, [2, 0])
        np.testing.assert_array_equal(scores.agree, [1, 0])
        np.testing.assert_array_equal(scores.contradict, [0, 1])
        np.testing.assert_array_equal(scores.strict, [True, False])
        np.testing.assert_array_equal(scores.safety, [True, True])

    def test_sirius_subject_is_not_sun(self):
        atoms = S.extract("Month III, the 15th, Sirius' first appearance.")
        self.assertEqual(len(atoms), 1)
        self.assertEqual((atoms[0].kind, atoms[0].planet), ("sirius", "sirius"))
        self.assertEqual(S.evidence_families(atoms[0]), ("seasonal_sirius",))

    def test_sirius_relabelling_does_not_change_physics(self):
        st = S.extract("Month III, the 15th, Sirius' first appearance.")[0]
        self.assertEqual(S.check(st, -149, ["III"]).result,
                         S.check(replace(st, planet="sun"), -149, ["III"]).result)

    def test_filtered_atom_does_not_keep_an_absent_family(self):
        def verdict(st, year, months):
            return S.Verdict(st, "agree" if year == -149 else "contradict", "fixture")
        with patch.object(S, "check", side_effect=verdict):
            scores = S.candidate_scores(self.atoms[:1], ["III"], [-149, -148])
        np.testing.assert_array_equal(scores.priority, [1, 0])

    def test_subdaily_retrograde_crossing_is_found(self):
        # Both daily endpoints are in Aries, but the parabola reaches Taurus
        # between them. A daily boolean mask would miss this interval entirely.
        spline = CubicSpline([0., 1.], np.radians([29., 29.]),
                             bc_type=((1, np.radians(8)), (1, np.radians(-8))))
        sky._sign_occupancy_for.cache_clear()
        try:
            with patch.object(sky, "_spline_date", return_value=spline), \
                 patch.object(sky, "BABYLONIAN_OFFSET_YEAR0_DEG", 0.), \
                 patch.object(sky, "BABYLONIAN_PRECESSION_DEG_PER_CENTURY", 0.):
                intervals = sky.sign_occupancy_intervals("mars", "Taurus", 0., 1.)
                self.assertEqual(len(intervals), 1)
                self.assertAlmostEqual(intervals[0][0], (1 - np.sqrt(.5)) / 2, places=7)
                self.assertAlmostEqual(intervals[0][1], (1 + np.sqrt(.5)) / 2, places=7)
        finally:
            sky._sign_occupancy_for.cache_clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
