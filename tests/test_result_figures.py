#!/usr/bin/env python3
"""Offline tests of the figure scripts in scripts/figures/.

The tests check that the curve of Fig. 4a reaches 100% at the last candidate
year, and that the rows of SI Fig. S1 match the configurations of the sensitivity audit.
They check that the cells of SI Fig. S2 sit on their years in a fixed order, and
that the labels of Figs. 1 to 3 do not overlap or leave the figure. The tests
use small fixtures and read no result table.

    python3 tests/test_result_figures.py
"""

from __future__ import annotations

import ast
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

import fig1_workflow as workflow
import fig2_diary as diary
import fig3_x301492 as tablet
import fig4_benchmark as performance
import figS1_sensitivity as ablation
import figS2_adart_iv as timeline


class ResultFiguresTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_recall_curve_reaches_complete_domain(self):
        for candidate_count in (701, 800):
            with self.subTest(candidate_count=candidate_count):
                fig, ax = plt.subplots()
                places = np.array([1, 617, 618, 700, candidate_count])
                performance.curve(ax, places, "blue", "test", False,
                                  candidate_count)
                x, y = ax.lines[0].get_data()
                self.assertEqual(x[-1], candidate_count)
                self.assertEqual(y[-1], 100.0)
                self.assertEqual(y[616], 40.0)

    def test_ablation_labels_cover_the_actual_audit(self):
        tree = ast.parse(
            (ROOT / "scripts" / "audit" / "audit_sensitivity.py")
            .read_text(encoding="utf-8"))
        config_node = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CONFIGS"
                    for t in node.targets))
        actual = {row["config"] for row in ast.literal_eval(config_node.value)}
        self.assertEqual({key for key, _, _ in ablation.CONFIGS}, actual)
        summary = pd.DataFrame(index=sorted(actual))
        ablation.validate_configurations(summary)
        self.assertEqual(len(ablation.layout()[0]), len(actual))

    def test_ablation_refuses_silent_configuration_omission(self):
        expected = [key for key, _, _ in ablation.CONFIGS]
        with self.assertRaises(ValueError):
            ablation.validate_configurations(pd.DataFrame(index=expected[:-1]))
        with self.assertRaises(ValueError):
            ablation.validate_configurations(
                pd.DataFrame(index=expected + ["new_unlabelled_configuration"]))

    def test_timeline_cells_have_correct_year_centres_and_stable_order(self):
        years = np.arange(-625, 76)
        source = pd.DataFrame({
            "claim": ["ranked", "ranked"],
            "strict_years_astro": [[int(years[0]), int(years[-1])],
                                 [int(years[0])]],
        })
        rendered = []
        def capture(fig, stem):
            rendered.append(fig)
            return stem.with_suffix(".pdf")
        for frame in (source, source.iloc[::-1]):
            with patch.object(timeline.pd, "read_parquet", return_value=frame), \
                 patch.object(timeline.cal, "candidate_years", return_value=years), \
                 patch.object(timeline.fs, "save", side_effect=capture):
                timeline.main()
        ax = rendered[0].axes[0]
        image = ax.images[0]
        left, right, _, _ = image.get_extent()
        step = (right - left) / len(years)
        self.assertEqual(left + step / 2, years[0])
        self.assertEqual(right - step / 2, years[-1])
        self.assertEqual(ax.get_xlim(), (years[0] - 0.5, years[-1] + 0.5))
        np.testing.assert_array_equal(
            rendered[0].axes[0].images[0].get_array(),
            rendered[1].axes[0].images[0].get_array())
        self.assertEqual(image.get_array()[:, 0].sum(), 2)
        self.assertEqual(image.get_array()[:, -1].sum(), 1)

    def test_diary_full_calendar_information_label_is_not_clipped(self):
        years = np.arange(-625, 76)
        source = pd.DataFrame({"tablet": [diary.TABLET], "text": ["Weather."]})
        scores = SimpleNamespace(
            outcomes=np.empty((len(years), 0), dtype=int),
            decisive=np.empty(0, dtype=bool), weights=np.empty(0))
        figures = []
        def capture(fig, stem):
            figures.append(fig)
            return stem.with_suffix(".pdf")
        with patch.object(diary.pd, "read_parquet", return_value=source), \
             patch.object(diary.cal, "candidate_years", return_value=years), \
             patch.object(diary.clauses, "load", return_value={diary.TABLET: []}), \
             patch.object(diary.S, "candidate_scores", return_value=scores), \
             patch.object(diary.plt, "imread", return_value=np.zeros((12, 12, 3))), \
             patch.object(diary.fs, "save", side_effect=capture):
            diary.main()
        fig = figures[0]
        fig.canvas.draw()
        text = next(t for t in fig.axes[-1].texts
                    if t.get_text() == "all 701, 0.0 bits")
        box = text.get_window_extent(fig.canvas.get_renderer())
        self.assertLessEqual(box.x1, fig.bbox.x1)

    def test_worked_example_count_headings_do_not_overlap(self):
        years = np.arange(-625, 76)
        source = pd.DataFrame({"tablet": [tablet.TABLET], "text": ["Test."]})
        statement = SimpleNamespace(kind="station", planet="jupiter",
                                    sign="Aquarius")
        scores = SimpleNamespace(
            outcomes=np.ones((len(years), 1), dtype=int),
            decisive=np.array([True]), weights=np.array([1.0]),
            strict=np.arange(len(years)) == 0,
            score=np.linspace(0, 1, len(years)),
            priority=np.ones(len(years), dtype=int))
        figures = []
        def capture(fig, stem):
            figures.append(fig)
            return stem.with_suffix(".pdf")
        with patch.object(tablet.pd, "read_parquet", return_value=source), \
             patch.object(tablet.cal, "candidate_years", return_value=years), \
             patch.object(tablet.clauses, "load", return_value={tablet.TABLET: [statement]}), \
             patch.object(tablet.S, "candidate_scores", return_value=scores), \
             patch.object(tablet.fs, "save", side_effect=capture):
            tablet.main()
        fig = figures[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = {t.get_text(): t for t in fig.axes[0].texts}
        alone = texts["alone"].get_window_extent(renderer)
        combined = texts["combined"].get_window_extent(renderer)
        self.assertLess(alone.x1, combined.x0)
        self.assertLessEqual(combined.x1, fig.bbox.x1)
        for label in fig.axes[0].get_yticklabels():
            self.assertGreaterEqual(label.get_window_extent(renderer).x0, 0)

    def test_workflow_labels_stay_apart_and_inside_their_panels(self):
        figures = []
        def capture(fig, stem):
            figures.append(fig)
            return stem.with_suffix(".pdf")
        with patch.object(workflow.fs, "save", side_effect=capture):
            workflow.main()
        fig = figures[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [(t.get_text(), t.get_window_extent(renderer))
                 for t in fig.axes[0].texts if t.get_text().strip()]
        # Font rasterizers differ by under a pixel between matplotlib builds,
        # so a touch within 1.5 pixels does not count as an overlap.
        tolerance = 1.5
        for (first, a), (second, b) in itertools.combinations(boxes, 2):
            self.assertFalse(a.padded(-tolerance).overlaps(b.padded(-tolerance)),
                             f"{first!r} overlaps {second!r}")
        for label, extent in boxes:
            self.assertTrue(fig.bbox.contains(extent.x0, extent.y0)
                            and fig.bbox.contains(extent.x1, extent.y1), label)
            x0, x1 = extent.x0 / fig.dpi, extent.x1 / fig.dpi
            y0, y1 = extent.y0 / fig.dpi, extent.y1 / fig.dpi
            if workflow.BOTTOM <= y0 and y1 <= workflow.TOP:
                slack = tolerance / fig.dpi
                self.assertTrue(any(left - slack <= x0 and x1 <= right + slack
                                    for left, right in workflow.PANELS), label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
