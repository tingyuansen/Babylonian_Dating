"""Offline tests of the read-only MCP server in scripts/agent/mcp_server.py.

Each tool runs once on a short text. The tests check the list of tools, the
fields and values of each output, and the refusal of bad input. Neither the
outputs nor the server's code may name a field that holds a published date. The
tools may read only the tables in data/physical/. Two tests need the mcp package
and are skipped without it.

    python3 tests/test_mcp_server.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from scripts.agent import mcp_server as server
from tools import sky

TEXT = "Month III. The 5th, Jupiter was in Leo. Night of the 14th, lunar eclipse."
ANSWER = re.compile(
    r"bce_first|bce_second|ancient_year|published|truth|reference", re.I)
NAMES = {"extract_clauses", "clause_verdict", "score_text",
         "babylonian_to_julian", "julian_to_babylonian", "body_position",
         "eclipses_near"}


def keys(value) -> set[str]:
    """Every field name at any depth of a tool's output."""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys(v)}
    return set()


class McpServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Each tool runs once, and each table that it opens is recorded.
        cls.tables = []
        read = pd.read_parquet

        def recorded(path, *args, **kwargs):
            cls.tables.append(Path(path).resolve())
            return read(path, *args, **kwargs)

        with patch.object(pd, "read_parquet", side_effect=recorded):
            clauses = server.extract_clauses(TEXT)
            cls.eclipse = next(c["index"] for c in clauses["clauses"]
                               if c["kind"] == "eclipse")
            date = server.babylonian_to_julian(150, "III", 14)
            julian = (date["julian_year_astro"], date["julian_month"],
                      date["julian_day"], "ASTRO")
            cls.out = {
                "extract_clauses": clauses,
                "clause_verdict": server.clause_verdict(TEXT, cls.eclipse, 150),
                "clause_verdict_all": server.clause_verdict(TEXT, cls.eclipse),
                "score_text": server.score_text(TEXT),
                "babylonian_to_julian": date,
                "julian_to_babylonian": server.julian_to_babylonian(*julian),
                "body_position": server.body_position("jupiter", *julian),
                "eclipses_near": server.eclipses_near(*julian),
            }

    @unittest.skipUnless(importlib.util.find_spec("mcp"),
                         "the mcp package is not installed")
    def test_lists_read_only_tools(self):
        tools = asyncio.run(server.build_server().list_tools())
        self.assertEqual({t.name for t in tools}, NAMES)
        self.assertTrue(all(t.annotations.readOnlyHint for t in tools))
        self.assertTrue(all(t.description for t in tools))

    @unittest.skipUnless(importlib.util.find_spec("mcp"),
                         "the mcp package is not installed")
    def test_call_through_the_protocol(self):
        content = asyncio.run(server.build_server().call_tool(
            "extract_clauses", {"text": TEXT}))
        self.assertEqual(json.loads(content[0].text),
                         json.loads(json.dumps(self.out["extract_clauses"])))

    def test_extracts_clauses_with_fields_and_rule(self):
        out = self.out["extract_clauses"]
        self.assertEqual(out["months"], ["III"])
        self.assertEqual({(c["kind"], c["planet"], c["sign"])
                          for c in out["clauses"]},
                         {("in_sign", "jupiter", "Leo"),
                          ("eclipse", "lunar", "")})
        for clause in out["clauses"]:
            self.assertEqual(clause["month"], "III")
            self.assertTrue(clause["parser_rule"])
            self.assertTrue(clause["decisive_type"])

    def test_verdict_in_one_year_and_in_all(self):
        one = self.out["clause_verdict"]
        self.assertEqual((one["year_astro"], one["verdict"]), (-149, "agree"))
        every = self.out["clause_verdict_all"]
        self.assertEqual(every["agree"] + every["contradict"]
                         + every["untestable"], 701)
        self.assertEqual(every["agree"], len(every["agreeing_years_astro"]))
        self.assertIn(-149, every["agreeing_years_astro"])
        self.assertGreater(every["bits"], 0)

    def test_score_is_the_pipeline_decision(self):
        out = self.out["score_text"]
        self.assertIn(out["decision"], {"committed", "ranked", "decline"})
        self.assertTrue(set(out["strict_set_astro"])
                        <= set(out["safety_set_astro"]))
        self.assertEqual(out["first_groups"][0]["tier"], "strict")
        self.assertEqual(len(out["clauses"]), 2)

    def test_calendar_sky_and_eclipses_agree(self):
        back = self.out["julian_to_babylonian"]
        self.assertEqual((back["babylonian_year_astro"], back["month"],
                          back["day"]), (-149, "III", 14))
        position = self.out["body_position"]
        self.assertIn(position["sign"], sky.SIGNS)
        self.assertAlmostEqual(position["degrees_in_sign"],
                               position["babylonian_longitude_deg"] % 30, 3)
        lunar = [e for e in self.out["eclipses_near"]["eclipses"]
                 if e["kind"] == "lunar" and abs(e["days_from_date"]) <= 1.5]
        self.assertTrue(lunar and lunar[0]["agreement_eligible"])

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):
            server.clause_verdict(TEXT, 5)
        with self.assertRaises(ValueError):
            server.clause_verdict(TEXT, 0, 700)
        with self.assertRaises(ValueError):
            server.babylonian_to_julian(150, "XIII", 1)
        with self.assertRaises(ValueError):
            server.body_position("jupiter", 900, 1, 1)

    def test_no_output_names_an_answer(self):
        for name, out in self.out.items():
            leaks = sorted(k for k in keys(out) if ANSWER.search(k))
            self.assertFalse(leaks, f"{name} returns {leaks}")

    def test_only_physical_tables_are_read(self):
        physical = (HERE / "data" / "physical").resolve()
        self.assertTrue(self.tables)
        for path in self.tables:
            self.assertEqual(path.parent, physical, path)

    def test_source_names_no_answer_column(self):
        # The scan that verify.py runs over the modules that build candidate
        # sets, with comments and docstrings removed.
        source = Path(server.__file__).read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#"))
        code = re.sub(r'"""(?:.|\n)*?"""', "", code)
        self.assertIsNone(re.search(
            r"bce_first|bce_second|ancient_year|reference_year|\btruth\b",
            code))
        # Nor does it name a directory that holds published dates.
        self.assertIsNone(
            re.search(r"[\"'/](results|audit|corpus)[\"'/]", code))


if __name__ == "__main__":
    unittest.main(verbosity=2)
