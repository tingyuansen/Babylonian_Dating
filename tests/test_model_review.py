"""Offline checks of the model review's code-side rules.

No model is called. The checks cover what code decides whatever the model
says: what the model is shown, which unobserved mark a clause gets, and when a
cited passage may change or drop a clause.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tools import statements as S  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "model_review", HERE / "scripts/agent/model_review.py")
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


def payload_for(text: str, st: S.Statement) -> dict:
    return v.review_payload(v.blind(text), v.record_token("X100000"), 0, st)


class BlindingTest(unittest.TestCase):
    TEXT = ("Year 38 , kings Antiochus and Se [ leucus ] . Month VII , the 1st . "
            "Night of the 3rd , the moon was 2 cubits above α Leonis .")

    def test_year_numbers_and_rulers_are_withheld(self):
        masked = v.blind(self.TEXT).masked
        self.assertNotIn("38", masked)
        self.assertNotIn("Antiochus", masked)
        self.assertNotIn("leucus", masked)
        self.assertIn("Year {year} , kings {ruler} and {ruler} [ ]", masked)
        self.assertIn("the moon was 2 cubits above α Leonis", masked)

    def test_offsets_map_back_to_the_original(self):
        blind = v.blind(self.TEXT)
        quote = "the moon was 2 cubits above α Leonis"
        start = blind.masked.index(quote)
        original_start = blind.to_original_start(start)
        original_end = blind.to_original_end(start + len(quote))
        self.assertEqual(self.TEXT[original_start:original_end], quote)
        token = blind.masked.index("{ruler}")
        self.assertEqual(self.TEXT[blind.to_original_start(token):
                                   blind.to_original_end(token + 7)], "Antiochus")

    def test_payload_has_no_record_identifier(self):
        st = S.extract(self.TEXT)[0]
        payload = payload_for(self.TEXT, st)
        self.assertNotIn("tablet", payload)
        self.assertTrue(payload["record"].startswith("R"))
        self.assertNotIn("X100000", str(payload))
        self.assertNotIn("38", payload["context_before"])


class UnobservedRuleTest(unittest.TestCase):
    def test_parser_rule_is_the_only_rule(self):
        text = "Month IX , night of the 14th , lunar eclipse ; I did not watch ."
        eclipse = [st for st in S.extract(text) if st.kind == "eclipse"][0]
        self.assertTrue(eclipse.omitted)
        self.assertEqual(eclipse.omitted, S.unwatched(text, "eclipse", eclipse.source_start,
                                                       eclipse.source_end))
        season = "Month IV , the 24th , solstice ; I did not watch ."
        st = [x for x in S.extract(season) if x.kind == "season"][0]
        self.assertFalse(st.omitted)
        self.assertFalse(S.unwatched(season, "season", st.source_start, st.source_end))

    def test_model_cannot_change_the_mark(self):
        text = "Month IV , the 24th , solstice ; I did not watch ."
        st = [x for x in S.extract(text) if x.kind == "season"][0]
        payload = payload_for(text, st)
        final = dict(v._original_review_final(payload), omitted=True)
        held = v._hold_code_fields(payload, {"action": "MODIFY", "final": final})
        self.assertFalse(held["final"]["omitted"])
        self.assertEqual(held["_code_field_overrides"]["omitted"]["proposed"], True)
        modified = v.apply_review(st, {"action": "MODIFY", "final": final}, text)
        self.assertFalse(modified.omitted)


class LocalEvidenceTest(unittest.TestCase):
    TEXT = ("The 7th , Mercury's first appearance in the east in Cancer . "
            "The 8th , Mars' first appearance in the east in Cancer .")

    def setUp(self):
        self.st = [x for x in S.extract(self.TEXT) if x.planet == "mars"][0]
        self.payload = payload_for(self.TEXT, self.st)

    def drop(self, quote, field="planet", value="Mercury"):
        return v.drop_check_errors(self.payload, {
            "verdict": "CONFIRM_DROP", "reason": "wrong_subject_or_target",
            "contradicted_field": field, "source_value": value,
            "evidence_quote": quote})

    def test_drop_quoting_another_sentence_is_rejected(self):
        errors = self.drop("Mercury's first appearance in the east in Cancer")
        self.assertIn("evidence_quote is not in the clause's own sentence", errors)

    def test_drop_must_contain_a_different_value(self):
        quote = "Mars' first appearance in the east in Cancer"
        self.assertTrue(self.drop(quote, value="Mars"))
        self.assertTrue(self.drop(quote, field="sign", value="Leo"))
        self.assertEqual(self.drop(quote, field="assertion", value="not"),
                         ["source_value 'not' is not in evidence_quote"])

    def test_modify_needs_the_new_value_in_the_quote(self):
        quote = "Mars' first appearance in the east in Cancer"
        final = v._original_review_final(self.payload)
        self.assertTrue(v._field_value_errors("sign", "Leo", final, quote))
        self.assertEqual(v._field_value_errors("sign", "Cancer", final, quote), [])
        self.assertTrue(v._field_value_errors("day", 7, final, "The 8th , Mars'"))
        self.assertEqual(v._field_value_errors("day", 8, final, "The 8th , Mars'"), [])
        self.assertTrue(v._field_value_errors("year_offset", 1, final, quote))
        self.assertFalse(v._quote_in_clause_sentence(
            self.payload, "Mercury's first appearance in the east in Cancer"))
        self.assertTrue(v._quote_in_clause_sentence(self.payload, quote))


class GroundingTest(unittest.TestCase):
    def test_antecedent_binds_to_the_nearest_earlier_mention(self):
        source = ("when Jupiter became stationary to the west . Month XII , "
                  "when Jupiter became stationary to the west , it became stationary in Cancer .")
        clause = source.index("it became")
        quote, start, end = v._ground_quote_to_exact_source(
            "when Jupiter became stationary to the west", source, before=clause)
        self.assertEqual(start, source.rindex("when Jupiter", 0, clause))
        self.assertEqual(v._ground_quote_to_exact_source(quote, source)[1], 0)

    def test_validate_final_canonicalizes_stars_and_signs(self):
        base = {"kind": "appulse", "planet": "moon", "target_planet": None,
                "star": "alpha Leonis", "sign": "Leo", "month": "I", "day": 3,
                "year_offset": 0, "measurement_cubits": 2, "measurement_fingers": None,
                "direction": "above", "station_phase": None, "omitted": False,
                "restored": False, "event_subtype": None}
        final = v.validate_final(base)
        self.assertEqual((final["star"], final["sign"]), ("α Leonis", ""))
        with self.assertRaises(ValueError):
            v.validate_final(dict(base, star="alpha Nonexistentis"))
        sign = v.validate_final(dict(base, kind="in_sign", star=None, sign="Scorpius",
                                     measurement_cubits=None, direction=None))
        self.assertEqual(sign["sign"], "Scorpio")


class SearchRulesTest(unittest.TestCase):
    def test_distance_must_stand_before_the_relation(self):
        quote = "the moon was 3 cubits in front of γ Virginis , the moon being 4 cubits low to the south"
        final = {"kind": "appulse", "measurement_cubits": 4, "measurement_fingers": None,
                 "direction": "in front of"}
        self.assertTrue(v._measurement_binding_errors(final, quote))
        self.assertEqual(v._measurement_binding_errors(dict(final, measurement_cubits=3), quote), [])
        self.assertEqual(v._bound_measurements("the moon was x ] +1 cubits [ in front of α Virginis"), [])
        self.assertEqual(v._bound_measurements("the moon was 1 ? 1/2 cubits behind β Capricorni"), [])
        self.assertEqual(v._bound_measurements("2 ? [ cubits ] 6 fingers [ below ] Jupiter"),
                         [(2.0, 6.0, "below")])

    def test_eclipse_report_is_one_event(self):
        text = ("Month XII , night of the 14th , lunar eclipse ; when it began on the east side , "
                "it made ( an eclipse ) ; ( the moon ) became eclipsed in Libra . Night of the 18th , "
                "the moon became eclipsed .")
        held = [st for st in S.extract(text) if st.kind == "eclipse"]
        later = S.Statement(kind="eclipse", planet="lunar", sign="", text="",
                            source_start=text.index("( the moon )"), source_end=text.index(" in Libra"))
        self.assertTrue(v._continues_existing_eclipse(text, later, held))
        other = S.Statement(kind="eclipse", planet="lunar", sign="", text="",
                            source_start=text.rindex("the moon became"), source_end=len(text) - 2)
        self.assertFalse(v._continues_existing_eclipse(text, other, held))

    def test_bracketed_or_queried_passage_counts_as_restored(self):
        def restored(text: str, passage: str) -> bool:
            start = text.index(passage)
            return v._passage_restored(text, start, start + len(passage))
        self.assertFalse(restored("Month II. Mars was in Taurus.", "Mars was in Taurus"))
        self.assertTrue(restored("Month II. Ju ] piter was in Leo.", "Ju ] piter was in Leo"))
        self.assertTrue(restored("Month II. Saturn was in Ge ] mini.", "Saturn was in Ge ] mini"))
        self.assertTrue(restored("Month II. Mars? was in Taurus.", "Mars? was in Taurus"))
        self.assertTrue(restored("Month II. [Mars was in Taurus.", "Mars was in Taurus"))


class ClientPlumbingTest(unittest.TestCase):
    def test_schema_check_used_in_json_mode(self):
        schema = v.DROP_VERIFY_FORMAT["schema"]
        good = {"verdict": "KEEP", "reason": "no_positive_reason",
                "contradicted_field": "none", "source_value": "",
                "explanation": "", "evidence_quote": ""}
        self.assertEqual(v._schema_errors(good, schema), [])
        self.assertTrue(v._schema_errors(dict(good, verdict="MAYBE"), schema))
        self.assertTrue(v._schema_errors({"verdict": "KEEP"}, schema))

    def test_limiter_halves_on_429_and_grows_on_success(self):
        limiter = v.AdaptiveLimiter(8, 32)
        limiter.acquire()
        limiter.release(ok=False, throttled=True, retry_after=0.01)
        self.assertEqual((limiter.limit, limiter.rate_limited), (4, 1))
        for _ in range(8):  # two rounds of successes at limit 4
            limiter.in_flight += 1
            limiter.release(ok=True)
        self.assertEqual(limiter.limit, 5)


if __name__ == "__main__":
    unittest.main()
