#!/usr/bin/env python3
"""Write the data of the interactive site in docs/ from the committed results.

The site tells, for one record at a time, how the astronomy narrows its date:
its clauses, the verdict of each clause in each of the 701 candidate years,
the clocks K, the decisive bits I_D and the supporting bits I_S of each year,
the strict and safety sets, and the decision. It holds no value of its own.
This script recomputes the per-year scores with tools.statements, checks that
they give the committed strict and safety sets of every record, and writes:

    docs/data/meta.json             the candidate years and one summary row per
                                    record, for the chooser
    docs/data/records/<id>.json     the story of one record, with its tablet lines
    docs/data/sky.json              the Babylonian longitude of the Sun and the
                                    five planets at the start and the end of
                                    every month of the 701 years

The site covers the 417 dated records of ADART I to III, whose translation is
that of ADsD (CC0). The undated ADART IV entries stay in data/results/. The
site tells one record at a time, so the paper's aggregate results stay in the
paper.

Each record also tells what the language model of step 1 did, from the
committed run in data/clauses/model_review_<model>/: the review of each parser
clause, each field change that the review proposed and what the checks did
with it, and the clauses that the search found in the text. The dates use the
final clauses of data/clauses/reviewed_clauses.parquet: the parser clauses as
the review leaves them, and the clauses that the search added. A parser clause
that a rule from a coding agent read is marked too
(data/clauses/parser_clauses.parquet).

Usage:
    python scripts/site/build_site_data.py
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, clauses as final_clauses, sky, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402

sys.path.insert(0, str(HERE / "scripts" / "site"))
import cuneiform  # noqa: E402  (the edition's readings drawn as signs)

sys.path.insert(0, str(HERE / "scripts" / "agent"))
# The year and ruler masker of step 1, the key under which its review keeps
# each parser clause, and the event subtype that the review gives a clause.
from model_review import blind, cache_key, infer_event_subtype  # noqa: E402


OUT = HERE / "docs" / "data"
N_YEARS = 701
VERDICT = {1: "a", -1: "c", 0: "."}   # agree, contradict, untestable
VOLUME = {"adsd/adart1": "ADART I", "adsd/adart2": "ADART II", "adsd/adart3": "ADART III"}
SKY_BODIES = ["sun", "mercury", "venus", "mars", "jupiter", "saturn"]
MONTHS = ["I", "II", "III", "IV", "V", "VI", "VI2", "VII", "VIII", "IX", "X", "XI", "XII", "XII2"]
# Records that the story opens with: the worked example of the paper (Fig. 3),
# the record of Fig. 2, a commitment on a lunar eclipse, and a commitment on
# two or more clocks.
FEATURED = ["X301492", "X301322", "X300993", "X104181"]
# The rulers of Babylon up to the Seleucid Era, in astronomical years (-625 is 626 BCE),
# as the tables of Parker and Dubberstein give them (data/source/parker_dubberstein/).
# A year that two rulers share goes to the earlier one.
REIGNS = [
    ("Nabopolassar", -625, -604), ("Nebuchadnezzar II", -603, -561), ("Amel-Marduk", -560, -559),
    ("Neriglissar", -558, -555), ("Nabonidus", -554, -538), ("Cyrus", -537, -529), ("Cambyses", -528, -521),
    ("Darius I", -520, -485), ("Xerxes", -484, -464), ("Artaxerxes I", -463, -423), ("Darius II", -422, -404),
    ("Artaxerxes II", -403, -358), ("Artaxerxes III", -357, -337), ("Arses", -336, -335),
    ("Darius III", -334, -330), ("Alexander III", -329, -322), ("Philip Arrhidaeus", -321, -316),
    ("Alexander IV", -315, -311),
]
# The periods of rule, for the band under the years. From 311 BCE the tables count in the
# Seleucid Era; Arsacid kings hold Babylon from 141 BCE.
PERIODS = [
    ("babylon", "Babylonian kings", "Babylon", -625, -538), ("persia", "Persian kings", "Persia", -537, -330),
    ("macedon", "Macedonian kings", "Macedon", -329, -311), ("seleucid", "Seleucid kings", "Seleucid", -310, -141),
    ("arsacid", "Arsacid kings", "Arsacid", -140, 75),
]
# Which years get a written reason for each clause verdict: the leading years,
# the published year, and the strict set when it is small enough to read.
DETAIL_TOP = 12
DETAIL_STRICT_MAX = 40

KIND_LABEL = {
    "appulse": "Moon or planet against a Normal Star",
    "planet_appulse": "Moon against a planet",
    "station_appulse": "Station against a Normal Star",
    "planet_planet_appulse": "Planet against a planet",
    "in_sign": "Planet in a sign",
    "entry": "Planet enters a sign",
    "station": "Station in a sign",
    "first_app": "First appearance in a sign",
    "last_app": "Last appearance in a sign",
    "acronychal": "Acronychal rising",
    "eclipse": "Eclipse",
    "season": "Solstice or equinox",
    "sirius": "Sirius date",
}
# The period, in years, after which a clock repeats closely enough that its
# agreeing years line up. Planets: the sidereal period. Lunar eclipses: the
# Saros. The solar year and the Moon against the stars: the 19-year cycle of
# the Babylonian calendar.
CLOCK_PERIOD = {
    "mars": 1.8809, "jupiter": 11.862, "saturn": 29.457, "venus": 8.0,
    "mercury": 46.0, "moon": 19.0, "lunar_eclipse": 18.03,
    "seasonal_sirius": 19.0, "solar": 18.03, "sun": 1.0,
}
# The committed run of step 1 that the site tells.
REVIEW_MODEL = "gpt-6-luna-2026-09-22"
REVIEW_RUN = HERE / "data" / "clauses" / f"model_review_{REVIEW_MODEL}"
# How code starts its reason when it rejects a change that the model accepted.
CODE_REJECTION = "ACCEPT was rejected by code because "
# The words of the labels of the clauses that the search found.
APPEARANCE = {"first_app": "first appearance", "last_app": "last appearance",
              "acronychal": "acronychal rising"}
SIRIUS_EVENT = {"sirius_rising": "first appearance", "sirius_setting": "last appearance",
                "sirius_acronychal": "acronychal rising"}


def rounded(values, digits: int) -> list:
    return [round(float(v), digits) for v in values]


def years_list(value) -> list[int]:
    return [int(y) for y in (value if value is not None else [])]


def load_review_run() -> dict:
    """The language model's part, from the committed run of step 1.

        review   the review record of each parser clause, under the key that
                 the review gave it (record, text, clause index and clause)
        found    the clauses that the search added, by record, in run order
        tiers    the clauses of the run's final list, by record, with the tier
                 of each one
        origin   the parser clauses, by record, with the origin of each rule
        clauses  the final clauses, by record, each with its origin, parser
                 index and review action
    """
    review = {}
    with (REVIEW_RUN / "review_cache.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                review[row["cache_key"]] = row
    search = pd.read_csv(REVIEW_RUN / "search_proposals.csv")
    bits = pd.read_parquet(REVIEW_RUN / "reviewed_clause_bits.parquet")
    parser = pd.read_parquet(HERE / "data" / "clauses" / "parser_clauses.parquet")
    return {
        "review": review,
        "found": {t: g for t, g in search[search.status == "added"].groupby("tablet", sort=False)},
        "tiers": {t: g.sort_values("statement_index") for t, g in bits.groupby("tablet")},
        "origin": {t: g.sort_values("statement_index") for t, g in parser.groupby("tablet")},
        "clauses": final_clauses.load(final_clauses.REVIEWED, provenance=True),
    }


def check_parser_clauses(tablet: str, sts: list, rows: pd.DataFrame | None) -> None:
    """Stop when the parser clauses of the run are not the clauses of the site.

    The review and the rule origins name a parser clause by its index, so each
    index must give the same passage at the same offsets as here.
    """
    count = 0 if rows is None else len(rows)
    if count != len(sts):
        raise RuntimeError(f"{tablet}: {count} parser clauses in the run, {len(sts)} here")
    for j, (st, row) in enumerate(zip(sts, rows.itertuples() if rows is not None else [])):
        if ((int(row.statement_index), int(row.source_char_start), int(row.source_char_end),
             row.normalized_text.strip())
                != (j, st.source_start, st.source_end, st.text.strip())):
            raise RuntimeError(f"{tablet}: parser clause {j} is not the clause of the run")


def field_change(j: int, st, record: dict, decision: dict) -> dict:
    """One field change that the review proposed, and what the checks did with it.

    The value before is the parser's, and the value after an accepted change is
    the run's. The run keeps no value for a rejected change, so that value is
    given only when it is certain: the other value of a true-or-false field,
    or the day or month that a rejection by code names.
    """
    field = decision["field"]
    before = (infer_event_subtype(st) if field == "event_subtype"
              else record["original_statement"].get(field))
    accepted = decision.get("decision") == "ACCEPT"
    explanation = decision.get("explanation") or ""
    by_code = explanation.startswith(CODE_REJECTION)
    reason = explanation[len(CODE_REJECTION):].rstrip(".") if by_code else decision.get("reason")
    change = {"j": j, "field": field, "from": before}
    named = re.match(rf"{field} (\S+) is not a {field} heading in the quote", reason or "")
    if accepted:
        change["to"] = record["review_final_fields"][field]
    elif isinstance(before, bool):
        change["to"] = not before
    elif by_code and named:
        # Code writes the value with repr(), so this reads it back exactly.
        change["to"] = ast.literal_eval(named.group(1))
    change["accepted"] = accepted
    # The passage that the confirming call cited, also when code rejected it,
    # else the passage of the first review.
    quote = (decision.get("evidence_quote") or decision.get("rejected_quote")
             or record.get("evidence_quote"))
    if quote:
        change["quote"] = quote
    change["why"] = reason
    return change


def model_review(tablet: str, text: str, sts: list, records: dict) -> tuple[dict, list]:
    """What the review did with each parser clause of one record.

    Each clause is looked up under the key that the review gave it, which holds
    the record, its text, the clause index and the clause. Returns the counts
    and the proposed field changes, and the final form of each clause (None
    when the review dropped it).
    """
    counts = {"KEEP": 0, "MODIFY": 0, "DROP": 0}
    changes, finals = [], []
    for j, st in enumerate(sts):
        record = records.get(cache_key(tablet, text, j, st, REVIEW_MODEL))
        if record is None:
            raise RuntimeError(f"{tablet}: parser clause {j} has no review in the step 1 run")
        counts[record["action"]] += 1
        finals.append(record["final_statement"])
        for decision in (record.get("modify_verification") or {}).get("decisions", []):
            changes.append(field_change(j, st, record, decision))
    review = {"kept": counts["KEEP"], "modified": counts["MODIFY"], "dropped": counts["DROP"]}
    if changes:
        review["changes"] = changes
    return review, finals


def place_quote(text: str, quote: str, start: int, end: int) -> tuple[int, int] | None:
    """Where a quote stands in the text: at its offsets, else at the exact match
    nearest to them. None when the text does not hold the quote."""
    if text[start:end] == quote:
        return start, end
    hits, at = [], text.find(quote)
    while quote and at >= 0:
        hits.append(at)
        at = text.find(quote, at + 1)
    if not hits:
        return None
    best = min(hits, key=lambda h: abs(h - start))
    return best, best + len(quote)


def fraction(value: float) -> str:
    """A number as the translation writes it: a whole part and a fraction, as 1 1/2."""
    exact = Fraction(value).limit_denominator(12)
    whole, part = divmod(exact.numerator, exact.denominator)
    rest = Fraction(part, exact.denominator)
    if not rest:
        return str(whole)
    return f"{whole} {rest}" if whole else str(rest)


def distance(cubits: float | None, unit: str | None) -> str:
    """A measured distance in the unit of the text: cubits, fingers or both.

    The run keeps every distance in cubits; one cubit is 24 fingers.
    """
    if cubits is None or (isinstance(cubits, float) and math.isnan(cubits)):
        return ""
    if unit in ("finger", "fingers"):
        fingers = fraction(cubits * 24)
        return f"{fingers} {'finger' if fingers == '1' else 'fingers'}"
    if unit == "cubit+finger":
        whole = math.floor(cubits + 1e-9)
        fingers = round((cubits - whole) * 24)
        parts = [f"{whole} {'cubit' if whole == 1 else 'cubits'}" if whole else "",
                 f"{fingers} {'finger' if fingers == 1 else 'fingers'}" if fingers else ""]
        return " ".join(p for p in parts if p)
    return f"{fraction(cubits)} {'cubits' if cubits > 1 + 1e-9 else 'cubit'}"


def found_label(st: dict) -> str:
    """A short label for a clause that the search found, from its fields only."""
    kind = st["kind"]
    rule = (st.get("parser_rule") or "").removesuffix("_bracket")
    body = (st.get("planet") or "").capitalize()
    sign = st.get("sign") or ""
    if kind in ("appulse", "planet_appulse", "planet_planet_appulse", "station_appulse"):
        other = st.get("star") or (st.get("target_planet") or "").capitalize()
        words = [body, "stationary" if kind == "station_appulse" else "",
                 distance(st.get("normalized_amount_cubits"), st.get("source_unit_label")),
                 st.get("direction") or "", other]
    elif kind == "in_sign":
        words = [body, f"in {sign}" if sign else ""]
    elif kind == "entry":
        words = [body, f"reaches {sign}" if sign else "reaches a sign"]
    elif kind == "station":
        words = [body, "stationary", f"in {sign}" if sign else ""]
    elif kind in APPEARANCE:
        words = [f"{body}, {APPEARANCE[kind]}" + (f", in {sign}" if sign else "")]
    elif kind == "eclipse":
        words = [body, "eclipse"]
    elif kind == "season":
        words = [rule.removeprefix("season_").capitalize()]
    elif kind == "sirius":
        words = ["Sirius,", SIRIUS_EVENT.get(rule, "date")]
    else:
        words = [KIND_LABEL.get(kind, kind)]
    label = " ".join(w for w in words if w)
    return label + ", not observed" if st.get("omitted") else label


def search_found(tablet: str, text: str, sts: list, finals: list,
                 rows: pd.DataFrame | None, tiers: pd.DataFrame | None,
                 index: dict) -> list[dict]:
    """The clauses that the search added to one record, placed in its text.

    The run's final list holds the reviewed parser clauses and the added ones,
    sorted as merge_recall_additions in scripts/agent/model_review.py sorts
    them, so its order gives the tier of each added clause in
    reviewed_clause_bits.parquet. index gives the place of each added clause in
    the final list, by its passage, kind and body. A quote that
    is not at its offsets is placed at the nearest exact match in the text; a
    quote that is not in the text is left out.
    """
    if rows is None:
        return []
    added = [(int(r.atomic_source_start), int(r.atomic_source_end),
              json.loads(r.final_statement_json), r.atomic_evidence_quote)
             for r in rows.itertuples()]
    final_list = [(st.source_start, st.source_end, final["kind"], final["planet"], None)
                  for st, final in zip(sts, finals) if final is not None]
    final_list += [(start, end, st["kind"], st["planet"], i)
                   for i, (start, end, st, _) in enumerate(added)]
    final_list.sort(key=lambda c: c[:4])
    if tiers is None or len(tiers) != len(final_list):
        raise RuntimeError(f"{tablet}: the run's final list does not match its scored clauses")
    decisive, restored = {}, {}
    for (_, _, kind, planet, i), row in zip(final_list, tiers.itertuples()):
        same = row.kind == kind and row.planet == planet
        if i is not None:
            st = added[i][2]
            same = same and row.parser_rule == st["parser_rule"]
            # The restored mark of the final clause, which code also sets from
            # the passage after the run.
            decisive[i], restored[i] = bool(row.decisive_tier), bool(row.restored)
        if not same:
            raise RuntimeError(f"{tablet}: the run's final list does not match its scored clauses")
    found = []
    for i, (start, end, st, quote) in enumerate(added):
        place = place_quote(text, quote, start, end) if isinstance(quote, str) else None
        if place is None:
            continue
        found.append({"start": place[0], "end": place[1], "quote": quote,
                      "what": found_label(st), "restored": restored[i],
                      "decisive": decisive[i],
                      "j": index[(start, end, st["kind"], st["planet"])]})
    return sorted(found, key=lambda c: (c["start"], c["end"]))


def record_story(tablet: str, text: str, years: list[int], saved: pd.Series,
                 published: int | None, review_run: dict) -> dict:
    """One record's clauses, verdict grid, scores, sets and decision, and what
    the language model of step 1 did with the record."""
    parsed = S.extract(text)
    items = review_run["clauses"].get(tablet, [])
    sts = [st for st, _, _, _ in items]
    months = S.months_in(text)
    story = {"id": tablet, "published_astro": published,
             "claim": saved.claim, "basis": saved.basis, "reason": saved.reason}
    story["text"] = text
    # The spans of year numbers and ruler names, which the language model sees
    # as placeholders. The site hides them until the edition's date is shown.
    story["hidden"] = [[int(a), int(b)] for a, b, _, _ in blind(text).spans]
    # The language model's part. The review names a parser clause by its index
    # in the parser's list, and the page by its place in the final list.
    origin = review_run["origin"].get(tablet)
    check_parser_clauses(tablet, parsed, origin)
    review, finals = model_review(tablet, text, parsed, review_run["review"])
    place = {index: j for j, (_, source, index, _) in enumerate(items) if source == "parser"}
    for change in review.get("changes", []):
        change["j"] = place[change["j"]]
    if parsed:
        story["review"] = review
    added = {(st.source_start, st.source_end, st.kind, st.planet): j
             for j, (st, source, _, _) in enumerate(items) if source == "search"}
    found = search_found(tablet, text, parsed, finals, review_run["found"].get(tablet),
                         review_run["tiers"].get(tablet), added)
    if found:
        story["found"] = found
    # What the parser clauses alone would give, for the comparison on the page.
    if len(parsed) != len(sts) or any(a != b for a, b in zip(parsed, sts)):
        alone = S.candidate_scores(parsed, months, years, "skill") if parsed else None
        alone_levels = S.evidence_levels(alone) if alone is not None else []
        story["parser_alone"] = {
            "clauses": len(parsed),
            "strict": int(alone.strict.sum()) if alone is not None else len(years),
            "first": len(alone_levels[0]["years_astro"]) if alone_levels else len(years)}
    # The sets and the first three groups of the ranking, as the results hold them.
    levels = json.loads(saved.evidence_levels_json) if isinstance(
        saved.evidence_levels_json, str) else []
    groups = [years_list(level.get("years_astro")) for level in levels[:3]]
    if not sts:
        # No clause, so no evidence: every candidate year stays in the
        # strict and safety sets. The page shows this case on its own. When the
        # results hold no ranking, all the candidate years are one group.
        story.update(no_clauses=True, clauses=[], grid=[], K=[], ID=[], IS=[], contradict=[],
                     strict=years_list(saved.strict_years_astro),
                     safety=years_list(saved.safety_years_astro),
                     groups=groups or [list(years)], details={})
        return story
    scores = S.candidate_scores(sts, months, years, "skill")
    year_arr = np.asarray(years)
    strict = [int(y) for y in year_arr[scores.strict]]
    if strict != years_list(saved.strict_years_astro):
        raise RuntimeError(f"{tablet}: the strict set does not match the committed result")
    safety = [int(y) for y in year_arr[scores.safety]]
    if safety != years_list(saved.safety_years_astro):
        raise RuntimeError(f"{tablet}: the safety set does not match the committed result")
    order = [int(year_arr[i]) for i in scores.order]
    clauses = []
    for j, (st, source, index, _) in enumerate(items):
        families = list(S.evidence_families(st))
        # A joint statement such as "Saturn and Mars were in Leo" gives one
        # clause per planet. Each clause shows the clock of its own body.
        clock = st.planet if st.planet in families else (families[0] if families else None)
        clauses.append({
            "kind": st.kind, "type": KIND_LABEL.get(st.kind, st.kind),
            "passage": st.text.strip(),
            "start": int(st.source_start), "end": int(st.source_end),
            "body": st.planet, "target": st.target_planet, "star": st.star,
            "sign": st.sign or None, "month": st.month, "day": st.day,
            "year_offset": int(st.year_offset),
            "joint": [p for p, _ in st.joint_positions if p != st.planet] if st.joint_positions else [],
            "restored": bool(st.restored), "unobserved": bool(st.omitted),
            "decisive": bool(scores.decisive[j] and scores.weights[j] > 0),
            "idle": bool(scores.decisive[j] and scores.weights[j] <= 0),
            "clock": clock,
            "period": CLOCK_PERIOD.get(clock) if clock else None,
            "agree": int(scores.agreeing[j]), "testable": int(scores.testable[j]),
            "bits": round(float(scores.weights[j]), 3),
            "rule": st.parser_rule,
        })
        # A rule that a coding agent proposed; the authors wrote and tested it.
        if source == "parser" and str(origin.rule_origin.iloc[index]).startswith("agent_suggested"):
            clauses[-1]["agent_rule"] = True
        if source == "search":
            clauses[-1]["found"] = True
    grid =["".join(VERDICT[int(v)] for v in scores.outcomes[:, j])
            for j in range(len(sts))]
    # A written reason for each clause in the years a reader will open first.
    wanted = set(order[:DETAIL_TOP]) | (set(strict) if len(strict) <= DETAIL_STRICT_MAX else set())
    if published is not None and published in years:
        wanted.add(published)
    details = {}
    for year in sorted(wanted):
        details[str(year)] = [S.check(st, year, months).detail for st in sts]
    story.update(
        clauses=clauses, grid=grid,
        K=[int(v) for v in scores.priority], ID=rounded(scores.score, 3),
        IS=rounded(scores.support, 3), contradict=[int(v) for v in scores.contradict],
        strict=strict, safety=safety, top=order[:DETAIL_TOP], groups=groups,
        details=details)
    return story


STATE = {"": "o", "damaged": "d", "broken": "b", "unread": "u"}


def glyph_line(translit: str, table: dict[str, str]) -> list[str]:
    """One line of the tablet as compact tokens: a state letter, then the text.

    o, d and b are a sign that is whole, damaged or broken; l is a reading with no
    glyph, kept in Latin letters; g is a gap where the clay is lost; u an unread
    sign; w a blank space on the clay; s a space between words.
    """
    out = []
    for kind, text, state, reading in cuneiform.to_cuneiform(translit, table):
        if kind == "space":
            out.append("s")
        elif kind == "gap":
            out.append({"unread": "u", "blank": "w"}.get(state, "g"))
        elif kind == "latin":
            out.append("l" + text)
        else:
            out.append(STATE.get(state, "o") + text)
    return out


def tablet_rows(text: str, rows: pd.DataFrame, table: dict[str, str]) -> list[dict] | None:
    """The lines of the tablet, each with its transliteration and the line of the
    translation it gives, or None when a line has no translation of its own.

    The record's text joins the translations of its lines and leaves out empty
    ones, so the lines are paired in order. A record whose lines do not pair up
    gets no tablet, and the site shows its translation alone.
    """
    lines = text.split("\n")
    out, k = [], 0
    for row in rows.itertuples():
        translation = row.translation if isinstance(row.translation, str) else ""
        tie = None
        if k < len(lines) and translation == lines[k]:
            tie, k = k, k + 1
        out.append({"label": row.label or "", "translit": row.translit or "", "line": tie,
                    "signs": glyph_line(row.translit or "", table)})
    return out if k == len(lines) and lines != [""] else None


def sky_table(years: list[int]) -> dict:
    """Where the Sun and the planets stood at the bounds of each month's window.

    The rules test a body over the whole month, so the site draws each body's
    track over the window that the sign rules test. Longitudes are in tenths of a degree
    of the Babylonian zodiac, from the committed ephemeris.
    """
    rows = []
    for year in years:
        by_month = []
        for name in MONTHS:
            start = cal.month_start_jd(year, name)
            if start is None:
                by_month.append(None)
                continue
            # The next month begins where this one ends; after the last month
            # of the year comes month I of the next year.
            index = next(i for i in range(1, 14) if cal.month_start_jd(year, i) == start)
            end = cal.month_start_jd(year, index + 1) or cal.month_start_jd(year + 1, 1) or start + 29.5
            # The bounds of the window the sign rules test: from the civil day
            # before the month begins to the start of the next month.
            by_month.append([int(round(10 * sky.babylonian_lon(b, jd))) % 3600
                             for jd in (start - 1.0, end) for b in SKY_BODIES])
        rows.append(by_month)
    return {"bodies": SKY_BODIES, "months": MONTHS, "signs": list(sky.SIGNS),
            "first_year": years[0], "lon": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    years = [int(y) for y in cal.candidate_years()]

    tablets = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet").set_index("tablet")
    signs = cuneiform.sign_table()
    lines = {tablet: group for tablet, group in
             pd.read_parquet(HERE / "data" / "corpus" / "lines.parquet").groupby("tablet", sort=False)}
    dates = pd.read_parquet(HERE / "data" / "results" / "tablet_dates.parquet").set_index("tablet")
    review_run = load_review_run()
    records_dir = args.out / "records"
    if records_dir.exists():
        shutil.rmtree(records_dir)
    records_dir.mkdir(parents=True)

    index, stories = [], []
    for tablet, row in tablets.iterrows():
        published = yr.bce_to_astro(int(row.bce_first)) if not math.isnan(row.bce_first) else None
        story = record_story(tablet, row.text or "", years, dates.loc[tablet], published,
                             review_run)
        stories.append(story)
        story["tablet"] = (tablet_rows(row.text, lines[tablet], signs)
                           if isinstance(row.text, str) and tablet in lines else None)
        story.update(volume=VOLUME[row.project],
                     designation=row.designation, museum_no=row.museum_no)
        (records_dir / f"{tablet}.json").write_text(
            json.dumps(story, ensure_ascii=False, separators=(",", ":")))
        decisive = [c for c in story["clauses"] if c["decisive"]]
        index.append({
            "id": tablet, "volume": story["volume"], "museum_no": row.museum_no,
            "n_clauses": len(story["clauses"]), "n_decisive": len(decisive),
            "clocks": max(story["K"]) if story["K"] else 0,
            "strict": len(story["strict"]),
            "first": len(story["groups"][0]) if story["groups"] else N_YEARS})
        if story.get("no_clauses"):
            index[-1]["no_clauses"] = True

    meta = {"years": [years[0], years[-1]], "featured": FEATURED, "records": index,
            "reigns": [{"name": n, "first": a, "last": b} for n, a, b in REIGNS],
            "periods": [{"key": k, "name": n, "short": h, "first": a, "last": b} for k, n, h, a, b in PERIODS]}
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
    (args.out / "sky.json").write_text(json.dumps(sky_table(years), separators=(",", ":")))
    size = sum(p.stat().st_size for p in records_dir.iterdir())
    print(f"{len(index)} records -> {records_dir} ({size / 1e6:.1f} MB), meta -> {args.out / 'meta.json'}")
    # The language model's part, counted over the records of the site.
    changes = [c for s in stories for c in s.get("review", {}).get("changes", [])]
    found = [c for s in stories for c in s.get("found", [])]
    added = sum(len(review_run["found"].get(s["id"], [])) for s in stories)
    print(f"step 1: {sum(s.get('review', {}).get('modified', 0) for s in stories)} parser clauses "
          f"modified; {len(changes)} proposed changes in "
          f"{sum(1 for s in stories if s.get('review', {}).get('changes'))} records "
          f"({sum(c['accepted'] for c in changes)} accepted); {len(found)} clauses found "
          f"({sum(c['decisive'] for c in found)} decisive, {added - len(found)} not placed); "
          f"{sum(c.get('agent_rule', False) for s in stories for c in s['clauses'])} "
          "parser clauses read by rules from a coding agent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
