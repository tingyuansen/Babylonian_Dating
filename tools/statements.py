#!/usr/bin/env python3
"""Read a diary's astronomical claims, and check them against a candidate year.

This is the piece that turns text into something testable. It handles the
statement forms that actually occur often enough to matter, and it is
deliberately explicit about the ones it cannot handle, because a statement
silently ignored is indistinguishable from a statement that agrees.

Six forms are read.

    planet in sign      "Jupiter was in Aquarius"
    station             "the 7th, Jupiter became stationary in Aquarius"
    sign entry          "The 29th, Mars reached Capricorn"
    last appearance     "Mercury's last appearance in the east in Libra"
    planet appulse      "the moon stood 2 cubits in front of Saturn"
    dated eclipse       "Night of the 14th, lunar eclipse"

Last appearances are checked only for the sign, not the date, because the
visibility threshold is not stated in Sachs and Hunger and would be a free
parameter. Eclipses are checked only when both a month and a day survive.  A
preserved eclipse agrees only when its relevant local visibility is robust
throughout the fixed Delta T trials.  Locally verified umbral lunar eclipses
enter the decisive tier.  Robustly local solar eclipses may agree without a
magnitude threshold, but remain supporting pending an independent forward
gate.  An explicitly omitted, excluded, or unseen prediction is retained as
untestable and is never matched as an observation.

Every check returns one of three verdicts. Agreed, contradicted, or untestable
with a reason. The third is a first-class outcome, not a failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from . import cal, eclipses, sky
from . import years as yr

# Tolerances in degrees, chosen from the retrospective residual audit in
# scripts/appulse_residuals.py and exposed to whole-corpus sensitivity tests.
# They are transparent development choices, not independent calibration.
#
# The Moon's is the looser of the two for a physical reason. This ephemeris is
# geocentric, and lunar parallax reaches about a degree, which is half a cubit
# and comparable to the distances being recorded. A topocentric grid at Babylon
# would tighten this and is the obvious next improvement.
TOL_MOON = 3.0
TOL_PLANET = 2.0
TOL_PLANET_PAIR = 2.0
# A solstice or equinox is recorded to the day, and the residual against the
# computed turning point has a 90th percentile of under four days once the
# scheme's own lateness is allowed for. The window is that residual, not a
# guess, and the sensitivity audit varies it.
TOL_SEASON_DAYS = 4.0
# An acronychal rising is the night a planet rises as the Sun sets, which is
# opposition. That is geometry rather than visibility, and the residual against
# the recorded dates has a 90th percentile of five degrees.
TOL_OPPOSITION_DEG = 10.0
# A Sirius date is a scheme date like a solstice, and the residual against the
# reconstructed scheme is a day or less, so the window is the same as the one
# the solstice check uses.
TOL_SIRIUS_DAYS = 3.0
# Exposed only so the whole-corpus ablation can reconstruct the legacy policy.
# Production inference always leaves this on.
PROPAGATE_RESTORED_MONTH_CONTEXT = True
# The candidate ordering settles decisive information before it counts
# contradictions. That is a choice rather than a necessity, and it decides the
# one active tablet where the method commits against the published chronology,
# so the alternative is exposed here for the sensitivity audit to measure.
ORDER_CONTRADICTIONS_FIRST = False
# Brackets mark the editors' restorations, and they interrupt clauses a grammar
# would otherwise recognise. Reading through them recovers about a fifth more
# clauses. Because the editors knew the dates, everything recovered this way is
# marked restored and can never reach the decisive tier or eliminate a year; it
# contributes supporting information only. The sensitivity audit measures the
# method without it.
BRACKET_TOLERANT = True
_BLANK_BRACKETS = str.maketrans("[]", "  ")
# The variant glyphs of theta and rho, read as the letters of the star catalogue.
_GREEK_VARIANTS = str.maketrans("ϑϱ", "θρ")
# Membership of the leading set is a Pareto frontier over independent decisive
# families and decisive information. Contradictions play no part in it, so a
# year that explains more can lead while contradicting more. Turning this on
# admits contradictions as a third frontier dimension, which the sensitivity
# audit measures.
FRONTIER_PENALISES_CONTRADICTIONS = False
# "Above" and "below" state a latitude, which no rule tests, so such a distance
# is tested in longitude alone. Production inference lets it be decisive.
# Turning this off keeps every such distance supporting, which the sensitivity
# audit measures.
LATITUDINAL_DISTANCES_DECISIVE = True
# A measured relation of the Moon on a stated day is tested from the midnight
# that begins the civil day before it to noon of the day after. Turning this on
# tests it only over the night that opens the Babylonian day, from sunset to
# sunrise at Babylon, or over the night one day later for the calendar
# uncertainty, and it agrees if either holds. The sensitivity audit measures it.
MOON_NIGHT_WINDOWS = False

SIGNS = "|".join(
    "Scorpi(?:o|u)s" if sign == "Scorpio" else sign
    for sign in sky.SIGNS
)
PLANETS = {"Jupiter": "jupiter", "Saturn": "saturn", "Mars": "mars",
           "Venus": "venus", "Mercury": "mercury"}
P = "|".join(PLANETS)

PATTERNS = [
    ("in_sign",  re.compile(
        rf"\b({P})\s+was\s+in\s+"
        rf"(?:(?:the\s+)?(?:beginning|middle|end)\s+of\s+)?({SIGNS})\b")),
    ("station",  re.compile(rf"\b({P})\s+became\s+stationary\s+in\s+({SIGNS})\b")),
    ("entry",    re.compile(rf"\b({P})\s+reached\s+({SIGNS})\b")),
    ("last_app", re.compile(rf"\b({P})[’'`s]*\s+last\s+appearance[^.;]{{0,40}}?"
                            rf"\bin\s+({SIGNS})\b")),
    # The mirror of the rule above. An appearance is a visibility event whose
    # date the tools cannot compute, but the sign it is reported in is an
    # ordinary position, and that is all this reads.
    ("first_app", re.compile(rf"\b({P})[’'`s]*\s+first\s+appearance[^.;]{{0,40}}?"
                             rf"\bin\s+({SIGNS})\b")),
]

MONTH_PATH = re.compile(
    rf"\b({P})\s*,\s*in\s+the\s+beginning\s+of\s+the\s+month\s*,\s*"
    rf"was\s+in\s+({SIGNS})\s*,\s*at\s+the\s+end\s+of\s+the\s+month\s*,\s*"
    rf"in\s+({SIGNS})\b")
MONTH_PATH_AFTER = re.compile(
    rf"\b({P})\s+was\s+in\s+({SIGNS})\s*,\s*(?:until|at)\s+the\s+end\s+of\s+"
    rf"the\s+month\s*,\s*in\s+({SIGNS})\b")
COORD_IN_SIGN = re.compile(
    rf"\b({P})\s+and\s+({P})\s+were\s+in\s+({SIGNS})\b")
TRIPLE_IN_SIGN = re.compile(
    rf"\b({P})\s*,\s*({P})\s+and\s+({P})\s+were\s+in\s+({SIGNS})\b")

# Headings read "Month VII ," or "[ Month VII ," or "[ Month ] VIII ,". The
# optional bracket and the optional closing bracket both occur.
MONTH_RE = re.compile(
    r"(?:\bMonth|\bMon\s*(?:\[|\])?\s*th)\s*\]?\s*"
    r"(XII(?:2|₂)?|VI(?:2|₂)?|XI|IX|VIII|VII|VI|IV|V|"
    r"III|II|I|X)(?:\s*\[\s*(I{1,3})(?=\s*[\],.]))?(?=\W|$)",
    re.IGNORECASE,
)

# "Night of the 19th", "The 21st". The day governs the statements after it in
# the same way a month heading does.
DAY_RE = re.compile(r"\b(?:[Nn]ight of the|[Tt]he)\s+(\d{1,2})(?:st|nd|rd|th)\b")

# Explicit year headings are used only for their relative difference. Deciding
# whether a bare year is Seleucid, Arsacid, or regnal is a separate task.
YEAR_RE = re.compile(r"\b[Yy]ear\s+(\d{1,3})\b")


def month_matches(text: str, start: int = 0, end: int | None = None):
    """Yield observation month headings, not lowercase narrative mentions.

    Most headings capitalize ``Month``. Two recurring edition forms use a
    lowercase heading after an explicit Year heading (including ``mon [ th``).
    A globally case-insensitive regex also captures hundreds of prose phrases
    such as ``Diary from month VII to the end of month XII`` and silently
    changes all later statement context. Lowercase forms are therefore admitted
    only at the start of a source line or after ``Year`` on that same line.
    """
    limit = len(text) if end is None else end
    for match in MONTH_RE.finditer(text, start, limit):
        spelling = text[match.start():match.end()]
        if spelling.startswith(("Month", "Mon")):
            yield match
            continue
        segment_start = max(
            text.rfind(mark, 0, match.start()) for mark in ("\n", ".", ";")
        ) + 1
        prefix = text[segment_start:match.start()]
        year_heading = re.search(
            r"\bYear\b[^.;\n]{0,160}$", prefix, re.IGNORECASE)
        no_earlier_month = not re.search(
            r"\bmon(?:th)?\b", prefix, re.IGNORECASE)
        heading_punctuation = bool(re.match(
            r"\s*[,.:]", text[match.end():match.end() + 8]))
        if heading_punctuation and (
                not prefix.strip(" []\t")
                or (year_heading and no_earlier_month)):
            yield match

# "the moon was 1 1/2 cubits behind beta Geminorum". Distances are cubits or
# fingers, 1 cubit being 2 degrees and 24 fingers to the cubit. Compound
# amounts such as "1 cubit 20 fingers" have their own stricter patterns. This
# keeps the single-unit grammar stable and makes the new rule independently
# auditable and removable.
#
# The four directions do not mean the same kind of thing. "in front of" is
# westward and "behind" eastward, so both give a longitude difference directly.
# "above" and "below" are latitude, which this ephemeris does not carry, but
# they still say the body was close to the star *in longitude*, and that is a
# real constraint. So the two cases are tested differently and the difference
# is recorded rather than smoothed over.
APPULSE = re.compile(
    rf"\b(moon|{P})\s+was\s+((?:\d+\s+)?(?:\d+/\d+\s+)?)"
    rf"(cubits?|fingers?)\s+(above|below|behind|in front of)\s+"
    rf"([\u03b1-\u03c9])\s*(\w+)")
COMPOUND_APPULSE = re.compile(
    rf"\b(moon|{P})\s+was\s+((?:\d+\s+)?(?:\d+/\d+\s+)?)cubits?\s+"
    rf"((?:\d+\s+)?(?:\d+/\d+\s+)?)fingers?\s+"
    rf"(above|below|behind|in front of)\s+([\u03b1-\u03c9])\s*(\w+)")

# Measured Moon-to-planet separations carry the same kind of dated geometric
# information as Normal-Star appulses.  Brackets around only the copular verb
# are harmless: the body, amount, direction, and target all survive in the
# manuscript.  We do not infer a distance or direction when either is absent.
PLANET_APPULSE = re.compile(
    rf"\bmoon\s+(?:\[\s*)?(?:was|stood)(?:\s*\])?\s+"
    rf"((?:\d+\s+)?(?:\d+/\d+\s+)?)"
    rf"(cubits?|fingers?)\s+(above|below|behind|in front of)\s+"
    rf"({P})\b")
COMPOUND_PLANET_APPULSE = re.compile(
    rf"\bmoon\s+(?:\[\s*)?(?:was|stood)(?:\s*\])?\s+"
    rf"((?:\d+\s+)?(?:\d+/\d+\s+)?)cubits?\s+"
    rf"((?:\d+\s+)?(?:\d+/\d+\s+)?)fingers?\s+"
    rf"(above|below|behind|in front of)\s+({P})\b")
PLANET_PLANET_APPULSE = re.compile(
    rf"\b({P})\s+was\s+((?:\d+\s+)?(?:\d+/\d+\s+)?)"
    rf"(cubits?|fingers?)\s+(above|below|behind|in front of)\s+({P})\b")
COMPOUND_PLANET_PLANET_APPULSE = re.compile(
    rf"\b({P})\s+was\s+((?:\d+\s+)?(?:\d+/\d+\s+)?)cubits?\s+"
    rf"((?:\d+\s+)?(?:\d+/\d+\s+)?)fingers?\s+"
    rf"(above|below|behind|in front of)\s+({P})\b")

# A measured position at an outer-planet station is syntactically distinct
# from an ordinary dated appulse. The planet is resolved only from a named
# same-clause antecedent (``when Jupiter became stationary ...``); anaphoric
# ``it`` is never filled from a distant planet name. Brackets may surround
# restored words, but restoration is retained and the whole grammar initially
# remains supporting pending its own forward residual gate.
_STATION_SPACE = r"\s*(?:[\[\]]\s*)*"
_STATION_NUMBER = r"((?:\d+\s+)?(?:\d+(?:/|⁄)\d+\s+)?)"
STATION_APPULSE = re.compile(
    rf"\bbecame\s+stationary{_STATION_SPACE}{_STATION_NUMBER}"
    rf"{_STATION_SPACE}(cubits?|fingers?){_STATION_SPACE}"
    rf"(above|below|behind|in front of){_STATION_SPACE}"
    rf"([\u03b1-\u03c9\u03f1]){_STATION_SPACE}(\w+)")
COMPOUND_STATION_APPULSE = re.compile(
    rf"\bbecame\s+stationary{_STATION_SPACE}{_STATION_NUMBER}"
    rf"{_STATION_SPACE}cubits?{_STATION_SPACE}{_STATION_NUMBER}"
    rf"{_STATION_SPACE}fingers?{_STATION_SPACE}"
    rf"(above|below|behind|in front of){_STATION_SPACE}"
    rf"([\u03b1-\u03c9\u03f1]){_STATION_SPACE}(\w+)")

# Brackets can split a restored word ("lu [ nar ] eclipse") or surround the
# whole body ("[ solar ] eclipse"). Untyped "eclipse" is deliberately ignored:
# inferring lunar from context belongs in the interpretive layer, not a regex.
ECLIPSE = re.compile(
    r"\b(lunar|solar|lu\s*\[\s*nar\s*\])\s*[\[\]]*\s*eclipse\b",
    re.IGNORECASE)
# "Mars' acronychal rising", either way round. Only the outer planets have one.
ACRONYCHAL = re.compile(
    rf"\b(?:({P})[’'`s]*\s+acronychal\s+rising"
    rf"|acronychal\s+rising\s+of\s+({P}))", re.IGNORECASE)

# "Sirius' first appearance", "Sirius' acronychal rising".
SIRIUS = re.compile(
    r"\bSirius[’'`s]*\s+(first\s+appearance|last\s+appearance"
    r"|acronychal\s+rising)", re.IGNORECASE)
SIRIUS_KIND = {"first appearance": "rising", "last appearance": "setting",
               "acronychal rising": "acronychal"}

# "The 26th, solstice" and "The 27th, equinox". The diaries never say which of
# the two solstices or which equinox, because the month heading already does.
SEASON = re.compile(r"\b(solstice|equinox)\b", re.IGNORECASE)
OMITTED = re.compile(
    r"\b(?:omitted|excluded|passed\s+by)\b|"
    r"\b(?:I\s+)?did\s+not\s+(?:see|watch)\b",
    re.IGNORECASE)


def unwatched(text: str, kind: str, start: int, end: int) -> bool:
    """Whether the scribe marks the event of a clause as not observed.

    This is the parser's only rule for the mark, and the language-model review
    imports it rather than keeping its own. ``start`` and ``end`` are the
    clause's offsets in ``text``. The mark follows the event. For an eclipse
    the search runs 180 characters on, or to the line end. A semicolon does not
    end it, because the result is often reported after one: "solar eclipse;
    when I watched I did not see it". For a measured station and an acronychal
    rising the search also stops at a sentence end: "Saturn's acronychal
    rising; I did not watch." Other kinds are never marked. Solstice, equinox
    and Sirius dates in particular come from an arithmetical scheme that the
    rule tests, so "I did not watch" after them leaves an observed claim.
    """
    if kind == "eclipse":
        stops, blank_brackets = ("\n",), True
    elif kind in ("station_appulse", "acronychal"):
        stops, blank_brackets = (".", "\n"), False
    else:
        return False
    limits = [min(len(text), end + 180)]
    limits.extend(p for mark in stops if (p := text.find(mark, end)) >= 0)
    window = text[start:min(limits)]
    if blank_brackets:
        window = re.sub(r"[\[\]]", " ", window)
    return bool(OMITTED.search(window))


def parse_amount(txt: str) -> float | None:
    """A distance written as '1 1/2' or '2 2/3' or '1/2', in units."""
    tot, seen = 0.0, False
    for tok in txt.replace("⁄", "/").split():
        if "/" in tok:
            a, b = tok.split("/")
            if not b or not a.isdigit() or not b.isdigit():
                return None
            tot += int(a) / int(b)
            seen = True
        elif tok.isdigit():
            tot += int(tok)
            seen = True
    return tot if seen else None


def parse_measure(cubits: str | None = None,
                  fingers: str | None = None) -> tuple[float, float, str] | None:
    """Normalize a single- or mixed-unit distance to cubits and degrees.

    Returning total cubits preserves the meaning of ``Statement.amount``
    across grammars. The conversion is exact in the source unit system:
    24 fingers = 1 cubit = 2 degrees. Unicode fractions and damaged unit words
    remain outside this rule rather than being guessed here.
    """
    cubit_amount = parse_amount(cubits or "") if cubits is not None else 0.0
    finger_amount = parse_amount(fingers or "") if fingers is not None else 0.0
    if cubit_amount is None or finger_amount is None:
        return None
    total_cubits = cubit_amount + finger_amount / 24.0
    if total_cubits <= 0:
        return None
    unit = "cubit+finger" if cubits is not None and fingers is not None else (
        "cubit" if cubits is not None else "finger")
    return total_cubits, total_cubits * 2.0, unit


@dataclass
class Statement:
    kind: str
    planet: str
    sign: str
    text: str
    month: str | None = None      # the heading that governs it, if any
    day: int | None = None        # the day heading that governs it
    year_offset: int = 0          # later year after a month-number rollover
    star: str | None = None       # for an appulse, the Normal Star
    target_planet: str | None = None  # for a measured Moon-to-planet appulse
    offset: float | None = None   # expected longitude minus the star, degrees
    latitudinal: bool = False     # "above"/"below", so longitude only says near
    omitted: bool = False         # predicted/excluded/unseen, never an observation
    restored: bool = False        # the governing day lies inside editor brackets
    source_start: int = -1        # offsets into the exact tablet translation
    source_end: int = -1
    parser_rule: str = ""         # stable rule name for provenance and ablation
    amount: float | None = None   # recorded number before unit conversion
    unit: str | None = None
    direction: str | None = None
    station_phase: str | None = None  # east=retrograde start; west=direct
    # Atomic source records stay intact; these fields link explicitly
    # coordinated sign conjuncts into one simultaneous scoring/removal unit.
    joint_positions: tuple[tuple[str, str], ...] = ()
    joint_source_start: int = -1
    joint_source_end: int = -1
    joint_day: int | None = None
    joint_restored: bool = False
    joint_omitted: bool = False


@dataclass
class Verdict:
    statement: Statement
    result: str          # "agree", "contradict", or "untestable"
    detail: str


@dataclass
class CandidateScores:
    """Every candidate year scored against the same tablet statements.

    ``agreeing`` and ``testable`` are per statement. Everything else is per
    candidate year except ``weights``, which is per statement too. Keeping the
    matrix calculation here prevents command-line harnesses from quietly using
    different definitions of a score or a top set.
    """

    years: np.ndarray
    priority: np.ndarray
    score: np.ndarray
    support: np.ndarray
    agree: np.ndarray
    contradict: np.ndarray
    untestable: np.ndarray
    weights: np.ndarray
    agreeing: np.ndarray
    testable: np.ndarray
    equivalent: np.ndarray
    decisive: np.ndarray
    weighting: str
    strict: np.ndarray
    safety: np.ndarray
    outcomes: np.ndarray
    scoring_units: np.ndarray  # per atomic record, shared by coordinated atoms

    @property
    def top(self) -> np.ndarray:
        """Boolean mask for the safety set used to measure recall."""
        return self.safety

    @property
    def order(self) -> np.ndarray:
        """Candidate indices, best first, with deterministic tie ordering."""
        # A dominated candidate must never be displayed before the strict
        # Pareto frontier, and a candidate outside the one-statement safety set
        # must never precede that set. np.lexsort uses the last key as primary;
        # the remaining quantities preserve the documented inspection order
        # inside each set. Contradictions do not change set membership.
        tier = np.where(self.strict, 0, np.where(self.safety, 1, 2))
        if ORDER_CONTRADICTIONS_FIRST:
            return np.lexsort(
                (self.years, -self.support, -self.score, self.contradict,
                 -self.priority, tier))
        return np.lexsort(
            (self.years, self.contradict, -self.support, -self.score,
             -self.priority, tier))


MONTH_ORDER = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VI2": 6.5, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XII2": 12.5,
}


def canonical_month(name: str) -> str:
    """Normalize the printed and ASCII forms of an intercalary subscript."""
    return name.upper().replace("₂", "2")


def canonical_sign(name: str) -> str:
    """Normalize the edition's Scorpius spelling to the model's Scorpio."""
    return "Scorpio" if name in {"Scorpio", "Scorpius"} else name


def inside_restoration(text: str, pos: int) -> bool:
    """Whether a character position lies inside an editorial bracket."""
    before = text[:pos]
    return before.rfind("[") > before.rfind("]")


def match_fields_restored(
        text: str, match: re.Match, groups: tuple[int, ...] | None = None,
        *, include_start: bool = True) -> bool:
    """Whether restoration touches a scientific field in a regex match.

    Editorial brackets around a copula such as ``moon [stood] 2 cubits`` do
    not reconstruct the observed body, amount, direction, or target.  A
    blanket bracket-in-clause rule therefore discards valid measurements.  We
    instead inspect the match start (normally the subject) and the explicitly
    captured scientific fields.  Callers may omit ``include_start`` only when
    the match begins with non-scientific connective prose.
    """
    if include_start and inside_restoration(text, match.start()):
        return True
    selected = groups or tuple(range(1, (match.lastindex or 0) + 1))
    for group in selected:
        start, end = match.span(group)
        if start < 0:
            continue
        if inside_restoration(text, start) or "[" in text[start:end]:
            return True
    return False


def month_from_match(match: re.Match) -> str:
    """Read a month token, including a restored split Roman suffix.

    The edition can print Month V [ II ] for VII and Month I [ II ] for III.
    Treating only the preserved prefix as the month moves every following
    observation by two lunations while still producing plausible astronomy.
    """
    return canonical_month(match.group(1) + (match.group(2) or ""))


def day_matches(text: str, start: int, end: int):
    """Day headings, excluding a preceding month's day in a month formula."""
    for match in DAY_RE.finditer(text, start, end):
        tail = text[match.end():match.end() + 45]
        if re.match(
                r"\s*(?:\(\s*)?of\s+(?:the\s+)?preceding\s+month",
                tail, re.IGNORECASE):
            continue
        yield match


def governing_context(text: str, pos: int) -> tuple[str | None, int | None, int]:
    """Month, day, and relative year governing a statement position.

    Headings govern everything after them until the next one, so a statement
    belongs to the last heading before it. When month numbers roll backwards,
    the tablet has crossed Nisannu into the next Babylonian year. A candidate
    base year of 145 BCE therefore tests a later Month I at 144 BCE.

    Day headings reset at a month boundary; an otherwise undated statement in a
    new month must not inherit the preceding month's final day.
    """
    last_month = None
    last_order = None
    last_month_pos = None
    year_offset = 0
    base_heading = None
    events = [(m.start(), "month", m) for m in month_matches(text, 0, pos)]
    events.extend((m.start(), "year", m)
                  for m in YEAR_RE.finditer(text, 0, pos))
    for _, kind, match in sorted(events, key=lambda item: item[0]):
        if kind == "year":
            # A number inside an open editorial bracket is reconstructed and
            # cannot safely control otherwise conclusive evidence.
            before = text[:match.start()]
            if before.rfind("[") > before.rfind("]"):
                continue
            number = int(match.group(1))
            if base_heading is None:
                base_heading = number
                year_offset = 0
            elif base_heading <= number <= base_heading + 10:
                year_offset = number - base_heading
            last_order = None
            continue
        month = month_from_match(match)
        order = MONTH_ORDER[month]
        if last_order is not None and order < last_order:
            year_offset += 1
        last_month = month
        last_order = order
        last_month_pos = match.start()

    day = None
    day_start = last_month_pos if last_month_pos is not None else 0
    for m in day_matches(text, day_start, pos):
        day = int(m.group(1))
    return last_month, day, year_offset


def governing_month_restored(text: str, pos: int) -> bool:
    """Whether the governing month heading is editorially restored."""
    if not PROPAGATE_RESTORED_MONTH_CONTEXT:
        return False
    heading = None
    for match in month_matches(text):
        if match.start() > pos:
            break
        heading = match
    if heading is None:
        return False
    # Judge the numeral, not the word ``Month``. In ``[ Month ] III`` the
    # label is supplied but III survives; in ``[ Month IX ... ]`` the numeral
    # itself is restored. A split suffix such as ``Month V [ II ]`` is also
    # uncertain because the normalized VII depends on restored characters.
    number_restored = inside_restoration(text, heading.start(1))
    suffix_restored = bool(
        heading.group(2)
        and inside_restoration(text, heading.start(2)))
    return number_restored or suffix_restored


def appulse_context(text: str, pos: int) -> tuple[
        str | None, int | None, int, bool]:
    """Context for a measured relation, refusing a day across a lacuna.

    A day may govern several semicolon-separated observations, but not a later
    broken passage.  Importing the last number seen anywhere in the month turns
    an undated appulse into rare, falsely precise evidence.
    """
    month, _, year_offset = governing_context(text, pos)
    last_month_pos = 0
    for match in month_matches(text):
        if match.start() > pos:
            break
        last_month_pos = match.start()
    day_match = None
    for match in day_matches(text, last_month_pos, pos):
        day_match = match
    month_restored = governing_month_restored(text, pos)
    if day_match is None:
        return month, None, year_offset, month_restored
    between = text[day_match.end():pos]
    without_lacunae = re.sub(r"\.{2,}", "", between)
    secure = (len(between) <= 260
              and "..." not in between
              and "." not in without_lacunae)
    if not secure:
        return month, None, year_offset, month_restored
    before = text[:day_match.start()]
    restored = (month_restored
                or before.rfind("[") > before.rfind("]")
                or bool(re.match(r"\s*\?", text[day_match.end():])))
    return month, int(day_match.group(1)), year_offset, restored


def within_month_entries(text: str, pos: int) -> bool:
    """Whether a statement sits in a dated entry rather than a month summary.

    It does if its own sentence holds a month heading, as in "Year 212, month
    VIII, Venus and Saturn were in Capricorn", or a night heading, as in "Night
    of the 27th, ... Venus was in Gemini". A month summary carries neither. It
    may name a day, as in "the 26th, Venus reached Aries", but not a night of
    observation. Older headings do not count, because a break in the tablet can
    hide the months between them.
    """
    sentence = max(text.rfind(mark, 0, pos) for mark in (".", ";", "\n")) + 1
    if any(month_matches(text, sentence, pos)):
        return True
    return re.search(r"\bNight of the \d", text[sentence:pos]) is not None


def preceding_summary_month(text: str, pos: int) -> str | None:
    """Infer the immediately preceding month from an adjacent month summary.

    A diary can begin mid-month, so the first surviving heading is the next
    month.  If a statement is followed by its own ``That month`` summary and
    then directly by Month VIII, its month is VII.  Only unambiguous predecessor
    pairs are admitted; VI/VII and intercalary boundaries are left unresolved.
    """
    next_heading = next(
        (match for match in month_matches(text) if match.start() > pos),
        None)
    if next_heading is None:
        return None
    between = text[pos:next_heading.start()]
    # Without a "That month" marker, nearness to the next heading alone does
    # not make a summary when the statement sits in its own month's entries.
    adjacent = len(between) <= 200 and not within_month_entries(text, pos)
    markers_after = len(re.findall(r"\bThat month\b", between))
    marker_before = bool(re.search(
        r"\bThat month\b", text[max(0, pos - 600):pos]))
    summarized = (len(between) <= 700
                  and (markers_after == 1
                       or (markers_after == 0 and marker_before)))
    first_fragment = (len(between) <= 700
                      and not any(month_matches(text, 0, pos)))
    if not (adjacent or summarized or first_fragment):
        return None
    previous = {
        "II": "I", "III": "II", "IV": "III", "V": "IV",
        "VI": "V", "VII": "VI|VI2", "VIII": "VII", "IX": "VIII", "X": "IX",
        "XI": "X", "XII": "XI",
    }
    return previous.get(month_from_match(next_heading))


def positional_context(text: str, pos: int) -> tuple[str | None, int, bool]:
    """Month, relative year, and unresolved-missing-month flag."""
    month, _, year_offset = governing_context(text, pos)
    # An adjacent end-of-month summary belongs to the month immediately before
    # the next heading even when a much older heading also survives.
    inferred = preceding_summary_month(text, pos)
    if inferred is not None:
        month = inferred
        return month, year_offset, False

    last_heading = 0
    for match in month_matches(text, 0, pos):
        last_heading = match.start()
    previous_day = None
    missing_month = False
    for match in day_matches(text, last_heading, pos):
        day = int(match.group(1))
        # A large day-number rollback without a heading means at least one
        # month heading is lost. The later summary cannot inherit the older
        # month merely because no replacement heading survives.
        if previous_day is not None and day < previous_day - 10:
            missing_month = True
        previous_day = day
    if missing_month:
        return None, year_offset, True
    return month, year_offset, False


def clause_at(text: str, start: int, end: int) -> str:
    """The punctuation-delimited clause containing a matched expression."""
    left = max(text.rfind(mark, 0, start) for mark in (".", ";", "\n"))
    rights = [text.find(mark, end) for mark in (".", ";", "\n")]
    rights = [p for p in rights if p >= 0]
    right = min(rights) if rights else len(text)
    return text[left + 1:right]


def eclipse_context(text: str, pos: int) -> tuple[
        str | None, int | None, int, bool]:
    """Month/year plus a day only when it governs this eclipse sentence.

    Diary day headings normally govern following clauses, but a broken eclipse
    sentence can occur after an unrelated completed day entry. Treating that
    older heading as the eclipse date manufactures conclusive evidence. A day
    is therefore secure here only if no sentence-ending period or semicolon
    intervenes. Runs of dots are lacuna markers, not sentence boundaries.
    """
    month, _, year_offset = governing_context(text, pos)
    last_month_pos = 0
    for m in month_matches(text):
        if m.start() > pos:
            break
        last_month_pos = m.start()
    day_match = None
    for m in day_matches(text, last_month_pos, pos):
        day_match = m
    month_restored = governing_month_restored(text, pos)
    if day_match is None:
        return month, None, year_offset, month_restored
    between = text[day_match.end():pos]
    without_lacunae = re.sub(r"\.{2,}", "", between)
    secure = (len(between) <= 240
              and "..." not in between
              and not re.search(r"[.;]", without_lacunae))
    if not secure:
        return month, None, year_offset, month_restored
    before = text[:day_match.start()]
    restored = (month_restored
                or before.rfind("[") > before.rfind("]"))
    return month, int(day_match.group(1)), year_offset, restored


def extract(text: str) -> list[Statement]:
    """Every statement this module knows how to test.

    Editorial brackets interrupt otherwise recognisable forms, so a clause such
    as ``[ ... Saturn ] was in Pisces`` is lost to a grammar that has no reason
    to expect a bracket in the middle of a subject. The text is therefore read
    twice, the second time with the brackets blanked to spaces, which preserves
    every offset and so leaves the month and day machinery untouched. Anything
    only the second pass finds is a reading of restored text, and is marked
    restored so that it can never reach the decisive tier. It contributes
    supporting information and nothing else.

    The translation writes theta and rho with two glyphs each, as in
    ``ϑ Leonis`` beside ``θ Leonis``. Both are read as the letter of the star
    catalogue, one character for one, so every offset stays the same.
    """
    text = text.translate(_GREEK_VARIANTS)
    out = _extract(text)
    if not BRACKET_TOLERANT or "[" not in text:
        return bind_joint_positions(text, out)
    known = {(st.kind, st.planet, st.sign, st.target_planet, st.star,
              st.month, st.day) for st in out}
    spans = [(st.source_start, st.source_end) for st in out]
    for st in _extract(text.translate(_BLANK_BRACKETS)):
        if (st.kind, st.planet, st.sign, st.target_planet, st.star,
                st.month, st.day) in known:
            continue
        # A pattern that matched a longer span once the brackets went is the
        # same assertion read further, not a second one.
        if any(st.source_start < end and start < st.source_end
               for start, end in spans):
            continue
        out.append(replace(st, restored=True,
                           parser_rule=st.parser_rule + "_bracket"))
    return bind_joint_positions(text, out)


def bind_joint_positions(text: str, statements: list[Statement]) -> list[Statement]:
    """Link only complete, source-grounded two/three-planet sign clauses.

    No proximity-based or tablet-wide pairing is permitted. This can also be
    called after cached textual review/recall: missing or modified conjuncts
    disable the joint group, rather than silently inventing an atomic record.
    """
    out = [replace(st, joint_positions=(), joint_source_start=-1,
                   joint_source_end=-1, joint_day=None,
                   joint_restored=False, joint_omitted=False) for st in statements]
    clean = text.translate(_BLANK_BRACKETS)
    triple_spans = [m.span() for m in TRIPLE_IN_SIGN.finditer(clean)]
    for pattern in (TRIPLE_IN_SIGN, COORD_IN_SIGN):
        for match in pattern.finditer(clean):
            if pattern is COORD_IN_SIGN and any(
                    a <= match.start() and match.end() <= b for a, b in triple_spans):
                continue
            groups = match.groups()
            positions = tuple((PLANETS[p], canonical_sign(groups[-1]))
                              for p in groups[:-1])
            if len({p for p, _ in positions}) != len(positions):
                continue
            members = [i for i, st in enumerate(out)
                       if st.kind == "in_sign"
                       and (st.planet, st.sign) in positions
                       and ((match.start() <= st.source_start
                             and st.source_end <= match.end())
                            or (st.source_start <= match.start()
                                and match.end() <= st.source_end))]
            if { (out[i].planet, out[i].sign) for i in members } != set(positions):
                continue
            contexts = {(out[i].month, out[i].year_offset) for i in members}
            if len(contexts) != 1:
                continue
            month, year_offset = next(iter(contexts))
            day_month, day, day_year, day_restored = eclipse_context(
                text, match.start())
            # Monthly summaries must not inherit a stale daily heading.
            if (day_month != month or day_year != year_offset
                    or preceding_summary_month(text, match.start()) is not None):
                day = None
                day_restored = False
            restored = any(out[i].restored for i in members) or (
                day is not None and day_restored)
            omitted = any(out[i].omitted for i in members)
            for i in members:
                out[i] = replace(
                    out[i], joint_positions=positions,
                    joint_source_start=match.start(), joint_source_end=match.end(),
                    joint_day=day, joint_restored=restored, joint_omitted=omitted)
    return out


def scoring_unit_key(st: Statement, index: int) -> tuple:
    """A source-group key, never just an identical candidate-year mask."""
    if st.joint_positions:
        return ("joint", st.joint_source_start, st.joint_source_end,
                st.joint_positions, st.month, st.joint_day, st.year_offset)
    return ("atomic", index)


def evidence_families(st: Statement) -> tuple[str, ...]:
    """Independent clocks, separate from the observed body's public label."""
    if st.kind in {"season", "sirius"}:
        return ("seasonal_sirius",)
    if st.kind == "eclipse" and st.planet == "lunar":
        return ("lunar_eclipse",)
    if st.joint_positions:
        return tuple(p for p, _ in st.joint_positions)
    if st.kind == "planet_appulse" and st.target_planet:
        return (st.target_planet,)
    return (st.planet,)


def _extract(text: str) -> list[Statement]:
    """One pass of every grammar over the text exactly as given."""
    out, seen = [], set()

    for m in ECLIPSE.finditer(text):
        raw = m.group(1).lower()
        kind = "lunar" if raw.startswith("lu") else "solar"
        key = ("eclipse", kind, m.start())
        if key in seen:
            continue
        seen.add(key)
        month, day, year_offset, restored = eclipse_context(text, m.start())
        clause = clause_at(text, m.start(), m.end())
        restored = restored or match_fields_restored(text, m)
        out.append(Statement(
            kind="eclipse",
            planet=kind,
            sign="",
            text=re.sub(r"\s+", " ", clause).strip(),
            month=month,
            day=day,
            year_offset=year_offset,
            omitted=unwatched(text, "eclipse", m.start(), m.end()),
            restored=restored,
            source_start=m.start(),
            source_end=m.end(),
            parser_rule="typed_eclipse",
        ))

    planet_patterns = (
        (COMPOUND_PLANET_APPULSE, "measured_moon_planet_compound", True),
        (PLANET_APPULSE, "measured_moon_planet", False),
    )
    for pattern, parser_rule, compound in planet_patterns:
        for m in pattern.finditer(text):
            if compound:
                cubits, fingers, direction, target = m.groups()
                parsed = parse_measure(cubits=cubits, fingers=fingers)
            else:
                amount, source_unit, direction, target = m.groups()
                parsed = parse_measure(
                    cubits=amount if source_unit.startswith("cubit") else None,
                    fingers=amount if source_unit.startswith("finger") else None)
            if parsed is None:
                continue
            amt, deg, unit = parsed
            if not compound:
                unit = source_unit
            lat_kind = direction in ("above", "below")
            off = (0.0 if lat_kind
                   else (-deg if direction == "in front of" else deg))
            key = ("planet_appulse", "moon", target, m.start())
            if key in seen:
                continue
            seen.add(key)
            month, day, year_offset, restored = appulse_context(text, m.start())
            restored = restored or match_fields_restored(text, m)
            out.append(Statement(
                kind="planet_appulse",
                planet="moon",
                sign="",
                text=re.sub(r"\s+", " ", m.group(0)),
                month=month,
                day=day,
                year_offset=year_offset,
                target_planet=PLANETS[target],
                offset=off,
                latitudinal=lat_kind,
                restored=restored,
                source_start=m.start(),
                source_end=m.end(),
                parser_rule=parser_rule,
                amount=amt,
                unit=unit,
                direction=direction,
            ))

    pair_patterns = (
        (COMPOUND_PLANET_PLANET_APPULSE,
         "measured_planet_planet_compound", True),
        (PLANET_PLANET_APPULSE, "measured_planet_planet", False),
    )
    for pattern, parser_rule, compound in pair_patterns:
        for m in pattern.finditer(text):
            if compound:
                subject, cubits, fingers, direction, target = m.groups()
                parsed = parse_measure(cubits=cubits, fingers=fingers)
            else:
                subject, amount, source_unit, direction, target = m.groups()
                parsed = parse_measure(
                    cubits=amount if source_unit.startswith("cubit") else None,
                    fingers=amount if source_unit.startswith("finger") else None)
            if parsed is None or subject == target:
                continue
            amount, degrees, unit = parsed
            if not compound:
                unit = source_unit
            latitudinal = direction in ("above", "below")
            offset = (0.0 if latitudinal else
                      (-degrees if direction == "in front of" else degrees))
            key = ("planet_planet_appulse", subject, target, m.start())
            if key in seen:
                continue
            seen.add(key)
            month, day, year_offset, restored = appulse_context(text, m.start())
            restored = restored or match_fields_restored(text, m)
            out.append(Statement(
                kind="planet_planet_appulse",
                planet=PLANETS[subject],
                sign="",
                text=re.sub(r"\s+", " ", m.group(0)),
                month=month,
                day=day,
                year_offset=year_offset,
                target_planet=PLANETS[target],
                offset=offset,
                latitudinal=latitudinal,
                restored=restored,
                source_start=m.start(),
                source_end=m.end(),
                parser_rule=parser_rule,
                amount=amount,
                unit=unit,
                direction=direction,
            ))

    station_patterns = (
        (COMPOUND_STATION_APPULSE,
         "measured_station_normal_star_compound", True),
        (STATION_APPULSE, "measured_station_normal_star", False),
    )
    for pattern, parser_rule, compound in station_patterns:
        for m in pattern.finditer(text):
            # Resolve the subject only from a named ``when <planet> became
            # stationary`` antecedent in this punctuation-delimited clause.
            clause_start = max(text.rfind(mark, 0, m.start())
                               for mark in (".", ";", "\n")) + 1
            prefix = text[clause_start:m.start()]
            clean_prefix = re.sub(r"[\[\]]", " ", prefix)
            antecedents = list(re.finditer(
                rf"\bwhen\s+({P})\s+became\s+stationary"
                rf"(?:\s+to\s+the\s+(east|west))?",
                clean_prefix))
            if not antecedents:
                continue
            antecedent = antecedents[-1]
            planet, phase = antecedent.groups()
            planet_start = text.rfind(
                planet, clause_start, m.start(),
            )
            if planet_start < 0 or "..." in text[planet_start:m.start()]:
                continue

            if compound:
                cubits, fingers, direction, greek, constellation = m.groups()
                parsed = parse_measure(cubits=cubits, fingers=fingers)
            else:
                amount, source_unit, direction, greek, constellation = m.groups()
                parsed = parse_measure(
                    cubits=amount if source_unit.startswith("cubit") else None,
                    fingers=amount if source_unit.startswith("finger") else None)
            if parsed is None:
                continue
            amount, degrees, unit = parsed
            if not compound:
                unit = source_unit
            greek = "ρ" if greek == "ϱ" else greek
            name = f"{greek} {constellation}"
            if sky.star(name) is None:
                continue
            latitudinal = direction in ("above", "below")
            offset = (0.0 if latitudinal else
                      (-degrees if direction == "in front of" else degrees))
            key = ("station_appulse", planet, name, m.start())
            if key in seen:
                continue
            seen.add(key)
            month, day, year_offset, context_restored = appulse_context(
                text, planet_start)
            governing = text[max(clause_start, planet_start - 100):m.start()]
            if re.search(r"\b(?:around(?:\s+until)?|until)\s+the\s+\d+", governing):
                day = None
            source = text[planet_start:m.end()]
            restored = (
                context_restored
                or inside_restoration(text, planet_start)
                or match_fields_restored(text, m)
            )
            out.append(Statement(
                kind="station_appulse",
                planet=PLANETS[planet],
                sign="",
                text=re.sub(r"\s+", " ", source),
                month=month,
                day=day,
                year_offset=year_offset,
                star=name,
                offset=offset,
                latitudinal=latitudinal,
                omitted=unwatched(text, "station_appulse", planet_start, m.end()),
                restored=restored,
                source_start=planet_start,
                source_end=m.end(),
                parser_rule=parser_rule,
                amount=amount,
                unit=unit,
                direction=direction,
                station_phase=phase,
            ))

    star_patterns = (
        (COMPOUND_APPULSE, "measured_normal_star_compound", True),
        (APPULSE, "measured_normal_star", False),
    )
    for pattern, parser_rule, compound in star_patterns:
        for m in pattern.finditer(text):
            if compound:
                (body, cubits, fingers, direction, greek,
                 constellation) = m.groups()
                parsed = parse_measure(cubits=cubits, fingers=fingers)
            else:
                (body, amount, source_unit, direction, greek,
                 constellation) = m.groups()
                parsed = parse_measure(
                    cubits=amount if source_unit.startswith("cubit") else None,
                    fingers=amount if source_unit.startswith("finger") else None)
            if parsed is None:
                continue
            amt, deg, unit = parsed
            if not compound:
                unit = source_unit
            name = f"{greek} {constellation}"
            if sky.star(name) is None:
                continue
            lat_kind = direction in ("above", "below")
            off = (0.0 if lat_kind
                   else (-deg if direction == "in front of" else deg))
            key = ("appulse", body, name, m.start())
            if key in seen:
                continue
            seen.add(key)
            month, day, year_offset, restored = appulse_context(text, m.start())
            restored = restored or match_fields_restored(text, m)
            out.append(Statement(
                kind="appulse",
                planet="moon" if body == "moon" else PLANETS[body],
                sign="",
                text=re.sub(r"\s+", " ", m.group(0)),
                month=month,
                day=day,
                year_offset=year_offset,
                star=name,
                offset=off,
                latitudinal=lat_kind,
                restored=restored,
                source_start=m.start(),
                source_end=m.end(),
                parser_rule=parser_rule,
                amount=amt,
                unit=unit,
                direction=direction,
            ))

    for rule, path_pattern in (("monthly_path", MONTH_PATH),
                               ("monthly_path_after", MONTH_PATH_AFTER)):
        for m in path_pattern.finditer(text):
            planet, first_sign, last_sign = m.groups()
            first_sign = canonical_sign(first_sign)
            last_sign = canonical_sign(last_sign)
            month, year_offset, uncertain = positional_context(text, m.start())
            uncertain = uncertain or match_fields_restored(text, m)
            for phase, sign, key_pos in (
                    ("beginning", first_sign, m.start()),
                    ("end", last_sign, m.start(3))):
                key = ("in_sign", planet, sign, key_pos)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Statement(
                    kind="in_sign",
                    planet=PLANETS[planet],
                    sign=sign,
                    text=(f"{planet} was in {sign} at the {phase} of the month"),
                    month=month,
                    year_offset=year_offset,
                    restored=uncertain,
                    source_start=m.start(),
                    source_end=m.end(),
                    parser_rule=rule,
                ))

    for m in COORD_IN_SIGN.finditer(text):
        first, second, sign = m.groups()
        sign = canonical_sign(sign)
        month, year_offset, uncertain = positional_context(text, m.start())
        uncertain = uncertain or match_fields_restored(text, m)
        for planet, key_pos in ((first, m.start(1)), (second, m.start(2))):
            key = ("in_sign", planet, sign, key_pos)
            if key in seen:
                continue
            seen.add(key)
            out.append(Statement(
                kind="in_sign",
                planet=PLANETS[planet],
                sign=sign,
                text=f"{planet} was in {sign}",
                month=month,
                year_offset=year_offset,
                restored=uncertain,
                source_start=m.start(),
                source_end=m.end(),
                parser_rule="coordinated_sign",
            ))

    for m in TRIPLE_IN_SIGN.finditer(text):
        first, second, third, sign = m.groups()
        sign = canonical_sign(sign)
        month, year_offset, uncertain = positional_context(text, m.start())
        uncertain = uncertain or match_fields_restored(text, m)
        for group, planet in enumerate((first, second, third), start=1):
            key = ("in_sign", planet, sign, m.start(group))
            if key in seen:
                continue
            seen.add(key)
            out.append(Statement(
                kind="in_sign",
                planet=PLANETS[planet],
                sign=sign,
                text=f"{planet} was in {sign}",
                month=month,
                year_offset=year_offset,
                restored=uncertain,
                source_start=m.start(),
                source_end=m.end(),
                parser_rule="triple_coordinated_sign",
            ))

    for m in SIRIUS.finditer(text):
        key = ("sirius", m.start())
        if key in seen:
            continue
        seen.add(key)
        month, day, year_offset, restored = eclipse_context(text, m.start())
        phase = SIRIUS_KIND[re.sub(r"\s+", " ", m.group(1).lower())]
        out.append(Statement(
            kind="sirius",
            # Public subject and scoring family are distinct: Sirius is not
            # the Sun, but this scheme still shares the seasonal clock.
            planet="sirius",
            sign="",
            text=re.sub(r"\s+", " ", clause_at(text, m.start(), m.end())),
            month=month,
            day=day,
            year_offset=year_offset,
            restored=restored,
            source_start=m.start(),
            source_end=m.end(),
            parser_rule=f"sirius_{phase}",
        ))

    for m in ACRONYCHAL.finditer(text):
        planet = m.group(1) or m.group(2)
        if planet.lower() not in ("jupiter", "saturn", "mars"):
            continue
        key = ("acronychal", planet.lower(), m.start())
        if key in seen:
            continue
        seen.add(key)
        month, day, year_offset, restored = eclipse_context(text, m.start())
        out.append(Statement(
            kind="acronychal",
            planet=PLANETS[planet.capitalize()],
            sign="",
            text=re.sub(r"\s+", " ", clause_at(text, m.start(), m.end())),
            month=month,
            day=day,
            year_offset=year_offset,
            restored=restored,
            source_start=m.start(),
            source_end=m.end(),
            omitted=unwatched(text, "acronychal", m.start(), m.end()),
            parser_rule="acronychal_rising",
        ))

    for m in SEASON.finditer(text):
        key = ("season", m.group(1).lower(), m.start())
        if key in seen:
            continue
        seen.add(key)
        month, day, year_offset, restored = eclipse_context(text, m.start())
        out.append(Statement(
            kind="season",
            planet="sun",
            sign="",
            text=re.sub(r"\s+", " ", clause_at(text, m.start(), m.end())),
            month=month,
            day=day,
            year_offset=year_offset,
            # The scribes computed solstice, equinox and Sirius dates with an
            # arithmetical scheme, and the rule tests that scheme. "I did not
            # watch" after such a date says the event was not seen, not that
            # the date is uncertain, so the clause stays an observed claim.
            omitted=False,
            restored=restored,
            source_start=m.start(),
            source_end=m.end(),
            parser_rule=f"season_{m.group(1).lower()}",
        ))

    for kind, pat in PATTERNS:
        for m in pat.finditer(text):
            planet, sign = m.group(1), canonical_sign(m.group(2))
            key = (kind, planet, sign, m.start())
            if key in seen:
                continue
            seen.add(key)
            month, year_offset, uncertain = positional_context(text, m.start())
            uncertain = uncertain or match_fields_restored(text, m)
            out.append(Statement(
                kind=kind,
                planet=PLANETS[planet],
                sign=sign,
                text=re.sub(r"\s+", " ", m.group(0)),
                month=month,
                year_offset=year_offset,
                restored=uncertain,
                source_start=m.start(),
                source_end=m.end(),
                parser_rule=kind,
            ))
    return out


def months_in(text: str) -> list[str]:
    """Month headings the tablet itself preserves.

    Taken from the text rather than the catalogue on purpose. The catalogue's
    `months_recorded` is an editor's reading, and when the point is to date a
    tablet from its own text, using it would quietly import part of the answer.
    """
    return list(dict.fromkeys(month_from_match(m) for m in month_matches(text)))


def window(year_astro: int, months: list[str]) -> np.ndarray | None:
    """Julian days a statement could refer to, given a candidate year.

    A backward scan asks this for every candidate year, and the tablets repeat
    the same few month sets, so the result is cached. The array is returned to
    callers that only read it.
    """
    return _window(year_astro, tuple(months))


@lru_cache(maxsize=1 << 16)
def _window(year_astro: int, months: tuple[str, ...]) -> np.ndarray | None:
    _, nmon, slot = cal._index()
    if year_astro not in nmon:
        return None
    n = nmon[year_astro]
    idx = [i for i in (cal._month_index(m, n, slot.get(year_astro)) for m in months)
           if i]
    if months and not idx:
        # The clause names a month this year lacks, such as a second Addaru
        # in a twelve-month year. Testing it over the whole year would admit
        # positions the text excludes, so the clause is untestable here.
        return None
    if not idx:
        idx = list(range(1, n + 1))
    # The window runs from the civil day before the first month begins, which
    # holds the sunset that opens the month, to the start of the month after
    # the last one. Joint positions use the same bounds (joint_windows).
    ends = []
    for i in idx:
        end = (cal.month_start_jd(year_astro, i + 1) if i < n
               else cal.month_start_jd(year_astro + 1, 1))
        ends.append(end if end is not None
                    else cal.month_start_jd(year_astro, i) + 30.0)
    lo = min(cal.month_start_jd(year_astro, i) for i in idx) - 1.0
    hi = max(ends)
    # Daily samples with both bounds included, so w[0] and w[-1] are the window.
    return np.append(np.arange(lo, hi, 1.0), hi)


def _year_shape(year_astro: int) -> tuple[int, str | None]:
    """The month count and doubled month of a Babylonian year."""
    _, months, doubled = cal._index()
    return months.get(year_astro, 12), doubled.get(year_astro)


def sirius_scheme_date(year_astro: int, phase: str) -> tuple[int, int] | None:
    """Babylonian month slot and day the Sirius scheme predicts.

    The scheme runs on the same summer solstice the seasonal dates use, offset
    by whole months and days in the civil calendar, so this is arithmetic on a
    date the Sun already fixed rather than a visibility calculation.
    """
    return _sirius_scheme_date(year_astro, phase, float(sky.TIME_OFFSET_DAYS))


@lru_cache(maxsize=1 << 12)
def _sirius_scheme_date(year_astro: int, phase: str,
                        time_offset: float) -> tuple[int, int] | None:
    # time_offset is in the cache key only, so that a sensitivity run that
    # shifts the time scale is never served dates computed without the shift.
    ahead, extra = sky.SIRIUS_OFFSETS[phase]
    # The count starts at a summer solstice, in month III or IV. The last
    # appearance, ten months and 27 days later, falls in the next year, so it
    # is counted from the solstice of the year before.
    for base in (year_astro, year_astro - 1):
        start = cal.month_start_jd(base, "I")
        if start is None:
            continue
        solstice = sky.summer_solstice(start - 20.0, start + 400.0)
        if solstice is None:
            continue
        jd = solstice + sky.SCHEME_OFFSET_DAYS
        n_base, _ = _year_shape(base)
        slot = day = None
        for index in range(1, n_base + 1):
            begins = cal.month_start_jd(base, index)
            ends = (cal.month_start_jd(base, index + 1) if index < n_base
                    else cal.month_start_jd(base + 1, 1))
            if begins is not None and ends is not None and begins <= jd < ends:
                slot, day = index, int(jd - begins) + 1
                break
        if slot is None:
            continue
        slot += ahead
        day += extra
        # The scheme counts thirty-tithi months, which is what makes it a scheme.
        while day > 30:
            day -= 30
            slot += 1
        if base == year_astro and slot <= n_base:
            return slot, day
        if base == year_astro - 1 and slot > n_base:
            slot -= n_base
            return (slot, day) if slot <= _year_shape(year_astro)[0] else None
    return None


def check(st: Statement, year_astro: int, months: list[str]) -> Verdict:
    """Test one statement against one candidate year.

    The statement's own governing month is used where it has one, and its
    whole year otherwise. ``months``, the months the tablet names, is no longer
    read here; callers still pass it, and it is kept for them.
    """
    statement_year = int(year_astro) + st.year_offset

    if st.omitted:
        return Verdict(
            st, "untestable",
            "the text marks this event omitted, unwatched, or unobserved")

    if st.joint_positions:
        return check_joint_position(st, statement_year)

    if st.kind == "eclipse":
        if st.month is None or st.day is None:
            return Verdict(st, "untestable",
                           "no secure month and day govern this eclipse")
        event = eclipses.on_babylonian_day(
            statement_year, st.month, st.day, st.planet)
        if event is None:
            return Verdict(st, "untestable",
                           f"{yr.year_label(statement_year)} has no month "
                           f"{st.month} in the calendar")
        if event.empty:
            return Verdict(
                st, "contradict",
                f"no catalogue {st.planet} eclipse on this day")
        eligible = event[event.agreement_eligible.fillna(False)]
        if not eligible.empty:
            first = eligible.iloc[0]
            if st.planet == "lunar":
                detail = (
                    f"NASA lunar {first.catalog_num}, type {first.eclipse_type}; "
                    f"{first.local_phase} phase locally visible at Babylon "
                    "in all Delta T trials")
            else:
                detail = (
                    f"NASA solar {first.catalog_num}, type {first.eclipse_type}; "
                    f"local {first.local_phase} eclipse, magnitude "
                    f"{first.local_magnitude_min:.3f}--"
                    f"{first.local_magnitude_max:.3f} across Delta T trials")
            return Verdict(st, "agree", detail)

        uncertain = event[
            event.local_status.eq("delta_t_sensitive")
            | event.local_status.eq("robust_visible")
        ]
        if not uncertain.empty:
            first = uncertain.iloc[0]
            return Verdict(
                st, "untestable",
                f"NASA {st.planet} {first.catalog_num}, type "
                f"{first.eclipse_type}; {first.local_decision_reason}")

        first = event.iloc[0]
        return Verdict(
            st, "contradict",
            f"NASA {st.planet} {first.catalog_num}, type {first.eclipse_type}; "
            "global date agrees but no phase is locally visible at Babylon "
            "in any Delta T trial")

    if st.kind == "sirius":
        if st.month is None or st.day is None:
            return Verdict(st, "untestable",
                           "no secure month and day govern this Sirius date")
        phase = st.parser_rule.removeprefix("sirius_").removesuffix("_bracket")
        predicted = sirius_scheme_date(statement_year, phase)
        wanted = cal._month_index(
            st.month.split("|")[0], *_year_shape(statement_year))
        if predicted is None or wanted is None:
            return Verdict(st, "untestable",
                           f"{yr.year_label(statement_year)} is not in the calendar")
        if predicted[0] == wanted and abs(predicted[1] - st.day) <= TOL_SIRIUS_DAYS:
            return Verdict(st, "agree",
                           f"scheme puts it on day {predicted[1]} of the same "
                           "month")
        return Verdict(st, "contradict",
                       f"scheme puts it in month slot {predicted[0]} "
                       f"day {predicted[1]}")

    if st.kind == "acronychal":
        if st.month is None or st.day is None:
            return Verdict(st, "untestable",
                           "no secure month and day govern this acronychal "
                           "rising")
        jd = cal.bab_to_jd(statement_year, st.month.split("|")[0], st.day)
        if jd is None:
            return Verdict(st, "untestable",
                           f"{yr.year_label(statement_year)} is not in the calendar")
        off = 180.0 - abs(float(sky.elongation(st.planet, jd)))
        if abs(off) <= TOL_OPPOSITION_DEG:
            return Verdict(st, "agree",
                           f"{st.planet} is {off:+.1f} degrees from opposition")
        return Verdict(st, "contradict",
                       f"{st.planet} is {off:+.1f} degrees from opposition")

    if st.kind == "season":
        if st.month is None or st.day is None:
            return Verdict(st, "untestable",
                           "no secure month and day govern this solstice "
                           "or equinox")
        jd = cal.bab_to_jd(statement_year, st.month.split("|")[0], st.day)
        if jd is None:
            return Verdict(st, "untestable",
                           f"{yr.year_label(statement_year)} is not in the calendar")
        turning = (st.parser_rule.removeprefix("season_")
                   .removesuffix("_bracket"))
        computed = sky.turning_point(jd, turning)
        offset = jd - computed - sky.SCHEME_OFFSET_DAYS
        if abs(offset) <= TOL_SEASON_DAYS:
            return Verdict(st, "agree",
                           f"{turning} within {offset:+.1f} days of "
                           "the computed turning point, after the scheme's "
                           "own offset")
        return Verdict(st, "contradict",
                       f"{turning} {offset:+.1f} days from the "
                       "computed turning point")

    # A clause with no month of its own, such as one before the tablet's first
    # month heading, is tested over its whole year. The months the tablet names
    # later need not include the month the clause belongs to.
    statement_months = st.month.split("|") if st.month else []
    w = window(statement_year, statement_months)
    if w is None:
        return Verdict(st, "untestable",
                       f"{yr.year_label(statement_year)} has no month "
                       f"{'|'.join(statement_months)} in the calendar")

    if st.kind == "station_appulse":
        if st.star is None or st.offset is None:
            return Verdict(st, "untestable", "measured station target is incomplete")
        star = sky.star(st.star)
        if star is None:
            return Verdict(st, "untestable", f"{st.star} not catalogued")
        stations = sky.stations(st.planet, w[0], w[-1])
        if st.station_phase in {"east", "west"}:
            kind = "retrograde" if st.station_phase == "east" else "direct"
            stations = stations[stations.kind.eq(kind)]
        if st.day is not None and st.month and "|" not in st.month:
            start = cal.month_start_jd(statement_year, st.month)
            if start is not None:
                centre = start + st.day - 1
                stations = stations[stations.jd.between(centre - 2, centre + 2)]
        if stations.empty:
            return Verdict(st, "contradict", "no matching station in the source window")
        want = (star[0] + st.offset) % 360.0
        gaps = np.abs((stations.lon.to_numpy() - want + 180.0) % 360.0 - 180.0)
        gap = float(gaps.min())
        return Verdict(
            st, "agree" if gap < TOL_PLANET else "contradict",
            f"{gap:.2f} deg from the measured station-star relation")

    if st.kind in {"appulse", "planet_appulse", "planet_planet_appulse"}:
        if st.day is None:
            return Verdict(st, "untestable", "no day number governs this")
        if st.month and "|" in st.month:
            return Verdict(st, "untestable", "appulse month is ambiguous")
        start = (cal.month_start_jd(statement_year, st.month)
                 if st.month else None)
        if start is None:
            return Verdict(st, "untestable", "no month heading governs this")
        t0, t1 = start + st.day - 2.0, start + st.day + 0.5
        spans = [(t0, t1)]
        if MOON_NIGHT_WINDOWS and st.planet == "moon":
            civil = start + st.day - 1
            spans = [sky.babylon_night(civil), sky.babylon_night(civil + 1.0)]

        if st.kind in {"planet_appulse", "planet_planet_appulse"}:
            if st.target_planet is None or st.offset is None:
                return Verdict(st, "untestable", "measured target is incomplete")
            # The Moon needs a fine grid; two planets move slowly enough for a
            # quarter-day grid. Both use the same wrapped relative-longitude
            # convention as the measured text.
            step = 0.01 if st.kind == "planet_appulse" else 0.25
            jds = np.concatenate([np.arange(lo, hi + 1e-9, step)
                                  for lo, hi in spans])
            relative = ((sky.lon(st.planet, jds)
                         - sky.lon(st.target_planet, jds) + 180.0)
                        % 360.0 - 180.0)
            gap = float(np.abs(relative - st.offset).min())
            tolerance = (TOL_MOON if st.kind == "planet_appulse"
                         else TOL_PLANET_PAIR)
            return Verdict(
                st, "agree" if gap < tolerance else "contradict",
                f"{gap:.2f} deg from the measured {st.planet}-"
                f"{st.target_planet} relation")

        pos = sky.star(st.star)
        if pos is None:
            return Verdict(st, "untestable", f"{st.star} not catalogued")
        # The Babylonian day begins at the preceding sunset, so a night
        # observation on day n falls on the evening before that day's daytime.
        # One day of slack covers that and the sunset boundary both.
        # The Moon moves about 13 degrees a day, so coarse half-day samples can
        # add roughly 3 degrees of numerical residual to an exact appulse.
        # One day either side of month start plus day minus one. That centre
        # is not assumed, it is measured. Across 742 lunar appulses that match
        # within a degree, the offset from it is zero for 476 and minus one for
        # 161, and the minus one group is the sunset boundary putting a night
        # observation on the preceding Julian day. A window narrower than this
        # cuts that group off and halves the agreement rate.
        want = (sky.star(st.star)[0] + st.offset) % 360.0

        if st.planet == "moon":
            # The Moon never retrogrades in longitude, so over any window it
            # sweeps one monotonic arc and the only question is whether the
            # target lies inside it. That is two evaluations rather than the
            # hundred and twenty five that sampling finely enough for a body
            # moving 13 degrees a day would need, and it is exact rather than
            # approximate. Sampling was making a backward scan take minutes.
            gaps = []
            for lo, hi in spans:
                a, b = sky.lon("moon", lo), sky.lon("moon", hi)
                swept = (b - a) % 360.0
                into = (want - a) % 360.0
                gaps.append(0.0 if into <= swept else min(
                    (into - swept), (360.0 - into)))
            gap = min(gaps)
        else:
            # Planets do retrograde, so the arc is not monotonic. They are slow
            # enough that a coarse sample is ample.
            jds = np.arange(t0, t1 + 1e-9, 0.25)
            gap = float(np.abs((sky.lon(st.planet, jds) - want + 180.0)
                               % 360.0 - 180.0).min())

        tol = TOL_MOON if st.planet == "moon" else TOL_PLANET
        return Verdict(st, "agree" if gap < tol else "contradict",
                       f"{gap:.2f} deg from where the text puts it")

    want = sky.SIGNS.index(st.sign)
    got = set(sky.sign_index(sky.babylonian_lon(st.planet, w)))

    if st.kind in ("in_sign", "last_app", "first_app"):
        # The verdict uses the exact sign intervals; the daily samples only
        # name the signs for the reason.
        ok = bool(sky.sign_occupancy_intervals(st.planet, st.sign, w[0], w[-1]))
        where = ", ".join(sky.SIGNS[i] for i in sorted(got))
        return Verdict(st, "agree" if ok else "contradict",
                       f"{st.planet} spans {where}")

    if st.kind == "station":
        s = sky.stations(st.planet, w[0] - 20, w[-1] + 20)
        if s.empty:
            return Verdict(st, "contradict", f"{st.planet} does not station here")
        hit = [r for _, r in s.iterrows()
               if sky.sign_index(r.lon_babylonian) == want]
        if hit:
            return Verdict(st, "agree",
                           f"stations in {st.sign} at JD {hit[0].jd:.0f}")
        seen = ", ".join(sky.SIGNS[int(sky.sign_index(r.lon_babylonian))]
                         for _, r in s.iterrows())
        return Verdict(st, "contradict", f"stations only in {seen}")

    if st.kind == "entry":
        e = sky.sign_entries(st.planet, w[0] - 20, w[-1] + 20)
        if not e.empty and (e["into"] == st.sign).any():
            j = e[e["into"] == st.sign].jd.iloc[0]
            return Verdict(st, "agree", f"enters {st.sign} at JD {j:.0f}")
        return Verdict(st, "contradict",
                       f"no entry into {st.sign}" if e.empty
                       else f"enters only {', '.join(e['into'])}")

    return Verdict(st, "untestable", f"no test for {st.kind}")


def joint_windows(st: Statement, statement_year: int) -> list[tuple[float, float]]:
    """Actual source month alternatives, not their convex-hull time span.

    Month starts name the associated civil daytime. One preceding civil day
    includes the first Babylonian night's sunset boundary. A securely attached
    day restricts the test to that civil day and its preceding night; no
    tablet-specific calendar shifts or fitted timing offsets are introduced.
    """
    if st.month is None:
        return []
    n, doubled = _year_shape(statement_year)
    windows = []
    for name in st.month.split("|"):
        slot = cal._month_index(name, n, doubled)
        if slot is None:
            continue
        start = cal.month_start_jd(statement_year, slot)
        end = (cal.month_start_jd(statement_year, slot + 1) if slot < n
               else cal.month_start_jd(statement_year + 1, "I"))
        if start is None or end is None:
            continue
        if st.joint_day is not None:
            civil = start + st.joint_day - 1
            if not start <= civil < end:
                continue
            windows.append((civil - 1.0, civil + 1.0))
        else:
            windows.append((start - 1.0, end))
    return windows


def check_joint_position(st: Statement, statement_year: int) -> Verdict:
    if st.joint_omitted:
        return Verdict(st, "untestable", "the coordinated source marks a conjunct unobserved")
    windows = joint_windows(st, statement_year)
    if not windows:
        return Verdict(st, "untestable", "no secure calendar window governs this joint position")
    try:
        overlap = [interval for lo, hi in windows
                   for interval in sky.joint_sign_intervals(st.joint_positions, lo, hi)]
    except (KeyError, ValueError) as exc:
        return Verdict(st, "untestable", str(exc))
    label = " and ".join(f"{p} in {s}" for p, s in st.joint_positions)
    if overlap:
        periods = "; ".join(f"JD {a:.6f}--{b:.6f}" for a, b in overlap)
        return Verdict(st, "agree", f"simultaneous {label}: {periods}")
    return Verdict(st, "contradict", f"no simultaneous overlap for {label} in the source window")


def is_decisive(st: Statement) -> bool:
    """Semantic evidence tier, independent of any candidate-year outcome."""
    # Mixed-unit parsing is supported by recurring source forms, but the dated
    # corpus contains no unrestored, fully contextualized outer-planet example
    # with a testable forward residual. Retain these facts as support until an
    # independent decisive-tier forward gate exists.
    if st.joint_positions:
        # A supporting inner-planet conjunct must not acquire decisive power
        # merely by being coordinated with an outer planet.
        return (not st.restored and not st.omitted
                and not st.joint_restored and not st.joint_omitted
                and all(p in {"mars", "jupiter", "saturn"}
                        for p, _ in st.joint_positions))
    if st.latitudinal and not LATITUDINAL_DISTANCES_DECISIVE:
        return False
    if (st.parser_rule.endswith("_compound")
            or st.parser_rule.startswith("measured_station_normal_star")
            or st.parser_rule.startswith("measured_planet_planet")):
        return False
    return bool(
        (st.planet in {"mars", "jupiter", "saturn"}
         and st.kind != "last_app" and not st.restored)
        or (st.kind == "planet_appulse"
            and st.target_planet in {"mars", "jupiter", "saturn"}
            and not st.restored)
        or (st.kind == "eclipse" and st.planet == "lunar"
            and not st.omitted and not st.restored)
        or (st.kind == "season" and not st.restored)
        or (st.kind == "sirius" and not st.restored)
    )


# Bits are summed in integers of 1e-12 bit and the sums rounded to 1e-9 bit.
# Integer sums do not depend on the order of addition, so every machine gets
# the same scores, and sums that differ only by floating-point rounding tie in
# the strict set, the ordering and the groups.
BIT_UNIT = 1e-12
BIT_QUANTUM = 1e-9


def bit_sums(agrees: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-year sums of the weights of agreeing clauses, machine-independent."""
    units = np.rint(np.asarray(weights, dtype=float) / BIT_UNIT).astype(np.int64)
    total = np.asarray(agrees, dtype=np.int64) @ units
    return np.rint(total * (BIT_UNIT / BIT_QUANTUM)) * BIT_QUANTUM


def candidate_scores(sts: list[Statement], months: list[str], years,
                     weighting: str = "skill") -> CandidateScores:
    """Score candidate years, measuring each statement's discrimination.

    A statement's agreeing fraction is the share of testable candidate years
    for which it agrees. The default weight is its Shannon information,
    ``-log2(fraction)``. Thus a statement true in every year contributes
    nothing to the ranking, one true in a quarter of years contributes two
    bits, and one true in a sixteenth contributes four. This is deliberately
    more conservative than raw inverse frequency, under which one rare bad
    extraction could overwhelm everything else on the tablet.

    ``skill`` uses the same information weights but applies the semantic rule
    already justified in the Dating Skill: Moon, Venus and Mercury evidence may
    rank candidates but may not eliminate a year that has stronger outer-planet
    support. Its top set is therefore determined by fully checked Mars, Jupiter
    and Saturn statements; ``last_app`` stays supporting until visibility is
    implemented rather than its sign alone. Supporting evidence orders
    candidates *inside* that set. If no outer-planet
    statement survives, supporting evidence may order the years for inspection
    but the primary score is tied across the full calendar: the method declines
    to turn Moon or inner-planet evidence into a year, as Skill section 6
    requires.

    ``count`` preserves the old one-agreement-one-vote baseline. ``inverse`` is
    exposed for calibration comparison, not used by default. ``equal`` keeps
    the tiers, clocks and sets of ``skill``, but gives each statement that
    agrees with some year one unit in place of its bits, shared in the same
    way; only the sensitivity audit uses it. A statement that
    is never testable or agrees with no year gets zero weight: it cannot
    distinguish among the candidates regardless of why it failed.
    """
    if weighting not in {"count", "information", "inverse", "skill", "equal"}:
        raise ValueError(
            f"unknown weighting {weighting!r}; use count, information, "
            "inverse, skill or equal")

    years = np.asarray(list(years), dtype=int)
    # A diagnostic ablation or caller may remove a conjunct. Such a list must
    # not continue testing an unrepresented planet through stale group fields.
    grouped: dict[tuple, list[int]] = {}
    for j, st in enumerate(sts):
        if st.joint_positions:
            grouped.setdefault(scoring_unit_key(st, j), []).append(j)
    sts = list(sts)
    for members in grouped.values():
        expected = set(sts[members[0]].joint_positions)
        actual = {(sts[j].planet, sts[j].sign) for j in members
                  if sts[j].kind == "in_sign"}
        if actual != expected:
            for j in members:
                sts[j] = replace(sts[j], joint_positions=(), joint_source_start=-1,
                                 joint_source_end=-1, joint_day=None,
                                 joint_restored=False, joint_omitted=False)
    unit_ids: dict[tuple, int] = {}
    scoring_units = np.array([
        unit_ids.setdefault(scoring_unit_key(st, j), len(unit_ids))
        for j, st in enumerate(sts)], dtype=int)
    first_in_unit = [int(np.flatnonzero(scoring_units == unit)[0])
                     for unit in range(len(unit_ids))]
    outcomes = np.zeros((len(years), len(sts)), dtype=np.int8)
    for i, year_astro in enumerate(years):
        for j, st in enumerate(sts):
            result = check(st, int(year_astro), months).result
            outcomes[i, j] = 1 if result == "agree" else (
                -1 if result == "contradict" else 0)

    agrees = outcomes == 1
    checked = outcomes != 0
    agreeing = agrees.sum(axis=0)
    testable = checked.sum(axis=0)
    weights = np.zeros(len(sts), dtype=float)
    useful = agreeing > 0

    if weighting == "count":
        weights[testable > 0] = 1.0
    elif weighting == "equal":
        weights[useful] = 1.0
    elif weighting in {"information", "skill"}:
        weights[useful] = np.log2(testable[useful] / agreeing[useful])
    else:
        weights[useful] = testable[useful] / agreeing[useful]

    decisive = np.array([is_decisive(st) for st in sts], dtype=bool)

    # Two extracted clauses sometimes reduce to exactly the same predicate in
    # the current deterministic layer. For example, until visibility itself is
    # built, "last appearance in Aries" and "was in Aries" can admit precisely
    # the same years. Repeating an identical candidate mask adds no ranking
    # information, so share one weight across equivalent columns. The count
    # mode remains byte-for-byte the historical baseline and does not do this.
    equivalent = np.ones(len(sts), dtype=int)
    if weighting != "count":
        groups: dict[bytes, list[int]] = {}
        for j in range(len(sts)):
            groups.setdefault(outcomes[:, j].tobytes(), []).append(j)
        for group in groups.values():
            equivalent[group] = len(group)
            # If a fully checked decisive clause and a supporting-only clause
            # reduce to the same candidate mask, keep the predicate in the
            # decisive tier rather than splitting half its evidence into the
            # weaker tier. Otherwise share one predicate weight equally.
            targets = [j for j in group if decisive[j]] or group
            group_weight = weights[group[0]]
            weights[group] = 0.0
            weights[targets] = group_weight / len(targets)
    if weighting == "count":
        # One source clause, not one vote per decomposed conjunct.
        for unit in range(len(unit_ids)):
            cols = np.flatnonzero(scoring_units == unit)
            weights[cols] /= len(cols)
    total = bit_sums(agrees, weights)
    priority = np.zeros(len(years), dtype=int)
    support = np.zeros(len(years), dtype=float)

    def distinct_bodies(active: np.ndarray) -> np.ndarray:
        """How many independent decisive evidence families each year explains."""
        count = np.zeros(len(years), dtype=int)
        for body in ("mars", "jupiter", "saturn", "lunar_eclipse", "seasonal_sirius"):
            cols = np.array([
                active[j]
                and body in evidence_families(sts[j])
                and weights[j] > 0
                for j in range(len(sts))
            ], dtype=bool)
            if cols.any():
                count += agrees[:, cols].any(axis=1)
        return count

    if weighting in {"skill", "equal"}:
        if decisive.any():
            priority = distinct_bodies(decisive)
            score = bit_sums(agrees[:, decisive], weights[decisive])
            support = bit_sums(agrees[:, ~decisive], weights[~decisive])
        else:
            score = np.zeros(len(years), dtype=float)
            support = total
    else:
        score = total

    unit_outcomes = outcomes[:, first_in_unit]
    contradictions = (unit_outcomes == -1).sum(axis=1)

    def pareto_frontier(body_count: np.ndarray,
                        information: np.ndarray) -> np.ndarray:
        """Years not dominated on independent bodies and information."""
        frontier = np.zeros(len(information), dtype=bool)
        higher_best = -np.inf
        for count in range(int(body_count.max()), -1, -1):
            same = body_count == count
            if not same.any():
                continue
            best = float(information[same].max())
            # Equal information at a higher body count dominates this tier.
            if best > higher_best + 1e-12:
                frontier |= same & np.isclose(
                    information, best, rtol=1e-12, atol=1e-12)
            higher_best = max(higher_best, best)
        if not FRONTIER_PENALISES_CONTRADICTIONS or not frontier.any():
            return frontier
        # Among the years the frontier retains, keep only those contradicting
        # fewest clauses. A contradiction may be a scribal error, a restoration
        # or a misread clause, which is why this is not the default.
        fewest = int(contradictions[frontier].min())
        return frontier & (contradictions == fewest)

    strict = (pareto_frontier(priority, score) if len(score)
              else np.zeros(0, dtype=bool))
    safety = strict.copy()
    if weighting in {"skill", "equal"} and decisive.any():
        # A year is in the safety set if it leads either with all decisive
        # statements or after removing any one of them. This is an explicit
        # one-bad-extraction budget, not a fitted score margin. It implements
        # the Skill's rule that a single wrongly promoted hard constraint must
        # not make the correct year unrecoverable.
        for unit in range(len(unit_ids)):
            cols = np.flatnonzero((scoring_units == unit) & decisive & (weights > 0))
            if not len(cols):
                continue
            active = decisive.copy()
            active[scoring_units == unit] = False
            ablated_priority = distinct_bodies(active)
            kept = decisive & (scoring_units != unit)
            ablated = bit_sums(agrees[:, kept], weights[kept])
            safety |= pareto_frontier(ablated_priority, ablated)

    return CandidateScores(
        years=years,
        priority=priority,
        score=score,
        support=support,
        agree=(unit_outcomes == 1).sum(axis=1),
        contradict=contradictions,
        untestable=(unit_outcomes == 0).sum(axis=1),
        weights=weights,
        agreeing=agreeing,
        testable=testable,
        equivalent=equivalent,
        decisive=decisive,
        weighting=weighting,
        strict=strict,
        safety=safety,
        outcomes=outcomes,
        scoring_units=scoring_units,
    )


def evidence_level_key(scores: CandidateScores, index: int) -> tuple:
    """Scientific ordering key with the arbitrary chronological tie-break removed."""
    tier = 0 if scores.strict[index] else (1 if scores.safety[index] else 2)
    families = -int(scores.priority[index])
    decisive = -round(float(scores.score[index]), 12)
    support = -round(float(scores.support[index]), 12)
    contradictions = int(scores.contradict[index])
    if ORDER_CONTRADICTIONS_FIRST:
        return (tier, families, contradictions, decisive, support)
    return (tier, families, decisive, support, contradictions)


def evidence_levels(
        scores: CandidateScores, max_levels: int | None = None) -> list[dict]:
    """Return ordered score levels and exact predicate-equivalent subgroups.

    A score level contains every candidate tied on the implemented scientific
    ordering fields. Its ``physical_groups_years_astro`` split that level by the
    complete agree/contradict/untestable vector.  A predicate-exact level has
    one such subgroup and can be called a physical evidence group without any
    arbitrary within-level decision.  If a future tablet has several subgroups
    at one rank, they remain explicitly co-ranked.
    """
    keys = [evidence_level_key(scores, i) for i in range(len(scores.years))]
    ordered_keys = sorted(set(keys))
    if max_levels is not None:
        ordered_keys = ordered_keys[:max_levels]
    tier_labels = {0: "strict", 1: "safety", 2: "outside"}
    levels: list[dict] = []
    for rank, key in enumerate(ordered_keys, start=1):
        indices = [i for i, candidate_key in enumerate(keys)
                   if candidate_key == key]
        by_signature: dict[bytes, list[int]] = {}
        for index in indices:
            by_signature.setdefault(
                scores.outcomes[index].tobytes(), []).append(
                    int(scores.years[index]))
        physical_groups = sorted(
            (sorted(years) for years in by_signature.values()),
            key=lambda years: (years[0], len(years)),
        )
        levels.append({
            "rank": rank,
            "tier": tier_labels[key[0]],
            "decisive_families": -key[1],
            "primary_bits": -key[2],
            "support_bits": -key[3],
            "contradictions": key[4],
            "years_astro": sorted(int(scores.years[i]) for i in indices),
            "physical_groups_years_astro": physical_groups,
            "predicate_exact": len(physical_groups) == 1,
        })
    return levels


def score(text: str, year_astro: int) -> dict:
    """Check every statement on a tablet against one candidate year."""
    months = months_in(text)
    sts = extract(text)
    vs = [check(s, year_astro, months) for s in sts]
    return {"year_astro": int(year_astro), "months": months, "verdicts": vs,
            "agree": sum(v.result == "agree" for v in vs),
            "contradict": sum(v.result == "contradict" for v in vs),
            "untestable": sum(v.result == "untestable" for v in vs),
            "n": len(vs)}
