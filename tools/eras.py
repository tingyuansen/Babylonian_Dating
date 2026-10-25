#!/usr/bin/env python3
"""Extract source-backed absolute chronology stated inside a tablet.

The common formula ``year 170, which is year 234`` identifies both eras without
requiring a king name: the epochs are independently pinned 64 years apart. A
second conservative form is ``year 143, king Antiochus``. Parker and
Dubberstein document continuous Seleucid-Era dating after Seleucus I and give
these Seleucid ruler headings in the same numbered era. A bare ``year 170``, a
generic ``king Arsaces``, a historical person or troop report, and any restored
year remain unresolved. Refusal is the important behavior.

Absolute years are returned in astronomical numbering so the same type works
across the BCE/AD boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from . import cal
except ImportError:  # permit ``python tools/eras.py`` as a direct self-check
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import cal

DUAL = re.compile(
    r"\byear\s+(\d{1,3})\b[^.;]{0,100}?"
    r"\b(?:which\s+is|corresponding\s+to)\s+year\s+(\d{1,3})\b",
    re.IGNORECASE)
SE_RULER_YEAR = re.compile(
    r"\byear\s+(\d{1,3})\s*,\s*"
    r"(?:\(\s*kings?\s*\)\s*|kings?\s+)?"
    r"(Seleucus|Antiochus|Demetrius)\b",
    re.IGNORECASE)
MONTH_HEADING = re.compile(
    r"\bMonth\s+(?:XII|XI|IX|VIII|VII|VI|IV|V|III|II|I|X)\b",
    re.IGNORECASE)
EDGE_OR_DIARY = re.compile(
    r"(?:\bDiary\b|\((?:b|t|l|r)\.e\.\s*\d*\))",
    re.IGNORECASE)


@dataclass(frozen=True)
class EraReading:
    ae_year: int | None
    se_year: int | None
    astro_year: int
    text: str
    system: str
    source_start: int


def _restored(text: str, pos: int) -> bool:
    before = text[:pos]
    return before.rfind("[") > before.rfind("]")


def _dating_formula(text: str, pos: int) -> bool:
    """Distinguish a tablet heading/edge from a dated historical report."""
    if not MONTH_HEADING.search(text, 0, pos):
        return True
    line_start = text.rfind("\n", 0, pos) + 1
    return bool(EDGE_OR_DIARY.search(text[line_start:pos]))


def extract(text: str) -> list[EraReading]:
    """Secure dual-era/named-SE readings, deduplicated in textual order."""
    readings = []
    seen = set()

    for match in DUAL.finditer(text):
        if _restored(text, match.start(1)) or _restored(text, match.start(2)):
            continue
        a, b = int(match.group(1)), int(match.group(2))
        ae_year, se_year = min(a, b), max(a, b)
        if se_year - ae_year != cal.SE_EPOCH - cal.AE_EPOCH:
            continue
        astro_se = cal.to_astro(se_year, "SE")
        astro_ae = cal.to_astro(ae_year, "AE")
        if astro_se != astro_ae:
            continue
        key = (ae_year, se_year, astro_se)
        if key in seen:
            continue
        seen.add(key)
        readings.append(EraReading(
            ae_year=ae_year,
            se_year=se_year,
            astro_year=astro_se,
            text=re.sub(r"\s+", " ", match.group(0)),
            system="dual_ae_se",
            source_start=match.start(),
        ))

    for match in SE_RULER_YEAR.finditer(text):
        if _restored(text, match.start(1)) or _restored(text, match.start(2)):
            continue
        if not _dating_formula(text, match.start()):
            continue
        se_year = int(match.group(1))
        astro_year = cal.to_astro(se_year, "SE")
        key = (None, se_year, astro_year)
        if key in seen:
            continue
        seen.add(key)
        readings.append(EraReading(
            ae_year=None,
            se_year=se_year,
            astro_year=astro_year,
            text=re.sub(r"\s+", " ", match.group(0)),
            system="named_seleucid_era",
            source_start=match.start(),
        ))

    return sorted(readings, key=lambda reading: reading.source_start)


def resolve(text: str) -> int:
    """One absolute astronomical year, refusing absent or conflicting evidence."""
    readings = extract(text)
    years = {reading.astro_year for reading in readings}
    if not years:
        raise cal.AmbiguousEra(
            "no unrestored dual AE/SE year statement survives")
    if len(years) != 1:
        raise cal.AmbiguousEra(
            f"the text contains multiple absolute years: {sorted(years)}")
    return years.pop()


def check() -> bool:
    # 78 BCE -> astronomical -77; 169 BCE -> astronomical -168.
    good = resolve("Year 170, which is year 234.") == -77
    good &= resolve("Year 143, king Antiochus.") == -168
    good &= not extract("Year 170, which is [ year 234 ].")
    good &= not extract("[ Year 143 ], king Antiochus.")
    # Explicitly test a Seleucid-Era year after the era boundary.
    good &= cal.to_astro(312, "SE") == 1
    try:
        resolve("Year 170, king Arsaces.")
        good = False
    except cal.AmbiguousEra:
        pass
    print(f"  dual-era extractor  {'ok' if good else 'FAIL'}")
    return bool(good)


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
