#!/usr/bin/env python3
"""Build the Parker--Dubberstein (1956) Babylonian calendar from OCR text.

The 1956 Brown University Press scan is image-only.  First run
``scripts/physical/ocr_parker_dubberstein.py`` to OCR PDF pages 43--63.  This
script then parses that OCR product, applies a small transparent ledger of
visually verified OCR corrections, and writes the physical calendar.

All computational years use astronomical numbering:

    626 BCE -> -625
    1 BCE   -> 0
    AD 1    -> 1
    AD 75   -> 75

The output must contain exactly 701 Babylonian years, 626 BCE through AD 75.
The final Babylonian year extends into Julian AD 76.

Which month a thirteen-month year doubled is read from the printed column
(U II or A II) into ``data/source/parker_dubberstein/intercalary_slots.csv`` by
``ocr_parker_dubberstein.py --slots``.  Every thirteen-month year must appear in
that ledger and no twelve-month year may.

Important: Parker--Dubberstein itself prints four month-start intervals of 28
or 31 days.  Those are preserved and listed in ``PRINTED_IRREGULAR_GAPS``;
any other interval outside 29--30 days, within a year or across a year
boundary, stops the build.  The source is not altered merely to force all
month lengths to 29/30 days.  Independent astronomical checking is done by
``check_calendar_against_sky.py`` after this build step.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import years as yr  # noqa: E402

DEFAULT_OCR = (
    ROOT / "data" / "source" / "parker_dubberstein" / "calendar_ocr.txt"
)
DEFAULT_OUT = ROOT / "data" / "physical" / "calendar.parquet"
DEFAULT_CORRECTIONS = ROOT / "data" / "physical" / "calendar_ocr_corrections.csv"
DEFAULT_SLOTS = ROOT / "data" / "source" / "parker_dubberstein" / "intercalary_slots.csv"

# Deliberately no word boundary before the month.  Tesseract occasionally
# fuses the printed rollover year to a month token, e.g. ``3891/27``.  The
# month normalizer below recovers the trailing valid month number.
DATE_RE = re.compile(r"(\d{1,3})/(\d{1,2})")
PAGE_MARKER = re.compile(r"^===== PDF PAGE (\d+) =====$")
PRE_HEAD = re.compile(r"^[*.,\s]*(Ac|\d{1,3})\s+(\d{3})\b")
SE_HEAD = re.compile(r"^[*.,\s]*(\d{1,3})\b")

# A very small number of rows are especially unstable across Tesseract builds.
# For these rows we transcribe the complete printed row directly from the 1956
# scan rather than trying to repair one token at a time.  This is still an OCR
# correction, not an astronomical reconstruction.
ROW_OVERRIDES = {
    yr.bce_to_astro(308): {
        "page": 53,
        "dates": [(3,30),(4,29),(5,29),(6,27),(7,26),(8,25),
                  (9,23),(10,23),(11,21),(12,21),(1,19),(2,18),(3,19)],
        "note": "complete row transcribed from printed SE 4 / 308 BCE row",
    },
    yr.bce_to_astro(125): {
        "page": 58,
        "dates": [(4,16),(5,15),(6,13),(7,13),(8,11),(9,10),
                  (10,9),(11,8),(12,8),(1,6),(2,5),(3,7)],
        "note": "complete row transcribed because Tesseract corrupts the final 2/5 and 3/7",
    },
    4: {
        "page": 61,
        "dates": [(4,10),(5,10),(6,9),(7,8),(8,7),(9,6),
                  (10,5),(11,3),(12,2),(1,1),(1,30),(3,1)],
        "note": "complete AD 4 row transcribed because rollover cells are unstable in OCR",
    },
    61: {
        "page": 63,
        "dates": [(4,10),(5,10),(6,8),(7,8),(8,6),(9,5),
                  (10,4),(11,3),(12,3),(1,1),(1,31),(3,1)],
        "note": "complete AD 61 row transcribed because rollover cells are unstable in OCR",
    },
}

# Visually checked against the 1956 scan.  These are OCR corrections only;
# they do not alter the printed Parker--Dubberstein dates.
# Keys are Babylonian years in astronomical numbering.  A year in
# ROW_OVERRIDES is replaced whole and never reaches this table.
SOURCE_FIXES = {
    yr.bce_to_astro(479): {
        "page": 47,
        "action": "replace_last",
        "date": (3, 21),
        "note": "OCR read printed 3/21 as 3/27",
    },
    yr.bce_to_astro(396): {
        "page": 50,
        "action": "insert_before_last",
        "date": (2, 1),
        "note": "OCR rendered printed 2/1 as letters",
    },
    yr.bce_to_astro(377): {
        "page": 50,
        "action": "insert_before_last",
        "date": (2, 1),
        "note": "OCR rendered printed 2/1 as 2/i",
    },
    yr.bce_to_astro(348): {
        "page": 51,
        "action": "insert_before_last",
        "date": (2, 11),
        "note": "OCR rendered printed 2/11 as non-numeric glyphs",
    },
    yr.bce_to_astro(342): {
        "page": 51,
        "action": "replace_last",
        "date": (3, 5),
        "note": "OCR misread printed 3/5 as 8/5",
    },
    yr.bce_to_astro(320): {
        "page": 52,
        "action": "insert_before_last",
        "date": (2, 1),
        "note": "OCR rendered printed 2/1 as letters",
    },
    yr.bce_to_astro(174): {
        "page": 56,
        "action": "insert_before_last",
        "date": (2, 7),
        "note": "OCR rendered printed 2/7 without a slash",
    },
    yr.bce_to_astro(54): {
        "page": 60,
        "action": "insert_before_last",
        "date": (2, 1),
        "note": "OCR dropped the day digit in printed 2/1",
    },
    0: {
        "page": 61,
        "action": "prepend",
        "date": (3, 26),
        "note": "OCR lost the printed Nisanu 1 date 3/26 for SE 311 = 1 BCE",
    },
    yr.bce_to_astro(605): {
        "page": 43, "action": "insert_before_last", "date": (2, 2),
        "note": "OCR rendered printed 2/2 as nonnumeric glyphs",
    },
    yr.bce_to_astro(410): {
        "page": 49, "action": "append_many", "dates": [(2, 7), (3, 7)],
        "note": "OCR rendered printed 2/7 and 3/7 without slashes",
    },
    yr.bce_to_astro(100): {
        "page": 59, "action": "insert_at", "index": 9, "date": (1, 1),
        "note": "OCR rendered printed 1/1 as i/1",
    },
    yr.bce_to_astro(7): {
        "page": 61, "action": "prepend", "date": (4, 2),
        "note": "OCR fused printed 7 and 4/2 as 74/2",
    },
    yr.bce_to_astro(2): {
        "page": 61, "action": "prepend", "date": (4, 6),
        "note": "OCR fused printed 2 and 4/6 as 24/6",
    },
    2: {
        "page": 61, "action": "prepend", "date": (4, 3),
        "note": "OCR fused printed AD 2 and 4/3 as 24/3",
    },
    7: {
        "page": 62, "action": "prepend", "date": (4, 8),
        "note": "OCR fused printed AD 7 and 4/8 as 74/8",
    },
}

# The only month-start intervals outside 29--30 days that the 1956 table
# prints, each checked on the scan.  Keyed by Babylonian year and the index of
# the month the interval ends at; index 1 is the interval from the last month
# start of the year before.
PRINTED_IRREGULAR_GAPS = {
    (yr.bce_to_astro(600), 8): 28.0,  # 9/12 -> 10/10, p. 43
    (yr.bce_to_astro(530), 1): 31.0,  # 531 BCE 2/23 -> 530 BCE 3/26, p. 45
    (yr.bce_to_astro(512), 9): 31.0,  # 10/28 -> 11/28, p. 46
    (yr.bce_to_astro(314), 5): 31.0,  # 7/3 -> 8/3, p. 52
}

# Standard nineteen-year cycle.  From 380 BCE onward the table must follow it
# in which years have thirteen months.  Its placement of the doubled month
# (second Ululu when the BCE year is 9 mod 19, otherwise second Addaru) is only
# a cross-check on the printed column: it matches every printed row from
# 380 BCE on but not all of the earlier ones.
CYCLE = 19
STANDARD_BCE_RESIDUES = {1, 4, 7, 9, 12, 15, 18}
STANDARD_ASTRO_RESIDUES = {(1 - value) % CYCLE for value in STANDARD_BCE_RESIDUES}
ULULU_RESIDUE_ASTRO = (1 - 9) % CYCLE
REGULAR_CYCLE_START_ASTRO = yr.bce_to_astro(380)


def tidy(line: str) -> str:
    """Repair mechanical OCR spacing in row heads without guessing dates."""
    line = line.strip()
    # OCR can prefix table rows with rules/bars/braces.
    line = re.sub(r"^[^A-Za-z0-9]+", "", line)

    # A stray OCR letter can precede a numeric table row, e.g. ``i 141 171``.
    line = re.sub(r"^[A-Za-z]\s+(?=\d)", "", line)

    # Examples: ``3.314`` -> ``3 314`` and ``42, 363`` -> ``42 363``.
    line = re.sub(r"^(\d{1,3})[.,]\s*(\d{3})(?=\s)", r"\1 \2", line)

    # Sometimes the three-digit BCE year is fused directly to Nisanu,
    # e.g. ``7 3524/6`` -> ``7 352 4/6``.
    line = re.sub(r"^(\d{1,3})\s+(\d{3})(?=\d{1,2}/)", r"\1 \2 ", line)

    # Tesseract occasionally fuses a one/two-digit regnal year and the
    # three-digit BCE year: ``4526`` -> ``4 526``, ``7515`` -> ``7 515``.
    fused = re.match(r"^(\d{4,6})\s+(?=\d{1,3}/)", line)
    if fused:
        blob = fused.group(1)
        prefix, suffix = blob[:-3], blob[-3:]
        if (prefix and 311 <= int(suffix) <= 626
                and 0 <= int(prefix) <= 100):
            line = f"{int(prefix)} {int(suffix)}" + line[fused.end(1):]

    return line


def extract_dates(line: str) -> list[tuple[int, int]]:
    """Extract plausible Julian month/day cells from one OCR row."""
    dates: list[tuple[int, int]] = []
    for month_text, day_text in DATE_RE.findall(line):
        month = int(month_text)
        day = int(day_text)

        # Do not reinterpret an impossible month such as ``44/7`` as
        # ``4/7``.  Such tokens are usually OCR damage to the printed
        # Julian rollover year (for example 441), not calendar dates.
        if 1 <= month <= 12 and 1 <= day <= 31:
            dates.append((month, day))

    return dates


def apply_source_fix(
    year_astro: int,
    dates: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], dict | None]:
    """Apply a visually verified correction when that OCR defect is present.

    Returns the dates and a ledger record, or no record when the reading
    already had the printed value and nothing changed.
    """
    fix = SOURCE_FIXES.get(int(year_astro))
    if fix is None:
        return dates, None

    original = list(dates)
    dates = list(dates)
    target = tuple(fix["date"]) if "date" in fix else None
    targets = [tuple(x) for x in fix.get("dates", [])]

    # If a different Tesseract version already read the correction correctly,
    # do nothing.  An insertion would otherwise duplicate a date already read.
    if target is not None and target in dates and fix["action"] != "replace_last":
        return dates, None
    if targets and all(t in dates for t in targets):
        return dates, None

    action = fix["action"]
    if action == "prepend":
        dates.insert(0, target)
    elif action == "append":
        dates.append(target)
    elif action == "insert_before_last":
        if not dates:
            raise RuntimeError(f"cannot insert into empty row {year_astro}")
        dates.insert(len(dates) - 1, target)
    elif action == "replace_last":
        if not dates:
            raise RuntimeError(f"cannot replace empty row {year_astro}")
        dates[-1] = target
    elif action == "insert_at":
        dates.insert(int(fix["index"]), target)
    elif action == "append_many":
        dates.extend(targets)
    else:
        raise RuntimeError(f"unknown OCR correction action {action!r}")

    if dates == original:
        return dates, None

    record = {
        "bab_year_astro": int(year_astro),
        "year_label": yr.year_label(int(year_astro)),
        "pdf_page": int(fix["page"]),
        "action": action,
        "corrected_month": int(target[0]) if target is not None else None,
        "corrected_day": int(target[1]) if target is not None else None,
        "note": str(fix["note"]),
    }
    return dates, record


def heading(line: str) -> str | None:
    """Return a ruler/era heading from the table, if this line is one."""
    s = " ".join(line.strip().split())
    if not s or "/" in s or any(ch.isdigit() for ch in s):
        return None
    up = s.upper()
    if up == "SELEUCID ERA":
        return "SELEUCID ERA"
    if not re.fullmatch(r"[A-Z][A-Z '\-]{2,40}", up):
        return None
    blocked = ("YEAR", "BABYLONIAN", "TABLE", "JULIAN", "CALENDAR")
    if up.startswith(blocked):
        return None
    return up


def parse_se_year(line: str, expected_se: int | None = None) -> int | None:
    """Recover the Seleucid-era year from an OCR row head robustly.

    Some scans turn the left table rule into a stray leading ``1``.  Rather
    than trusting the first integer, inspect the numeric prefix before the
    first month/day token and look for a pair satisfying the printed
    Seleucid/BCE or Seleucid/AD relation.
    """
    date_match = DATE_RE.search(line)
    prefix = line[:date_match.start()] if date_match else line
    nums = [int(x) for x in re.findall(r"\d{1,3}", prefix)]
    if not nums:
        return None

    # The Seleucid table is sequential.  When the expected SE year is
    # explicitly present in the OCR prefix, prefer it over a corrupted civil
    # year in the second column (e.g. printed ``309 3`` OCRed as ``309 93``).
    if expected_se is not None and expected_se in nums:
        return expected_se

    # Best case: an adjacent printed pair is internally consistent.
    for a, b in zip(nums, nums[1:]):
        if 1 <= a <= 311 and 1 <= b <= 311 and a + b == 312:
            return a
        if 312 <= a <= 386 and 1 <= b <= 75 and a - b == 311:
            return a

    # With a normal two-column head, use the printed civil year to repair a
    # damaged SE number (e.g. SE 10 OCRed as 40 while BCE 302 is intact).
    if len(nums) >= 2:
        first, second = nums[0], nums[1]
        derived = []
        if 1 <= second <= 311:
            derived.append(312 - second)
        if 1 <= second <= 75:
            derived.append(311 + second)
        derived = [x for x in derived if 1 <= x <= 386]
        if expected_se is not None and expected_se in derived:
            return expected_se
        if derived:
            if first in derived:
                return first
            return min(derived, key=lambda x: abs(x - first))

    # Fused civil-year/Nisanu cells leave only the SE number before the first
    # slash (e.g. ``305 74/2`` or ``313 24/3``).
    first = nums[0]
    return first if 1 <= first <= 386 else None


def parse_rows(text: str) -> tuple[list[dict], list[dict]]:
    """Parse all 701 calendar rows from the OCR product."""
    rows: list[dict] = []
    corrections: list[dict] = []
    current_page: int | None = None
    current_heading: str | None = None
    in_seleucid = False
    expected_se_year: int | None = None

    for raw in text.splitlines():
        marker = PAGE_MARKER.match(raw.strip())
        if marker:
            current_page = int(marker.group(1))
            continue

        line = tidy(raw)
        if not line:
            continue

        h = heading(line)
        if h:
            current_heading = h
            if h == "SELEUCID ERA":
                in_seleucid = True
                expected_se_year = 1
            continue

        dates = extract_dates(line)
        if len(dates) < 8:
            continue

        if in_seleucid:
            se_year = parse_se_year(line, expected_se_year)
            if se_year is None:
                continue
            bab_year_astro = se_year - 311
            regnal = str(se_year)
            expected_se_year = se_year + 1
        else:
            match = PRE_HEAD.match(line)
            if not match:
                continue
            bce_year = int(match.group(2))
            if not 311 <= bce_year <= 626:
                continue
            bab_year_astro = 1 - bce_year
            regnal = match.group(1)

        override = ROW_OVERRIDES.get(int(bab_year_astro))
        if override is not None:
            printed = [tuple(x) for x in override["dates"]]
            # Logged only when this OCR run actually misread the row.
            if printed != dates:
                corrections.append({
                    "bab_year_astro": int(bab_year_astro),
                    "year_label": yr.year_label(int(bab_year_astro)),
                    "pdf_page": int(override["page"]),
                    "action": "replace_row",
                    "corrected_month": None,
                    "corrected_day": None,
                    "note": str(override["note"]),
                })
            dates = printed
        else:
            dates, correction = apply_source_fix(bab_year_astro, dates)
            if correction is not None:
                corrections.append(correction)

        rows.append({
            "bab_year_astro": int(bab_year_astro),
            "regnal": regnal,
            "ruler": current_heading,
            "source_page": current_page,
            "dates": dates,
            "source_line": raw.strip(),
        })

    return rows, corrections


def to_jd(year: int, month: int, day: int) -> float:
    """Proleptic Julian date -> JD; year uses astronomical numbering."""
    a = (14 - month) // 12
    yy = year + 4800 - a
    mm = month + 12 * a - 3
    return day + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - 32083 - 0.5


def row_to_calendar(row: dict) -> list[dict]:
    """Expand one source row into one record per Babylonian month start."""
    year = int(row["bab_year_astro"])
    julian_year = year
    previous: tuple[int, int] | None = None
    rolled = False
    out: list[dict] = []

    for index, (month, day) in enumerate(row["dates"], start=1):
        if previous is not None and (month, day) <= previous and not rolled:
            julian_year += 1
            rolled = True

        out.append({
            "bab_year_astro": year,
            "regnal": row["regnal"],
            "ruler": row["ruler"],
            "source_page": row["source_page"],
            "month_index": index,
            "julian_year_astro": int(julian_year),
            "julian_month": int(month),
            "julian_day": int(day),
            "n_months": len(row["dates"]),
        })
        previous = (month, day)

    return out


def cycle_slot(bab_year_astro: int) -> str:
    """The month the standard nineteen-year cycle would double."""
    return "VI2" if bab_year_astro % CYCLE == ULULU_RESIDUE_ASTRO else "XII2"


def load_slots(path: Path) -> dict[int, str]:
    """Printed doubled month of each thirteen-month year, from the ledger."""
    ledger = pd.read_csv(path)
    bad = ledger[~ledger.printed_slot.isin(["VI2", "XII2"])]
    if len(bad):
        raise RuntimeError(f"{path}: unknown slot(s) {bad.to_dict('records')}")
    if ledger.bab_year_astro.duplicated().any():
        raise RuntimeError(f"{path}: a year appears more than once")
    return {int(y): str(v)
            for y, v in zip(ledger.bab_year_astro, ledger.printed_slot)}


def intercalary_slot(bab_year_astro: int, n_months: int,
                     slots: dict[int, str]) -> str | None:
    """The doubled month as printed, or None in a twelve-month year."""
    if n_months != 13:
        return None
    return slots[int(bab_year_astro)]



def validate_rows(rows: list[dict]) -> bool:
    """Hard structural checks on the OCR transcription."""
    ok = True
    by_year = {int(row["bab_year_astro"]): row for row in rows}
    expected = set(int(y) for y in yr.candidate_years())
    present = set(by_year)

    missing = sorted(expected - present)
    extra = sorted(present - expected)
    duplicates = len(rows) - len(present)

    print(f"{len(rows)} rows parsed; {len(present)} unique candidate years")
    if missing:
        print("  FAIL missing years:", [yr.year_label(y) for y in missing])
        ok = False
    if extra:
        print("  FAIL unexpected years:", [yr.year_label(y) for y in extra])
        ok = False
    if duplicates:
        print(f"  FAIL {duplicates} duplicate candidate-year row(s)")
        counts = {}
        for row in rows:
            y = int(row["bab_year_astro"])
            counts[y] = counts.get(y, 0) + 1
        for y, count in sorted(counts.items()):
            if count > 1:
                print(f"    duplicate {yr.year_label(y)}: {count} rows")
                for row in rows:
                    if int(row["bab_year_astro"]) == y:
                        print(f"      p.{row.get('source_page')}: {row.get('source_line')}")
        ok = False

    bad_counts = [
        (year, len(row["dates"]))
        for year, row in sorted(by_year.items())
        if len(row["dates"]) not in (12, 13)
    ]
    if bad_counts:
        print("  FAIL rows not containing 12 or 13 month starts:")
        for year, count in bad_counts:
            print(f"    {yr.year_label(year)}: {count}")
        ok = False

    bad_gaps: list[tuple[int, int, float]] = []
    printed: set[tuple[int, int]] = set()
    previous_end: dict[int, float] = {}
    for year, row in sorted(by_year.items()):
        if len(row["dates"]) not in (12, 13):
            continue
        expanded = row_to_calendar(row)
        jd = [
            to_jd(item["julian_year_astro"], item["julian_month"], item["julian_day"])
            for item in expanded
        ]
        previous_end[year] = jd[-1]
        # Interval ending at each month start, the first one across the
        # year boundary.
        gaps = {i: b - a for i, (a, b) in enumerate(zip(jd, jd[1:]), start=2)}
        if year - 1 in previous_end:
            gaps[1] = jd[0] - previous_end[year - 1]
        for index, gap in sorted(gaps.items()):
            if PRINTED_IRREGULAR_GAPS.get((year, index)) == gap:
                printed.add((year, index))
            elif gap not in (29.0, 30.0):
                bad_gaps.append((year, index, gap))

    if bad_gaps:
        print("  FAIL month-start gaps outside 29--30 days that the table "
              "does not print:")
        for year, index, gap in bad_gaps:
            print(f"    {yr.year_label(year)}, interval ending at month "
                  f"{index}: {gap:.0f} days")
        ok = False
    stale = sorted(set(PRINTED_IRREGULAR_GAPS) - printed)
    if stale:
        print("  FAIL printed 28/31-day interval(s) not found:",
              [(yr.year_label(y), i) for y, i in stale])
        ok = False
    if not bad_gaps and not stale:
        print("  month-start gaps: 29 or 30 days, bar the "
              f"{len(printed)} printed 28/31-day intervals")

    # From 380 BCE onward, the table follows the standard nineteen-year cycle.
    cycle_mismatch = []
    for year, row in sorted(by_year.items()):
        if year < REGULAR_CYCLE_START_ASTRO:
            continue
        expected_intercalary = year % CYCLE in STANDARD_ASTRO_RESIDUES
        actual_intercalary = len(row["dates"]) == 13
        if expected_intercalary != actual_intercalary:
            cycle_mismatch.append(year)
    if cycle_mismatch:
        print("  FAIL standard-cycle mismatch:",
              [yr.year_label(y) for y in cycle_mismatch])
        ok = False

    return ok


def validate_slots(rows: list[dict], slots: dict[int, str]) -> bool:
    """The slot ledger must cover exactly the thirteen-month years."""
    ok = True
    thirteen = {int(r["bab_year_astro"]) for r in rows if len(r["dates"]) == 13}
    missing = sorted(thirteen - set(slots))
    extra = sorted(set(slots) - thirteen)
    if missing:
        print("  FAIL 13-month year(s) missing from the slot ledger:",
              [yr.year_label(y) for y in missing])
        ok = False
    if extra:
        print("  FAIL slot ledger names year(s) without 13 months:",
              [yr.year_label(y) for y in extra])
        ok = False

    disagree = [y for y in sorted(thirteen & set(slots))
                if cycle_slot(y) != slots[y]]
    print(f"  doubled month as printed; the nineteen-year cycle rule "
          f"disagrees in {len(disagree)} of {len(thirteen)} years")
    for y in disagree:
        print(f"    {yr.year_label(y)}: printed {slots[y]}, "
              f"cycle {cycle_slot(y)}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-text", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corrections-out", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--slots", type=Path, default=DEFAULT_SLOTS)
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="validate the OCR parse but do not write physical outputs",
    )
    args = parser.parse_args()

    if not args.ocr_text.exists():
        print(f"OCR source not found: {args.ocr_text}")
        print("Run scripts/physical/ocr_parker_dubberstein.py first.")
        return 1

    if not args.slots.exists():
        print(f"doubled-month ledger not found: {args.slots}")
        print("Run scripts/physical/ocr_parker_dubberstein.py --slots first.")
        return 1

    text = args.ocr_text.read_text(encoding="utf-8", errors="replace")
    rows, corrections = parse_rows(text)
    slots = load_slots(args.slots)

    print(f"OCR source: {args.ocr_text}")
    print(f"visually verified OCR fixes applied: {len(corrections)}")

    ok = validate_rows(rows)
    ok = validate_slots(rows, slots) and ok
    if not ok:
        print("calendar not written")
        return 1

    print("  candidate calendar structure: ok")

    if args.parse_only:
        print("parse-only check complete; no files written")
        return 0

    flat = [item for row in rows for item in row_to_calendar(row)]
    df = pd.DataFrame(flat)
    df["intercalary_month"] = [
        intercalary_slot(int(year), int(count), slots)
        for year, count in zip(df.bab_year_astro, df.n_months)
    ]
    df = df.sort_values(["bab_year_astro", "month_index"]).reset_index(drop=True)

    # Final diagnostics before writing.
    counts = df.groupby("bab_year_astro").n_months.first()
    span = df.groupby("bab_year_astro").julian_year_astro.nunique()
    n_years = int(df.bab_year_astro.nunique())
    n_intercalary = int(counts.eq(13).sum())

    print(f"  candidate years covered : {n_years} "
          f"({yr.year_label(int(df.bab_year_astro.min()))} .. "
          f"{yr.year_label(int(df.bab_year_astro.max()))})")
    print(f"  years of 12 or 13 months: {int(counts.isin([12, 13]).sum())}/{n_years}")
    print(f"  intercalary years        : {n_intercalary}/{n_years} "
          f"({n_intercalary / n_years:.3f})")
    print(f"  years spanning 2 Julian : {int((span == 2).sum())}, "
          f"spanning 1: {int((span == 1).sum())}")

    last = df[df.bab_year_astro == yr.LAST_CANDIDATE_ASTRO]
    if last.empty or int(last.julian_year_astro.max()) != 76:
        print("  FAIL AD 75 Babylonian year does not extend into Julian AD 76")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.corrections_out.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(args.out, index=False)
    pd.DataFrame(corrections).to_csv(args.corrections_out, index=False)

    print(f"written to {args.out}")
    print(f"OCR correction ledger -> {args.corrections_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
