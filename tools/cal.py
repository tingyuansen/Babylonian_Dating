#!/usr/bin/env python3
"""Babylonian calendar and era arithmetic.

All computational years use astronomical numbering:

    626 BCE -> -625
    1 BCE   -> 0
    AD 1    -> 1

Historical BCE numbers are converted only at data/input boundaries. This avoids
the one-year ambiguity at the BCE/AD transition and lets the same API work over
the full 626 BCE--AD 75 candidate interval.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import years as yr

HERE = Path(__file__).resolve().parents[1]

# Historical epoch constants retained because their difference is useful for
# recognizing dual-era formulae. The conversion itself returns astro years.
SE_EPOCH = 312   # SE 1 began in the Babylonian year beginning 311 BCE
AE_EPOCH = 248   # AE 1 began in the Babylonian year beginning 247 BCE

MONTHS = ["I", "II", "III", "IV", "V", "VI", "VI2", "VII", "VIII", "IX", "X",
          "XI", "XII", "XII2"]


class AmbiguousEra(ValueError):
    """Raised when a year number is given without a resolvable era."""


@lru_cache(maxsize=1)
def _cal() -> pd.DataFrame:
    return pd.read_parquet(HERE / "data" / "physical" / "calendar.parquet")


@lru_cache(maxsize=1)
def _index() -> tuple[dict, dict, dict]:
    """Month starts, month count per year, and doubled month, built once."""
    df = _cal()
    required = {
        "bab_year_astro", "julian_year_astro", "julian_month", "julian_day",
        "month_index", "n_months", "intercalary_month",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            "calendar.parquet lacks columns " + ", ".join(sorted(missing))
            + "; rebuild it with scripts/physical/build_calendar.py")

    y = df.julian_year_astro.to_numpy(dtype=int)
    mo = df.julian_month.to_numpy(dtype=int)
    d = df.julian_day.to_numpy(dtype=int)
    a = (14 - mo) // 12
    yy, mmn = y + 4800 - a, mo + 12 * a - 3
    jd = d + (153 * mmn + 2) // 5 + 365 * yy + yy // 4 - 32083 - 0.5

    starts = {
        (int(b), int(i)): float(j)
        for b, i, j in zip(df.bab_year_astro, df.month_index, jd)
    }
    n = {
        int(b): int(v)
        for b, v in zip(df.bab_year_astro, df.n_months)
    }
    slot = {
        int(b): (None if v is None or (isinstance(v, float) and v != v)
                 else str(v))
        for b, v in zip(df.bab_year_astro, df.intercalary_month)
    }
    return starts, n, slot


def candidate_years() -> np.ndarray:
    """The canonical 701 Babylonian candidate years, as astro-year integers.

    The function also verifies that the built calendar actually contains the
    entire configured interval. This makes an incomplete calendar a hard error
    instead of silently shrinking every downstream candidate set.
    """
    expected = yr.candidate_years()
    _, nmon, _ = _index()
    available = set(nmon)
    missing = [int(y) for y in expected if int(y) not in available]
    if missing:
        labels = ", ".join(yr.year_label(y) for y in missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise RuntimeError(
            f"calendar is missing {len(missing)} candidate year(s): "
            f"{labels}{suffix}"
        )
    return expected.copy()


def to_jd(y: int, mo: int, d: int, hour: float = 0.0) -> float:
    """Julian day from a proleptic Julian date, astronomical year numbering."""
    a = (14 - mo) // 12
    yy, mmn = y + 4800 - a, mo + 12 * a - 3
    jdn = d + (153 * mmn + 2) // 5 + 365 * yy + yy // 4 - 32083
    return jdn - 0.5 + hour / 24.0


def from_jd(jd: float) -> tuple[int, int, int]:
    """Proleptic Julian date from a Julian day, astronomical year numbering."""
    z = int(jd + 0.5)
    b = z + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e)
    mo = e - 1 if e < 14 else e - 13
    y = c - 4716 if mo > 2 else c - 4715
    return y, mo, day


def to_astro(year: int, era: str) -> int:
    """Absolute Babylonian year in astronomical numbering from an era year.

    ``era`` must be ``SE``, ``AE`` or ``regnal``. A regnal year cannot be
    resolved without a king and therefore raises deliberately.
    """
    era = era.upper()
    year = int(year)
    if era == "SE":
        return year - 311
    if era == "AE":
        return year - 247
    if era == "REGNAL":
        raise AmbiguousEra(
            "a regnal year needs a king before it means anything, "
            "and the king is not something this tool knows"
        )
    raise AmbiguousEra(
        f"era {era!r} not recognised. Say SE, AE or regnal explicitly. "
        "A bare year number on a tablet is not evidence of which one it is."
    )


def month_start_jd(year_astro: int, month: str | int) -> float | None:
    """First Julian day of a Babylonian month, or None if not tabulated."""
    year_astro = int(year_astro)
    starts, nmon, slot = _index()
    if year_astro not in nmon:
        return None
    if isinstance(month, str):
        idx = _month_index(month, nmon[year_astro], slot.get(year_astro))
        if idx is None:
            return None
    else:
        idx = int(month)
    return starts.get((year_astro, idx))


def _month_index(name: str, n_months: int,
                 doubled: str | None = None) -> int | None:
    """Position of a named month within a 12- or 13-month Babylonian year."""
    name = name.upper().replace("_", "").strip()
    base = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI",
            "XII"]
    if n_months == 13 and doubled is None:
        doubled = "XII2"
    if name == "VI2":
        return 7 if doubled == "VI2" else None
    if name == "XII2":
        return 13 if doubled == "XII2" else None
    if name not in base:
        return None
    i = base.index(name) + 1
    if doubled == "VI2" and i >= 7:
        return i + 1
    return i


def bab_to_jd(year_astro: int, month: str | int, day: int) -> float | None:
    """Julian day of a Babylonian year, month and day.

    ``year_astro`` is the astronomical year in which the Babylonian year begins.
    The returned Julian day is the civil daytime associated with that Babylonian
    date; the Babylonian day itself begins at the preceding sunset.
    """
    start = month_start_jd(year_astro, month)
    if start is None:
        return None
    return start + (int(day) - 1)


def check() -> bool:
    """Checks that must hold if the era and calendar arithmetic are right."""
    ok = True

    if to_astro(234, "SE") != to_astro(170, "AE"):
        print("  FAIL the two era epochs disagree")
        ok = False

    # SE 162 corresponds to 150/149 BCE; the Babylonian year begins in astro -149.
    if to_astro(162, "SE") != -149:
        print(f"  FAIL SE 162 gives {to_astro(162, 'SE')}, expected -149")
        ok = False

    # Crossing the era boundary must be exact: SE 311 = 1 BCE, SE 312 = AD 1.
    if to_astro(311, "SE") != 0 or to_astro(312, "SE") != 1:
        print("  FAIL SE conversion across BCE/AD boundary")
        ok = False

    try:
        to_astro(12, "regnal")
        print("  FAIL a regnal year was converted without a king")
        ok = False
    except AmbiguousEra:
        pass

    for y, m, d in [(-149, 10, 5), (-651, 4, 1), (0, 12, 31),
                    (1, 1, 1), (-300, 2, 29), (75, 6, 1)]:
        if from_jd(to_jd(y, m, d)) != (y, m, d):
            print(f"  FAIL round trip on {y}-{m}-{d}")
            ok = False

    # The candidate calendar must be complete.
    try:
        ys = candidate_years()
        if len(ys) != yr.CANDIDATE_YEAR_COUNT:
            print(f"  FAIL candidate calendar has {len(ys)} years")
            ok = False
    except Exception as exc:
        print(f"  FAIL candidate calendar: {exc}")
        return False

    # Known intercalary examples translated from historical BCE to astro years.
    y142 = yr.bce_to_astro(142)
    y150 = yr.bce_to_astro(150)
    y145 = yr.bce_to_astro(145)
    starts, nmon, slot = _index()
    if slot.get(y142) != "VI2" or slot.get(y150) is not None:
        print(f"  FAIL the doubled month is not recorded: 142 BCE -> "
              f"{slot.get(y142)}, 150 BCE -> {slot.get(y150)}")
        ok = False
    else:
        if _month_index("VI2", nmon[y142], slot[y142]) != 7:
            print("  FAIL VI2 does not resolve in a second-Ululu year")
            ok = False
        if _month_index("VII", nmon[y142], slot[y142]) != 8:
            print("  FAIL month VII is not displaced in a second-Ululu year")
            ok = False
        if _month_index("VII", nmon[y145], slot[y145]) != 7:
            print("  FAIL month VII moved in a second-Addaru year")
            ok = False
        gap = month_start_jd(y142, "VII") - month_start_jd(y142, "VI2")
        if gap not in (29.0, 30.0):
            print(f"  FAIL VI2 to VII is {gap} days, not one lunation")
            ok = False

    # Validate every built candidate year, not just an arbitrary prefix.
    # Parker & Dubberstein's 1956 table prints four month-start intervals of 28
    # or 31 days, each checked on the scan. Any other interval outside 29--30
    # days, within a year or across a year boundary, signals a bad parse, a
    # missing month start, or a year rollover error. Keyed by Babylonian year
    # and the month the interval ends at; month 1 is the interval from the last
    # month start of the year before. scripts/physical/build_calendar.py keeps
    # the same list.
    printed = {(-599, 8): 28.0, (-529, 1): 31.0, (-511, 9): 31.0, (-313, 5): 31.0}
    df = _cal()
    bad_gaps: list[tuple[int, int, float]] = []
    found: dict[tuple[int, int], float] = {}
    previous_last = None

    for year in sorted(int(y) for y in df.bab_year_astro.unique()):
        group = df[df.bab_year_astro == year]
        n_months = int(group.n_months.iloc[0])

        if n_months not in (12, 13):
            print(
                f"  FAIL {yr.year_label(year)} has {n_months} months, "
                "expected 12 or 13"
            )
            ok = False
            previous_last = None
            continue

        jds = [
            month_start_jd(year, i)
            for i in range(1, n_months + 1)
        ]

        missing_months = [
            i for i, value in enumerate(jds, start=1) if value is None
        ]
        if missing_months:
            print(
                f"  FAIL missing month start(s) in {yr.year_label(year)}: "
                + ", ".join(str(i) for i in missing_months)
            )
            ok = False
            previous_last = None
            continue

        # None values were excluded above.
        numeric_jds = [float(value) for value in jds if value is not None]
        starts = ([previous_last] if previous_last is not None else []) + numeric_jds
        first_index = 1 if previous_last is not None else 2
        for month_index, (a, b) in enumerate(zip(starts, starts[1:]),
                                             start=first_index):
            gap = b - a
            if gap in (29.0, 30.0):
                continue
            if printed.get((year, month_index)) == gap:
                found[(year, month_index)] = gap
            else:
                bad_gaps.append((year, month_index, gap))
        previous_last = numeric_jds[-1]

    missing = sorted(set(printed) - set(found))
    if bad_gaps or missing:
        if bad_gaps:
            print(
                f"  FAIL {len(bad_gaps)} month interval(s) outside 29--30 days "
                "that the printed table does not show"
            )
        for year, month_index, gap in bad_gaps[:20]:
            print(
                f"    {yr.year_label(year)}, interval ending at month "
                f"{month_index}: {gap:.0f} days"
            )
        if len(bad_gaps) > 20:
            print(f"    ... and {len(bad_gaps) - 20} more")
        if missing:
            print("  FAIL printed 28/31-day interval(s) not found: "
                  + ", ".join(f"{yr.year_label(y)} month {m}" for y, m in missing))
        ok = False
    else:
        print(
            "  calendar month-start gaps  ok "
            f"({len(found)} printed 28/31-day intervals)"
        )

    print(f"  era and calendar arithmetic  {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
