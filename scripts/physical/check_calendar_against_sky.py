#!/usr/bin/env python3
"""Check the printed Parker--Dubberstein calendar against the Moon.

Parker and Dubberstein is the authority for the Babylonian calendar and stays
that way. But OCR can misread digits in ways that look plausible in isolation.
A wrong month start is a wrong date for every statement in that month.

A Babylonian month began on the evening the new crescent first became visible,
roughly one day after conjunction of Sun and Moon. Every tabulated month start
should therefore sit a small, consistent distance after a computed conjunction.
This script checks every retained row independently against the ephemeris. It
does not repair the calendar; repairs are performed only by build_calendar.py.
It exits nonzero if any month start lies less than LAG_MIN or more than
LAG_MAX days after the preceding conjunction, unless that row is listed in
CHECKED_AGAINST_PRINT as read off the printed table.

The calendar uses astronomical year numbering throughout.

Usage:
    python scripts/physical/check_calendar_against_sky.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]

# Accepted lag of a month start behind the preceding conjunction, in days.
# The tabulated starts run from 0.38 days (600 BCE month 8, after a printed
# 28-day month) to 3.29 days (512 BCE month 9, after a printed 31-day month).
LAG_MIN = 0.3
LAG_MAX = 3.4

# Month starts outside those limits that were checked against the printed
# table and found to be what Parker and Dubberstein print, keyed by
# (bab_year_astro, month_index) with a note of the check.  None at present.
CHECKED_AGAINST_PRINT: dict[tuple[int, int], str] = {}


def label(year: int) -> str:
    """Historical label for an astronomical year."""
    return f"{1 - year} BCE" if year <= 0 else f"AD {year}"


def to_jd(y, mo, d):
    """Proleptic Julian date to JD; y is an astronomical year."""
    a = (14 - mo) // 12
    yy, mmn = y + 4800 - a, mo + 12 * a - 3
    return d + (153 * mmn + 2) // 5 + 365 * yy + yy // 4 - 32083 - 0.5


def from_jd(jd):
    """JD to proleptic Julian date; returned year is astronomical."""
    z = int(jd + 0.5)
    b = z + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)
    day = b - d - int(30.6001 * e)
    mo = e - 1 if e < 14 else e - 13
    y = c - 4716 if mo > 2 else c - 4715
    return y, mo, day


def conjunctions(eph: pd.DataFrame) -> np.ndarray:
    """Julian days of Sun-Moon conjunction by zero-crossing interpolation."""
    d = (eph.moon_lon.to_numpy() - eph.sun_lon.to_numpy() + 180.0) % 360.0 - 180.0
    jd = eph.jd.to_numpy()
    i = np.where((d[:-1] <= 0) & (d[1:] > 0))[0]
    frac = -d[i] / (d[i + 1] - d[i])
    return jd[i] + frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", type=Path,
                    default=HERE / "data" / "physical" / "calendar.parquet")
    ap.add_argument("--eph", type=Path,
                    default=HERE / "data" / "physical" / "ephemeris.parquet")
    args = ap.parse_args()

    cal = pd.read_parquet(args.cal).sort_values(
        ["bab_year_astro", "month_index"])
    eph = pd.read_parquet(args.eph)
    conj = conjunctions(eph)
    print(f"{len(conj):,} conjunctions computed over the ephemeris span")

    jd = np.array([
        to_jd(int(y), int(m), int(d))
        for y, m, d in zip(
            cal.julian_year_astro,
            cal.julian_month,
            cal.julian_day,
        )
    ], dtype=float)
    cal = cal.assign(jd=jd)

    # Distance from each tabulated month start back to the preceding conjunction.
    k = np.searchsorted(conj, jd) - 1
    inrange = (k >= 0) & (k < len(conj))
    lag = np.full(len(jd), np.nan)
    lag[inrange] = jd[inrange] - conj[k[inrange]]

    good = lag[np.isfinite(lag)]
    if not len(good):
        print("no calendar rows fall within the ephemeris conjunction span")
        return 1

    print("\nmonth start minus preceding conjunction, in days")
    print(f"  median {np.median(good):.2f}, "
          f"16th to 84th percentile {np.percentile(good, 16):.2f} to "
          f"{np.percentile(good, 84):.2f}")
    cal = cal.assign(lag=lag)
    for name, row in (("least", cal.loc[cal.lag.idxmin()]),
                      ("greatest", cal.loc[cal.lag.idxmax()])):
        print(f"  {name} {row.lag:.2f}: {label(int(row.bab_year_astro))} "
              f"month {int(row.month_index)}")

    ok = True
    unchecked = cal[~np.isfinite(lag)]
    if len(unchecked):
        print(f"  FAIL {len(unchecked)} rows outside the ephemeris span")
        ok = False
    off = cal[(cal.lag < LAG_MIN) | (cal.lag > LAG_MAX)]
    listed = [(int(y), int(i)) in CHECKED_AGAINST_PRINT
              for y, i in zip(off.bab_year_astro, off.month_index)]
    print(f"  {len(off)} of {len(good)} rows outside {LAG_MIN}--{LAG_MAX} days, "
          f"{sum(listed)} of them checked against the print")
    for (_, row), seen in zip(off.iterrows(), listed):
        if not seen:
            print(f"  FAIL {label(int(row.bab_year_astro))} month "
                  f"{int(row.month_index)} starts {row.lag:.2f} days after "
                  "conjunction")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
