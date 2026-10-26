#!/usr/bin/env python3
"""Compare the month starts of the calendar with the computed new moons.

A Babylonian month began on the evening when the new crescent was first seen.
This was about one day after the conjunction of the Sun and the Moon. So each
month start of the calendar should come a short time after a computed
conjunction. An OCR error in a date of Parker and Dubberstein shows up as a
month start too early or too late. This is the check against the new moons in
SI Appendix, section S3.

Reads data/physical/calendar.parquet and data/physical/ephemeris.parquet, and
writes no file. The OCR corrections are in build_calendar.py. The script
prints the lag of the month starts behind the conjunction before them. It
fails if a lag is less than LAG_MIN or more than LAG_MAX days. A month start
listed in CHECKED_AGAINST_PRINT does not fail.

Years use astronomical numbering.

Usage:
    python3 scripts/physical/check_calendar_against_sky.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]

# Limits on the lag of a month start behind the conjunction before it, in
# days.  The least lag in the table is 0.38 days, in 600 BCE month 8, after a
# printed month of 28 days.  The greatest is 3.29 days, in 512 BCE month 9,
# after a printed month of 31 days.
LAG_MIN = 0.3
LAG_MAX = 3.4

# Month starts outside these limits that match the printed table on a check
# of the scan.  The key is (bab_year_astro, month_index), and the value is a
# note of the check.  It is empty, because all month starts lie within the
# limits.
CHECKED_AGAINST_PRINT: dict[tuple[int, int], str] = {}


def label(year: int) -> str:
    """Return the BCE or AD label of an astronomical year."""
    return f"{1 - year} BCE" if year <= 0 else f"AD {year}"


def to_jd(y, mo, d):
    """Return the Julian day of a proleptic Julian date, with an astronomical year."""
    a = (14 - mo) // 12
    yy, mmn = y + 4800 - a, mo + 12 * a - 3
    return d + (153 * mmn + 2) // 5 + 365 * yy + yy // 4 - 32083 - 0.5


def conjunctions(eph: pd.DataFrame) -> np.ndarray:
    """Return the Julian days of the conjunctions of the Sun and the Moon.

    The longitude of the Moon minus that of the Sun changes from negative to
    positive at a conjunction.  The function interpolates linearly between
    the two daily rows of the ephemeris around that change.
    """
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

    # Lag of each month start behind the conjunction before it.
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
