#!/usr/bin/env python3
"""Validate the project's formal Babylonian zodiac without fitting a boundary.

Mean tropical longitudes of date are converted to Babylonian sidereal
longitude with Britton (2010), eq. (5):

    Delta lambda* = 3.20 deg - 1.3828 deg Y,

where Y is measured in Julian centuries from year 0.0.  Formal signs are ideal
30-degree bins in the converted longitude: Aries 0--30 deg through Pisces
330--360 deg.  No parameter is fitted to the project's dated corpus.

This script is deliberately a *validation* against dated planet-in-sign
statements, not a calibration.  It reports how often those statements are
compatible with DE441 positions in the preserved tablet months.  The dated
corpus is therefore still retrospective evidence about the implementation, but
it cannot move or optimize the sign boundaries.

Usage:
    python scripts/physical/validate_zodiac.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, sky  # noqa: E402
from tools import years as yr  # noqa: E402

SIGNS = "|".join(sky.SIGNS)
PLANETS = {
    "Jupiter": "jupiter",
    "Saturn": "saturn",
    "Mars": "mars",
    "Venus": "venus",
    "Mercury": "mercury",
}

STMT = re.compile(rf"\b({'|'.join(PLANETS)})\s+was\s+in\s+({SIGNS})\b")


def statements(tablets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in tablets.iterrows():
        if not isinstance(r.text, str):
            continue
        for planet, sign in STMT.findall(r.text):
            rows.append({
                "tablet": r.tablet,
                "year_astro": yr.bce_to_astro(int(r.bce_first)),
                "planet": PLANETS[planet],
                "sign": sign,
                "months": r.months_recorded,
            })
    return pd.DataFrame(rows).drop_duplicates()


def window(year_astro: int, months) -> np.ndarray | None:
    """Daily Julian-day window covered by the catalogue's preserved months."""
    df = cal._cal()
    row = df[df.bab_year_astro == year_astro]
    if row.empty:
        return None

    idx = []
    if isinstance(months, str):
        doubled = row.intercalary_month.iloc[0]
        doubled = None if not isinstance(doubled, str) else doubled
        idx = [
            i for i in (
                cal._month_index(m, int(row.n_months.iloc[0]), doubled)
                for m in months.split()
            ) if i
        ]
    if not idx:
        return None

    lo = min(cal.month_start_jd(year_astro, i) for i in idx)
    hi = max(cal.month_start_jd(year_astro, i) for i in idx)
    return np.arange(lo, hi + 30.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--minimum-agreement", type=float, default=0.90,
        help="fail below this overall retrospective agreement fraction",
    )
    args = ap.parse_args()

    if not 0.0 <= args.minimum_agreement <= 1.0:
        ap.error("--minimum-agreement must lie between 0 and 1")

    # The conversion requires the tropical longitude-of-date grid.
    sky._grid_date()

    tb = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet")
    tb = tb[tb.bce_first.notna() & tb.text.notna()]
    st = statements(tb)
    print(f"{len(st)} planet-in-sign statements from "
          f"{st.tablet.nunique()} dated tablets")

    rows = []
    for _, s in st.iterrows():
        w = window(int(s.year_astro), s.months)
        if w is None:
            continue
        got = set(
            int(i)
            for i in sky.sign_index(sky.babylonian_lon(s.planet, w))
        )
        want = sky.SIGNS.index(s.sign)
        rows.append({
            "tablet": s.tablet,
            "planet": s.planet,
            "sign": s.sign,
            "agree": want in got,
        })

    d = pd.DataFrame(rows)
    print(f"{len(d)} of them fall in a year and months we can place")
    if d.empty:
        print("FAIL no statements could be evaluated")
        return 1

    agreement = float(d.agree.mean())
    print("\nformal zodiac: ideal 30-degree Babylonian sidereal signs")
    print("Britton (2010) offset: 3.20 deg - 1.3828 deg * Y")
    print(f"overall agreement: {int(d.agree.sum())}/{len(d)} "
          f"({agreement*100:.1f}%)")

    print("\nby planet")
    for planet in ["saturn", "jupiter", "mars", "venus", "mercury"]:
        q = d[d.planet.eq(planet)]
        if q.empty:
            continue
        print(f"  {planet:<8} {int(q.agree.sum()):4d}/{len(q):<4d} "
              f"{q.agree.mean()*100:5.1f}%")

    # Anti-regression: test Britton's relation at three exact project epochs.
    epoch_jd = np.array([
        cal.to_jd(-500, 1, 1),
        cal.to_jd(0, 1, 1),
        cal.to_jd(2000, 1, 1),
    ])
    offset = np.asarray(sky.babylonian_offset_deg(epoch_jd), float)
    expected = np.array([10.114, 3.200, -24.456])
    relation_ok = np.allclose(offset, expected, atol=1e-6)
    print(
        "\nBritton offset check: "
        f"year -500={offset[0]:.3f} deg, "
        f"year 0={offset[1]:.3f} deg, "
        f"year 2000={offset[2]:.3f} deg  "
        f"{'ok' if relation_ok else 'FAIL'}"
    )

    # The tropical conversion input must itself remain a genuine of-date
    # frame, rather than a hidden constant shift of the fixed J2000 grid.
    grid, _ = sky._grid_date()
    sample = np.array([grid[0] + 30.0, grid[-1] - 30.0])
    shift = ((np.asarray(sky.lon_date("sun", sample))
              - np.asarray(sky.lon("sun", sample)) + 180.0) % 360.0 - 180.0)
    change = abs(float(shift[1] - shift[0]))
    print(f"\nof-date/J2000 solar-frame shift changes by {change:.2f} deg "
          "across the ephemeris")

    ok = agreement >= args.minimum_agreement and relation_ok and change > 5.0
    print(f"zodiac validation  {'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
