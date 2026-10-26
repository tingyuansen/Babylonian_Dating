#!/usr/bin/env python3
"""Measure how far stated distances to Normal Stars lie from computed ones.

A clause such as "the moon was 1 1/2 cubits behind beta Geminorum" agrees with
a year if the computed position lies within a tolerance of the stated one. The
tolerances of 3 degrees for the Moon and 2 degrees for a planet were chosen
from the residuals that this script measures (Methods, SI Appendix, section S4).
The residuals use the published years of ADART I to III, which include the
benchmark, so the sensitivity audit also tests other tolerances (section S7).

Reads data/corpus/tablets.parquet and writes data/physical/appulse_residuals.parquet.
Each parser clause that places the Moon or a planet against a Normal Star, with a
month and a day, enters. The script computes the body's longitude near the
stated day of the published year. It records the smallest gap to the stated
position, and prints percentiles of the gaps for each body.

Most residuals come from errors of observation and from lunar parallax. Parallax
reaches about one degree, and the geocentric grid does not include it. A tail of
large residuals points to clauses with a wrong day or month.

The script compares longitudes only. For "in front of" and "behind", the stated
distance is in longitude. For "above" and "below" it is in latitude, which
tools.sky does not give, so the residual says only how near in longitude the
body was to the star.

Usage:
    python3 scripts/physical/appulse_residuals.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, sky, statements as S   # noqa: E402
from tools import years as yr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=HERE / "data" / "physical" / "appulse_residuals.parquet")
    args = ap.parse_args()

    tb = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet")
    tb = tb[tb.bce_first.notna() & tb.text.notna()]

    rows, skipped = [], {"no day": 0, "no month": 0, "no calendar": 0}
    for _, r in tb.iterrows():
        year_astro = yr.bce_to_astro(int(r.bce_first))
        for st in S.extract(r.text):
            if st.kind != "appulse":
                continue
            if st.day is None:
                skipped["no day"] += 1
                continue
            if st.month is None:
                skipped["no month"] += 1
                continue
            start = cal.month_start_jd(year_astro + st.year_offset, st.month)
            if start is None:
                skipped["no calendar"] += 1
                continue
            # The Moon moves about 13 degrees a day. A step of 0.02 days, about
            # a quarter of a degree for the Moon, keeps the step out of the
            # residuals.
            jds = np.arange(start + st.day - 2.0, start + st.day + 0.5, 0.02)
            want = (sky.star(st.star)[0] + st.offset) % 360.0
            d = np.abs((sky.lon(st.planet, jds) - want + 180.0) % 360.0 - 180.0)
            rows.append({"tablet": r.tablet, "year_astro": year_astro, "body": st.planet,
                         "star": st.star, "latitudinal": st.latitudinal,
                         "gap": float(d.min())})

    d = pd.DataFrame(rows)
    d.to_parquet(args.out, index=False)
    print(f"{len(d)} appulses placed to a day, "
          f"skipped {skipped}")

    for body in ("moon", "venus", "mars", "mercury", "jupiter", "saturn"):
        g = d[d.body == body]
        if len(g) < 5:
            continue
        print(f"\n{body}, {len(g)} statements")
        for lab, sub in (("longitudinal", g[~g.latitudinal]),
                         ("above/below ", g[g.latitudinal])):
            if not len(sub):
                continue
            q = np.percentile(sub.gap, [50, 68, 90])
            print(f"  {lab}  n={len(sub):4d}  median {q[0]:5.2f}  "
                  f"68th {q[1]:5.2f}  90th {q[2]:5.2f}  "
                  f"within 3 deg {100*(sub.gap < 3).mean():.0f}%")

    print("\nThe residual distribution documents the retrospective tolerance "
          "choice. Large-tail cases require context, source, and physical-model "
          "review rather than an automatically wider threshold; freeze the "
          "choice before prospective evaluation.")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
