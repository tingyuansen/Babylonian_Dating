#!/usr/bin/env python3
"""Extract the Normal Star catalogue from the ADART volume I introduction.

The tablets give the Moon's and the planets' positions as distances from about
thirty reference stars near the ecliptic, so without this table the largest
single class of statement in the corpus cannot be tested at all.

The input is data/source/adart_i/normal_stars.txt, the 32 rows of the table
as `pdftotext -layout` reads them from the scan of the introduction. The scan
itself is not distributed. The numeric fields survive `pdftotext`. Greek
letters require reconstruction because the PDF font maps them onto
punctuation, so "eta Piscium" arrives as `!"Piscium`.

Punctuation to Greek is recovered rather than guessed. Six of the entries are
stars whose identity is not in doubt, and they pin the mapping. `LUGAL`, "The
King", is Regulus, and its listed ecliptic latitude of 0.35 degrees matches
Regulus sitting almost exactly on the ecliptic. The Jaw of the Bull is
Aldebaran, the rear Twin star is Pollux, the front Twin is Castor. Those fix
alpha, beta and eta, and every remaining glyph then resolves consistently
across all twelve constellations.

The table gives ecliptic longitude and latitude at -600, -300 and 0, in the
equinox of each date. Each epoch is reduced to the fixed J2000 ecliptic with
the Lieske precession build_ephemeris.py uses for its of-date frame, inverted,
so the stars and the planets share one frame. Two independent checks follow
from that and neither is anything the transcription knows.

  1. The three reduced longitudes must agree. Of date, longitude advances by
     about 4.15 degrees per 300 years, which is precession.
  2. The three reduced latitudes must agree. Of date, latitude is very nearly
     constant, precession being almost a rotation about the ecliptic pole.

Both catch misread digits. The catalogue prints one longitude as 67.77 where it
means 167.77, and one latitude as 6.19 where it means 5.19, and the checks find
both without being told.

Output is in the fixed J2000 ecliptic frame, to match the ephemeris. A star's
position there is constant, so the three epochs collapse to one number and their
disagreement is a measure of the transcription's quality, and of the stars'
proper motion over 600 years, which the table includes.

Usage:
    python3 build_normal_stars.py
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

from scripts.physical.build_ephemeris import (  # noqa: E402
    precess_ecliptic_j2000_to_date,
)

# The font's punctuation for each Greek letter, fixed by the six named stars.
GLYPH = {"!": "eta", "#": "beta", "$": "alpha", "%": "zeta", "&": "mu",
         "'": "gamma", "(": "theta", ")": "delta", "*": "epsilon", "+": "rho"}

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
         "epsilon": "ε", "zeta": "ζ", "eta": "η", "mu": "μ", "rho": "ρ",
         "theta": "θ"}

EPOCHS = [-600, -300, 0]
J2000_JD = 2451545.0

ROW = re.compile(r"^([!#$%&'()*+])\"?\s*([A-Z][a-z]+)\s+(.*)$")
PAIR = re.compile(r"(-?\s?\d+\.\d+)\s*/\s*(-?\s?\d+\.\d+)")

# Stars whose identity is beyond doubt, with their modern J2000 ecliptic
# latitude. Used only to verify, never to build.
ANCHORS = {("alpha", "Tauri"): -5.47, ("beta", "Geminorum"): 6.68,
           ("alpha", "Geminorum"): 10.09, ("alpha", "Leonis"): 0.47,
           ("alpha", "Virginis"): -2.06, ("alpha", "Scorpii"): -4.57}


def to_date(lon: float, lat: float, year: float) -> tuple[float, float]:
    """Fixed J2000 ecliptic -> mean ecliptic and equinox of ``year``."""
    jd = J2000_JD + (year - 2000) * 365.25
    lon_d, lat_d = precess_ecliptic_j2000_to_date(lon, lat, jd)
    return float(lon_d), float(lat_d)


def to_j2000(lon: float, lat: float, year: float) -> tuple[float, float]:
    """Mean ecliptic and equinox of ``year`` -> fixed J2000 ecliptic.

    Inverts the ephemeris's precession by iteration rather than with a second
    formula, so a star reduced here and precessed back by build_ephemeris.py
    lands where it started.
    """
    lon0, lat0 = lon, lat
    for _ in range(20):
        lon_d, lat_d = to_date(lon0, lat0, year)
        dlon = (lon - lon_d + 180.0) % 360.0 - 180.0
        dlat = lat - lat_d
        lon0, lat0 = (lon0 + dlon) % 360.0, lat0 + dlat
        if abs(dlon) < 1e-10 and abs(dlat) < 1e-10:
            break
    return lon0, lat0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path,
                    default=HERE / "data" / "source" / "adart_i" / "normal_stars.txt")
    ap.add_argument("--out", type=Path,
                    default=HERE / "data" / "physical" / "normal_stars.parquet")
    args = ap.parse_args()

    text = args.text.read_text(encoding="utf-8")

    rows = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        m = ROW.match(line)
        if not m:
            continue
        pairs = PAIR.findall(m.group(3))
        if len(pairs) != 3:
            continue
        lon = [float(a.replace(" ", "")) for a, _ in pairs]
        lat = [float(b.replace(" ", "")) for _, b in pairs]
        rows.append({"greek": GLYPH[m.group(1)], "constellation": m.group(2),
                     "lon": lon, "lat": lat})
    print(f"{len(rows)} Normal Stars read")

    fixed = []
    for r in rows:
        lon, lat = list(r["lon"]), list(r["lat"])

        # Longitude must advance by precession alone. A step far from that is a
        # misread leading digit, and the other two epochs say what it should be.
        #
        # Both comparisons are made against the *median* of the untouched
        # values and correct at most one epoch. Comparing each value against
        # the mean of the other two flags all three when one is wrong, and then
        # corrects using values already corrupted by the same outlier.
        # Longitudes also run through 360, beta Arietis sitting at 357.9 and
        # 2.0, so the arithmetic has to be circular or the wrap looks like an
        # error and gets "fixed" by 180 degrees.
        pos = [to_j2000(L, B, e) for L, B, e in zip(lon, lat, EPOCHS)]
        ref = [x for x, _ in pos]
        k = int(np.argsort([
            sum(abs((x - y + 180) % 360 - 180) for y in ref) for x in ref])[0])
        off = [abs((x - ref[k] + 180) % 360 - 180) for x in ref]
        if max(off) > 1.0:
            i = int(np.argmax(off))
            pred = to_date(*pos[k], EPOCHS[i])[0]
            print(f"  {r['greek']:<8}{r['constellation']:<12} longitude "
                  f"{lon[i]} at {EPOCHS[i]} corrected to {pred:.2f}")
            lon[i] = pred
            fixed.append("lon")
            pos = [to_j2000(L, B, e) for L, B, e in zip(lon, lat, EPOCHS)]

        # Reduced latitude is nearly constant, so an outlier is a misread digit.
        med = float(np.median([b for _, b in pos]))
        dev = [abs(b - med) for _, b in pos]
        if max(dev) > 0.25:
            i = int(np.argmax(dev))
            pred = to_date(pos[i][0], med, EPOCHS[i])[1]
            print(f"  {r['greek']:<8}{r['constellation']:<12} latitude "
                  f"{lat[i]} at {EPOCHS[i]} corrected to {pred:.2f}")
            lat[i] = pred
            fixed.append("lat")
            pos = [to_j2000(L, B, e) for L, B, e in zip(lon, lat, EPOCHS)]

        # Refer to the fixed J2000 ecliptic, where the position is constant.
        j2000 = [x for x, _ in pos]
        c = np.mean(np.exp(1j * np.radians(j2000)))
        r["lon_j2000"] = float(np.degrees(np.angle(c)) % 360.0)
        r["spread"] = float(max(abs((x - r["lon_j2000"] + 180) % 360 - 180)
                                for x in j2000))
        r["lat_j2000"] = float(np.mean([b for _, b in pos]))

    df = pd.DataFrame([{
        "star": f"{GREEK[r['greek']]} {r['constellation']}",
        "greek": r["greek"], "constellation": r["constellation"],
        "lon": r["lon_j2000"], "lat": r["lat_j2000"], "spread": r["spread"],
    } for r in rows]).sort_values("lon").reset_index(drop=True)

    ok = True
    print(f"\n{len(fixed)} values corrected against precession")
    print(f"epoch-to-epoch spread after conversion to J2000: "
          f"median {df.spread.median():.3f} deg, max {df.spread.max():.3f} deg")
    if df.spread.max() > 0.2:
        print("  ! a star's three epochs still disagree, transcription suspect")
        ok = False

    print("\nverification against stars whose identity is not in doubt")
    for (g, c), want in ANCHORS.items():
        row = df[(df.greek == g) & (df.constellation == c)]
        if row.empty:
            print(f"  {g} {c:<12} MISSING")
            ok = False
            continue
        got = float(row.lat.iloc[0])
        good = abs(got - want) < 0.5
        ok &= good
        print(f"  {GREEK[g]} {c:<12} latitude {got:+6.2f}, modern {want:+6.2f}"
              f"  {'ok' if good else 'FAIL'}")

    if not ok:
        print("\nCHECKS FAILED, not writing")
        return 1
    df.drop(columns="spread").to_parquet(args.out, index=False)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
