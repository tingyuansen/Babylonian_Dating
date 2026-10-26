#!/usr/bin/env python3
"""Build the table of the Normal Stars from the introduction to ADART I.

The Normal Stars are about thirty reference stars near the ecliptic. The
diaries give positions of the Moon and the planets as distances from them.
These clauses are the most common type in the corpus (SI Appendix, Table S3).
The rules of step 2 read the star positions from this table (SI Appendix,
section S4 and Table S2).

Input and output:

    data/source/adart_i/normal_stars.txt    the 32 rows of the table, as
                                            ``pdftotext -layout`` reads them
                                            from the scan of the introduction
    data/physical/normal_stars.parquet      one row per star, with its
                                            longitude and latitude in the
                                            J2000 ecliptic

The scan is not distributed. pdftotext reads the numbers correctly, but the
PDF font maps the Greek letters onto punctuation. So "eta Piscium" reads as
``!"Piscium``. GLYPH gives the letter of each mark.

Stars of certain identity fix the first marks. Regulus, "The King" (LUGAL),
lies almost on the ecliptic, and the table gives it a latitude of 0.35
degrees. The Jaw of the Bull is Aldebaran, the rear Twin is Pollux and the
front Twin is Castor. These stars fix alpha and beta, and the other marks then
give one consistent set of stars in all eleven constellations of the table.

The table gives the ecliptic longitude and latitude of each star at the years
-600, -300 and 0, in the equinox of each date. The script converts each epoch
to the fixed J2000 ecliptic with the inverse of the precession in
build_ephemeris.py. The stars and the planets then share one frame. This gives
two checks on the transcription.

  1. The three converted longitudes must agree. In the frame of date, the
     longitude grows by about 4.15 degrees in 300 years from precession.
  2. The three converted latitudes must agree. In the frame of date, the
     latitude stays almost constant, because precession is almost a rotation
     about the ecliptic pole.

These checks find misread digits. The input gives alpha Virginis a longitude
of 67.77 for 167.77, and beta Tauri a latitude of 6.19 for about 5.2. The
script finds and corrects both.

In the J2000 frame the position of a star is fixed, so the script stores the
mean of the three epochs. Their spread comes from errors of transcription and
from the proper motion of the star over 600 years, which the table includes.
The script writes nothing if the three longitudes of a star spread more than
0.2 degrees from their mean. It also writes nothing if the latitude of one of
six stars of certain identity differs from its modern value by 0.5 degrees or
more.

Usage:
    python3 scripts/physical/build_normal_stars.py
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

# The Greek letter for each punctuation mark of the PDF font.
GLYPH = {"!": "eta", "#": "beta", "$": "alpha", "%": "zeta", "&": "mu",
         "'": "gamma", "(": "theta", ")": "delta", "*": "epsilon", "+": "rho"}

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
         "epsilon": "ε", "zeta": "ζ", "eta": "η", "mu": "μ", "rho": "ρ",
         "theta": "θ"}

EPOCHS = [-600, -300, 0]
J2000_JD = 2451545.0

ROW = re.compile(r"^([!#$%&'()*+])\"?\s*([A-Z][a-z]+)\s+(.*)$")
PAIR = re.compile(r"(-?\s?\d+\.\d+)\s*/\s*(-?\s?\d+\.\d+)")

# Six stars of certain identity, with their modern J2000 ecliptic latitudes.
# The script uses them only as a check of the output.
ANCHORS = {("alpha", "Tauri"): -5.47, ("beta", "Geminorum"): 6.68,
           ("alpha", "Geminorum"): 10.09, ("alpha", "Leonis"): 0.47,
           ("alpha", "Virginis"): -2.06, ("alpha", "Scorpii"): -4.57}


def to_date(lon: float, lat: float, year: float) -> tuple[float, float]:
    """Convert a J2000 ecliptic position to the mean ecliptic and equinox of ``year``."""
    jd = J2000_JD + (year - 2000) * 365.25
    lon_d, lat_d = precess_ecliptic_j2000_to_date(lon, lat, jd)
    return float(lon_d), float(lat_d)


def to_j2000(lon: float, lat: float, year: float) -> tuple[float, float]:
    """Convert a position in the mean ecliptic and equinox of ``year`` to the J2000 ecliptic.

    The function inverts the precession of build_ephemeris.py by iteration.
    That precession then takes the result back to the input position.
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

        # In J2000 the three longitudes should agree. A value more than 1
        # degree from the others is a misread digit, and the other two epochs
        # give its value. The reference is the epoch closest to the other two,
        # so one wrong value cannot move it. At most one epoch is corrected.
        # Differences are taken modulo 360 degrees, because beta Arietis runs
        # from 357.88 to 2.02.
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

        # In J2000 the three latitudes should agree. A value more than 0.25
        # degrees from their median is a misread digit.
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

        # The circular mean of the three J2000 longitudes, and the largest
        # distance of one of them from it.
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
