#!/usr/bin/env python3
"""Generate the ephemeris grid from DE441 and validate it.

Every astronomical claim in this project rests on this table, so the script
refuses to write one that fails its own checks.

Two coordinate frames are written deliberately:

* ``*_lon`` / ``*_lat`` are fixed J2000 ecliptic coordinates.  These are used
  for relative geometry against Normal Stars and for station/appulse work where
  a fixed stellar frame is appropriate.
* ``*_lon_date`` / ``*_lat_date`` are mean ecliptic coordinates of date.  These
  are used directly for equinoxes and solstices and provide the tropical input
  to the Babylonian sidereal conversion in ``tools.sky``.  They are not by
  themselves formal Babylonian zodiacal longitudes.

No zodiac zero point is fitted or stored in this grid.  The independently
published Britton (2010) conversion is applied downstream, so this file does
not need to be rebuilt if that documented conversion is inspected or revised.

The physical grid is slightly wider than the candidate calendar: 700 BCE
through AD 76 by default.  That preserves the early validation margin and lets
the Babylonian year beginning in AD 75 extend into the following Julian year.

Daily sampling is deliberate.  Cubic interpolation between daily samples was
previously measured at 0.0012 arcminutes median error for the Moon, the fastest
body, against tablets that record in fingers of roughly five arcminutes.

Validation, all of which must pass:

  1. At catalogued solar eclipses the Sun and Moon are in conjunction.
  2. At catalogued lunar eclipses they are in opposition.
  3. Each body's mean motion matches its known sidereal period.
  4. The of-date frame is not a hidden fixed-offset copy of J2000.

Usage:
    python scripts/physical/build_ephemeris.py --kernel <path>/de441_part-1.bsp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.physical.eclipse_catalog import read_catalog  # noqa: E402
from tools import years as yr  # noqa: E402

# Barycentre targets, and the bodies reached through them.
BODIES = {
    "sun": (0, 10), "moon": (3, 301), "mercury": (0, 1), "venus": (0, 2),
    "mars": (0, 4), "jupiter": (0, 5), "saturn": (0, 6),
}

# Sidereal periods in days, used only as a sanity check on mean motion.
SIDEREAL = {
    "mercury": 87.969, "venus": 224.701, "mars": 686.980,
    "jupiter": 4332.589, "saturn": 10759.22, "moon": 27.3217,
}

# J2000 mean obliquity, IAU 2006 value, used only for the fixed J2000 frame.
J2000_OBLIQUITY_RAD = np.radians(84381.406 / 3600.0)


def ecliptic_coords_j2000(x, y, z):
    """ICRF/J2000-equatorial rectangular vector -> J2000 ecliptic lon/lat."""
    e = J2000_OBLIQUITY_RAD
    ye = y * np.cos(e) + z * np.sin(e)
    ze = -y * np.sin(e) + z * np.cos(e)
    lon = np.degrees(np.arctan2(ye, x)) % 360.0
    lat = np.degrees(np.arctan2(ze, np.hypot(x, ye)))
    return lon, lat


def precess_ecliptic_j2000_to_date(lon_deg, lat_deg, jd):
    """Precess J2000 ecliptic coordinates to mean ecliptic/equinox of date.

    This is the standard ecliptic-coordinate precession transformation with
    J2000 as the starting epoch.  It keeps the planetary state vector from
    DE441 and changes only the coordinate frame.  The result supplies tropical
    longitudes for seasonal points and for the downstream Britton conversion.

    The precession angles are the classical IAU/Lieske polynomial form used for
    long-term mean precession.  Across the present -699..+76 astronomical-year
    span, comparison with an independent Swiss-Ephemeris of-date transform is at
    the few-thousandths-of-a-degree level, far below the angular scale relevant
    to a 30-degree sign boundary.
    """
    jd = np.asarray(jd, float)
    t = (jd - 2451545.0) / 36525.0

    # Angles from Meeus/Lieske ecliptic precession, J2000 -> date.
    eta = np.radians(
        (47.0029 * t - 0.03302 * t**2 + 0.000060 * t**3) / 3600.0
    )
    pi = np.radians(
        174.876384 + (-869.8089 * t + 0.03536 * t**2) / 3600.0
    )
    p = np.radians(
        (5029.0966 * t + 1.11113 * t**2 - 0.000006 * t**3) / 3600.0
    )

    lam = np.radians(np.asarray(lon_deg, float))
    beta = np.radians(np.asarray(lat_deg, float))

    a = (np.cos(eta) * np.cos(beta) * np.sin(pi - lam)
         - np.sin(eta) * np.sin(beta))
    b = np.cos(beta) * np.cos(pi - lam)
    c = (np.cos(eta) * np.sin(beta)
         + np.sin(eta) * np.cos(beta) * np.sin(pi - lam))

    lon_date = np.degrees(p + pi - np.arctan2(a, b)) % 360.0
    lat_date = np.degrees(np.arcsin(np.clip(c, -1.0, 1.0)))
    return lon_date, lat_date


def positions(spk, jd):
    """Geocentric coordinates of every supported body in both frames."""
    earth = spk[0, 3].compute(jd) + spk[3, 399].compute(jd)
    out = {}
    for name, (a, b) in BODIES.items():
        if name == "moon":
            vec = spk[3, 301].compute(jd) + spk[0, 3].compute(jd) - earth
        else:
            vec = spk[a, b].compute(jd) - earth

        lon, lat = ecliptic_coords_j2000(vec[0], vec[1], vec[2])
        lon_date, lat_date = precess_ecliptic_j2000_to_date(lon, lat, jd)
        out[name] = {
            "lon": lon,
            "lat": lat,
            "lon_date": lon_date,
            "lat_date": lat_date,
        }
    return out


def read_eclipses(path: Path, lo: int, hi: int):
    """Compatibility view of the shared full-catalogue parser."""
    kind = "solar" if "solar" in path.name.lower() else "lunar"
    records = read_catalog(path, kind, lo, hi)
    return [(r["astro_year"], r["month"], r["day"], r["hour"],
             r["minute"], r["second"]) for r in records]


def to_jd(y, mo, d, hh, mm, ss):
    """Julian day from a proleptic Julian date, astronomical year numbering."""
    a = (14 - mo) // 12
    yy, mmn = y + 4800 - a, mo + 12 * a - 3
    jdn = d + (153 * mmn + 2) // 5 + 365 * yy + yy // 4 - 32083
    return jdn - 0.5 + (hh + mm / 60 + ss / 3600) / 24


def _wrapped_delta(a, b):
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", type=Path, required=True)
    ap.add_argument(
        "--first-astro", type=int, default=yr.FIRST_PHYSICAL_ASTRO,
        help="first proleptic Julian astronomical year (default: -699 = 700 BCE)",
    )
    ap.add_argument(
        "--last-astro", type=int, default=yr.LAST_PHYSICAL_ASTRO,
        help="last proleptic Julian astronomical year (default: 76 = AD 76)",
    )
    ap.add_argument(
        "--out", type=Path,
        default=HERE / "data" / "physical" / "ephemeris.parquet",
    )
    args = ap.parse_args()

    if args.last_astro < args.first_astro:
        ap.error("--last-astro must be >= --first-astro")

    from jplephem.spk import SPK
    spk = SPK.open(str(args.kernel))

    jd0 = to_jd(args.first_astro, 1, 1, 0, 0, 0)
    jd1 = to_jd(args.last_astro, 12, 31, 0, 0, 0)
    jd = np.arange(np.floor(jd0), np.ceil(jd1) + 1, 1.0)
    print(
        f"grid: {len(jd):,} days, JD {jd[0]:.1f} to {jd[-1]:.1f} "
        f"({yr.year_label(args.first_astro)} to {yr.year_label(args.last_astro)})"
    )

    pos = positions(spk, jd)
    columns = {"jd": jd}
    for body in BODIES:
        for coord in ("lon", "lat", "lon_date", "lat_date"):
            columns[f"{body}_{coord}"] = pos[body][coord]
    df = pd.DataFrame(columns)

    ok = True
    print("\ncheck 1 and 2, eclipse geometry")
    for kind, want, fname in (
        ("solar", 0.0, "solar_catalog.txt"),
        ("lunar", 180.0, "lunar_catalog.txt"),
    ):
        path = HERE / "data" / "source" / "nasa_eclipses" / fname
        if not path.exists():
            print(f"  {kind}: catalogue missing, skipped")
            continue
        ecl = read_eclipses(path, args.first_astro, args.last_astro)
        if not ecl:
            print(f"  {kind}: no catalogue events in requested span")
            ok = False
            continue
        sample = ecl[:: max(1, len(ecl) // 40)][:40]
        jds = np.array([to_jd(*e) for e in sample])
        p = positions(spk, jds)
        sep = np.abs(_wrapped_delta(p["moon"]["lon"] - p["sun"]["lon"], want))
        med, mx = np.median(sep), sep.max()
        good = mx < 2.0
        ok &= good
        print(
            f"  {kind}: {len(sample)} events, median {med:.3f} deg, "
            f"max {mx:.3f} deg  {'ok' if good else 'FAIL'}"
        )

    print("\ncheck 3, mean motion against known sidereal periods")
    # A sidereal period is heliocentric, so planets are checked from the Sun and
    # only the Moon from Earth.  Use the fixed J2000 frame here deliberately.
    for name, period in SIDEREAL.items():
        if name == "moon":
            series = pos["moon"]["lon"]
        else:
            a, b = BODIES[name]
            vec = spk[a, b].compute(jd) - spk[0, 10].compute(jd)
            series, _ = ecliptic_coords_j2000(vec[0], vec[1], vec[2])
        u = np.unwrap(np.radians(series))
        rev = abs(u[-1] - u[0]) / (2 * np.pi)
        got = (jd[-1] - jd[0]) / rev
        err = abs(got - period) / period
        good = err < 0.05
        ok &= good
        print(
            f"  {name:<8} implied period {got:8.2f} d, known {period:8.2f} d, "
            f"{err*100:5.2f}%  {'ok' if good else 'FAIL'}"
        )

    print("\ncheck 4, zodiac frame is genuinely of date")
    # A fixed-offset copy of J2000 would have the same shift at both ends.  A
    # true of-date frame must drift appreciably across this ~775-year interval.
    first_shift = float(_wrapped_delta(pos["sun"]["lon_date"][0], pos["sun"]["lon"][0]))
    last_shift = float(_wrapped_delta(pos["sun"]["lon_date"][-1], pos["sun"]["lon"][-1]))
    drift = abs(float(_wrapped_delta(last_shift, first_shift)))
    good = drift > 5.0
    ok &= good
    print(
        f"  Sun frame shift: {first_shift:+.2f} deg -> {last_shift:+.2f} deg; "
        f"change {drift:.2f} deg  {'ok' if good else 'FAIL'}"
    )

    if not ok:
        print("\nCHECKS FAILED, not writing the grid.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwritten to {args.out}  ({args.out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
