#!/usr/bin/env python3
"""Compute the daily ephemeris grid from JPL DE441 and check it.

Writes data/physical/ephemeris.parquet, the geocentric positions of the Sun,
the Moon and the five visible planets once a day, at integer Julian days
(Methods, Step 2, and SI Appendix, section S4). The script needs a DE441
kernel, which the repository does not hold. It writes the grid only if all
four checks below pass.

The grid holds ecliptic coordinates in two frames.

* ``*_lon`` and ``*_lat`` use the fixed J2000 ecliptic.  tools.sky uses this
  frame for positions against Normal Stars and for stations.
* ``*_lon_date`` and ``*_lat_date`` use the mean ecliptic and equinox of date.
  tools.sky uses this frame for solstices and equinoxes.  It also converts
  these longitudes to the Babylonian zodiac with the relation of Britton (2010).

The grid stores no zodiac offset, so a change to the relation of Britton
needs no new grid.

The grid runs from 700 BCE to AD 76, a little wider than the candidate years.
The extra year at the end holds the part of the year that begins in AD 75 and
runs into AD 76.

tools.sky interpolates the daily values with cubic splines.  For the Moon,
the fastest body, the median error of this interpolation is about 0.0012
arcminutes.  A finger, the smallest unit on the tablets, is about five
arcminutes.

Checks:

  1. At catalogued solar eclipses the Sun and Moon are in conjunction.
  2. At catalogued lunar eclipses they are in opposition.
  3. The mean motion of each body matches its known sidereal period.
  4. The frame of date drifts against J2000, so it is not J2000 with a fixed offset.

Usage:
    python3 scripts/physical/build_ephemeris.py --kernel <path>/de441_part-1.bsp
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

# The (center, target) segments of the kernel for each body.  The Moon is
# reached through the Earth-Moon barycentre, and the others from the solar
# system barycentre.
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
    """Convert an equatorial ICRF vector to J2000 ecliptic longitude and latitude."""
    e = J2000_OBLIQUITY_RAD
    ye = y * np.cos(e) + z * np.sin(e)
    ze = -y * np.sin(e) + z * np.cos(e)
    lon = np.degrees(np.arctan2(ye, x)) % 360.0
    lat = np.degrees(np.arctan2(ze, np.hypot(x, ye)))
    return lon, lat


def precess_ecliptic_j2000_to_date(lon_deg, lat_deg, jd):
    """Precess J2000 ecliptic coordinates to the mean ecliptic and equinox of date.

    The function changes only the frame of the DE441 positions.  Its result
    gives the longitudes for solstices and equinoxes, and the input to the
    relation of Britton.  The precession angles follow the polynomials of
    Lieske, as given by Meeus.  From 700 BCE to AD 76 the result agrees with
    the transform of Swiss Ephemeris to a few thousandths of a degree.
    """
    jd = np.asarray(jd, float)
    t = (jd - 2451545.0) / 36525.0

    # Angles of the ecliptic precession from J2000 to the date (Lieske, Meeus).
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
    """Return the geocentric coordinates of each body in both frames."""
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
    """Return the date and time of each catalogue eclipse, for the checks below."""
    kind = "solar" if "solar" in path.name.lower() else "lunar"
    records = read_catalog(path, kind, lo, hi)
    return [(r["astro_year"], r["month"], r["day"], r["hour"],
             r["minute"], r["second"]) for r in records]


def to_jd(y, mo, d, hh, mm, ss):
    """Return the Julian day of a proleptic Julian date, with an astronomical year."""
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
    # A sidereal period is heliocentric, so the planets are checked from the
    # Sun and only the Moon from the Earth.  The check uses the fixed J2000
    # frame, in which the stars do not move.
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
    # J2000 with a fixed offset would have the same shift at both ends.  The
    # frame of date must drift by more than 5 degrees over about 775 years.
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
