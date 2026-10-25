#!/usr/bin/env python3
"""Planetary positions and derived events used by the dating system.

The ephemeris deliberately carries two ecliptic frames, and the sky layer
derives a third longitude convention from them:

* :func:`lon` is fixed J2000 ecliptic longitude.  It is used for relative
  geometry against the fixed-J2000 Normal Star catalogue and for other
  sidereal angular relations.
* :func:`lon_date` is mean tropical ecliptic longitude of date.  It is used
  directly for equinoxes and solstices.
* :func:`babylonian_lon` converts tropical longitude of date to the
  Babylonian sidereal zodiac with Britton's revised Huber relation.  Formal
  zodiacal signs are ideal 30-degree bins in this longitude.

No zodiac parameter is fitted to the project's dated corpus.  Normal-Star and
constellation evidence remains separate from the formal sign bins.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import brentq

from . import cal as _cal

HERE = Path(__file__).resolve().parents[1]

BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]

# Formal signs, in their ordinary 30-degree order in Babylonian longitude.
# Normal Stars and constellation/star-region language are separate evidence and
# are not inferred from these sign bins.
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
         "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


@lru_cache(maxsize=1)
def _ephemeris() -> pd.DataFrame:
    return pd.read_parquet(HERE / "data" / "physical" / "ephemeris.parquet")


@lru_cache(maxsize=1)
def _grid():
    """Fixed-J2000 longitude grid, retained for stellar relative geometry."""
    df = _ephemeris()
    jd = df.jd.to_numpy()
    out = {b: np.unwrap(np.radians(df[f"{b}_lon"].to_numpy())) for b in BODIES}
    return jd, out


@lru_cache(maxsize=1)
def _grid_date():
    """Mean tropical longitude-of-date grid."""
    df = _ephemeris()
    required = [f"{b}_lon_date" for b in BODIES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "ephemeris.parquet predates the longitude-of-date migration; "
            "rebuild it with scripts/physical/build_ephemeris.py. Missing: "
            + ", ".join(missing)
        )
    jd = df.jd.to_numpy()
    out = {
        b: np.unwrap(np.radians(df[f"{b}_lon_date"].to_numpy()))
        for b in BODIES
    }
    return jd, out


# The grid is computed at Julian days derived from civil dates, and the kernel
# expects dynamical time, so the two differ by Delta T.  The day-long windows a
# clause is tested in are wider than this, so it is not applied by default; the
# sensitivity audit can still test its effect.
TIME_OFFSET_DAYS = 0.0


def delta_t_days(bc_year: float) -> float:
    """Delta T in days, from the ancient parabola used by the project."""
    centuries = (1.0 - bc_year - 1825.0) / 100.0
    return (-320.0 + 32.5 * centuries * centuries) / 86400.0


def _offset_jd(jd):
    return np.asarray(jd, float) + TIME_OFFSET_DAYS if TIME_OFFSET_DAYS else jd


def lon(body: str, jd) -> np.ndarray | float:
    """Ecliptic longitude in degrees in the fixed J2000 frame."""
    q = _offset_jd(jd)
    v = np.degrees(_spline(body)(q)) % 360.0
    return float(v) if np.isscalar(jd) or np.ndim(jd) == 0 else v


def lon_date(body: str, jd) -> np.ndarray | float:
    """Mean tropical ecliptic longitude of date in degrees."""
    q = _offset_jd(jd)
    v = np.degrees(_spline_date(body)(q)) % 360.0
    return float(v) if np.isscalar(jd) or np.ndim(jd) == 0 else v


# Britton (2010), Archive for History of Exact Sciences 64, 617--663,
# eq. (5), p. 630:
#
#   Delta lambda* = 3.20 deg - 1.3828 deg Y +/- 0.09 deg,
#
# where Y is measured in Julian centuries from year 0.0 and Delta lambda* is
# Babylonian sidereal longitude minus modern tropical longitude.  Britton's
# relation revises Huber's earlier estimate; none of these values is fitted to
# the present corpus.
BABYLONIAN_OFFSET_YEAR0_DEG = 3.20
BABYLONIAN_PRECESSION_DEG_PER_CENTURY = 1.3828
BABYLONIAN_OFFSET_UNCERTAINTY_DEG = 0.09
BABYLONIAN_EPOCH_JD = _cal.to_jd(0, 1, 1)


def babylonian_offset_deg(jd) -> np.ndarray | float:
    """Babylonian-minus-tropical longitude offset from Britton (2010).

    Dates use the project's proleptic Julian calendar and astronomical year
    numbering.  The offset is intentionally exposed so audits can report both
    the tropical input and the converted Babylonian longitude.
    """
    q = np.asarray(_offset_jd(jd), dtype=float)
    centuries = (q - BABYLONIAN_EPOCH_JD) / 36525.0
    value = (
        BABYLONIAN_OFFSET_YEAR0_DEG
        - BABYLONIAN_PRECESSION_DEG_PER_CENTURY * centuries
    )
    return float(value) if np.isscalar(jd) or np.ndim(jd) == 0 else value


def babylonian_lon(body: str, jd) -> np.ndarray | float:
    """Longitude in the Babylonian sidereal zodiac, in degrees.

    Mean tropical longitude of date is converted using Britton's revised
    Huber relation.  Formal signs are then ordinary 30-degree bins beginning
    with Aries at 0 degrees in this Babylonian longitude.
    """
    value = (
        np.asarray(lon_date(body, jd), dtype=float)
        + np.asarray(babylonian_offset_deg(jd), dtype=float)
    ) % 360.0
    return float(value) if np.isscalar(jd) or np.ndim(jd) == 0 else value


@lru_cache(maxsize=len(BODIES))
def _spline(body: str) -> CubicSpline:
    grid, series = _grid()
    if body not in series:
        raise KeyError(f"{body!r} is not in the grid. Have {BODIES}")
    return CubicSpline(grid, series[body])


@lru_cache(maxsize=len(BODIES))
def _spline_date(body: str) -> CubicSpline:
    grid, series = _grid_date()
    if body not in series:
        raise KeyError(f"{body!r} is not in the of-date grid. Have {BODIES}")
    return CubicSpline(grid, series[body])


@lru_cache(maxsize=1)
def _stars() -> dict:
    """Normal Stars, fixed-J2000 ecliptic longitude/latitude keyed by name."""
    df = pd.read_parquet(HERE / "data" / "physical" / "normal_stars.parquet")
    return {r.star: (float(r.lon), float(r.lat)) for _, r in df.iterrows()}


def star(name: str):
    """Fixed-J2000 longitude and latitude of a Normal Star, if catalogued."""
    return _stars().get(name)


def sign_index(babylonian_longitude):
    """Ideal 30-degree sign index from Babylonian sidereal longitude.

    0 = Aries for [0, 30), 1 = Taurus for [30, 60), ..., 11 = Pisces for
    [330, 360).  Conversion to this longitude is performed by
    :func:`babylonian_lon`; this function intentionally has no offset argument.
    """
    return np.floor(
        (np.asarray(babylonian_longitude, float) % 360.0) / 30.0
    ).astype(int)


def sign(body: str, jd) -> str:
    """Formal Babylonian zodiacal sign occupied by a body at ``jd``."""
    i = sign_index(babylonian_lon(body, jd))
    return SIGNS[int(i)]


def _sign_occupancy(body: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete half-open sign intervals in the interpolator's time frame.

    Root brackets include every daily spline knot AND every longitude
    extremum. Thus a brief crossing followed by retrograde re-entry between
    daily samples cannot be missed. The secular zodiac offset is included
    before finding extrema and sign-boundary roots.

    The zodiac constants are passed into the cache key, so a sensitivity run
    that changes them cannot be served intervals computed with the old ones.
    """
    return _sign_occupancy_for(body, BABYLONIAN_OFFSET_YEAR0_DEG,
                               BABYLONIAN_PRECESSION_DEG_PER_CENTURY)


@lru_cache(maxsize=4 * len(BODIES))
def _sign_occupancy_for(body: str, offset_year0_deg: float,
                        precession_deg_per_century: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _spline_date(body)
    coeff = np.degrees(source.c.copy())
    rate = precession_deg_per_century / 36525.0
    coeff[2] -= rate
    coeff[3] += (offset_year0_deg
                 - rate * (source.x[:-1] - BABYLONIAN_EPOCH_JD))
    longitude = PPoly(coeff, source.x, extrapolate=False)
    extrema = longitude.derivative().roots(extrapolate=False)
    extrema = extrema[np.isfinite(extrema)]
    knots = np.unique(np.concatenate((source.x, extrema)))
    values = longitude(knots)
    bins = np.floor(values / 30.0).astype(np.int64)
    crossings = []
    for i in np.flatnonzero(bins[:-1] != bins[1:]):
        a, b = float(knots[i]), float(knots[i + 1])
        lo, hi = sorted((int(bins[i]), int(bins[i + 1])))
        for boundary in range(lo + 1, hi + 1):
            target = boundary * 30.0
            crossings.append(brentq(
                lambda q: float(longitude(q)) - target,
                a, b, xtol=1e-8))
    edges = np.unique(np.concatenate((
        [source.x[0]], np.asarray(crossings), [source.x[-1]])))
    starts, ends = edges[:-1], edges[1:]
    signs = sign_index(longitude((starts + ends) / 2.0))
    return starts, ends, signs


def sign_occupancy_intervals(body: str, sign_name: str,
                             jd0: float, jd1: float) -> list[tuple[float, float]]:
    """Intervals during which a planet occupies a sign inside [jd0, jd1).

    Returned bounds are Julian days in the caller's time frame. Missing
    ephemeris coverage raises ValueError, so it cannot become a contradiction.
    Boundary contacts of zero duration are not overlapping intervals.
    """
    if sign_name not in SIGNS or jd1 <= jd0:
        raise ValueError("invalid sign or time interval")
    starts, ends, signs = _sign_occupancy(body)
    offset = float(TIME_OFFSET_DAYS)
    q0, q1 = jd0 + offset, jd1 + offset
    if q0 < starts[0] or q1 > ends[-1]:
        raise ValueError("time interval is outside the ephemeris")
    left = int(np.searchsorted(ends, q0, side="right"))
    right = int(np.searchsorted(starts, q1, side="left"))
    want = SIGNS.index(sign_name)
    return [(max(jd0, float(starts[i]) - offset),
             min(jd1, float(ends[i]) - offset))
            for i in range(left, right) if signs[i] == want]


def joint_sign_intervals(positions: tuple[tuple[str, str], ...],
                         jd0: float, jd1: float) -> list[tuple[float, float]]:
    """Intersect simultaneous sign occupancy for two or more named planets."""
    if len(positions) < 2 or len({p for p, _ in positions}) != len(positions):
        raise ValueError("a joint position requires distinct named planets")
    overlap = [(float(jd0), float(jd1))]
    for body, sign_name in positions:
        intervals = sign_occupancy_intervals(body, sign_name, jd0, jd1)
        overlap = [(max(a, c), min(b, d))
                   for a, b in overlap for c, d in intervals
                   if min(b, d) - max(a, c) > 1e-8]
        if not overlap:
            break
    return overlap


# The diaries record solstices and equinoxes, and the scribes computed them by
# an arithmetical scheme rather than watching. The scheme runs late against the
# true event, and the offset below is the median measured across the dated
# corpus, a retrospective calibration. It is unrelated to zodiac boundaries.
SCHEME_OFFSET_DAYS = 2.0


def _turning_points() -> dict[str, np.ndarray]:
    """Every solstice and equinox in the grid, computed once per time offset.

    Turning points are defined directly in ecliptic longitude of date, so the
    Sun crosses 0/90/180/270 degrees without any J2000 precession workaround.
    The time offset is passed into the cache key, so a sensitivity run that
    shifts it cannot be served turning points computed without the shift.
    """
    return _turning_points_for(float(TIME_OFFSET_DAYS))


@lru_cache(maxsize=4)
def _turning_points_for(time_offset: float) -> dict[str, np.ndarray]:
    grid, _ = _grid_date()
    days = np.arange(grid[0] + 1.0, grid[-1] - 1.0, 0.25)
    tropical = np.asarray(lon_date("sun", days), float)
    out: dict[str, list[float]] = {"solstice": [], "equinox": []}
    for target, kind in ((90.0, "solstice"), (270.0, "solstice"),
                         (0.0, "equinox"), (180.0, "equinox")):
        offset = (tropical - target + 180.0) % 360.0 - 180.0
        crossing = np.where(np.diff(np.sign(offset)) > 0)[0]
        before, after = offset[crossing], offset[crossing + 1]
        out[kind].extend(days[crossing] + 0.25 * before / (before - after))
    return {kind: np.sort(np.array(values)) for kind, values in out.items()}


SIRIUS_OFFSETS = {
    "rising": (0, 21),
    "acronychal": (6, 11),
    "setting": (10, 27),
}


def summer_solstice(jd0: float, jd1: float) -> float | None:
    """Julian day of the summer solstice between two days, if there is one."""
    for candidate in _turning_points()["solstice"]:
        if not jd0 <= candidate <= jd1:
            continue
        tropical = float(lon_date("sun", candidate)) % 360.0
        if abs((tropical - 90.0 + 180.0) % 360.0 - 180.0) < 1.0:
            return float(candidate)
    return None


def turning_point(jd, kind: str):
    """Julian day of the solstice or equinox nearest each given day."""
    table = _turning_points()[kind]
    days = np.atleast_1d(np.asarray(jd, float))
    right = np.clip(np.searchsorted(table, days), 1, len(table) - 1)
    left = right - 1
    nearer = np.where(np.abs(table[right] - days)
                      < np.abs(days - table[left]), right, left)
    found = table[nearer]
    return float(found[0]) if np.isscalar(jd) or np.ndim(jd) == 0 else found


def stations(body: str, jd0: float = None, jd1: float = None) -> pd.DataFrame:
    """Dates a planet turned retrograde or direct over the requested span.

    Station timing is found in the fixed stellar frame.  ``lon`` is fixed J2000
    for Normal-Star geometry, ``lon_date`` is the tropical longitude used as
    the conversion input, and ``lon_babylonian`` classifies formal signs.
    """
    if body in ("sun", "moon"):
        raise ValueError("the Sun and Moon do not station")
    grid, series = _grid()
    m = np.ones(len(grid), bool)
    if jd0 is not None:
        m &= grid >= jd0 - 2
    if jd1 is not None:
        m &= grid <= jd1 + 2
    g, s = grid[m], series[body][m]
    v = np.diff(s)
    turn = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0] + 1
    station_jd = g[turn]
    return pd.DataFrame({
        "jd": station_jd,
        "kind": np.where(v[turn] < 0, "retrograde", "direct"),
        "lon": np.degrees(s[turn]) % 360.0,
        "lon_date": np.asarray(lon_date(body, station_jd), float),
        "lon_babylonian": np.asarray(
            babylonian_lon(body, station_jd), float
        ),
    })


def sign_entries(body: str, jd0: float, jd1: float) -> pd.DataFrame:
    """Dates a body crossed from one ideal 30-degree sign into the next."""
    grid, _ = _grid_date()
    m = (grid >= jd0) & (grid <= jd1)
    g = grid[m]
    idx = sign_index(babylonian_lon(body, g))
    ch = np.where(np.diff(idx) != 0)[0]
    return pd.DataFrame({
        "jd": g[ch + 1],
        "into": [SIGNS[i] for i in idx[ch + 1]],
        "from": [SIGNS[i] for i in idx[ch]],
    })


def elongation(body: str, jd) -> np.ndarray | float:
    """Angular distance from the Sun in fixed-frame longitude, signed, degrees."""
    d = (np.asarray(lon(body, jd), float) - np.asarray(lon("sun", jd), float)
         + 180.0) % 360.0 - 180.0
    return float(d) if np.ndim(d) == 0 else d


# Babylon, as in scripts/physical/build_eclipses.py (Pedersen 2021, p. 12).
BABYLON_LAT_DEG = 32.0 + 32.0 / 60.0 + 30.0 / 3600.0
BABYLON_LON_DEG = 44.0 + 25.0 / 60.0 + 30.0 / 3600.0
# The Sun sets or rises when its centre is at an apparent altitude of 0
# degrees, the visibility rule of the eclipse tool. Refraction at the horizon
# is 34 arcminutes.
SUN_HORIZON_DEG = -34.0 / 60.0


def babylon_night(civil_jd: float) -> tuple[float, float]:
    """The night that opens the Babylonian day of a civil day, at Babylon.

    ``civil_jd`` is the Julian day at 0h of the civil date, as
    :func:`tools.cal.bab_to_jd` gives it. The night runs from sunset on the
    date before to sunrise on the date. Both times are found in UT, moved to
    dynamical time by the Delta T parabola of :func:`delta_t_days`, and
    returned in the frame that :func:`lon` reads. No rule of the baseline uses
    them, and only the sensitivity audit does.
    """
    return _babylon_night(float(civil_jd), float(TIME_OFFSET_DAYS))


@lru_cache(maxsize=1 << 16)
def _babylon_night(civil_jd: float,
                   time_offset: float) -> tuple[float, float]:
    # time_offset is in the cache key, as for the turning points.
    return (_sun_on_horizon(civil_jd - 1.0, True) - time_offset,
            _sun_on_horizon(civil_jd, False) - time_offset)


def _sun_on_horizon(civil_jd: float, setting: bool) -> float:
    """Dynamical time of sunset or sunrise at Babylon on one local date.

    Declination and the equation of time come from the Sun's longitude of
    date and the mean obliquity (Meeus, Astronomical Algorithms, eqs. 22.2,
    25.2). Three passes settle the time to well under a minute.
    """
    def dynamical(hours: float) -> float:
        ut = civil_jd + hours / 24.0 - BABYLON_LON_DEG / 360.0
        return ut + delta_t_days(1.0 - (2000.0 + (ut - 2451545.0) / 365.25))

    latitude = np.radians(BABYLON_LAT_DEG)
    local_hours = 18.0 if setting else 6.0
    for _ in range(3):
        tt = dynamical(local_hours)
        t = (tt - 2451545.0) / 36525.0
        obliquity = np.radians(
            23.0 + 26.0 / 60.0
            + (21.448 - 46.815 * t - 0.00059 * t**2 + 0.001813 * t**3)
            / 3600.0)
        sun = np.radians(float(lon_date("sun", tt - TIME_OFFSET_DAYS)))
        declination = np.arcsin(np.sin(obliquity) * np.sin(sun))
        ascension = np.degrees(np.arctan2(np.cos(obliquity) * np.sin(sun),
                                          np.cos(sun)))
        mean = 280.46646 + 36000.76983 * t + 0.0003032 * t**2
        equation = (mean - ascension + 180.0) % 360.0 - 180.0
        hour_angle = np.degrees(np.arccos(
            (np.sin(np.radians(SUN_HORIZON_DEG))
             - np.sin(latitude) * np.sin(declination))
            / (np.cos(latitude) * np.cos(declination))))
        local_hours = (12.0 + (hour_angle if setting else -hour_angle) / 15.0
                       - equation / 15.0)
    return float(dynamical(local_hours))


def check() -> bool:
    """Independent and anti-regression checks for the sky layer."""
    ok = True
    grid, fixed = _grid()

    # Both interpolators must reproduce their daily source grids.
    probe = grid[1000:1010]
    err = np.abs(((lon("moon", probe)
                   - np.degrees(fixed["moon"][1000:1010]) % 360.0)
                  + 180.0) % 360.0 - 180.0)
    if err.max() > 1e-6:
        print(f"  FAIL J2000 interpolation disagrees with grid by {err.max():.2e} deg")
        ok = False

    date_grid, date_series = _grid_date()
    date_probe = date_grid[1000:1010]
    date_err = np.abs(((lon_date("moon", date_probe)
                        - np.degrees(date_series["moon"][1000:1010]) % 360.0)
                       + 180.0) % 360.0 - 180.0)
    if date_err.max() > 1e-6:
        print(f"  FAIL of-date interpolation disagrees with grid by {date_err.max():.2e} deg")
        ok = False

    # Lock the formal sign convention after conversion: exact 30-degree bins.
    boundary_test = sign_index(np.array([0.0, 29.999999, 30.0, 359.999999]))
    expected = np.array([0, 0, 1, 11])
    good = np.array_equal(boundary_test, expected)
    ok &= good
    print(f"  ideal 30-degree Babylonian signs  {'ok' if good else 'FAIL'}")

    # Lock Britton's independently published epoch relation.  These dates use
    # the same proleptic-Julian/astronomical-year convention as the project.
    epoch_probe = np.array([
        _cal.to_jd(-500, 1, 1),
        _cal.to_jd(0, 1, 1),
        _cal.to_jd(2000, 1, 1),
    ])
    offset_probe = np.asarray(babylonian_offset_deg(epoch_probe), float)
    expected_offset = np.array([10.114, 3.200, -24.456])
    good = np.allclose(offset_probe, expected_offset, atol=1e-6)
    ok &= good
    print(
        "  Britton zodiac offset -500/0/2000: "
        f"{offset_probe[0]:.3f}/{offset_probe[1]:.3f}/"
        f"{offset_probe[2]:.3f} deg  {'ok' if good else 'FAIL'}"
    )

    # The of-date frame must not be a disguised constant shift of J2000.
    q = np.array([grid[0] + 10.0, grid[-1] - 10.0])
    shift = ((np.asarray(lon_date("sun", q)) - np.asarray(lon("sun", q))
              + 180.0) % 360.0 - 180.0)
    frame_drift = abs(float(shift[1] - shift[0]))
    good = frame_drift > 5.0
    ok &= good
    print(f"  of-date frame drift across ephemeris {frame_drift:.1f} deg, "
          f"expect >5  {'ok' if good else 'FAIL'}")

    # Mercury and Venus never stray far from the Sun.
    s = np.arange(grid[0] + 10, grid[0] + 100000, 7.0)
    for body, limit in (("mercury", 30.0), ("venus", 48.5)):
        e = np.abs(elongation(body, s)).max()
        good = e < limit
        ok &= good
        print(f"  {body:<8} greatest elongation {e:5.1f} deg, "
              f"limit {limit}  {'ok' if good else 'FAIL'}")

    # Historical Julian-date sanity check for the date frame.
    equinox = turning_point(_cal.to_jd(-44, 3, 20), "equinox")
    month, day = _cal.from_jd(equinox)[1:]
    good = (month, day) == (3, 23)
    ok &= good
    print(f"  vernal equinox of 45 BC on {month}/{day} Julian, "
          f"expect 3/23  {'ok' if good else 'FAIL'}")

    for body, syn in (("mars", 780), ("jupiter", 399), ("saturn", 378)):
        st = stations(body, grid[0], grid[0] + 36525)
        expect = 2 * 36525 / syn
        good = abs(len(st) - expect) / expect < 0.1
        ok &= good
        print(f"  {body:<8} {len(st):3d} stations per century, "
              f"expect {expect:.0f}  {'ok' if good else 'FAIL'}")

    print(f"  sky tool  {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
