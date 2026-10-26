"""Read the NASA five-millennium catalogues of lunar and solar eclipses.

The catalogues are those of Espenak and Meeus, in data/source/nasa_eclipses/.
The two files have different text formats. read_catalog returns the date and
place of greatest eclipse, and also the Delta T, magnitudes and durations.
build_eclipses.py uses these to compute the visibility at Babylon (SI
Appendix, section S4). build_ephemeris.py uses the dates in its checks.
Catalogue times are in dynamical time (TT or TD).
"""

from __future__ import annotations

import re
from pathlib import Path

MON = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"])}


def coordinate(token: str) -> float:
    """Convert a catalogue coordinate such as ``24S`` or ``143.8W`` to degrees.

    South and west are negative.
    """
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([NSEW])", token)
    if not m:
        raise ValueError(f"not a catalogue coordinate: {token!r}")
    value = float(m.group(1))
    return -value if m.group(2) in {"S", "W"} else value


def number(token: str) -> float | None:
    """Return a numeric catalogue field, or None for ``-``."""
    return None if token == "-" else float(token)


def read_catalog(path: Path, kind: str, lo: int, hi: int) -> list[dict]:
    """Return the records of one catalogue from astronomical year ``lo`` to ``hi``.

    ``kind`` is "lunar" or "solar", and selects the format of the file.
    """
    if kind not in {"lunar", "solar"}:
        raise ValueError("kind must be lunar or solar")
    if kind == "lunar":
        row = re.compile(
            r"^\s*(\d+)\s+(-?\d+)\s+(\w{3})\s+(\d+)\s+"
            r"(\d+):(\d+):(\d+)\s+(.*)$")
    else:
        row = re.compile(
            r"^\s*(\d+)\s+(\d+)\s+(-?\d+)\s+(\w{3})\s+(\d+)\s+"
            r"(\d+):(\d+):(\d+)\s+(.*)$")

    records = []
    for line in path.open(encoding="utf-8", errors="ignore"):
        match = row.match(line)
        if not match:
            continue
        groups = match.groups()
        if kind == "lunar":
            cat, year, mon, day, hour, minute, second, tail = groups
            plate = None
        else:
            cat, plate, year, mon, day, hour, minute, second, tail = groups
        year = int(year)
        if not lo <= year <= hi:
            continue
        fields = tail.split()
        # After the time of greatest eclipse, both formats give Delta T,
        # lunation, Saros and eclipse type.  The later fields differ.
        delta_t_seconds = float(fields[0])
        lunation = int(fields[1])
        saros = int(fields[2])
        eclipse_type = fields[3]
        if kind == "lunar":
            gamma = float(fields[5])
            penumbral_magnitude = float(fields[6])
            umbral_magnitude = float(fields[7])
            penumbral_duration_minutes = number(fields[8])
            partial_duration_minutes = number(fields[9])
            total_duration_minutes = number(fields[10])
            latitude, longitude = coordinate(fields[-2]), coordinate(fields[-1])
            global_magnitude = umbral_magnitude
        else:
            gamma = float(fields[5])
            global_magnitude = float(fields[6])
            penumbral_magnitude = None
            umbral_magnitude = None
            penumbral_duration_minutes = None
            partial_duration_minutes = None
            total_duration_minutes = None
            latitude, longitude = coordinate(fields[7]), coordinate(fields[8])
        records.append({
            "catalog_num": int(cat),
            "plate": int(plate) if plate is not None else None,
            "kind": kind,
            "astro_year": year,
            "month": MON[mon],
            "day": int(day),
            "hour": int(hour),
            "minute": int(minute),
            "second": int(second),
            "delta_t_seconds": delta_t_seconds,
            "lunation": lunation,
            "saros": saros,
            "eclipse_type": eclipse_type,
            "gamma": gamma,
            "global_magnitude": global_magnitude,
            "penumbral_magnitude": penumbral_magnitude,
            "umbral_magnitude": umbral_magnitude,
            "penumbral_duration_minutes": penumbral_duration_minutes,
            "partial_duration_minutes": partial_duration_minutes,
            "total_duration_minutes": total_duration_minutes,
            "greatest_lat": latitude,
            "greatest_lon": longitude,
        })
    return records

