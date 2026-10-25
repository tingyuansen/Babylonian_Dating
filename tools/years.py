#!/usr/bin/env python3
"""Central year conventions for the Babylonian-dating project.

All astronomical computation uses astronomical year numbering:

    626 BCE -> -625
    1 BCE   -> 0
    AD 1    -> 1
    AD 75   -> 75

The candidate calendar is the full Parker--Dubberstein 1956 interval used by
this project, 626 BCE through AD 75 inclusive (701 Babylonian years).

Physical resources are deliberately built a little wider. The early margin
retains the existing 700 BCE coverage used by validation/calibration material;
the late margin reaches AD 76 because a Babylonian year beginning in AD 75 can
extend into the following Julian year.
"""

from __future__ import annotations

import numpy as np

# Candidate Babylonian-year domain.
FIRST_CANDIDATE_ASTRO = -625   # 626 BCE
LAST_CANDIDATE_ASTRO = 75      # AD 75
CANDIDATE_YEAR_COUNT = LAST_CANDIDATE_ASTRO - FIRST_CANDIDATE_ASTRO + 1

# Physical-resource domain (Julian/ephemeris/eclipses).
FIRST_PHYSICAL_ASTRO = -699    # 700 BCE
LAST_PHYSICAL_ASTRO = 76       # AD 76


def candidate_years() -> np.ndarray:
    """Return the canonical 701 candidate Babylonian years."""
    return np.arange(
        FIRST_CANDIDATE_ASTRO,
        LAST_CANDIDATE_ASTRO + 1,
        dtype=int,
    )


def bce_to_astro(year_bc: int) -> int:
    """Convert a positive historical BCE year to astronomical numbering."""
    year_bc = int(year_bc)
    if year_bc < 1:
        raise ValueError("BCE year must be a positive historical year number")
    return 1 - year_bc


def astro_to_bce(year_astro: int) -> int:
    """Convert an astronomical year to a positive BCE year.

    Raises for AD years because there is no meaningful positive-BCE encoding
    for them. Use :func:`year_label` for display across the era boundary.
    """
    year_astro = int(year_astro)
    if year_astro > 0:
        raise ValueError(f"astronomical year {year_astro} is AD, not BCE")
    return 1 - year_astro


def year_label(year_astro: int) -> str:
    """Human-readable historical label for one astronomical year."""
    year_astro = int(year_astro)
    if year_astro <= 0:
        return f"{1 - year_astro} BCE"
    return f"AD {year_astro}"


def babylonian_span_label(year_astro: int) -> str:
    """Label a Babylonian year by the two Julian-era years it overlaps."""
    return f"{year_label(year_astro)}/{year_label(int(year_astro) + 1)}"


def check() -> bool:
    """Basic invariants for the central year convention."""
    ok = True
    ok &= bce_to_astro(626) == -625
    ok &= bce_to_astro(1) == 0
    ok &= astro_to_bce(-625) == 626
    ok &= astro_to_bce(0) == 1
    ys = candidate_years()
    ok &= len(ys) == 701
    ok &= int(ys[0]) == -625 and int(ys[-1]) == 75
    ok &= year_label(-625) == "626 BCE"
    ok &= year_label(0) == "1 BCE"
    ok &= year_label(1) == "AD 1"
    print(f"  year convention  {'ok' if ok else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
