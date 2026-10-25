#!/usr/bin/env python3
"""Build the eclipse table and local circumstances at Babylon.

The archived NASA five-millennium catalogues supply the global event inventory,
greatest-eclipse time in dynamical time, catalogue Delta T, and global eclipse
properties.  Swiss Ephemeris then calculates the local circumstances at one
fixed Babylon coordinate for the catalogue Delta T and for its plus/minus one
standard-error alternatives.  The output distinguishes robust local
visibility, robust local non-visibility, and Delta-T-sensitive cases.

The site is the approximate centre of Babylon published by Olof Pedersen,
``Babylon: The Great City`` (2021), p. 12: 32 deg 32 min 30 sec N,
44 deg 25 min 30 sec E (WGS84).  Altitude is set to zero because the historical
local horizon is not reconstructed; this is a geometric-horizon calculation.

Only a locally visible umbral phase of a lunar eclipse is eligible for the
inference code's decisive tier.  A solar eclipse is agreement-eligible whenever
its local visibility is robust to all three Delta T trials; its computed
magnitude is retained as a descriptive quantity, not used as an admission
threshold.  The statement scorer keeps every solar eclipse in the supporting
tier until an independent forward gate is available.

Years are stored in astronomical numbering throughout. The default physical
span is 700 BCE through AD 76, slightly wider than the 626 BCE--AD 75 candidate
calendar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import swisseph as swe

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.physical.eclipse_catalog import read_catalog  # noqa: E402
from tools import cal, sky  # noqa: E402
from tools import years as yr  # noqa: E402

BABYLON_LAT = 32.0 + 32.0 / 60.0 + 30.0 / 3600.0
BABYLON_LON = 44.0 + 25.0 / 60.0 + 30.0 / 3600.0
BABYLON_ALT_M = 0.0
BABYLON_GEOPOS = (BABYLON_LON, BABYLON_LAT, BABYLON_ALT_M)

# Espenak and Meeus give this standard-error expression for Delta T over the
# interval that contains the entire project calendar (1000 BCE to AD 1200).
# Three fixed trials keep the classification reproducible and transparent.
DELTA_T_SIGMA_MULTIPLIERS = (-1.0, 0.0, 1.0)
LOCAL_MATCH_TOLERANCE_DAYS = 1.0
LOCALITY_SCHEMA = "babylon-local-eclipse-v2"


def delta_t_sigma_seconds(astro_year: int) -> float:
    """One-sigma Delta T uncertainty used by the NASA eclipse canon."""
    return 0.8 * ((float(astro_year) - 1820.0) / 100.0) ** 2


def _lunar_visible_phase(flags: int) -> str:
    """Highest lunar-eclipse phase that is locally above the horizon."""
    total_visible = bool(
        flags & (swe.ECL_TOTBEG_VISIBLE | swe.ECL_TOTEND_VISIBLE)
        or (flags & swe.ECL_TOTAL and flags & swe.ECL_MAX_VISIBLE)
    )
    if total_visible:
        return "total"
    partial_visible = bool(
        flags & (swe.ECL_PARTBEG_VISIBLE | swe.ECL_PARTEND_VISIBLE)
        or (flags & (swe.ECL_TOTAL | swe.ECL_PARTIAL)
            and flags & swe.ECL_MAX_VISIBLE)
    )
    if partial_visible:
        return "partial"
    return "penumbral"


def _solar_visible_phase(flags: int) -> str:
    if flags & swe.ECL_TOTAL:
        return "total"
    if flags & swe.ECL_ANNULAR:
        return "annular"
    if flags & swe.ECL_ANNULAR_TOTAL:
        return "hybrid"
    return "partial"


def _local_trial(row, delta_t_seconds: float) -> dict:
    """Calculate whether this catalogue event is locally observable."""
    swe.set_delta_t_userdef(float(delta_t_seconds) / 86400.0)
    catalogue_ut = float(row.jd) - float(delta_t_seconds) / 86400.0
    function = (swe.lun_eclipse_when_loc if row.kind == "lunar"
                else swe.sol_eclipse_when_loc)
    flags, times, attributes = function(
        catalogue_ut - 2.0, BABYLON_GEOPOS, swe.FLG_MOSEPH)
    local_max_ut = float(times[0])
    local_max_tt = local_max_ut + float(delta_t_seconds) / 86400.0
    matched = abs(local_max_tt - float(row.jd)) <= LOCAL_MATCH_TOLERANCE_DAYS
    if not matched:
        return {
            "visible": False,
            "phase": None,
            "magnitude": None,
            "apparent_altitude": None,
            "local_max_ut": None,
            "flags": 0,
        }
    phase = (_lunar_visible_phase(flags) if row.kind == "lunar"
             else _solar_visible_phase(flags))
    magnitude = float(attributes[0] if row.kind == "lunar"
                      else attributes[8])
    return {
        "visible": True,
        "phase": phase,
        "magnitude": magnitude,
        "apparent_altitude": float(attributes[6]),
        "local_max_ut": local_max_ut,
        "flags": int(flags),
    }


def local_circumstances(row) -> dict:
    """Summarize three fixed Delta T trials for one catalogue event."""
    sigma = delta_t_sigma_seconds(int(row.astro_year))
    delta_ts = [
        float(row.delta_t_seconds) + multiplier * sigma
        for multiplier in DELTA_T_SIGMA_MULTIPLIERS
    ]
    try:
        trials = [_local_trial(row, value) for value in delta_ts]
    finally:
        swe.set_delta_t_userdef(swe.DELTAT_AUTOMATIC)

    visible = [bool(trial["visible"]) for trial in trials]
    status = ("robust_visible" if all(visible)
              else "robust_not_visible" if not any(visible)
              else "delta_t_sensitive")
    phase_rank = {"penumbral": 0, "partial": 1, "annular": 2,
                  "total": 3, "hybrid": 3}
    phases = [trial["phase"] for trial in trials if trial["phase"] is not None]
    conservative_phase = (min(phases, key=phase_rank.get) if phases else None)
    magnitudes = [float(trial["magnitude"]) for trial in trials
                  if trial["magnitude"] is not None]
    nominal = trials[1]

    if row.kind == "lunar":
        agreement_eligible = (
            status == "robust_visible"
            and all(phase_rank[trial["phase"]] >= phase_rank["partial"]
                    for trial in trials)
        )
        decision_reason = (
            "umbral phase locally visible in all Delta T trials"
            if agreement_eligible
            else "only penumbral or Delta-T-sensitive local visibility"
            if any(visible) else "no local visibility in any Delta T trial")
    else:
        agreement_eligible = status == "robust_visible"
        decision_reason = (
            "local solar eclipse visible in all Delta T trials; magnitude is "
            "reported descriptively without an admission threshold"
            if agreement_eligible else
            "local solar visibility is Delta-T-sensitive"
            if any(visible) else "no local visibility in any Delta T trial")

    return {
        "locality_schema": LOCALITY_SCHEMA,
        "babylon_lat": BABYLON_LAT,
        "babylon_lon": BABYLON_LON,
        "babylon_alt_m": BABYLON_ALT_M,
        "delta_t_sigma_seconds": sigma,
        "jd_ut": float(row.jd) - float(row.delta_t_seconds) / 86400.0,
        "local_status": status,
        "local_phase": conservative_phase,
        "local_magnitude": nominal["magnitude"],
        "local_magnitude_min": min(magnitudes) if magnitudes else None,
        "local_magnitude_max": max(magnitudes) if magnitudes else None,
        "local_apparent_altitude": nominal["apparent_altitude"],
        "local_jd_ut": nominal["local_max_ut"],
        "local_flags": nominal["flags"],
        "visible_dt_minus": visible[0],
        "visible_babylon": visible[1],
        "visible_dt_plus": visible[2],
        "agreement_eligible": agreement_eligible,
        "local_decision_reason": decision_reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-astro", type=int, default=yr.FIRST_PHYSICAL_ASTRO)
    ap.add_argument("--last-astro", type=int, default=yr.LAST_PHYSICAL_ASTRO)
    ap.add_argument("--out", type=Path,
                    default=HERE / "data" / "physical" / "eclipses.parquet")
    args = ap.parse_args()

    if args.last_astro < args.first_astro:
        ap.error("--last-astro must be >= --first-astro")
    rows = []
    for kind, name in (
        ("lunar", "lunar_catalog.txt"),
        ("solar", "solar_catalog.txt"),
    ):
        rows.extend(read_catalog(
            HERE / "data" / "source" / "nasa_eclipses" / name,
            kind,
            args.first_astro,
            args.last_astro,
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("no eclipses parsed in requested interval")
        return 1

    df["jd"] = [
        cal.to_jd(
            int(row.astro_year), int(row.month), int(row.day),
            float(row.hour) + float(row.minute) / 60.0
            + float(row.second) / 3600.0,
        )
        for row in df.itertuples()
    ]

    # Human-readable convenience only; astro_year remains the canonical field.
    df["year_label"] = [yr.year_label(int(value)) for value in df.astro_year]

    print("calculating local circumstances at Babylon", flush=True)
    locality = []
    for index, row in enumerate(df.itertuples(), start=1):
        locality.append(local_circumstances(row))
        if index % 250 == 0 or index == len(df):
            print(f"  {index:,}/{len(df):,} eclipses", flush=True)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(locality)], axis=1)
    df["visible_babylon"] = df.visible_babylon.astype("boolean")
    df["visible_dt_minus"] = df.visible_dt_minus.astype("boolean")
    df["visible_dt_plus"] = df.visible_dt_plus.astype("boolean")
    df["agreement_eligible"] = df.agreement_eligible.astype("boolean")

    lunar = df.kind == "lunar"

    ok = True
    print(
        f"{len(df)} eclipses from {yr.year_label(args.first_astro)} "
        f"to {yr.year_label(args.last_astro)}"
    )
    for kind, want in (("solar", 0.0), ("lunar", 180.0)):
        part = df[df.kind == kind]
        separation = np.abs(
            (sky.lon("moon", part.jd.to_numpy())
             - sky.lon("sun", part.jd.to_numpy()) - want + 180.0) % 360.0
            - 180.0
        )
        med, maximum = float(np.median(separation)), float(separation.max())
        good = maximum < 2.0
        ok &= good
        print(
            f"  {kind:<5} {len(part):4d} events, geometry median {med:.3f}°, "
            f"max {maximum:.3f}°  {'ok' if good else 'FAIL'}"
        )

    statuses = df.groupby(["kind", "local_status"]).size()
    eligible = df.groupby("kind").agreement_eligible.sum()
    for kind in ("lunar", "solar"):
        counts = {status: int(statuses.get((kind, status), 0)) for status in
                  ("robust_visible", "robust_not_visible", "delta_t_sensitive")}
        print(f"  {kind:<5} local statuses {counts}; "
              f"agreement-eligible {int(eligible.get(kind, 0))}")

    good = (df.local_status.isin(
        {"robust_visible", "robust_not_visible", "delta_t_sensitive"}).all()
        and (~df.agreement_eligible
             | df.local_status.eq("robust_visible")).all()
        and df.loc[~lunar, "agreement_eligible"].equals(
            df.loc[~lunar, "local_status"].eq("robust_visible")
            .astype("boolean"))
        and df.loc[lunar & df.agreement_eligible, "local_phase"].isin(
            {"partial", "total"}).all()
        and df.visible_babylon.notna().all())
    ok &= bool(good)
    print(f"  local-classification schema  {'ok' if good else 'FAIL'}")

    if not ok:
        print("checks failed, not writing eclipse table")
        return 1

    columns = [
        "catalog_num", "plate", "kind", "astro_year", "year_label", "month",
        "day", "hour", "minute", "second", "jd", "jd_ut",
        "delta_t_seconds", "delta_t_sigma_seconds", "lunation", "saros",
        "eclipse_type", "gamma", "global_magnitude",
        "penumbral_magnitude", "umbral_magnitude",
        "penumbral_duration_minutes", "partial_duration_minutes",
        "total_duration_minutes", "greatest_lat", "greatest_lon",
        "locality_schema", "babylon_lat", "babylon_lon", "babylon_alt_m",
        "local_status", "local_phase", "local_magnitude",
        "local_magnitude_min", "local_magnitude_max",
        "local_apparent_altitude", "local_jd_ut", "local_flags",
        "visible_dt_minus", "visible_babylon", "visible_dt_plus",
        "agreement_eligible", "local_decision_reason",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[columns].sort_values("jd").to_parquet(args.out, index=False)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
