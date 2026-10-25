#!/usr/bin/env python3
"""Global eclipse events and Delta-T-robust local visibility at Babylon."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import cal, sky

HERE = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _events() -> pd.DataFrame:
    df = pd.read_parquet(HERE / "data" / "physical" / "eclipses.parquet")
    required = {
        "jd_ut", "locality_schema", "local_status", "local_phase",
        "visible_babylon", "agreement_eligible", "delta_t_sigma_seconds",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            "eclipses.parquet predates local eclipse Version 3; rebuild it "
            "with scripts/physical/build_eclipses.py. Missing: "
            + ", ".join(sorted(missing))
        )
    return df


def events(jd0: float, jd1: float, kind: str | None = None,
           visible_babylon: bool | None = None) -> pd.DataFrame:
    """Eclipses in a Julian-day interval.

    Times are compared in Universal Time.  ``visible_babylon`` is the nominal
    Delta T trial for either eclipse kind; use ``local_status`` and
    ``agreement_eligible`` on returned rows for the uncertainty-aware physical
    result.  Solar magnitude is descriptive and is not an admission threshold.
    Evidence-tier admission is separately controlled by
    :func:`tools.statements.is_decisive`.
    """
    df = _events()
    mask = (df.jd_ut >= jd0) & (df.jd_ut <= jd1)
    if kind is not None:
        if kind not in {"lunar", "solar"}:
            raise ValueError("kind must be lunar or solar")
        mask &= df.kind == kind
    if visible_babylon is not None:
        mask &= df.visible_babylon.fillna(False) == visible_babylon
    return df.loc[mask].copy()


def on_babylonian_day(year_astro: int, month: str, day: int,
                      kind: str) -> pd.DataFrame | None:
    """Catalogue events compatible with a stated Babylonian day.

    ``year_astro`` is the astronomical year in which the Babylonian year begins.
    ``month`` may list alternatives as "VI|VI2"; each is looked up. The
    calendar reconstruction and sunset boundary each carry about a day of
    uncertainty in the dated forward corpus, so the lookup allows 1.5 days
    either side. Adjacent eclipses of the same kind are separated by months.

    Returns None when the candidate year has none of the stated months, for
    example a second Addaru in a twelve-month year. That is a gap in the
    calendar, not an absent eclipse, so the caller must not read it as a
    contradiction.
    """
    if kind not in {"lunar", "solar"}:
        raise ValueError("kind must be lunar or solar")
    jds = [cal.bab_to_jd(int(year_astro), m, int(day)) for m in month.split("|")]
    jds = [jd for jd in jds if jd is not None]
    if not jds:
        return None
    return pd.concat([events(jd - 1.5, jd + 1.5, kind) for jd in jds]
                     ).drop_duplicates("catalog_num")


def check() -> bool:
    """Independent geometry and schema checks on the built table."""
    df = _events()
    ok = not df.empty and set(df.kind) == {"lunar", "solar"}
    lunar = df[df.kind == "lunar"]
    solar = df[df.kind == "solar"]
    ok &= df.visible_babylon.notna().all()
    ok &= df.local_status.isin(
        {"robust_visible", "robust_not_visible", "delta_t_sensitive"}).all()
    ok &= df.locality_schema.eq("babylon-local-eclipse-v2").all()
    ok &= (~df.agreement_eligible
           | df.local_status.eq("robust_visible")).all()
    ok &= lunar.loc[lunar.agreement_eligible, "local_phase"].isin(
        {"partial", "total"}).all()
    # There is deliberately no solar-magnitude gate.  Every solar event that
    # is locally visible in all three Delta T trials can return an agreement;
    # the statement scorer still keeps all solar eclipses supporting-only.
    ok &= solar.agreement_eligible.equals(
        solar.local_status.eq("robust_visible").astype("boolean"))
    for part, want in ((solar, 0.0), (lunar, 180.0)):
        probe = part.iloc[::max(1, len(part) // 40)].head(40)
        separation = np.abs(
            (sky.lon("moon", probe.jd.to_numpy())
             - sky.lon("sun", probe.jd.to_numpy()) - want + 180.0) % 360.0
            - 180.0)
        ok &= float(separation.max()) < 2.0
    status = df.groupby(["kind", "local_status"]).size()
    summary = ", ".join(
        f"{kind} {int((df.kind == kind).sum())} "
        f"({int(status.get((kind, 'robust_visible'), 0))} robust local)"
        for kind in ("lunar", "solar")
    )
    print(f"  eclipse tool  {'ok' if ok else 'FAIL'}  {summary}")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
