#!/usr/bin/env python3
"""Calibrate elongation-only first/last appearance thresholds, then hold out.

This script is deliberately an admission gate, not a parameter factory. It fits
only tablets in ``testable_checked`` (whose era statements independently pin
their dates), then reports the untouched ``testable`` cohort. A threshold is
not promoted into ``tools/statements.py`` unless that holdout is reliable.

The ephemeris stores longitude but not local altitude. The model here therefore
tests the narrow claim available to it: the signed Sun-planet elongation at an
appearance date. Three residual quantiles expose sensitivity instead of hiding
one favourable tolerance.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import cal, sky, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402

BODY = "|".join(S.PLANETS)
APPEARANCE = re.compile(
    rf"\b({BODY})[’'s]*\s+(first|last)\s+appearance\b([^.;\n]{{0,100}})",
    re.IGNORECASE)


def observations() -> pd.DataFrame:
    tablets = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet")
    cohorts = pd.read_parquet(HERE / "data" / "corpus" / "cohorts.parquet").set_index(
        "tablet")
    rows = []
    dated = tablets[tablets.bce_first.notna() & tablets.text.notna()]
    for tablet in dated.itertuples():
        for match in APPEARANCE.finditer(tablet.text):
            body = S.PLANETS[match.group(1).title()]
            which, tail = match.group(2).lower(), match.group(3)
            east = re.search(r"\b(?:in\s+the\s+)?east\b", tail, re.I)
            west = re.search(r"\b(?:in\s+the\s+)?west\b", tail, re.I)
            direction = "east" if east else ("west" if west else None)
            clean_tail = re.sub(r"[\[\]]", " ", tail)
            omitted = (bool(S.OMITTED.search(clean_tail))
                       or bool(re.search(r"\b(?:expected|ideal)\b", tail, re.I)))
            month, day, offset, restored = S.eclipse_context(
                tablet.text, match.start())
            orientation = 0
            if direction == "east":
                orientation = -1
            elif direction == "west":
                orientation = 1
            elif body in {"mars", "jupiter", "saturn"}:
                orientation = -1 if which == "first" else 1
            if (omitted or restored or month is None or day is None
                    or orientation == 0):
                continue
            jd = cal.bab_to_jd(
                yr.bce_to_astro(int(tablet.bce_first)) + offset, month, day)
            if jd is None:
                continue
            rows.append({
                "tablet": tablet.tablet,
                "cohort": cohorts.loc[tablet.tablet].cohort,
                "body": body,
                "which": which,
                "orientation": orientation,
                "jd": jd,
                "oriented_elongation": orientation * sky.elongation(body, jd),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantiles", default="0.75,0.9,1.0")
    ap.add_argument("--min-train", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=HERE / "data" / "physical" / "visibility_calibration.parquet")
    args = ap.parse_args()
    quantiles = [float(q) for q in args.quantiles.split(",")]
    obs = observations()
    keys = ["body", "which", "orientation"]
    train = obs[(obs.cohort == "testable_checked")
                & (obs.oriented_elongation > 0)]

    parameters = {}
    for key, group in train.groupby(keys):
        if len(group) < args.min_train:
            continue
        values = group.oriented_elongation.to_numpy(float)
        threshold = float(np.median(values))
        residual = np.abs(values - threshold)
        parameters[key] = (threshold, residual, len(values))

    details = []
    print(f"{len(obs)} secure, oriented appearance statements; "
          f"{len(parameters)} categories have >= {args.min_train} training rows")
    for quantile in quantiles:
        outcome = collections.defaultdict(list)
        for row in obs.itertuples():
            key = (row.body, row.which, row.orientation)
            if key not in parameters:
                continue
            threshold, residual, n_train = parameters[key]
            tolerance = max(2.0, float(np.quantile(residual, quantile)))
            probe = np.arange(row.jd - 1.5, row.jd + 1.501, 0.1)
            oriented = row.orientation * sky.elongation(row.body, probe)
            gap = float(np.min(np.abs(oriented - threshold)))
            outcome[row.cohort].append(gap <= tolerance)
            details.append({
                "quantile": quantile, "body": row.body, "which": row.which,
                "orientation": row.orientation, "threshold": threshold,
                "tolerance": tolerance, "category_train_n": n_train,
                "tablet": row.tablet, "cohort": row.cohort, "gap": gap,
                "agree": gap <= tolerance,
            })
        print(f"\nresidual quantile {quantile:.2f}")
        for cohort in ("testable_checked", "testable"):
            values = outcome[cohort]
            print(f"  {cohort:<18} {sum(values):3d}/{len(values):3d} "
                  f"({100*np.mean(values):5.1f}%)")

    result = pd.DataFrame(details)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
