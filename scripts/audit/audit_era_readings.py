#!/usr/bin/env python3
"""Read the era and regnal years that the texts state, and check them at the published years.

The output is data/benchmark/era_readings.parquet. Dating never reads it, because
a year stated in the text enters only after the candidate sets are final.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import eras  # noqa: E402
from tools import years as yr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=HERE / "data" / "benchmark" / "era_readings.parquet")
    args = ap.parse_args()
    tablets = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet")
    rows = []
    for tablet in tablets[tablets.text.notna()].itertuples():
        for reading in eras.extract(tablet.text):
            dated = pd.notna(tablet.bce_first)
            published_bce = int(tablet.bce_first) if dated else None
            published = yr.bce_to_astro(published_bce) if dated else None
            # "150/149" is the one Babylonian year beginning in 150 BCE;
            # bce_second is the Julian year it ends in, not a second year.
            # A tablet spanning two years ("303/301") ends with the year
            # beginning in bce_second + 1.
            last = (yr.bce_to_astro(int(tablet.bce_second) + 1)
                    if dated and pd.notna(tablet.bce_second) else published)
            lo, hi = ((published, max(published, last))
                      if dated else (None, None))
            rows.append({
                "tablet": tablet.tablet, "ae_year": reading.ae_year,
                "se_year": reading.se_year, "astro_year": reading.astro_year,
                "system": reading.system, "text": reading.text,
                "source_start": reading.source_start,
                "published_bce": published_bce,
                "published_astro": published,
                "forward_agree": (lo <= reading.astro_year <= hi)
                if dated else pd.NA,
            })
    result = pd.DataFrame(rows)
    result["ae_year"] = result.ae_year.astype("Int64")
    result["se_year"] = result.se_year.astype("Int64")
    result["forward_agree"] = result.forward_agree.astype("boolean")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)

    dated = result[result.forward_agree.notna()]
    undated = result[result.forward_agree.isna()]
    print(f"{len(result)} secure absolute chronology readings on "
          f"{result.tablet.nunique()} tablets")
    print(f"  dated forward check: {int(dated.forward_agree.sum())}/{len(dated)}")
    print(f"  undated tablets resolved: {undated.tablet.nunique()}")
    multi = result.groupby("tablet").astro_year.nunique().gt(1)
    print(f"  multi-year tablets retained as ranges: {int(multi.sum())}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
