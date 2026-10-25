#!/usr/bin/env python3
"""Date the dated addenda of ADART IV once, as the undated entries are dated.

This is the dating step of the frozen test that
data/benchmark/addenda_registration.json registers. It runs main() of
date_adart_iv.py unchanged, with its input and output paths set to the
addenda. Each text is therefore dated by the same call as an undated entry:
the astronomy pass reads only the identifier and the blind astronomy text of
build_adart_iv_addenda.py, and the check of stated years runs after it. This
script reads no published year.

Outputs:
    data/results/adart_iv_addenda_dates.parquet   one row per text
    data/results/adart_iv_addenda_summary.json    counts and hashes
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.inference import date_adart_iv as run  # noqa: E402

SOURCE = HERE / "data" / "corpus" / "adart_iv_addenda_tablets.parquet"
OUT = HERE / "data" / "results" / "adart_iv_addenda_dates.parquet"
SUMMARY = HERE / "data" / "results" / "adart_iv_addenda_summary.json"


def main() -> int:
    run.SOURCE, run.OUT, run.SUMMARY = SOURCE, OUT, SUMMARY
    return run.main()


if __name__ == "__main__":
    raise SystemExit(main())
