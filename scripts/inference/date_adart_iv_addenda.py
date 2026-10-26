#!/usr/bin/env python3
"""Date the dated addenda of ADART IV once, in the way the undated entries are dated.

This is the dating step of the registered test in
data/benchmark/addenda_registration.json. It runs main() of date_adart_iv.py,
the ADART IV dating of SI Appendix, section S9, with the input and output paths
of the addenda. It dates with the parser clauses, as the registered rules do.
The astronomy pass reads only the identifier and the blind text that
build_adart_iv_addenda.py writes. The check of stated years runs after it.

This script reads no published year. Its input holds the ADART IV text, which
is not distributed. The Makefile lists the script under REGISTERED_TEST,
outside make rebuild.

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
    run.CLAUSES = None
    return run.main()


if __name__ == "__main__":
    raise SystemExit(main())
