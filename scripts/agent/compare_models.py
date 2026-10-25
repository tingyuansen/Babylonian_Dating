#!/usr/bin/env python3
"""Compare three language models in the pilot of the model review, on ten records.

Before the full run, gpt-6-luna and two models of the previous release, the
small gpt-5.6-luna and the larger gpt-5.6-terra, reviewed and searched the
same ten records of ADART I to III with the same prompts and checks. Their
outputs are in data/clauses/model_review_pilot/. This script tests each
model's accepted modifications and search additions at the published year
with the current rules, and writes the counts to
data/clauses/model_review_pilot/comparison.json.

It makes no model call. The pilot used an earlier version of the search's
normalization prompt, so its clauses are not those of the full run.

Usage:
    python scripts/agent/compare_models.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import statements as S  # noqa: E402
from tools import years as yr  # noqa: E402

PILOT = HERE / "data" / "clauses" / "model_review_pilot"
MODELS = ["gpt-6-luna-2026-09-22", "gpt-5.6-luna-2026-07-09", "gpt-5.6-terra-2026-07-09"]
FIELDS = {field.name for field in dataclasses.fields(S.Statement)}


def statement(saved: str, start=None, end=None) -> S.Statement:
    """A clause from the JSON the pilot saved, with its source offsets."""
    values = json.loads(saved)
    values.setdefault("text", values.get("normalized_text", ""))
    if "offset_degrees" in values:
        values["offset"] = values.pop("offset_degrees")
    if "normalized_amount_cubits" in values:
        values["amount"] = values.pop("normalized_amount_cubits")
    values["sign"] = values.get("sign") or ""
    if start is not None and not (isinstance(start, float) and math.isnan(start)):
        values["source_start"], values["source_end"] = int(start), int(end)
    return S.Statement(**{key: value for key, value in values.items() if key in FIELDS})


def tally(verdicts: list[str]) -> dict:
    return {result: verdicts.count(result) for result in ("agree", "contradict", "untestable")}


def main() -> int:
    tablets = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet").set_index("tablet")

    def verdict(st: S.Statement, tablet: str) -> str:
        published = yr.bce_to_astro(int(tablets.at[tablet, "bce_first"]))
        return S.check(st, published, S.months_in(tablets.at[tablet, "text"])).result

    out = {}
    for model in MODELS:
        summary = json.loads((PILOT / f"{model}_summary.json").read_text())
        additions = pd.read_csv(PILOT / f"{model}_additions.csv")
        added = additions[additions.status == "added"]
        changes = pd.read_csv(PILOT / f"{model}_changes.csv")
        modified = changes[changes.action == "MODIFY"]
        calls = summary["all_cached_record_calls"]
        out[model] = {
            "records": len(summary["tablets"]),
            "calls": int(calls["n_calls"]),
            "cost_usd": round(float(calls["cost_usd"]), 2),
            "parser_clauses": int(summary["review"]["clauses"]),
            "modified": int(len(modified)),
            "dropped": int((changes.action == "DROP").sum()),
            "added": int(len(added)),
            "added_at_published_year": tally([
                verdict(statement(r.final_statement_json, r.atomic_source_start,
                                  r.atomic_source_end), r.tablet)
                for r in added.itertuples()]),
            "modified_at_published_year": tally([
                verdict(statement(r.final_statement_json), r.tablet)
                for r in modified.itertuples()]),
        }
    (PILOT / "comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    for model, row in out.items():
        a = row["added_at_published_year"]
        print(f"{model}: US${row['cost_usd']:.2f}, {row['modified']} modified, "
              f"{row['added']} added, {a['agree']} of {a['agree'] + a['contradict']} "
              "testable additions agree at the published year")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
