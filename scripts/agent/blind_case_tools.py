#!/usr/bin/env python3
"""Give an agent read-only commands that parse and score the text of one record.

The script runs in a blind workspace that holds a copy of tools/ and a
case.json with the record's identifier and translated text. The workspace holds
no published date. Each command prints JSON:

    summary            the record's length, months and number of clauses
    statements         the parser's clauses, with their fields (step 1)
    rank               strict set, safety set, first three groups (step 3)
    check YEAR         the verdict of each clause in one candidate year (step 2)
    compare YEAR ...   the verdicts of each clause in several years
    ablate INDEX ...   the rank output with some clauses removed
    search TERM        up to 20 passages of the text that hold a term
    context INDEX      the text around one clause

A year is BCE unless --era gives AD or ASTRO (astronomical numbering).
scripts/agent/mcp_server.py uses input_year and statement_record of this file.

Usage, inside the workspace:
    python blind_case_tools.py rank
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tools import cal, statements as S  # noqa: E402
from tools import years as yr  # noqa: E402


def load_case() -> tuple[dict, str, list[S.Statement], list[str], np.ndarray]:
    """Read case.json, with its clauses, months and the 701 candidate years."""
    case = json.loads((HERE / "case.json").read_text())
    text = case["text"]
    sts = S.extract(text)
    months = S.months_in(text)
    years = np.array([int(y) for y in cal.candidate_years()], dtype=int)
    return case, text, sts, months, years


def statement_record(index: int, st: S.Statement) -> dict:
    """The fields of one clause, with its index in the parser's list."""
    out = asdict(st)
    out["index"] = index
    return out


def scores_record(sts, months, years) -> dict:
    """The sets, the first three groups and the first 40 ranked years of the clauses.

    In each ranked year, decisive_families is the number of clocks K,
    primary_bits the decisive bits I_D and support_bits the supporting bits I_S.
    """
    scores = S.candidate_scores(sts, months, years)
    order = scores.order
    levels = S.evidence_levels(scores, max_levels=3)

    def candidate(i: int) -> dict:
        return {
            "year_astro": int(years[i]),
            "year_label": yr.year_label(int(years[i])),
            "decisive_families": int(scores.priority[i]),
            "primary_bits": round(float(scores.score[i]), 4),
            "support_bits": round(float(scores.support[i]), 4),
            "agree": int(scores.agree[i]),
            "contradict": int(scores.contradict[i]),
            "untestable": int(scores.untestable[i]),
            "strict": bool(scores.strict[i]),
            "safety": bool(scores.safety[i]),
        }

    informative = bool((scores.weights > 0).any())
    return {
        "informative": informative,
        "decline_reason": (None if informative else
                           "no testable weighted astronomical predicate"),
        "strict_count": int(scores.strict.sum()),
        "strict_years_astro": [int(y) for y in years[scores.strict]],
        "strict_year_labels": [yr.year_label(int(y)) for y in years[scores.strict]],
        "safety_count": int(scores.safety.sum()),
        "safety_years_preview_astro": [int(y) for y in years[scores.safety][:100]],
        "safety_year_labels_preview": [yr.year_label(int(y)) for y in years[scores.safety][:100]],
        "first_three_evidence_levels": levels,
        "ranked_preview": ([candidate(int(i)) for i in order[:40]]
                           if informative else []),
    }


def input_year(value: int, era: str) -> int:
    """Convert a year of era BCE, AD or ASTRO to astronomical numbering (1 BCE is 0)."""
    era = era.upper()
    if era == "ASTRO":
        return int(value)
    if era == "BCE":
        return yr.bce_to_astro(int(value))
    if int(value) < 1:
        raise ValueError("AD year must be positive")
    return int(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    sub.add_parser("statements")
    sub.add_parser("rank")
    check = sub.add_parser("check")
    check.add_argument("year", type=int)
    check.add_argument("--era", choices=["BCE", "AD", "ASTRO"], default="BCE")
    compare = sub.add_parser("compare")
    compare.add_argument("years", nargs="+", type=int)
    compare.add_argument("--era", choices=["BCE", "AD", "ASTRO"], default="BCE")
    ablate = sub.add_parser("ablate")
    ablate.add_argument("indices", nargs="*", type=int)
    search = sub.add_parser("search")
    search.add_argument("term")
    context = sub.add_parser("context")
    context.add_argument("index", type=int)
    context.add_argument("--chars", type=int, default=400)
    args = ap.parse_args()

    case, text, sts, months, years = load_case()
    if args.command == "summary":
        out = {
            "tablet": case["tablet"], "characters": len(text),
            "months": months, "statement_count": len(sts),
            "available_commands": [
                "summary", "statements", "rank", "check YEAR",
                "compare YEAR...", "ablate INDEX...", "search TERM",
                "context INDEX",
            ],
        }
    elif args.command == "statements":
        out = [statement_record(i, st) for i, st in enumerate(sts)]
    elif args.command == "rank":
        out = scores_record(sts, months, years)
    elif args.command == "check":
        year_astro = input_year(args.year, args.era)
        if year_astro not in set(years):
            raise SystemExit("candidate is outside the calendar")
        out = {
            "year_astro": year_astro,
            "year_label": yr.year_label(year_astro),
            "verdicts": [
                {"index": i, "result": v.result, "detail": v.detail,
                 "statement": statement_record(i, st)}
                for i, st in enumerate(sts)
                for v in [S.check(st, year_astro, months)]
            ],
        }
    elif args.command == "compare":
        compare_years = [input_year(value, args.era) for value in args.years]
        if any(year not in set(years) for year in compare_years):
            raise SystemExit("candidate is outside the calendar")
        out = []
        for i, st in enumerate(sts):
            out.append({
                "index": i, "text": st.text,
                "candidates": {
                    yr.year_label(year): asdict(S.check(st, year, months))
                    for year in compare_years
                },
            })
    elif args.command == "ablate":
        bad = [i for i in args.indices if i < 0 or i >= len(sts)]
        if bad:
            raise SystemExit(f"invalid statement indices: {bad}")
        removed = set(args.indices)
        kept = [st for i, st in enumerate(sts) if i not in removed]
        out = {"removed": sorted(removed), **scores_record(kept, months, years)}
    elif args.command == "search":
        out = []
        for match in re.finditer(re.escape(args.term), text, re.IGNORECASE):
            lo, hi = max(0, match.start() - 300), min(len(text), match.end() + 300)
            out.append({"offset": match.start(),
                        "context": re.sub(r"\s+", " ", text[lo:hi])})
            if len(out) == 20:
                break
    else:
        if args.index < 0 or args.index >= len(sts):
            raise SystemExit("invalid statement index")
        needle = sts[args.index].text
        pos = text.find(needle)
        if pos < 0:
            words = needle.split()
            pos = text.find(" ".join(words[:3])) if words else -1
        if pos < 0:
            out = {"index": args.index, "context": None}
        else:
            lo = max(0, pos - args.chars)
            hi = min(len(text), pos + len(needle) + args.chars)
            out = {"index": args.index, "offset": pos,
                   "context": text[lo:hi]}
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
