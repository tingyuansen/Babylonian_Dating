#!/usr/bin/env python3
"""Classify what chronological evidence each tablet actually preserves.

This is the input to the evaluation cohort. A tablet can only be scored fairly
against the published date if we know what a dater had to work with, and that
cannot be read off the catalogue. It has to come from the text.

Two passes. A deterministic one first, because most of the signal is in fixed
vocabulary and a regex is free, reproducible and does not hallucinate. Then a
model pass in parallel over the same tablets, which catches what patterns miss,
principally era statements phrased unusually and planetary positions written
across a damaged line.

Where the two disagree the record keeps both, since the disagreements are
exactly the tablets worth looking at by hand.

Fields written per tablet:

    era_regex / era_llm        an era or regnal year stated in the text
    outer_regex / outer_llm    a position for Jupiter, Saturn or Mars
    inner, lunar, eclipse      other evidence classes, regex only
    n_joins                    fragments composing the tablet, from museum_no
    agree                      whether the two passes concur on era and outer

Usage:
    python3 classify_evidence.py                # regex only, seconds
    python3 classify_evidence.py --llm          # adds the model pass
    python3 classify_evidence.py --llm --limit 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "scripts"))

OUTER = r"(Jupiter|Saturn|Mars)"
INNER = r"(Venus|Mercury)"
SIGNS = (r"(Aries|Taurus|Gemini|Cancer|Leo|Virgo|Libra|Scorpi[ou]s|Sagittarius|"
         r"Capricorn|Aquarius|Pisces)")

# The diaries write planets as Sumerograms. About a sixth of the corpus is
# transliterated but untranslated, so evidence must be findable in either form.
SUMERO = {
    "jupiter": r"MUL[₂2]?\.?BABBAR|SAG\.ME\.GAR",
    "saturn": r"GENNA|SAG\.U[ŠS]",
    # Mars only as the Akkadian word. AN alone is the determinative for god and
    # heaven, one of the commonest signs in the corpus, and matching it fired on
    # 57 of 61 transliterated tablets while the real word matched none.
    "mars": r"ṣal-bat-a-nu|MUL\.AN|\bAN\b(?=\s*(?:ina|ana|MEŠ|KI))",
    "mercury": r"GU[₄4]\.UD",
    "venus": r"dele-bat|DELE\.BAT",
}
OUTER_SUM = "|".join(SUMERO[k] for k in ("jupiter", "saturn", "mars"))
INNER_SUM = "|".join(SUMERO[k] for k in ("mercury", "venus"))

PAT_TL = {
    "era": re.compile(r"\bMU\b\s*[\d.]+", re.I),
    "outer": re.compile(OUTER_SUM),
    "inner": re.compile(INNER_SUM),
    "lunar": re.compile(r"\bsin\b|\b30\b", re.I),
    "eclipse": re.compile(r"AN\.KU[₁1][₀0]|attal", re.I),
}

PAT = {
    # A regnal or era year. The diaries write these as "year N of king X" or as
    # a bare "year N" heading a section.
    "era": re.compile(r"\byear\s+\d+\b", re.I),
    # A planet placed in a constellation, or a station, which is equally dating.
    "outer": re.compile(rf"{OUTER}[^.]{{0,80}}(?:{SIGNS}|stationary|retrograde)"
                        rf"|(?:{SIGNS})[^.]{{0,40}}{OUTER}", re.I),
    "inner": re.compile(rf"{INNER}[^.]{{0,80}}(?:{SIGNS}|first appearance|"
                        rf"last appearance)", re.I),
    # The Moon measured against a reference star, the commonest and weakest.
    "lunar": re.compile(r"\bmoon\b[^.]{0,90}(cubit|finger|below|above|"
                        r"in front of|behind)", re.I),
    "eclipse": re.compile(r"\beclipse\b", re.I),
}

SYSTEM = """You are reading a translated Babylonian astronomical diary.
Answer only with a JSON object, no prose, with exactly these keys:

  "era": true if the text states a year of a king or of a named era
         (Seleucid, Arsacid), for example "year 50" or "year 12 of king
         Alexander". A day or month number is not an era year.
  "outer": true if the text gives a position, station, retrograde motion or
         visibility phenomenon for Jupiter, Saturn or Mars.
  "note": at most 12 words saying what the decisive phrase was, or "" if neither.

Damaged text is marked with brackets. A statement inside brackets is a modern
restoration; count it, but say so in the note.

Some tablets are given as Akkadian transliteration with no translation. Read
those directly. The planets appear as Sumerograms: MUL2.BABBAR or SAG.ME.GAR is
Jupiter, GENNA or SAG.US is Saturn, AN or salbatanu is Mars, GU4.UD is Mercury,
dele-bat is Venus. MU followed by a number is a year."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "corpus" / "evidence.parquet")
    args = ap.parse_args()

    t = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet")
    lines = pd.read_parquet(HERE / "data" / "corpus" / "lines.parquet")
    lines = lines.assign(translation=lines.translation.fillna(""),
                         translit=lines.translit.fillna(""))
    text = (lines.groupby("tablet").translation
                 .apply(lambda s: " ".join(x for x in s if x)).to_dict())
    tlit = (lines.groupby("tablet").translit
                 .apply(lambda s: " ".join(x for x in s if x)).to_dict())

    t = t[t.project.isin(["adsd/adart1", "adsd/adart2", "adsd/adart3"])].copy()
    t["text"] = t.tablet.map(text).fillna("")
    t["tlit"] = t.tablet.map(tlit).fillna("")
    # Tablets ADsD transliterated but never translated are read in Akkadian.
    t["lang"] = t.text.str.len().gt(0).map({True: "en", False: "tlit"})
    t["n_joins"] = t.museum_no.fillna("").apply(
        lambda s: len(re.split(r"\s*\+\s*", s)) if s.strip() else 0)
    for name, rx in PAT.items():
        en = t.text.apply(lambda s, r=rx: bool(r.search(s)))
        tl = t.tlit.apply(lambda s, r=PAT_TL[name]: bool(r.search(s)))
        t[f"{name}_regex"] = en | (t.lang.eq("tlit") & tl)

    if args.limit:
        t = t.head(args.limit)

    if args.llm:
        from agent.llm import ask_many, CHEAP
        have = t[(t.text.str.len() > 0) | (t.tlit.str.len() > 0)]
        print(f"model pass over {len(have)} tablets, {args.workers} in parallel")
        # Long tablets are truncated. The dating evidence clusters early and a
        # full tablet would cost more than the answer is worth.
        prompts = [(a if a else "TRANSLITERATION ONLY, no translation exists:\n" + b)[:6000]
                   for a, b in zip(have.text, have.tlit)]
        out = ask_many(SYSTEM, prompts, model=CHEAP, workers=args.workers,
                       max_tokens=200)
        era, outer, note = {}, {}, {}
        bad = 0
        for tid, raw in zip(have.tablet, out):
            if not raw:
                bad += 1
                continue
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                bad += 1
                continue
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                bad += 1
                continue
            era[tid] = bool(d.get("era"))
            outer[tid] = bool(d.get("outer"))
            note[tid] = str(d.get("note", ""))[:90]
        print(f"  {len(era)} parsed, {bad} unusable")
        t["era_llm"] = t.tablet.map(era)
        t["outer_llm"] = t.tablet.map(outer)
        t["note"] = t.tablet.map(note)
        both = t.dropna(subset=["era_llm"])
        t["agree"] = (t.era_regex == t.era_llm) & (t.outer_regex == t.outer_llm)
        print(f"  regex and model agree on {int(both.era_regex.eq(both.era_llm).sum())}"
              f"/{len(both)} for era, "
              f"{int(both.outer_regex.eq(both.outer_llm).sum())}/{len(both)} for outer planets")

    out = t.drop(columns=["text", "tlit"])

    # A deterministic rerun preserves existing model columns unless --llm is
    # requested, keeping rebuild behavior monotonic.
    if not args.llm and args.out.exists():
        prev = pd.read_parquet(args.out)
        keep = [c for c in ("era_llm", "outer_llm", "note", "agree")
                if c in prev.columns and c not in out.columns]
        if keep:
            out = out.merge(prev[["tablet", *keep]], on="tablet", how="left")
            print(f"  carried forward from the previous model pass: {', '.join(keep)}")

    out.to_parquet(args.out, index=False)
    print(f"\n{len(t)} diaries -> {args.out}")
    for name in PAT:
        print(f"  {name:<8} {int(t[f'{name}_regex'].sum()):>3}")
    print(f"  joins    {int((t.n_joins > 1).sum()):>3} composed of 2+ fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
