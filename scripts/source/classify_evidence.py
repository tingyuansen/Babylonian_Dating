#!/usr/bin/env python3
"""Mark the kinds of chronological evidence that the text of each diary keeps.

Reads data/corpus/tablets.parquet and lines.parquet, and writes
data/corpus/evidence.parquet. build_cohort.py groups the records by these
marks, and scripts/audit/build_benchmark.py copies them into the benchmark
ledger. The marks come from the text, because the catalogue does not record
what evidence a tablet keeps.

The marks come from regular expressions on the English translation. For a
record without a translation, the script searches the transliteration. The
columns era_llm, outer_llm, note and agree come from an earlier pass of a
language model over the same texts, with the prompt in MODEL_PROMPT. The script
keeps these columns from the existing file. The cohorts use only the regular
expressions.

Fields written per tablet:

    era_regex / era_llm        an era or regnal year stated in the text
    outer_regex / outer_llm    a position for Jupiter, Saturn or Mars
    inner_regex, lunar_regex, eclipse_regex
                               other kinds of evidence, regular expressions only
    n_joins                    number of joined fragments, from museum_no
    agree                      whether the two passes agree on era and outer

Usage:
    python3 scripts/source/classify_evidence.py
    python3 scripts/source/classify_evidence.py --limit 40    # the first 40 records only
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]

OUTER = r"(Jupiter|Saturn|Mars)"
INNER = r"(Venus|Mercury)"
SIGNS = (r"(Aries|Taurus|Gemini|Cancer|Leo|Virgo|Libra|Scorpi[ou]s|Sagittarius|"
         r"Capricorn|Aquarius|Pisces)")

# The transliteration writes the planets as Sumerograms. About one record in
# seven has a transliteration but no translation, so the patterns also search
# these forms.
SUMERO = {
    "jupiter": r"MUL[₂2]?\.?BABBAR|SAG\.ME\.GAR",
    "saturn": r"GENNA|SAG\.U[ŠS]",
    # Mars matches as the Akkadian word, as MUL.AN, or as AN before ina, ana,
    # MEŠ or KI. AN alone is the determinative for god and heaven, one of the
    # most common signs. A pattern for AN alone matched 57 of the 61
    # transliterated tablets.
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
    # An outer planet in a sign, or its station or retrograde motion.
    "outer": re.compile(rf"{OUTER}[^.]{{0,80}}(?:{SIGNS}|stationary|retrograde)"
                        rf"|(?:{SIGNS})[^.]{{0,40}}{OUTER}", re.I),
    "inner": re.compile(rf"{INNER}[^.]{{0,80}}(?:{SIGNS}|first appearance|"
                        rf"last appearance)", re.I),
    # The Moon measured against a reference star. This kind is the most common
    # and carries the least information.
    "lunar": re.compile(r"\bmoon\b[^.]{0,90}(cubit|finger|below|above|"
                        r"in front of|behind)", re.I),
    "eclipse": re.compile(r"\beclipse\b", re.I),
}

# The prompt of the earlier model pass that wrote era_llm, outer_llm and note.
# The script does not call a model.
MODEL_PROMPT = """You are reading a translated Babylonian astronomical diary.
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
    ap.add_argument("--limit", type=int, default=0)
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
    # A record without English text is searched in its transliteration.
    t["lang"] = t.text.str.len().gt(0).map({True: "en", False: "tlit"})
    t["n_joins"] = t.museum_no.fillna("").apply(
        lambda s: len(re.split(r"\s*\+\s*", s)) if s.strip() else 0)
    for name, rx in PAT.items():
        en = t.text.apply(lambda s, r=rx: bool(r.search(s)))
        tl = t.tlit.apply(lambda s, r=PAT_TL[name]: bool(r.search(s)))
        t[f"{name}_regex"] = en | (t.lang.eq("tlit") & tl)

    if args.limit:
        t = t.head(args.limit)

    out = t.drop(columns=["text", "tlit"])

    # The model columns of the existing output are kept, so a rebuild does not
    # lose them.
    if args.out.exists():
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
