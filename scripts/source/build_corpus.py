#!/usr/bin/env python3
"""Join the ADsD catalogue to the fetched text, one record per tablet.

This script builds the corpus of 417 dated diary records of ADART I to III
(Methods, Corpus, and SI Appendix, section S1). It reads only the files that
fetch_adsd.py and fetch_texts.py write to data/source/adsd/. ADsD also holds
the other genres of ADART V and VI, which the paper does not use.

The published year comes from the catalogue field `date_bce`, written as
`262/1`. This is one Babylonian year, which spans two Julian years.
The table keeps both years, as bce_first and bce_second. The benchmark scores
bce_first, the Julian year in which the Babylonian year begins.

Outputs:

    data/corpus/tablets.parquet   one row per tablet, with the catalogue fields
    data/corpus/lines.parquet     one row per line, with transliteration and translation

Fields of tablets.parquet:

    tablet, project, designation, bce_first, bce_second, ancient_year,
    museum_no, months_recorded, text_comments, copy, photo, accession_no,
    period, provenience, lon, lat, n_lines, text

At the end, the script compares the range of dates in each volume with the
range that the edition prints. It exits with status 1 if they differ.

Usage:
    python3 scripts/source/build_corpus.py
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
PUBLISHED_SPAN = {"adsd/adart1": (652, 262),
                  "adsd/adart2": (261, 165),
                  "adsd/adart3": (164, 61)}


def load_catalogue(raw: Path) -> dict[str, dict]:
    """Return the catalogue entries, each with the ADART volume that holds it.

    The catalogues of the volumes overlap, because each lists the whole diary
    corpus. The project field of the first file read would put every tablet in
    ADART I. The proxies map in the master corpus.json gives the volume of most
    tablets. For the others, the digit after the X of the identifier gives the
    volume. This rule agrees with the proxies map for each tablet in the map.
    """
    owner: dict[str, str] = {}
    master = raw / "adsd.zip"
    if master.exists():
        with zipfile.ZipFile(master) as z:
            owner.update(json.loads(z.read("adsd/corpus.json")).get("proxies", {}))

    cat: dict[str, dict] = {}
    for zp in sorted(raw.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.endswith("catalogue.json"):
                    continue
                for tid, rec in json.loads(z.read(name)).get("members", {}).items():
                    if tid not in cat:
                        rec = dict(rec)
                        proj = owner.get(tid)
                        if not proj and len(tid) > 1 and tid[1].isdigit():
                            proj = f"adsd/adart{tid[1]}"
                        rec["project"] = proj or rec.get("project", "")
                        cat[tid] = rec
    return cat


def split_date(date_bce: str, designation: str = "") -> tuple[
        int | None, int | None]:
    """Parse the catalogue date field into the first and the last year BCE.

    Almost all entries give one Babylonian year as the two Julian years that
    it spans, with the second year shortened. For example, `262/1` is
    262/261 and `310/09` is 310/309. One tablet is an excerpt of several years,
    written `303-301`, and the function returns both ends as written.
    """
    d = (date_bce or "").strip()
    m = re.match(r"^(\d+)-(\d+)$", d)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)/(\d+)$", d)
    if not m:
        return None, None
    first = int(m.group(1))
    suffix = m.group(2)
    if str(first - 1).endswith(suffix):
        return first, first - 1

    # Two catalogue rows have a malformed short form. Their designations give
    # the year in astronomical numbering, so -73 is the Babylonian year 74/73 BCE.
    negative = re.search(r"(?:^|\s)-(\d+)", designation or "")
    if negative:
        astronomical = int(negative.group(1))
        return astronomical + 1, astronomical
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=HERE / "data" / "source" / "adsd")
    ap.add_argument("--texts", type=Path, default=HERE / "data" / "source" / "adsd" / "texts.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "corpus" / "tablets.parquet")
    args = ap.parse_args()

    cat = {tid: c for tid, c in load_catalogue(args.raw).items()
           if c.get("project") in PUBLISHED_SPAN}
    texts = {}
    if args.texts.exists():
        for line in args.texts.open():
            rec = json.loads(line)
            if rec["n_lines"] or rec["tablet"] not in texts:
                texts[rec["tablet"]] = rec
    print(f"catalogue {len(cat)} entries, text for {sum(1 for t in texts.values() if t['n_lines'])}")

    rows = []
    for tid, c in sorted(cat.items()):
        first, second = split_date(
            c.get("date_bce", ""), c.get("designation", ""))
        coord = c.get("pleiades_coord", "")
        lon = lat = None
        m = re.match(r"\[([-\d.]+),([-\d.]+)\]", coord)
        if m:
            lon, lat = float(m.group(1)), float(m.group(2))
        t = texts.get(tid, {})
        rows.append({
            "tablet": tid,
            "project": c.get("project", ""),
            "designation": c.get("designation", ""),
            "bce_first": first,
            "bce_second": second,
            "ancient_year": c.get("ancient_year", ""),
            "museum_no": c.get("museum_no", ""),
            "months_recorded": c.get("months_recorded", ""),
            # The notes of the editors. Some name an eclipse and its date, or
            # note an error on the tablet.
            "text_comments": c.get("text_comments", ""),
            "copy": c.get("copy", ""),
            "photo": c.get("photo", ""),
            "accession_no": c.get("accession_no", ""),
            "period": c.get("period", ""),
            "provenience": c.get("provenience", ""),
            "lon": lon, "lat": lat,
            "n_lines": t.get("n_lines", 0),
            "lines": t.get("lines", []),
            "text": t.get("text", ""),
        })

    import pandas as pd
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The years are floats, so that a missing date can be NaN.
    pd.DataFrame([{k: v for k, v in r.items() if k != "lines"} for r in rows]) \
        .astype({"bce_first": "float64", "bce_second": "float64"}) \
        .to_parquet(args.out.parent / "tablets.parquet", index=False)
    pd.DataFrame([dict(tablet=r["tablet"], project=r["project"], **ln)
                  for r in rows for ln in r["lines"]]) \
        .to_parquet(args.out.parent / "lines.parquet", index=False)

    by_proj = defaultdict(list)
    for r in rows:
        if r["bce_first"]:
            by_proj[r["project"]].append(r["bce_first"])
    print(f"\n{len(rows)} tablets -> tablets.parquet, lines.parquet")
    print(f"  with text  : {sum(1 for r in rows if r['n_lines'])}")
    print(f"  with a date: {sum(1 for r in rows if r['bce_first'])}")
    ok = True
    for proj, (hi, lo) in PUBLISHED_SPAN.items():
        got = by_proj.get(proj, [])
        if not got:
            print(f"  {proj}: NO DATES"); ok = False; continue
        match = (max(got), min(got)) == (hi, lo)
        ok &= match
        print(f"  {proj}: {len(got):>3} dated, BCE {max(got)}..{min(got)}, "
              f"published {hi}..{lo}  {'ok' if match else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
