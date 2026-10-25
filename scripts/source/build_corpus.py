#!/usr/bin/env python3
"""Join the ADsD catalogue to the fetched text, one record per tablet.

This is the only corpus builder. It reads nothing outside this directory, so
the whole dataset can be rebuilt from `fetch_adsd.py` and `fetch_texts.py`
without reference to any earlier project.

Only the diaries of ADART I to III enter. ADsD also holds the other genres of
ADART V and VI, which the paper does not use.

Ground truth comes from the catalogue's `date_bce`, written as `262/1`, which is
the Babylonian year across the two Julian years it spans. Both are kept, since
which one a method should be scored against is a choice the evaluation has to
make explicitly rather than inherit.

Output is two parquet tables in `data/`.

    tablets.parquet   one row per tablet, everything a stratification needs
    lines.parquet     one row per line, with transliteration and translation

Record fields:

    tablet, project, designation, bce_first, bce_second, ancient_year,
    museum_no, months_recorded, period, provenience, lon, lat,
    n_lines, lines[{label, translit, translation}], text

The volume-span check at the end must pass. It compares the dates recovered here
against the spans printed on the published editions and exits non-zero if they
disagree.

Usage:
    python3 build_corpus.py
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
    """Catalogue entries, with each tablet assigned to its true volume.

    The per-volume catalogues overlap, since each sub-project lists the whole
    diary corpus. Taking the project field from whichever file is read first
    puts every tablet in ADART I. The master corpus.json carries a proxies map
    from tablet to owning sub-project and that is the authority, but it does
    not cover every tablet. For the rest the identifier settles it: the digit
    after the X is the volume number, which holds for every tablet the proxies
    map does cover.
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
    """Parse the catalogue's date field into first and last BCE year.

    Almost every entry is a single Babylonian year written across the two Julian
    years it spans, with the second abbreviated: `262/1` is 262/261 and `310/09`
    is 310/309. One tablet is a multi-year excerpt written `303-301`, which is a
    range rather than a span, so both ends are taken as given.
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

    # Two catalogue rows have transposed/malformed abbreviated spans. Their
    # designations preserve the astronomical negative year: -73 means
    # astronomical year -73, or the Babylonian year 74/73 BCE.
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
            # The editors' own notes. Some identify an eclipse with a date, or
            # flag an error in the tablet, which is the closest thing we have to
            # a record of how a date was reached.
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

    # Two tables rather than nested JSON. One row per tablet for everything a
    # selection or a stratification needs, one row per line for the text. Both
    # are parquet, so they load in a second and carry their own schema.
    import pandas as pd
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The years stay floating point, so that a missing date is NaN.
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
