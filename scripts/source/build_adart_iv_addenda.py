#!/usr/bin/env python3
"""Build the blind input of the frozen test on the dated addenda of ADART IV.

The section "Dated Diaries: Addenda" (printed pages 1 to 92) prints each text
over two facing pages: the transliteration on the left page, the translation
on the right page. A heading "No. -NNN" gives the number of the diary. Each
text then starts with its museum number, with a letter when the number has
more than one text. A text with a letter can also start without a new number.

This script segments the right pages at these headings, one row per text, and
applies to each translation the filters of build_adart_iv_corpus.py. It
imports those functions and replaces only the page range and the
segmentation, because the undated section prints both columns on one page.
The OCR corrections table is applied as it is, and holds no line of these
pages.

The published year of each text is the number of its heading. It goes only to
data/benchmark/addenda_published_years.parquet, which the scorer reads after
dating. That file also records whether ADART IV calls the text a join, a
duplicate or a continuation of a text of ADART I to III, or whether a museum
number of the text is in the ADART I to III corpus.

Outputs:
    data/corpus/adart_iv_addenda_tablets.parquet   one row per text, the input
    data/corpus/adart_iv_addenda_lines.parquet     the lines of the right pages
    data/corpus/adart_iv_addenda_summary.json      counts and hashes
    data/benchmark/addenda_published_years.parquet published years and overlap
"""

from __future__ import annotations

import json
import re
import sys
from bisect import bisect_right
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from scripts.source.build_adart_iv_corpus import (  # noqa: E402
    CHRONOLOGY, CORRECTIONS, EDITORIAL_START, MANIFEST, OCR_LINES,
    PRINTED_OFFSET, SOURCE, STRUCTURE, astronomy_only, basic_text,
    catalogue_key, converted_lines, foreign_heading, normalize_translation,
    parser_line, sha256, without_chronology)

# PDF pages 88--179 are printed pages 1--92, the section "Dated Diaries:
# Addenda". A right page has an even PDF page number.
PDF_FIRST = 88
PDF_LAST = 179
# The pages before the section hold the corrections to Volumes I to III.
FRONT_LAST = PDF_FIRST - 1
ADSD = HERE / "data" / "corpus" / "tablets.parquet"
OUT_TABLETS = HERE / "data" / "corpus" / "adart_iv_addenda_tablets.parquet"
OUT_LINES = HERE / "data" / "corpus" / "adart_iv_addenda_lines.parquet"
OUT_SUMMARY = HERE / "data" / "corpus" / "adart_iv_addenda_summary.json"
OUT_YEARS = HERE / "data" / "benchmark" / "addenda_published_years.parquet"

ENTRY_HEADING = re.compile(r"^No\.\s*-\s*(\d+)([A-Z]?)(?:\s*/\s*(\d+))?$")
MUSEUM = (r"(?:BM|Rm|VAT|WML|Abbey Museum)\s+[A-Z]?\d[\w-]*"
          r"(?:\s*\+\s*(?:(?:BM|Rm)\s+)?\d+)*")
TEXT_HEADING = re.compile(rf"^(?:([A-Z])\s*:\s*)?({MUSEUM})$")
# One museum number, with the prefix that a join repeats or leaves out.
MUSEUM_PART = re.compile(r"^(BM|Rm|VAT|WML|Abbey Museum|Sp|DT|K)\s*(\S.*)$")
# A left-page line is compared with the right page at the same height. A
# heading on the left page can stand a few pixels above the one on the right.
LEFT_TOLERANCE = 50
# The words with which ADART IV links a text to another text.
LINK_WORDS = re.compile(
    r"duplicat|\bjoin|continu|parallel|same (?:month|tablet)|lacunae",
    re.IGNORECASE)
ADART_NUMBER = re.compile(r"No\.\s*-\s*\d+")


def right_page(page: int) -> bool:
    """A right page of the section holds the translation."""
    return page % 2 == 0


def position(row: dict) -> int:
    """Reading position of a right-page row, or of the left-page row beside it."""
    page = row["pdf_page"] + (0 if right_page(row["pdf_page"]) else 1)
    shift = 0 if right_page(row["pdf_page"]) else LEFT_TOLERANCE
    return page * 10_000 + row["y"] + shift


def museum_numbers(value: str) -> list[str]:
    """Each museum number of a join such as ``BM 41479 + Rm 710``."""
    numbers, prefix = [], None
    for part in re.split(r"\s*\+\s*", value or ""):
        part = part.strip()
        match = MUSEUM_PART.match(part)
        if match:
            prefix, rest = match.group(1), match.group(2).strip()
        else:
            rest = part
        if prefix and re.match(r"^[A-Z]?\d", rest):
            numbers.append(f"{prefix} {rest}")
    return numbers


def published_years(digits: str, second: str | None) -> list[int]:
    """Astronomical years of a heading: No. -212/1 gives -212 and -211."""
    years = [-int(digits)]
    if second:
        years.append(-int(digits[:-len(second)] + second))
    return years


def apply_corrections_in_range(rows: list[dict]) -> int:
    """Apply the checked OCR corrections of these pages, as the corpus does."""
    applied = 0
    for fix in pd.read_csv(CORRECTIONS).itertuples():
        if not PDF_FIRST <= fix.pdf_page <= PDF_LAST:
            continue
        hits = [row for row in rows if row["pdf_page"] == fix.pdf_page
                and fix.ocr_text in row["raw_text"]]
        if len(hits) != 1:
            raise RuntimeError(
                f"OCR correction for PDF page {fix.pdf_page} matches "
                f"{len(hits)} lines, not one: {fix.ocr_text!r}")
        hits[0]["raw_text"] = hits[0]["raw_text"].replace(
            fix.ocr_text, fix.printed_text, 1)
        hits[0]["display_text"] = basic_text(hits[0]["raw_text"])
        applied += 1
    return applied


def find_texts(rows: list[dict]) -> list[dict]:
    """Find the start of each text on the right pages.

    A diary number must be followed by a museum number. A text that starts
    under a number begins at the number, and the next text with a letter
    begins at its museum number.
    """
    texts, entry, pending = [], None, None
    right = sorted((row for row in rows if right_page(row["pdf_page"])),
                   key=lambda row: (row["pdf_page"], row["y"], row["x"]))
    for row in right:
        number = ENTRY_HEADING.match(row["display_text"])
        if number:
            if pending is not None:
                raise RuntimeError(f"two diary numbers in a row: {row['display_text']!r}")
            entry = {"entry_no": "-" + number.group(1) + number.group(2)
                     + (f"/{number.group(3)}" if number.group(3) else ""),
                     "published_years_astro": published_years(
                         number.group(1), number.group(3))}
            pending = row
            continue
        heading = TEXT_HEADING.match(catalogue_key(row["display_text"]))
        if heading:
            if entry is None:
                raise RuntimeError(f"museum number before any diary number: "
                                   f"{row['display_text']!r}")
            start = pending if pending is not None else row
            texts.append({
                **entry,
                "letter": heading.group(1) or "",
                "museum_no": catalogue_key(heading.group(2)),
                "new_number": pending is not None,
                "header_ids": {start["source_line_id"], row["source_line_id"]},
                "entry_header_id": pending["source_line_id"] if pending else "",
                "pdf_page": row["pdf_page"],
                "position": position(start),
            })
            pending = None
        elif pending is not None:
            raise RuntimeError(f"diary number {entry['entry_no']} is not followed "
                               "by a museum number")
    return texts


def left_headings_agree(rows: list[dict]) -> list[str]:
    """Each diary number stands on a left page and on the right page beside it."""
    found = {}
    for row in rows:
        number = ENTRY_HEADING.match(row["display_text"])
        if number:
            side = "right" if right_page(row["pdf_page"]) else "left"
            spread = row["pdf_page"] + (0 if side == "right" else 1)
            found.setdefault((spread, row["display_text"].replace(" ", "")),
                             set()).add(side)
    return sorted(f"{text} (PDF page {spread})"
                  for (spread, text), sides in found.items()
                  if sides != {"left", "right"})


def own_numbers(text: dict, left_rows: list[dict]) -> list[str]:
    """Museum numbers of a text, from its right heading and its left heading.

    The left heading can name more joined pieces than the right one, so the
    first left-page line of the text that holds its first number is read too.
    """
    numbers = museum_numbers(text["museum_no"])
    digits = re.search(r"\d[\w-]*", numbers[0]).group(0)
    for row in left_rows:
        line = normalize_translation(row["display_text"])
        line = re.sub(r"\([^()]*\)", " ", line)
        for part in re.split(r"&|\\\\|\|", line):
            if re.search(rf"(?<![\d]){re.escape(digits)}(?![\d])", part):
                found = re.search(MUSEUM, part)
                if found:
                    numbers += museum_numbers(found.group(0))
                    return sorted(set(numbers))
    return sorted(set(numbers))


def main() -> int:
    source_sha256 = json.loads(MANIFEST.read_text(encoding="utf-8"))["source_sha256"]
    file_id = json.loads(MANIFEST.read_text(encoding="utf-8"))["mathpix_file_id"]
    pages = json.loads(OCR_LINES.read_text(encoding="utf-8"))["pages"]
    if len(pages) != 669 or [page["page"] for page in pages] != list(range(1, 670)):
        raise RuntimeError("Mathpix line OCR does not contain exactly pages 1..669")
    rows = [row for page in pages[PDF_FIRST - 1:PDF_LAST]
            for row in converted_lines(page)]
    corrections_applied = apply_corrections_in_range(rows)
    texts = find_texts(rows)
    if not texts:
        raise RuntimeError("no addenda headings found")
    unpaired = left_headings_agree(rows)

    positions = [text["position"] for text in texts]
    tablet_ids = [f"ADART_IV_ADD_{serial:02d}_"
                  + re.sub(r"[^A-Za-z0-9]+", "_", text["museum_no"]).strip("_")
                  for serial, text in enumerate(texts, start=1)]
    for text, tablet in zip(texts, tablet_ids):
        text["tablet"] = tablet

    # The translation of each text: the right pages, read as the corpus of the
    # undated diaries reads its right-hand column.
    line_rows = []
    editorial = {tablet: False for tablet in tablet_ids}
    left_by_text: dict[str, list[dict]] = {tablet: [] for tablet in tablet_ids}
    for row in sorted(rows, key=lambda item: (item["pdf_page"], item["y"], item["x"])):
        index = bisect_right(positions, position(row)) - 1
        if index < 0:
            continue
        text = texts[index]
        tablet = text["tablet"]
        if not right_page(row["pdf_page"]):
            left_by_text[tablet].append(row)
            continue
        normalized = normalize_translation(row["display_text"])
        role = "translation"
        if text["entry_header_id"] and row["source_line_id"] == text["entry_header_id"]:
            role = "entry_header"
        elif row["source_line_id"] in text["header_ids"]:
            role = "catalogue_header"
        elif EDITORIAL_START.match(normalized):
            role = "editorial_heading"
            editorial[tablet] = True
        elif editorial[tablet]:
            role = "editorial_excluded"
        elif STRUCTURE.match(normalized):
            role = "structure"
        chronology = bool(CHRONOLOGY.search(normalized)) if role == "translation" else False
        included = role == "translation"
        line_rows.append({
            "tablet": tablet,
            "museum_no": text["museum_no"],
            **row,
            "normalized_text": normalized,
            "astronomy_text": astronomy_only(normalized) if included else "",
            "parser_text": parser_line(astronomy_only(normalized)) if included else "",
            "chronology_present": chronology,
            "role": role,
            "included_in_translation": included,
            "included_in_astronomy": included and bool(astronomy_only(normalized)),
        })
    lines = pd.DataFrame(line_rows)

    tablet_rows, truncated = [], 0
    for text in texts:
        group = lines[lines.tablet.eq(text["tablet"])]
        admitted = group[group.included_in_translation]
        translation = "\n".join(admitted.normalized_text)
        astronomy = " ".join(admitted.loc[admitted.included_in_astronomy, "parser_text"])
        cut = foreign_heading(astronomy, text["museum_no"])
        if cut is not None:
            astronomy = astronomy[:cut].strip()
            truncated += 1
        astronomy = without_chronology(astronomy)
        if CHRONOLOGY.search(astronomy):
            raise RuntimeError(f"{text['tablet']}: chronology left in the astronomy text")
        if ADART_NUMBER.search(astronomy):
            raise RuntimeError(f"{text['tablet']}: a diary number is in the astronomy text")
        tablet_rows.append({
            "tablet": text["tablet"],
            "project": "adart_iv/addenda",
            "museum_no": text["museum_no"],
            "pdf_page_start": int(group.pdf_page.min()),
            "pdf_page_end": int(group.pdf_page.max()),
            "printed_page_start": int(group.printed_page.min()),
            "printed_page_end": int(group.printed_page.max()),
            "n_translation_lines": int(len(admitted)),
            "n_translation_chars": len(translation),
            "translation": translation,
            "astronomy_text": astronomy,
            "chronology_text": "\n".join(
                admitted.loc[admitted.chronology_present, "normalized_text"]),
            "source_pdf": str(SOURCE.relative_to(HERE)),
            "source_pdf_sha256": source_sha256,
            "ocr_lines": str(OCR_LINES.relative_to(HERE)),
            "ocr_lines_sha256": sha256(OCR_LINES),
            "mathpix_file_id": file_id,
            "segmentation_rule": "addenda_right_page_diary_and_museum_heading_v1",
        })
    tablets = pd.DataFrame(tablet_rows)
    if tablets.tablet.duplicated().any() or len(tablets) != len(texts):
        raise RuntimeError("segmentation did not produce one unique row per text")

    # The published year and the links to ADART I to III. Nothing here enters
    # the input above.
    adsd: dict[str, list[str]] = {}
    for record in pd.read_parquet(ADSD, columns=["designation", "museum_no"]).itertuples():
        for number in museum_numbers(record.museum_no):
            adsd.setdefault(number, []).append(record.designation)
    front = [row for page in pages[:FRONT_LAST] for row in converted_lines(page)]
    front_addenda = [basic_text(row["raw_text"]) for row in front
                     if "Addenda" in row["raw_text"]]
    year_rows = []
    for text, record in zip(texts, tablet_rows):
        numbers = own_numbers(text, left_by_text[text["tablet"]])
        matches = sorted(f"{number} = {designation}" for number in numbers
                         for designation in adsd.get(number, []))
        corrigenda = [line for line in front_addenda
                      if any(re.search(re.escape(number).replace(r"\ ", r"\s*")
                                       + r"(?![\d])", line) for number in numbers)]
        notes = [normalize_translation(row["display_text"])
                 for row in left_by_text[text["tablet"]]]
        notes += lines.loc[lines.tablet.eq(text["tablet"])
                           & lines.role.isin(["editorial_heading", "editorial_excluded"]),
                           "normalized_text"].tolist()
        links = [note for note in notes if LINK_WORDS.search(note)
                 and (ADART_NUMBER.search(note)
                      or any(number in adsd for number in museum_numbers(
                          " + ".join(re.findall(MUSEUM, note)))))]
        # The year must not be in the input, as a number or as a BCE number.
        leaked = [year for year in text["published_years_astro"]
                  for value in (abs(year), abs(year) + 1)
                  if re.search(rf"(?<![\d]){value}(?![\d])", record["astronomy_text"])]
        year_rows.append({
            "tablet": text["tablet"],
            "entry_no": text["entry_no"],
            "letter": text["letter"],
            "museum_no": text["museum_no"],
            "starts_new_number": text["new_number"],
            "published_years_astro": text["published_years_astro"],
            "museum_numbers": numbers,
            "adsd_museum_matches": matches,
            "corrigenda_evidence": corrigenda,
            "note_evidence": links,
            "overlaps_adart_i_iii": bool(matches or corrigenda or links),
            "year_number_in_input": sorted(set(leaked)),
        })
    years = pd.DataFrame(year_rows)

    OUT_TABLETS.parent.mkdir(parents=True, exist_ok=True)
    tablets.to_parquet(OUT_TABLETS, index=False)
    lines.to_parquet(OUT_LINES, index=False)
    years.to_parquet(OUT_YEARS, index=False)
    summary = {
        "schema": "adart-iv-addenda-corpus-v1",
        "section": "ADART IV Dated Diaries: Addenda only",
        "pdf_pages": [PDF_FIRST, PDF_LAST],
        "printed_pages": [PDF_FIRST - PRINTED_OFFSET, PDF_LAST - PRINTED_OFFSET],
        "texts": len(tablets),
        "texts_with_translation": int(tablets.n_translation_lines.gt(0).sum()),
        "texts_without_translation": tablets.loc[
            tablets.n_translation_lines.eq(0), "museum_no"].tolist(),
        "texts_with_internal_chronology": int(
            tablets.chronology_text.str.strip().ne("").sum()),
        "texts_truncated_at_a_foreign_heading": truncated,
        "diary_numbers_without_a_heading_on_both_pages": len(unpaired),
        "translation_lines": int(lines.included_in_translation.sum()),
        "astronomy_input_lines": int(lines.included_in_astronomy.sum()),
        "editorial_lines_excluded": int(
            lines.role.isin(["editorial_heading", "editorial_excluded"]).sum()),
        "ocr_corrections_applied": corrections_applied,
        "source_pdf": str(SOURCE.relative_to(HERE)),
        "source_pdf_sha256": source_sha256,
        "ocr_lines": str(OCR_LINES.relative_to(HERE)),
        "ocr_lines_sha256": sha256(OCR_LINES),
        "ocr_corrections": str(CORRECTIONS.relative_to(HERE)),
        "ocr_corrections_sha256": sha256(CORRECTIONS),
        "outputs": {
            str(path.relative_to(HERE)): sha256(path)
            for path in (OUT_TABLETS, OUT_LINES, OUT_YEARS)},
    }
    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
