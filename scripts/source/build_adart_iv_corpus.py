#!/usr/bin/env python3
"""Build an answer-hidden corpus from the ADART IV Mathpix line OCR.

ADART IV prints transliteration on the left and English translation on the
right.  Every undated entry begins with the same catalogue heading in both
columns.  This script uses that redundant layout signal, rather than a list of
expected tablets, to segment the complete section.  Only the right-hand
translation is admitted to astronomical inference; editorial ``Date`` and
``Comments`` matter is excluded, and text-internal ruler/year language is
retained separately for a post-astronomy chronology pass.

Outputs:
    data/corpus/adart_iv_tablets.parquet  one row per printed entry
    data/corpus/adart_iv_lines.parquet    line geometry and inclusion decision
    data/corpus/adart_iv_corpus_summary.json      hashes, counts, and segmentation audit
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from bisect import bisect_right
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
# The scan of the book is not distributed. Its SHA-256 comes from the Mathpix
# manifest, which recorded it when the scan was read.
SOURCE = HERE / "data" / "source" / "adart_iv" / "scan.pdf"
OCR_DIR = HERE / "data" / "source" / "adart_iv"
OCR_LINES = OCR_DIR / "mathpix_lines.json"
MANIFEST = OCR_DIR / "mathpix_manifest.json"
# OCR errors found by comparing a decisive passage with the page image. Each
# row was checked by eye, and the committed Mathpix output stays unchanged.
CORRECTIONS = HERE / "data" / "source" / "adart_iv" / "ocr_corrections.csv"
OUT_TABLETS = HERE / "data" / "corpus" / "adart_iv_tablets.parquet"
OUT_LINES = HERE / "data" / "corpus" / "adart_iv_lines.parquet"
OUT_SUMMARY = HERE / "data" / "corpus" / "adart_iv_corpus_summary.json"

# PDF pages 182--662 are printed pages 95--575, the complete Undated Diaries
# section.  Pages 88--178 are dated addenda and pages 664--666 a late appendix;
# neither is allowed into this prospective corpus.
PDF_FIRST = 182
PDF_LAST = 662
PRINTED_OFFSET = 87

CATALOGUE = re.compile(
    r"^(?:BM\s+\d|Rm\s+\d|VAT\s+\d|CBS\s+\d|MNB\s+\d|U\s+\d|"
    r"Birmingham City Museum A\.)",
    re.IGNORECASE,
)
STRUCTURE = re.compile(
    r"^[\s'´‘’]*?(?:Obv|Rev|Flake|Upper edge|Lower edge|Left edge|"
    r"Right edge|Edge|Column|Col\.)[\s.'´‘’]*$",
    re.IGNORECASE,
)
EDITORIAL_START = re.compile(r"^(?:Comments?|Date|Dating)\s*:?[\s.]*$",
                             re.IGNORECASE)
# A year, a king, a ruler or an era. Any mention of a year counts, so that
# complete, damaged and spelled-out year numbers all match.
CHRONOLOGY = re.compile(
    r"\byears?\b|\bkings?\b|\b(?:Seleucus|Antiochus|Demetrius|Artaxerxes|"
    r"Arsaces|Alexander|Darius|Xerxes|Arses|Umaku[sš]|Philip|Antigonus|Cyrus|"
    r"Cambyses|Nebuchadnezzar|Nabonidus|Nabopolassar|Neriglissar|Mithradates|"
    r"Phraates)\b|\b(?:Seleucid|Arsacid)\s+Era\b|\bSE\s*\d",
    re.IGNORECASE,
)
# The end of a part of a text: a semicolon, a comma that is not inside a
# number such as 1,21, or a full stop before a space.
BOUNDARY = re.compile(r";|,(?!\d)|\.(?=\s)")


GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "omicron": "ο", "pi": "π", "varpi": "π", "rho": "ρ",
    "varrho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
}
FRACTIONS = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5", "⅙": "1/6",
    "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 for a retained source or OCR artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def basic_text(value: str) -> str:
    """Normalize width and spacing without changing scholarly content."""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("．", ".").replace("，", ",").replace("；", ";")
    return re.sub(r"\s+", " ", value).strip()


def catalogue_key(value: str) -> str:
    """Canonical heading used to match the two printed columns."""
    value = basic_text(value)
    # The left heading often adds an accession number in parentheses while the
    # right repeats only the museum number.
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    value = re.sub(r"\s*\+\s*", " + ", value)
    return value


def normalize_translation(value: str) -> str:
    """Turn Mathpix display markup into conservative parser-facing English."""
    # Preserve mixed-number boundaries before NFKC expands a glyph such as 1½
    # to the ambiguous string 11⁄2.
    for glyph, ascii_fraction in FRACTIONS.items():
        value = re.sub(rf"(?<=\d){re.escape(glyph)}", f" {ascii_fraction}", value)
        value = value.replace(glyph, ascii_fraction)
    value = basic_text(value).replace("⁄", "/")
    value = re.sub(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
                   r"\1/\2", value)
    for name, glyph in GREEK.items():
        value = re.sub(rf"\\{name}\b", glyph, value)
    # Ordinal suffixes and common wrappers are the only TeX constructs needed
    # by the statement grammar. Unknown commands are unwrapped, never guessed.
    value = re.sub(
        r"\^\s*\{?\s*\\text\s*\{\s*(st|nd|rd|th)\s*\}\s*\}?",
        r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "°", value)
    previous = None
    while value != previous:
        previous = value
        value = re.sub(r"\\(?:mathrm|text|operatorname)\s*\{([^{}]*)\}",
                       r"\1", value)
    value = re.sub(r"\\(?:left|right)\b", "", value)
    value = value.replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\[", "[").replace(r"\]", "]")
    value = value.replace("{", "").replace("}", "")
    # Depending on Mathpix nesting, a wrapper pass can leave a harmless caret
    # before an ordinal suffix. Removing only these four suffix forms restores
    # printed English such as ``3rd`` without erasing uncertainty markers.
    value = re.sub(r"\s*\^\s*(st|nd|rd|th)\b", r"\1", value,
                   flags=re.IGNORECASE)
    # Mathpix sometimes separates a plain-text ordinal suffix from its digit.
    # The suffix has no other grammatical role after an integer in these
    # translations, so this is a lossless normalization rather than a guess.
    value = re.sub(r"(?<=\d)\s+(st|nd|rd|th)\b", r"\1", value,
                   flags=re.IGNORECASE)
    value = re.sub(r"\\[A-Za-z]+\b", "", value)
    value = value.replace("\\", "")
    value = value.replace("[....]", "[ ... ]").replace("[.....]", "[ ... ]")
    value = re.sub(r"\.{2,}", "...", value)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def without_chronology(value: str) -> str:
    """Cut each part of a text that names a year, a king, a ruler or an era.

    A part runs from one boundary to the next. The rest of the text is kept
    character for character.
    """
    if not CHRONOLOGY.search(value):
        return value
    while (match := CHRONOLOGY.search(value)):
        before = [b.end() for b in BOUNDARY.finditer(value, 0, match.start())]
        start = before[-1] if before else 0
        after = BOUNDARY.search(value, match.end())
        value = value[:start] + value[after.end() if after else len(value):]
    return re.sub(r"\s{2,}", " ", value).strip()


def astronomy_only(value: str) -> str:
    """Remove chronology from one line, and keep the astronomy around it."""
    if not CHRONOLOGY.search(value):
        return value
    month = re.search(r"\bMonth\b", value, re.IGNORECASE)
    if not month:
        return ""
    # A common first line is ``Year ..., kings .... Month IX, ...``. The text
    # before the month heading goes, and so does any later part that names a
    # year or a ruler, as in ``Month VII, year 12``.
    return without_chronology(value[month.start():])


def parser_line(value: str) -> str:
    """Remove a printed line label while retaining the scholarly line text.

    Mathpix correctly preserves edition line numbers such as ``2`` and ``10'``
    at the left edge of the English column. They are provenance, not words in
    the diary. The geometry ledger keeps the untouched normalized line; only
    the reflowed parser input drops this anchored label.
    """
    return re.sub(
        r"^\s*(?:\(?[a-z]+\.\s*)?\d{1,3}[\u2032'\u00b4\u2019]?\s+", "", value,
        count=1, flags=re.IGNORECASE).strip()


def converted_lines(page: dict) -> list[dict]:
    """Flatten Mathpix objects while retaining page geometry and stable IDs."""
    rows = []
    for line in page["lines"]:
        raw = line.get("text", "").strip()
        if not line.get("conversion_output") or not raw:
            continue
        region = line["region"]
        rows.append({
            "pdf_page": int(page["page"]),
            "printed_page": int(page["page"]) - PRINTED_OFFSET,
            "source_line_id": line.get("id", ""),
            "ocr_line": int(line.get("line", 0)),
            "x": int(region["top_left_x"]),
            "y": int(region["top_left_y"]),
            "width": int(region["width"]),
            "height": int(region["height"]),
            "mathpix_type": line.get("type", ""),
            "raw_text": raw,
            "display_text": basic_text(raw),
        })
    return rows


def apply_corrections(rows: list[dict]) -> list[dict]:
    """Apply the checked OCR corrections, each to exactly one line of its page."""
    for fix in pd.read_csv(CORRECTIONS).itertuples():
        hits = [row for row in rows if row["pdf_page"] == fix.pdf_page
                and fix.ocr_text in row["raw_text"]]
        if len(hits) != 1:
            raise RuntimeError(
                f"OCR correction for PDF page {fix.pdf_page} matches "
                f"{len(hits)} lines, not one: {fix.ocr_text!r}")
        row = hits[0]
        row["raw_text"] = row["raw_text"].replace(fix.ocr_text, fix.printed_text, 1)
        row["display_text"] = basic_text(row["raw_text"])
    return rows


def find_headers(rows: list[dict]) -> list[dict]:
    """Find entry starts from matching left/right catalogue headings."""
    headers = []
    by_page: dict[int, list[dict]] = {}
    for row in rows:
        by_page.setdefault(row["pdf_page"], []).append(row)
    for page, page_rows in sorted(by_page.items()):
        left = [row for row in page_rows if row["x"] < 1000]
        right = [row for row in page_rows if row["x"] >= 1000]
        for candidate in left:
            key = catalogue_key(candidate["display_text"])
            if not CATALOGUE.match(key):
                continue
            match = next((row for row in right
                          if catalogue_key(row["display_text"]) == key
                          and abs(row["y"] - candidate["y"]) < 80), None)
            if match is None:
                continue
            # Mathpix occasionally emits the same heading twice as overlapping
            # text/table objects. The duplicates are not necessarily adjacent
            # in its object order, so check all headings already found here.
            if any(existing["pdf_page"] == page
                   and existing["museum_no"] == key
                   and abs(existing["y"] - match["y"]) < 10
                   for existing in headers):
                continue
            headers.append({
                "pdf_page": page,
                "printed_page": page - PRINTED_OFFSET,
                "y": match["y"],
                "museum_no": key,
                "source_line_id": match["source_line_id"],
                "position": page * 10_000 + match["y"],
            })
    return sorted(headers, key=lambda row: row["position"])


FOREIGN_HEADING = re.compile(
    r"\b(?:BM|Rm|MNB|VAT|DT|K|Sp|SH)\s*\d{3,6}(?:\s*\+\s*\d{3,6})*",
    re.IGNORECASE)


def foreign_heading(text: str, own: str) -> int | None:
    """Offset of the first catalogue heading belonging to another tablet."""
    mine = re.sub(r"\s+", " ", own or "").strip().lower()
    for match in FOREIGN_HEADING.finditer(text or ""):
        if re.sub(r"\s+", " ", match.group(0)).strip().lower() != mine:
            return match.start()
    return None


def safe_id(museum_no: str, serial: int) -> str:
    """Make a readable unique identifier without importing catalogue order."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", museum_no).strip("_")
    return f"ADART_IV_D_{serial:04d}_{cleaned}"


def main() -> int:
    source_sha256 = json.loads(MANIFEST.read_text(encoding="utf-8"))["source_sha256"]
    pages = json.loads(OCR_LINES.read_text(encoding="utf-8"))["pages"]
    if len(pages) != 669 or [page["page"] for page in pages] != list(range(1, 670)):
        raise RuntimeError("Mathpix line OCR does not contain exactly pages 1..669")
    rows = apply_corrections([row for page in pages[PDF_FIRST - 1:PDF_LAST]
                              for row in converted_lines(page)])
    headers = find_headers(rows)
    if not headers:
        raise RuntimeError("no paired ADART IV catalogue headings found")

    positions = [header["position"] for header in headers]
    tablet_ids = [safe_id(header["museum_no"], i)
                  for i, header in enumerate(headers, start=1)]
    line_rows = []
    editorial: dict[str, bool] = {tablet: False for tablet in tablet_ids}
    for row in sorted(rows, key=lambda item: (item["pdf_page"], item["y"], item["x"])):
        if row["x"] < 1000:
            continue
        position = row["pdf_page"] * 10_000 + row["y"]
        index = bisect_right(positions, position) - 1
        if index < 0:
            continue
        tablet = tablet_ids[index]
        header = headers[index]
        display = row["display_text"]
        normalized = normalize_translation(display)
        role = "translation"
        if (row["source_line_id"] == header["source_line_id"]
                or catalogue_key(display) == header["museum_no"]):
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
            "museum_no": header["museum_no"],
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
    tablet_rows = []
    truncated = 0
    for index, (tablet, group) in enumerate(lines.groupby("tablet", sort=False)):
        header = headers[index]
        admitted = group[group.included_in_translation]
        translation = "\n".join(admitted.normalized_text)
        # Source line breaks are column wraps, not clause boundaries. The line
        # ledger retains every boundary and coordinate for exact audit.
        astronomy = " ".join(
            admitted.loc[admitted.included_in_astronomy, "parser_text"])
        # A heading is normally printed in both columns and is segmented by
        # pairing them. Where the left column is damaged Mathpix sometimes
        # merges the right-hand heading into the body of a translation line, so
        # the pairing fails and the next tablet's text continues inside this
        # entry. Pooling two tablets into one candidate set would silently
        # assume they are contemporary, so the parser-facing text stops at the
        # first such heading. The material after it is dropped rather than
        # misattributed, and the full translation and line ledger keep it.
        cut = foreign_heading(astronomy, header["museum_no"])
        if cut is not None:
            astronomy = astronomy[:cut].strip()
            truncated += 1
        # A year formula can wrap from one source line into the next, so the
        # joined text is filtered once more, and then checked.
        astronomy = without_chronology(astronomy)
        if CHRONOLOGY.search(astronomy):
            raise RuntimeError(f"{tablet}: chronology left in the astronomy text")
        chronology = "\n".join(
            admitted.loc[admitted.chronology_present, "normalized_text"])
        tablet_rows.append({
            "tablet": tablet,
            "project": "adart_iv/undated",
            "designation": "D",
            "museum_no": header["museum_no"],
            "pdf_page_start": int(group.pdf_page.min()),
            "pdf_page_end": int(group.pdf_page.max()),
            "printed_page_start": int(group.printed_page.min()),
            "printed_page_end": int(group.printed_page.max()),
            "n_translation_lines": int(len(admitted)),
            "n_translation_chars": len(translation),
            "translation": translation,
            "astronomy_text": astronomy,
            "chronology_text": chronology,
            "source_pdf": str(SOURCE.relative_to(HERE)),
            "source_pdf_sha256": source_sha256,
            "ocr_lines": str(OCR_LINES.relative_to(HERE)),
            "ocr_lines_sha256": sha256(OCR_LINES),
            "mathpix_file_id": json.loads(
                MANIFEST.read_text(encoding="utf-8"))["mathpix_file_id"],
            "segmentation_rule": (
                "paired_catalogue_heading_right_translation_v3_truncated"),
        })
    tablets = pd.DataFrame(tablet_rows)
    print(f"  {truncated} entries truncated at an unpaired catalogue heading")

    if len(tablets) != len(headers) or tablets.tablet.duplicated().any():
        raise RuntimeError("segmentation did not produce one unique row per header")
    # A handful of extremely small fragments are printed without an English
    # translation in the right column. They remain explicit corpus rows with
    # zero admitted lines; silently dropping them would inflate coverage.
    entries_without_translation = tablets.loc[
        tablets.n_translation_lines.eq(0), "museum_no"].tolist()

    OUT_TABLETS.parent.mkdir(parents=True, exist_ok=True)
    tablets.to_parquet(OUT_TABLETS, index=False)
    lines.to_parquet(OUT_LINES, index=False)
    summary = {
        "schema": "adart-iv-undated-corpus-v2-reflow",
        "section": "ADART IV Undated Diaries only",
        "pdf_pages": [PDF_FIRST, PDF_LAST],
        "printed_pages": [PDF_FIRST - PRINTED_OFFSET,
                          PDF_LAST - PRINTED_OFFSET],
        "entries": len(tablets),
        "entries_with_translation": int(tablets.n_translation_lines.gt(0).sum()),
        "entries_without_translation": entries_without_translation,
        "entries_with_internal_chronology": int(
            tablets.chronology_text.str.strip().ne("").sum()),
        "translation_lines": int(lines.included_in_translation.sum()),
        "astronomy_input_lines": int(lines.included_in_astronomy.sum()),
        "editorial_lines_excluded": int(
            lines.role.isin(["editorial_heading", "editorial_excluded"]).sum()),
        "source_pdf": str(SOURCE.relative_to(HERE)),
        "source_pdf_sha256": source_sha256,
        "ocr_lines": str(OCR_LINES.relative_to(HERE)),
        "ocr_lines_sha256": sha256(OCR_LINES),
        "ocr_corrections": str(CORRECTIONS.relative_to(HERE)),
        "ocr_corrections_sha256": sha256(CORRECTIONS),
        "outputs": {
            str(OUT_TABLETS.relative_to(HERE)): sha256(OUT_TABLETS),
            str(OUT_LINES.relative_to(HERE)): sha256(OUT_LINES),
        },
    }
    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
