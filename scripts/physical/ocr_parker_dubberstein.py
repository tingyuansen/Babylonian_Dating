#!/usr/bin/env python3
"""OCR the calendar tables in Parker & Dubberstein (1956).

The 1956 Brown University Press scan is image-only.  The Babylonian-to-Julian
conversion tables occupy PDF pages 43--63 inclusive.  Dense table pages are
best handled here with Tesseract page-segmentation mode 6, which preserves each
printed row as one logical OCR line much more reliably than mode 4.

Output:
    data/source/parker_dubberstein/calendar_ocr.txt

With ``--slots`` the script instead reads which month each thirteen-month year
doubled.  The tables print a second Ululu (U II) and a second Addaru (A II) in
separate columns, so the doubled month is a matter of where the row's seventh
and thirteenth month starts sit on the page, and the plain OCR text loses that.
The month starts themselves come from build_calendar.py's parse of the text
above; Tesseract's word boxes only say which column each one is printed in.

Output with ``--slots``:
    data/source/parker_dubberstein/intercalary_slots.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIRST_PAGE = 43
LAST_PAGE = 63

# The scan is under copyright and not distributed. A reader with the book puts
# it here, and git ignores it.
DEFAULT_PDF = ROOT / "data" / "source" / "parker_dubberstein" / "scan.pdf"
DEFAULT_OUT = (
    ROOT / "data" / "source" / "parker_dubberstein" / "calendar_ocr.txt"
)
DEFAULT_SLOTS = ROOT / "data" / "source" / "parker_dubberstein" / "intercalary_slots.csv"

DATE_TOKEN = re.compile(r"^\D*(\d{1,2})/(\d{1,2})\D*$")

# Thirteen-month rows whose printed column was looked at on a crop of the
# scan.  They are every row where the printed column disagrees with the
# nineteen-year-cycle rule build_calendar.py once used (second Ululu only when
# the BCE year is 9 mod 19), plus a spread of agreeing rows across the pages.
# The reading from word positions must agree with each one.
CHECKED_BY_EYE = {
    # printed under U II where the cycle rule put a second Addaru
    "621 BCE": "VI2", "616 BCE": "VI2", "611 BCE": "VI2", "607 BCE": "VI2",
    "603 BCE": "VI2", "600 BCE": "VI2", "596 BCE": "VI2", "584 BCE": "VI2",
    "574 BCE": "VI2", "564 BCE": "VI2", "546 BCE": "VI2", "537 BCE": "VI2",
    "530 BCE": "VI2", "527 BCE": "VI2", "519 BCE": "VI2", "511 BCE": "VI2",
    # printed under A II where the cycle rule put a second Ululu
    "579 BCE": "XII2", "560 BCE": "XII2", "541 BCE": "XII2",
    "522 BCE": "XII2", "446 BCE": "XII2", "427 BCE": "XII2",
    # agreeing rows
    "598 BCE": "VI2", "550 BCE": "XII2", "465 BCE": "VI2", "414 BCE": "XII2",
    "351 BCE": "VI2", "292 BCE": "XII2", "218 BCE": "VI2", "153 BCE": "XII2",
    "85 BCE": "VI2", "15 BCE": "XII2", "AD 68": "VI2",
}


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"{name!r} was not found on PATH.")


def render_page(pdf: Path, page: int, dpi: int, base: Path) -> Path:
    """Render one PDF page to ``<base>.png`` with pdftoppm."""
    render = subprocess.run(
        [
            "pdftoppm",
            "-f", str(page),
            "-l", str(page),
            "-singlefile",
            "-r", str(dpi),
            "-png",
            str(pdf),
            str(base),
        ],
        capture_output=True,
        text=True,
    )
    if render.returncode != 0:
        raise SystemExit(render.stderr)
    return base.with_suffix(".png")


def page_words(pdf: Path, page: int, dpi: int, psm: int,
               tmp: Path) -> pd.DataFrame:
    """Tesseract word boxes for one page."""
    image = render_page(pdf, page, dpi, tmp / f"page_{page:03d}")
    base = tmp / f"page_{page:03d}_words"
    ocr = subprocess.run(
        ["tesseract", str(image), str(base), "-l", "eng",
         "--psm", str(psm), "tsv"],
        capture_output=True,
        text=True,
    )
    if ocr.returncode != 0:
        raise SystemExit(ocr.stderr)
    words = pd.read_csv(base.with_suffix(".tsv"), sep="\t",
                        quoting=csv.QUOTE_NONE, dtype={"text": str})
    return words[words.text.notna() & (words.conf >= 0)]


def date_lines(words: pd.DataFrame) -> list[dict]:
    """Month/day cells of each OCR line, with each cell's left edge.

    The printed columns are left-aligned, so a cell's left edge says which
    column it is in.
    """
    lines = []
    for _, line in words.groupby(["block_num", "par_num", "line_num"]):
        cells = {}
        for word in line.sort_values("left").itertuples():
            m = DATE_TOKEN.match(word.text.strip())
            if m and 1 <= int(m[1]) <= 12 and 1 <= int(m[2]) <= 31:
                cells.setdefault((int(m[1]), int(m[2])), float(word.left))
        if len(cells) >= 8:
            lines.append(cells)
    return lines


def read_slots(pdf: Path, dpi: int, psm: int) -> pd.DataFrame:
    """Printed column of the doubled month in every thirteen-month year."""
    sys.path.insert(0, str(ROOT))
    from scripts.physical.build_calendar import DEFAULT_OCR, parse_rows
    from tools import years as yr

    rows, _ = parse_rows(DEFAULT_OCR.read_text(encoding="utf-8",
                                               errors="replace"))
    out = []
    with tempfile.TemporaryDirectory(prefix="pd1956_slots_") as tmp_name:
        for page in range(FIRST_PAGE, LAST_PAGE + 1):
            print(f"word boxes page {page}")
            on_page = {int(r["bab_year_astro"]): [tuple(d) for d in r["dates"]]
                       for r in rows if r["source_page"] == page}

            # Rows nineteen years apart share most of their dates, so a line
            # is assigned only to the row it reproduces strictly best.
            matched: dict[int, dict] = {}
            for cells in date_lines(page_words(pdf, page, dpi, psm,
                                               Path(tmp_name))):
                score = sorted(
                    ((len(set(cells) & set(dates)), year)
                     for year, dates in on_page.items()),
                    reverse=True,
                )
                if score[0][0] >= 10 and score[0][0] > score[1][0]:
                    matched.setdefault(score[0][1], cells)

            # Column positions from twelve-month rows read completely.
            full = [
                [matched[y][d] for d in dates]
                for y, dates in on_page.items()
                if len(dates) == 12 and y in matched
                and all(d in matched[y] for d in dates)
            ]
            ululu, tashritu, addaru = (
                float(np.median([x[i] for x in full])) for i in (5, 6, 11)
            )
            step = tashritu - ululu

            for year, dates in on_page.items():
                if len(dates) != 13:
                    continue
                cells = matched.get(year, {})
                # Seventh start: under U II (VI2) or under Tashritu (XII2).
                # Thirteenth start: under Addaru (VI2) or under A II (XII2).
                x7, x13 = cells.get(dates[6]), cells.get(dates[12])
                votes = []
                if x7 is not None:
                    votes.append("VI2" if x7 < tashritu - step / 4 else "XII2")
                if x13 is not None:
                    votes.append("XII2" if x13 > addaru + step / 4 else "VI2")
                slot = votes[0] if votes and len(set(votes)) == 1 else None
                label = yr.year_label(year)
                eye = CHECKED_BY_EYE.get(label)
                if slot is None and eye is None:
                    raise SystemExit(
                        f"{label} (p.{page}): printed column not readable "
                        f"from word positions {votes}; look at the scan and "
                        "add the row to CHECKED_BY_EYE"
                    )
                if slot is not None and eye is not None and slot != eye:
                    raise SystemExit(
                        f"{label}: word positions say {slot}, "
                        f"the scan was read by eye as {eye}"
                    )
                # Positions as fractions of the Ululu-to-Tashritu spacing.
                f7 = (x7 - ululu) / step if x7 is not None else np.nan
                f13 = (x13 - addaru) / step if x13 is not None else np.nan
                print(f"  {label:>8}  {slot or eye:<4}  seventh at {f7:5.2f}"
                      f"  thirteenth at {f13:5.2f}")
                if slot is None:
                    established = "checked by eye"
                elif eye is None:
                    established = "column position"
                else:
                    established = "column position, checked by eye"
                out.append({
                    "bab_year_astro": year,
                    "year_label": label,
                    "pdf_page": page,
                    "printed_slot": slot or eye,
                    "established": established,
                })

    unused = set(CHECKED_BY_EYE) - {row["year_label"] for row in out}
    if unused:
        raise SystemExit(f"CHECKED_BY_EYE names no 13-month row: {unused}")
    return pd.DataFrame(out).sort_values("bab_year_astro", ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument(
        "--slots",
        action="store_true",
        help="write the doubled-month ledger instead of the OCR text",
    )
    args = parser.parse_args()

    require_program("pdftoppm")
    require_program("tesseract")

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}")
        return 1

    if args.slots:
        out = args.out or DEFAULT_SLOTS
        slots = read_slots(args.pdf, args.dpi, args.psm)
        out.parent.mkdir(parents=True, exist_ok=True)
        slots.to_csv(out, index=False)
        print()
        print(f"written to {out}")
        print(slots.printed_slot.value_counts().to_string())
        print(slots.established.value_counts().to_string())
        return 0

    args.out = args.out or DEFAULT_OUT
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pages: list[str] = []

    with tempfile.TemporaryDirectory(prefix="pd1956_ocr_") as tmp_name:
        tmp = Path(tmp_name)

        for page in range(FIRST_PAGE, LAST_PAGE + 1):
            number = page - FIRST_PAGE + 1
            total = LAST_PAGE - FIRST_PAGE + 1
            print(f"OCR page {page} ({number}/{total})")

            image_file = render_page(args.pdf, page, args.dpi,
                                     tmp / f"page_{page:03d}")

            text_base = tmp / f"page_{page:03d}_ocr"
            text_file = text_base.with_suffix(".txt")

            ocr = subprocess.run(
                [
                    "tesseract",
                    str(image_file),
                    str(text_base),
                    "-l", "eng",
                    "--psm", str(args.psm),
                ],
                capture_output=True,
                text=True,
            )
            if ocr.returncode != 0:
                print(ocr.stderr)
                return ocr.returncode

            if not text_file.exists():
                print(f"Tesseract did not create {text_file}")
                return 1

            text = text_file.read_text(encoding="utf-8", errors="replace")
            pages.append(f"\n\n===== PDF PAGE {page} =====\n\n{text}")

    output = "".join(pages)
    args.out.write_text(output, encoding="utf-8")

    print()
    print(f"written to {args.out}")
    print(f"OCR characters: {len(output):,}")
    print(f"Tesseract page-segmentation mode: {args.psm}")

    if len(output) < 50_000:
        print("WARNING: OCR output is unexpectedly short.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
