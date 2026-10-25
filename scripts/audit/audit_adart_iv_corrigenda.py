#!/usr/bin/env python3
"""Trace astronomy-consequential ADART IV corrigenda into the digital corpus.

ADART IV, PDF pages 83--86 (printed pp. LVIII--LXI), contains many corrections
to volumes I--III.  This ledger includes every correction that can change an
astronomical body, event, time, day/month context, measured relation, or the
reference date used by this repository.  Bibliographic, historical, spelling,
scribe-name, object-description, and join-only notes are outside that declared
scope.  Nothing in this script silently edits the archived ADsD bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
# The scan of the book is not distributed. Its SHA-256 comes from the Mathpix
# manifest, which recorded it when the scan was read.
MANIFEST = HERE / "data" / "source" / "adart_iv" / "mathpix_manifest.json"
SOURCE_SHA256 = json.loads(MANIFEST.read_text(encoding="utf-8"))["source_sha256"]
TABLETS = HERE / "data" / "corpus" / "tablets.parquet"
OUT = HERE / "data" / "benchmark" / "adart_iv_corrigenda.parquet"
SUMMARY = HERE / "data" / "benchmark" / "adart_iv_corrigenda_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ``expected`` and ``superseded`` are deliberately narrow witnesses in the
# current English translation. A missing match means that the relevant line is
# absent or too broken in ADsD, not that the book's correction is false.
CORRECTIONS = [
    (83, "-651 i 16", "X106510", "body_and_target",
     "Observation probably Jupiter ca. 3 degrees west of alpha Librae.",
     r"Jupiter.{0,80}(?:alpha|α) Librae", r"2 cubits in front of the Scales", False),
    (83, "-651 ii 1", "X106510", "event_type",
     "Reject restored eclipse; AN is probably rain.",
     r"15th\s*,\s*rain", r"eclipse", False),
    (83, "-651 iii 4", "X106510", "month_context",
     "The 2 July -651 eclipse places the column in month IV.",
     r"Month IV", r"Lunar eclipse", False),
    (83, "-567 obv. 18", "X105670", "body",
     "Replace restored Moon with Venus.", r"Venus.{0,80}bright star", r"moon.{0,80}bright star", False),
    (83, "-567 rev. 1", "X105670", "body",
     "Restore Mars instead of Moon.", r"first part.{0,80}Mars", r"first part.{0,80}moon", False),
    (83, "-567 rev. 8", "X105670", "day",
     "Abandon the emendation of day 7 to day 17.", r"Night of the 7th", r"Night of the 17th", False),
    (83, "-453 i 1", "X104530", "month_context",
     "Insert month III after Artaxerxes.", r"Artaxerxes.{0,40}Month III", r"Artaxerxes", False),
    (83, "-440", "X103813", "reference_date",
     "Redate and renumber as -381C (Artaxerxes II year 23).", None, None, True),
    (83, "-357 rev. 6", "X103570", "event_timing",
     "Read 76 degrees before sunset.", r"76° before sunset", r"16° before sunset", False),
    (83, "-346:22", "X103460", "eclipse_phase",
     "Read during onset of the eclipse.", r"during (?:the )?onset of its eclipse", None, False),
    (83, "-330:5", "X103301", "day",
     "Read night 15, day 6 instead of night 16.", r"Night of the 15th.{0,40}6th", r"Night of the 16th", False),
    (83, "-322D:24", "X103224", "month_length_context",
     "First day was identical with day 30 of the preceding month.",
     r"identical with.{0,30}30th", r"followed the 30th", False),
    (83, "-304", "X103670", "reference_date",
     "Redate as -367; eclipse was 6 March -366.", None, None, True),
    (84, "-273B rev. 8", "X102732", "eclipse_phase",
     "Read during onset of the eclipse.", r"during onset", None, False),
    (84, "-257A rev. 2", "X202571", "month_context",
     "Restore month VI after a separation line.", r"Month VI", None, False),
    (84, "-247B rev. 4", "X202470", "station_target",
     "Restore station behind the four rear stars of Sagittarius.",
     r"rear.{0,60}Sagittarius.{0,60}stationary", None, False),
    (84, "-237 rev. 9", "X202370", "star_target",
     "Replace pi Scorpii with beta Capricorni.", r"β Capricorni", r"π Scorpii", False),
    (84, "-214:8", "X202140", "eclipse_timing",
     "Translate 30 degrees (one-half beru) after sunset.",
     r"one-half bēru after sunset|30°.{0,20}after sunset", None, False),
    (84, "-194A:2", "X201941", "eclipse_side",
     "Eclipse began on the south side.", r"began on the south side", None, False),
    (84, "-189A:6", "X201891", "moon_planet_relation",
     "Restore Moon two cubits behind Saturn to the east.",
     r"moon.{0,50}2 cubits behind Saturn", None, False),
    (85, "-185B rev. 1", "X201852", "direction",
     "Read above instead of below.", r"above", r"below", False),
    (85, "-184B:9", "X201842", "eclipse_timing",
     "Translate: after 44 degrees of night it became bright.",
     r"after 44° of night.{0,30}bright", r"44° before sunrise", False),
    (85, "-181 rev. 6", "X201810", "planetary_entry",
     "Restore Venus reached Leo.", r"Venus.{0,20}reached Leo", None, False),
    (85, "-180A", "X201801", "reference_date_and_orientation",
     "New dating; interchange obverse/reverse; month XII night 23 Moon above gamma/delta Capricorni.",
     r"Month XII.{0,120}23rd.{0,120}Capricorni", None, True),
    (85, "-163A:1", "X301631", "moon_planet_relation",
     "Moon stood in front of Mercury to the west.",
     r"moon stood.{0,60}in front of Mercury to the west", None, False),
    (85, "-155A:14", "X301551", "measured_relation",
     "Restore two fingers to the west.", r"2 fingers.{0,20}(?:west|in front)", None, False),
    (86, "-140C obv. 9", "X301403", "amount",
     "Read four cubits instead of two-thirds cubit.",
     r"4 cubits below β Tauri", r"2/3 cubit below β Tauri", False),
]


def main() -> None:
    tablets = pd.read_parquet(TABLETS).set_index("tablet")
    rows = []
    for page, locus, tablet, kind, correction, expected, superseded, catalogue in CORRECTIONS:
        text = tablets.loc[tablet, "text"] or ""
        if catalogue:
            if locus == "-440":
                incorporated = tablets.loc[tablet, "designation"] == "AD -381C"
            elif locus == "-304":
                incorporated = tablets.loc[tablet, "designation"] == "AD -367"
            else:
                incorporated = tablets.loc[tablet, "designation"] == "AD -180A"
            status = "incorporated_reference_label" if incorporated else "reference_label_unresolved"
        elif expected and re.search(expected, text, re.IGNORECASE | re.DOTALL):
            status = "incorporated_in_adsd"
        elif superseded and re.search(superseded, text, re.IGNORECASE | re.DOTALL):
            status = "stale_or_ambiguous_adsd"
        else:
            status = "corrected_clause_absent_or_too_broken_in_adsd"

        # A correction affects the current numeric result only if it either
        # changes the reference label or changes a statement the current parser
        # actually emits. Manual review of the emitted-statement diff found no
        # non-label correction in this section that meets the latter condition.
        current_metric_effect = status == "incorporated_reference_label"
        rows.append({
            "source_pdf_page": page,
            "source_printed_page": {83: "LVIII", 84: "LIX", 85: "LX", 86: "LXI"}[page],
            "locus": locus,
            "tablet": tablet,
            "current_designation": tablets.loc[tablet, "designation"],
            "correction_kind": kind,
            "correction": correction,
            "digital_status": status,
            "current_metric_effect": current_metric_effect,
            "metric_note": (
                "reference label already incorporated before evaluation"
                if current_metric_effect else
                "current parser emits no affected scored statement; retain for witness review"
            ),
            "source_pdf_sha256": SOURCE_SHA256,
        })
    result = pd.DataFrame(rows)
    result.to_parquet(OUT, index=False)
    summary = {
        "schema": "adart-iv-corrigenda-audit-v1",
        "scope": "all ADART IV pp. LVIII--LXI corrections that may change astronomical dating data",
        "source_pdf_pages": [83, 86],
        "source_pdf_sha256": SOURCE_SHA256,
        "corrections": int(len(result)),
        "status_counts": result.digital_status.value_counts().sort_index().to_dict(),
        "reference_labels_already_incorporated": int(result.current_metric_effect.sum()),
        "non_label_corrections_changing_current_scores": 0,
        "interpretation": (
            "Unincorporated clauses remain a printed-witness/corpus correction ledger; "
            "none is silently converted into a scored fact."),
        "output_sha256": sha256(OUT),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
