#!/usr/bin/env python3
"""Check that the repository is complete and that its results follow from its data.

It checks the Python packages and the data tables, and runs the self-check of
each astronomical tool. It tests the parser on short texts and dates record
X301492 (Fig. 3), whose year is known. Then it checks the whole tree:

- Each SHA-256 that a result records matches the file it names.
- Each summary follows from the rows it summarizes, and each table that a fast
  script writes is the same when that script writes it again.
- The benchmark, the public results, the ranked candidates, the sensitivity
  baseline and the held-out test come from one run, record by record.
- The repository holds only what the READMEs, the Makefile and
  data/columns.csv describe, and each folder README names each file in its
  folder.

Run it from the repository root. ``make check`` runs the tests in tests/ and
then this file:

    python3 tests/verify.py

Each check prints one line, "ok" or "FAIL". Exit code 0 means that all checks
pass. If a check in the first sections fails, such as a missing package or data
table, the run stops there. Any later FAIL means that a committed result, or the
content of the tree, does not match the code and data.

The book scans, and the files that hold text read from ADART IV, are not
distributed. data/source/not_distributed.csv records the SHA-256 of each. A
file that is present is checked against that value, and an absent file is
replaced by that value, with a printed note.
"""

from __future__ import annotations

import ast
import csv
import fnmatch
import importlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

REQUIRED = ["numpy", "pandas", "scipy", "pyarrow", "swisseph", "requests",
            "matplotlib"]
OPTIONAL = {"jplephem": "only needed to rebuild the ephemeris from a kernel"}
# Row counts are fixed here only where the sources or the design fix them.
# The other tables are checked below against the code that writes them or the
# rows that they summarize.
DATA = {
    "corpus/tablets.parquet": 417,
    "corpus/lines.parquet": None,
    "corpus/evidence.parquet": 417,
    "corpus/cohorts.parquet": 417,
    "physical/calendar.parquet": None,
    "physical/ephemeris.parquet": None,
    "physical/eclipses.parquet": 3770,
    "physical/normal_stars.parquet": 32,
    "physical/appulse_residuals.parquet": None,
    "physical/visibility_calibration.parquet": None,
    "clauses/parser_clauses.parquet": None,
    "results/tablet_dates.parquet": None,
    "results/adart_iv_dates.parquet": 867,
    "benchmark/benchmark_records.parquet": 417,
    "benchmark/benchmark_results.parquet": 179,
    "benchmark/benchmark_clause_bits.parquet": None,
    "benchmark/ranking_candidates.parquet": None,
    "benchmark/sensitivity_runs.parquet": None,
    "benchmark/sensitivity_summary.parquet": None,
    "benchmark/held_out_runs.parquet": None,
    "benchmark/era_readings.parquet": None,
    "benchmark/adart_iv_corrigenda.parquet": None,
}

# The benchmark was fixed before any candidate set was compared with a
# published date. These values pin its membership, so no rebuild can move a
# record in or out. Each is the SHA-256 of sorted record identifiers, one per
# line. The first covers the 179 benchmark records, and the second the 112 of
# them that had a decisive clause when the benchmark was fixed.
FROZEN_ELIGIBLE_IDS_SHA256 = (
    "d2914bbc371a2faabc30a7a03320b8f112739161cbbc0a38c0ea63b2b4e04fae")
FROZEN_YEAR_BEARING_IDS_SHA256 = (
    "e39b6845256bb10c9597ec17635e831efc124bae7e1d2aaf77f5dfff7a8f76be")

# What may sit at the top of the tree.
TOP_LEVEL = {"README.md", "LICENSE", "CITATION.cff", "Makefile", "requirements.txt",
             ".gitignore", "data", "docs", "scripts", "skills",
             "tests", "tools"}
TRANSIENT_DIRS = {"tmp", "output", ".pytest_cache", ".mypy_cache",
                  ".ruff_cache", ".ipynb_checkpoints", ".cache"}
# The folders that have a README, which names each file in the folder.
FOLDERS = ("data", "docs", "scripts", "skills", "tests", "tools")
# Markdown files besides the README at the root.
DOCUMENTS = {f"{folder}/README.md" for folder in FOLDERS} | {"skills/dating_protocol.md"}
FILE_TOKEN = re.compile(
    r"[\w./+-]+\.(?:py|parquet|jsonl?|csv|md|tex|pdf|png|jpg|ttf|txt|zip|bib"
    r"|mmd|sty|cls|bst|ldf)")

fails: list[str] = []
warns: list[str] = []


def head(t: str) -> None:
    print(f"\n{t}\n{'-' * len(t)}")


def ok(label: str, good: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if good else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not good:
        fails.append(label)
    return good


def not_distributed() -> dict[str, str]:
    """The files the public tree leaves out, with their recorded SHA-256.

    These are the book scans, and the files that hold text read from ADART IV.
    Each is a row of data/source/not_distributed.csv, which names the file by
    its path from the root and gives its SHA-256.
    """
    table = HERE / "data" / "source" / "not_distributed.csv"
    if not table.exists():
        return {}
    with table.open(encoding="utf-8", newline="") as handle:
        return {row["path"]: row["sha256"] for row in csv.DictReader(handle)}


NOT_DISTRIBUTED = not_distributed()
_digests: dict[str, str] = {}


def digest(rel: str) -> str:
    """SHA-256 of a file in the tree, by its path from the root.

    A file that is not distributed gives the value that
    data/source/not_distributed.csv records for it. Hashes recorded elsewhere
    are then checked against this one value. Any other missing file gives
    "missing", which matches no recorded hash.
    """
    if rel not in _digests:
        path = HERE / rel
        if path.is_file():
            _digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif rel in NOT_DISTRIBUTED:
            warns.append(f"{rel} not distributed")
            _digests[rel] = NOT_DISTRIBUTED[rel]
        else:
            _digests[rel] = "missing"
    return _digests[rel]


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def frames_equal(a, b) -> str:
    """Empty if two tables hold the same values, else the first difference."""
    import pandas as pd
    try:
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-12)
        return ""
    except AssertionError as exc:
        return " ".join(str(exc).split())[:160]


def same_json(a, b) -> bool:
    """Equal JSON values, with floats equal to a relative 1e-9."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))
    return a == b


def first_difference(a, b, path: str = "") -> str:
    """Where two JSON values first differ, or empty if they agree."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}/{key} is on one side only"
            found = first_difference(a[key], b[key], f"{path}/{key}")
            if found:
                return found
        return ""
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{index}]")
            if found:
                return found
        return ""
    return "" if same_json(a, b) else f"{path or 'the value'} differs"


def plain(value):
    """A value as JSON would store it, so that numpy types compare."""
    return json.loads(json.dumps(value, default=lambda v: v.item()))


def ignore_patterns() -> list[str]:
    path = HERE / ".gitignore"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [line.strip() for line in lines
            if line.strip() and not line.lstrip().startswith("#")]


def ignored(rel: str, patterns: list[str]) -> bool:
    """Whether .gitignore keeps a path out of the repository."""
    parts = rel.split("/")
    for pattern in patterns:
        bare = pattern.strip("/")
        if "/" in bare:
            if any(fnmatch.fnmatch("/".join(parts[:i]), bare)
                   for i in range(1, len(parts) + 1)):
                return True
        elif any(fnmatch.fnmatch(part, bare) for part in parts):
            return True
    return False


def tree() -> tuple[list[str], list[str]]:
    """Every file the repository carries, and the directories that hold them.

    In a git checkout these are the files git tracks or would add, so that
    nothing git ignores, such as a local tool folder, counts. Without git, the
    files are those that .gitignore does not exclude. Paths are from the root.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=HERE, capture_output=True, check=True).stdout.decode("utf-8")
        files = sorted({rel for rel in listed.split("\0")
                        if rel and (HERE / rel).is_file()})
    except (OSError, subprocess.CalledProcessError):
        patterns = ignore_patterns()
        files = []
        for root, subdirs, names in os.walk(HERE):
            base = Path(root).relative_to(HERE).as_posix()
            prefix = "" if base == "." else base + "/"
            subdirs[:] = sorted(d for d in subdirs if d != ".git"
                                and not ignored(prefix + d, patterns))
            files += [prefix + n for n in names
                      if n != ".git" and not ignored(prefix + n, patterns)]
        files.sort()
    dirs = sorted({"/".join(rel.split("/")[:depth]) for rel in files
                   for depth in range(1, rel.count("/") + 1)})
    return files, dirs


def zodiac_guard_violations() -> list[str]:
    """The source files that bypass or refit the zodiac relation of Britton (Eq. 8).

    A hit is a file that names another zodiac offset or a zodiac fitted to the
    diaries. A file that takes a sign directly from a longitude of date is also
    a hit. tools/sky.py must keep the functions and the two constants of the
    relation.
    """
    forbidden = [
        "ARIES" + "0",
        "aries" + "0",
        "--" + "aries" + "0",
        "zodiac_" + "23",
        "zodiac_" + "27",
        "calibrate_" + "zodiac",
    ]
    roots = [HERE / "tools", HERE / "scripts", HERE / "skills"]
    files = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(root.rglob("*.py"))
        files.extend(root.rglob("*.md"))
    files.append(HERE / "README.md")
    direct_tropical_sign = re.compile(
        r"sign_index\s*\(\s*(?:sky\.)?lon_date\s*\("
    )
    hits = []
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        reasons = []
        if any(token in text for token in forbidden):
            reasons.append("legacy/fitted zodiac token")
        if path.suffix == ".py" and direct_tropical_sign.search(text):
            reasons.append("direct sign_index(lon_date(...))")
        if reasons:
            hits.append(
                f"{path.relative_to(HERE)} ({'; '.join(reasons)})"
            )

    sky_source = (HERE / "tools" / "sky.py").read_text(encoding="utf-8")
    for marker in (
        "def babylonian_offset_deg(",
        "def babylonian_lon(",
        "BABYLONIAN_OFFSET_YEAR0_DEG = 3.20",
        "BABYLONIAN_PRECESSION_DEG_PER_CENTURY = 1.3828",
    ):
        if marker not in sky_source:
            hits.append(f"tools/sky.py (missing {marker})")
    return sorted(set(hits))


def main() -> int:
    t0 = time.time()
    print("Babylonian_Dating, cold start verification")

    head("zodiac-frame migration")
    zodiac_violations = zodiac_guard_violations()
    ok("published Babylonian zodiac is not bypassed or corpus-fitted",
       not zodiac_violations, ", ".join(zodiac_violations))

    head("environment")
    for m in REQUIRED:
        try:
            importlib.import_module(m)
            ok(m, True)
        except ImportError:
            ok(m, False, "pip install -r requirements.txt")
    for m, why in OPTIONAL.items():
        try:
            importlib.import_module(m)
            print(f"  ok    {m}")
        except ImportError:
            print(f"  --    {m} absent, {why}")
            warns.append(m)

    head("data present")
    import pandas as pd
    for name, rows in DATA.items():
        p = HERE / "data" / name
        if not p.exists():
            ok(name, False, "missing, see README for how to rebuild")
            continue
        n = len(pd.read_parquet(p))
        ok(name, rows is None or n == rows, f"{n:,} rows")

    if fails:
        print("\nEnvironment or data incomplete. Fix that before the rest.")
        return 1

    scratch = tempfile.TemporaryDirectory(prefix="verify_")
    try:
        return audit(t0, Path(scratch.name))
    finally:
        scratch.cleanup()


def audit(t0: float, scratch: Path) -> int:
    import pandas as pd

    # Tables that a script writes in a few seconds are written again into a
    # scratch directory while the other checks run. They are compared at the end.
    # Each job lists the files it writes, and {0}, {1}, ... in its command
    # stand for their places in the scratch directory.
    rebuilds = [
        (["data/physical/calendar.parquet", "data/physical/calendar_ocr_corrections.csv"],
         ["scripts/physical/build_calendar.py", "--out", "{0}",
          "--corrections-out", "{1}"]),
        (["data/physical/normal_stars.parquet"],
         ["scripts/physical/build_normal_stars.py", "--out", "{0}"]),
        (["data/physical/appulse_residuals.parquet"],
         ["scripts/physical/appulse_residuals.py", "--out", "{0}"]),
        (["data/physical/visibility_calibration.parquet"],
         ["scripts/physical/calibrate_visibility.py", "--out", "{0}"]),
        (["data/corpus/cohorts.parquet"],
         ["scripts/source/build_cohort.py", "--out", "{0}"]),
        (["data/benchmark/era_readings.parquet"],
         ["scripts/audit/audit_era_readings.py", "--out", "{0}"]),
        (["data/clauses/parser_clauses.parquet"],
         ["scripts/audit/build_parser_clauses.py", "--out", "{0}"]),
    ]
    running = [
        (outputs, subprocess.Popen(
            [sys.executable, *(argument.format(*(str(scratch / Path(rel).name)
                                                 for rel in outputs))
                               for argument in command)],
            cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, env=dict(os.environ, MPLBACKEND="Agg")))
        for outputs, command in rebuilds]

    head("calendar and era arithmetic")
    from tools import cal
    from tools import years as yr
    ok("tools/cal.py self-check", cal.check())
    calendar = pd.read_parquet(HERE / "data" / "physical" / "calendar.parquet")
    ok("calendar search domain is complete",
       set(calendar.bab_year_astro.astype(int)) == set(int(y) for y in yr.candidate_years())
       and calendar.groupby("bab_year_astro").size().isin([12, 13]).all())
    from tools import eras
    ok("tools/eras.py self-check", eras.check())
    era_readings = pd.read_parquet(HERE / "data" / "benchmark" / "era_readings.parquet")
    named_se = era_readings[era_readings.system.eq("named_seleucid_era")]
    ok("named Seleucid-ruler chronology is sourced and separately gated",
       len(named_se) > 0
       and named_se.forward_agree.isna().eq(named_se.published_astro.isna()).all())
    from scripts.source.build_corpus import split_date
    ok("catalogue abbreviated BCE spans are validated",
       split_date("150/49") == (150, 149)
       and split_date("73/3", "-73") == (74, 73)
       and split_date("73/3") == (None, None))
    from scripts.audit.build_benchmark import check_blind
    try:
        check_blind()
        blind_code = True
    except RuntimeError:
        blind_code = False
    ok("the dating code reads no published date or catalogue field", blind_code)
    manifest_path = HERE / "data" / "source" / "adsd" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_ok = True
    for part, record in manifest["files"].items():
        path = HERE / "data" / "source" / "adsd" / f"{part}.zip"
        raw_ok &= (path.exists() and path.stat().st_size == record["bytes"]
                   and hashlib.sha256(path.read_bytes()).hexdigest()
                   == record["sha256"])
    ok("raw ADsD archives match the fresh fetch manifest", raw_ok,
       manifest["fetched_utc"])
    if not all(record["tls_verified"] for record in manifest["files"].values()):
        warns.append("ADsD TLS authenticity not independently verified")

    head("ephemeris, positions and derived events")
    from tools import sky
    ok("tools/sky.py self-check", sky.check())

    from tools import eclipses
    ok("tools/eclipses.py self-check", eclipses.check())

    head("the calendar against the Moon")
    # A Babylonian month begins at first crescent, so each calendar month
    # start should fall a day or two after a computed new moon. The
    # calendar comes from the printed tables of Parker and Dubberstein, and the
    # new moons come from JPL DE441. The two sources are independent.
    import numpy as np
    sys.path.insert(0, str(HERE / "scripts"))
    from scripts.physical.check_calendar_against_sky import conjunctions, to_jd
    c = pd.read_parquet(HERE / "data" / "physical" / "calendar.parquet")
    conj = conjunctions(pd.read_parquet(HERE / "data" / "physical" / "ephemeris.parquet"))
    jd = np.array([to_jd(y, m, d) for y, m, d in
                   zip(c.julian_year_astro, c.julian_month, c.julian_day)], float)
    k = np.searchsorted(conj, jd) - 1
    lag = jd[k >= 0] - conj[k[k >= 0]]
    med = float(np.median(lag))
    ok("month starts follow a conjunction", 1.0 < med < 2.5,
       f"median lag {med:.2f} d over {len(lag):,} months")

    head("Normal Stars against modern positions")
    for name, want in [("α Tauri", -5.47), ("α Leonis", 0.47),
                       ("β Geminorum", 6.68)]:
        got = sky.star(name)
        ok(f"{name} ecliptic latitude", got is not None
           and abs(got[1] - want) < 0.5,
           f"{got[1]:+.2f} against {want:+.2f}" if got else "not catalogued")

    head("dating a tablet whose answer is known")
    from tools import statements as S
    tb = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet")
    r = tb[tb.tablet == "X301492"].iloc[0]

    # Short texts that test the rules of the parser for headings,
    # restorations and clause types (SI Appendix, section S2), and some
    # verdicts. The first test puts Month I after Month XII in the next
    # Babylonian year. A day heading of the old month must not carry across
    # that boundary.
    context_text = ("Month XII, The 5th, Jupiter was in Pisces. "
                    "Month I, Mars was in Aries.")
    pos = context_text.index("Mars")
    ok("statement context crosses the year and resets the day",
       S.governing_context(context_text, pos) == ("I", None, 1))
    first_day = ("Month XI, the 1st (of which followed the 30th of the "
                 "preceding month), the moon stood 1 cubit behind Saturn.")
    pos = first_day.index("moon")
    ok("the preceding month's day 30 does not replace day 1",
       S.governing_context(first_day, pos) == ("XI", 1, 0))
    ok("a restored split Roman month is recombined",
       S.months_in("Month V [ II ], Night of the 8th") == ["VII"])
    ok("a split Month word is recombined",
       S.months_in("[ Mon ] th XI, the 1st") == ["XI"])
    lower_month = S.extract(
        "month VII, the 3rd, the moon was 1 cubit behind α Tauri.")[0]
    ok("lowercase month headings govern preserved observations",
       lower_month.month == "VII" and not lower_month.restored)
    ok("lowercase narrative month references are not headings",
       S.months_in(
           "Diary from month VII to the end of month XII.") == [])
    split_month = S.extract(
        "mon [ th VII, the 3rd, the moon was 1 cubit behind α Tauri.")[0]
    ok("a split restored month numeral is recognized and propagated",
       split_month.month == "VII" and split_month.restored)
    restored_month = S.extract(
        "[ Month IX, the 3rd, the moon was 1 cubit behind α Tauri.")[0]
    ok("a governing restored month numeral withholds decisive evidence",
       restored_month.month == "IX" and restored_month.restored)
    preserved_numeral = S.extract(
        "[ Month ] III, the 3rd, the moon was 1 cubit behind α Tauri.")[0]
    ok("a preserved month numeral is not downgraded with its label",
       preserved_numeral.month == "III" and not preserved_numeral.restored)
    restored_copula = S.extract(
        "Month X, the 3rd, the moon [stood] 2 cubits behind Saturn.")[0]
    ok("a restored copula does not downgrade preserved measured fields",
       not restored_copula.restored)
    restored_coordination = S.extract(
        "Month VIII. [ Jupiter and Mars were in Libra ].")
    ok("restoration spans propagate through coordinated summaries",
       len(restored_coordination) == 2
       and all(st.restored for st in restored_coordination))
    scorpio_alias = S.extract("Month X. Mars reached Scorpius.")
    ok("the edition's Scorpius spelling normalizes to Scorpio",
       len(scorpio_alias) == 1
       and scorpio_alias[0].kind == "entry"
       and scorpio_alias[0].sign == "Scorpio")
    opening_summary = (
        "Saturn was in the end of Pisces; Mars, in the beginning of the "
        "month, was in Virgo, at the end of the month, in Libra. That month, "
        "the river rose. Month VIII, the 1st.")
    opening_sts = S.extract(opening_summary)
    ok("an opening monthly summary inherits the preceding month",
       len(opening_sts) == 3
       and all(st.month == "VII" for st in opening_sts)
       and {(st.planet, st.sign) for st in opening_sts}
       == {("saturn", "Pisces"), ("mars", "Virgo"), ("mars", "Libra")})
    adjacent_summary = "Mars was in Cancer. [ ... ] Month II, the 1st."
    adjacent_st = S.extract(adjacent_summary)[0]
    ok("an adjacent pre-heading summary inherits the preceding month",
       adjacent_st.month == "I")
    missing_heading = (
        "Month II, the 1st. [ ... many damaged entries ... ] At that time, "
        "Jupiter was in Sagittarius. [ Mon ] th XI, the 1st.")
    missing_st = S.extract(missing_heading)[0]
    ok("an adjacent summary overrides a stale older month heading",
       missing_st.month == "X")
    before_marker = (
        "That month, the equivalent was barley. Saturn was in Leo; Mars was "
        "in Aquarius, until the end of the month, in Pisces. Month III.")
    before_sts = S.extract(before_marker)
    ok("a nearby summary marker and monthly path precede the next heading",
       len(before_sts) == 3
       and all(st.month == "II" for st in before_sts)
       and {(st.planet, st.sign) for st in before_sts}
       == {("saturn", "Leo"), ("mars", "Aquarius"), ("mars", "Pisces")})
    coordinated = S.extract(
        "At that time, Jupiter and Mercury were in Cancer. That month, the "
        "river fell. Month IV.")
    ok("coordinated planets in one sign are both retained",
       len(coordinated) == 2
       and all(st.month == "III" for st in coordinated)
       and {st.planet for st in coordinated} == {"jupiter", "mercury"})
    missing_month = S.extract(
        "Month VIII, The 25th, clouds. [ ... ] Night of the 4th, clouds. "
        "At that time, Jupiter and Mars were in Capricorn.")
    ok("a day rollback marks a lost month rather than reusing a stale heading",
       len(missing_month) == 2
       and all(st.month is None and st.restored for st in missing_month))
    intercalary_prior = S.extract(
        "Saturn and Mars were in Libra. Month VII, the 1st.")
    ok("a pre-VII summary retains the VI/intercalary-VI ambiguity",
       len(intercalary_prior) == 2
       and all(st.month == "VI|VI2" for st in intercalary_prior)
       and S.check(intercalary_prior[1], yr.bce_to_astro(137), ["VII"]).result == "agree")
    triple = S.extract(
        "Venus, Mercury and Mars were in Virgo. That month, prices changed. "
        "Month VII.")
    ok("three coordinated planets are all retained",
       len(triple) == 3 and {st.planet for st in triple}
       == {"venus", "mercury", "mars"})
    explicit_years = ("Year 212, Month VIII, Jupiter was in Pisces. "
                      "Year 213, Month I, Mars was in Aries. "
                      "Year 214, Month I, Saturn was in Aquarius.")
    pos = explicit_years.index("Saturn")
    ok("explicit year headings control relative year context",
       S.governing_context(explicit_years, pos) == ("I", None, 2))
    ok("Unicode intercalary month is normalized and usable",
       S.months_in("Month XII₂, The 14th") == ["XII2"]
       and cal.month_start_jd(yr.bce_to_astro(99), "XII2") is not None)
    eclipse_st = S.extract(
        "Month I, Night of the 14th, lunar eclipse which was omitted.")[0]
    ok("an omitted eclipse is never matched as an observation",
       eclipse_st.omitted
       and S.check(eclipse_st, yr.bce_to_astro(150), ["I"]).result == "untestable")
    unseen_solar = S.extract(
        "Month V, The 28th, solar eclipse; when I watched I did not see it.")[0]
    ok("an unwitnessed solar eclipse remains non-observational",
       unseen_solar.omitted
       and S.check(unseen_solar, yr.bce_to_astro(150), ["V"]).result
       == "untestable")
    restored_st = S.extract(
        "Month I, [ Night of the 14th, lunar eclipse.")[0]
    ok("a restored eclipse day is not decisive", restored_st.restored)
    repeated = S.extract(
        "Month I, Mars was in Aries. Month II, Mars was in Aries.")
    ok("repeated observations are retained", len(repeated) == 2)
    moving = S.extract(
        "Month X, Night of the 3rd, the moon [stood] 2 cubits "
        "in front of Saturn to the west.")
    ok("a measured Moon-to-planet appulse is extracted",
       len(moving) == 1
       and moving[0].kind == "planet_appulse"
       and moving[0].target_planet == "saturn")
    ok("the Moon-Saturn appulse discriminates the known year",
       S.check(moving[0], yr.bce_to_astro(97), ["X"]).result == "agree"
       and S.check(moving[0], yr.bce_to_astro(443), ["X"]).result == "contradict")
    compound_star = S.extract(
        "Month I, The 3rd, the moon was 1 cubit 20 fingers behind α Tauri.")
    ok("compound cubit-plus-finger Normal-Star distance is exact",
       len(compound_star) == 1
       and compound_star[0].parser_rule == "measured_normal_star_compound"
       and abs(compound_star[0].amount - (1 + 20 / 24)) < 1e-12
       and abs(compound_star[0].offset - 2 * (1 + 20 / 24)) < 1e-12
       and compound_star[0].unit == "cubit+finger")
    compound_planet = S.extract(
        "Month I, The 3rd, the moon stood 1 cubit 8 fingers in front of Mars.")
    ok("compound unit grammar is shared with Moon-to-planet measurements",
       len(compound_planet) == 1
       and compound_planet[0].parser_rule == "measured_moon_planet_compound"
       and abs(compound_planet[0].offset + 2 * (1 + 8 / 24)) < 1e-12)
    ok("compound grammar does not admit a headless fragment",
       not S.extract(
           "Month I, The 3rd, 1 cubit 8 fingers above α Tauri."))
    station_relation = S.extract(
        "Month VII, around until the 13th, when Mars became stationary to "
        "the east, it became stationary 2/3 cubit above η Geminorum.")
    ok("a named station-to-star relation is parsed conservatively",
       len(station_relation) == 1
       and station_relation[0].kind == "station_appulse"
       and station_relation[0].planet == "mars"
       and station_relation[0].station_phase == "east"
       and station_relation[0].day is None
       and not S.is_decisive(station_relation[0]))
    unicode_station = S.extract(
        "Month VII, the 13th, when Jupiter became stationary to the west, "
        "it became stationary 2 1⁄2 cubits behind α Virginis.")
    ok("station grammar normalizes a Unicode fraction without guessing",
       len(unicode_station) == 1
       and abs(unicode_station[0].amount - 2.5) < 1e-12
       and unicode_station[0].station_phase == "west")
    anaphoric_station = S.extract(
        "It became stationary 1 cubit behind α Virginis. Month VIII, "
        "Saturn was in Leo.")
    ok("an anaphoric station subject is not filled from distant context",
       not any(st.kind == "station_appulse" for st in anaphoric_station)
       and len(anaphoric_station) == 1
       and anaphoric_station[0].kind == "in_sign"
       and anaphoric_station[0].planet == "saturn")
    unwatched_station = S.extract(
        "Month II, when Saturn became stationary to the west, it became "
        "stationary 1 1/2 cubits in front of γ Virginis; I did not watch.")
    ok("an unwatched measured station remains untestable",
       len(unwatched_station) == 1 and unwatched_station[0].omitted
       and S.check(unwatched_station[0], yr.bce_to_astro(150), ["II"]).result == "untestable")
    planet_pair = S.extract(
        "Month VII, the 13th, Venus was 10 fingers below Saturn.")
    ok("an explicit planet-to-planet measurement is parsed as supporting",
       len(planet_pair) == 1
       and planet_pair[0].kind == "planet_planet_appulse"
       and planet_pair[0].planet == "venus"
       and planet_pair[0].target_planet == "saturn"
       and abs(planet_pair[0].amount - 10 / 24) < 1e-12
       and not S.is_decisive(planet_pair[0]))
    restored_unit = S.extract(
        "Month VII, the 13th, Venus was 8 [ fingers ] above Jupiter.")
    ok("planet-pair grammar refuses anaphora, and reads a restored unit "
       "only as support",
       not S.extract("Month VII, the 13th, it stood 1 cubit behind Saturn.")
       and len(restored_unit) == 1 and restored_unit[0].restored
       and not S.is_decisive(restored_unit[0]))
    leaked = S.extract(
        "Month XII, The 21st, clouds. [...] the moon stood 1 cubit "
        "behind Jupiter.")
    ok("an appulse does not inherit a day across a lacuna",
       len(leaked) == 1 and leaked[0].day is None)

    res = S.score(r.text, yr.bce_to_astro(150))
    ok("X301492 forward at its published year", res["contradict"] == 0,
       f"{res['agree']} agreed, {res['contradict']} contradicted")

    years = [int(y) for y in cal.candidate_years()]
    sts = [s for s in S.extract(r.text) if s.kind != "appulse"]
    months = S.months_in(r.text)
    scores = S.candidate_scores(sts, months, years)
    tied = list(scores.years[scores.strict])
    evidence_levels = S.evidence_levels(scores, max_levels=3)
    ok("X301492 backward recovers 150 BC uniquely",
       tied == [yr.bce_to_astro(150)]
       and evidence_levels[0]["predicate_exact"]
       and evidence_levels[0]["years_astro"] == [yr.bce_to_astro(150)],
       f"top score {scores.score.max():.2f} bits, {len(tied)} year(s) tie")

    dated = pd.read_parquet(HERE / "data" / "results" / "tablet_dates.parquet")
    ok("every tablet has one public claim record",
       dated.tablet.is_unique and set(dated.tablet) == set(tb.tablet)
       and set(dated.claim) <= {"committed", "ranked", "decline"}
       and dated.reason.fillna("").str.len().gt(0).all())
    ok("date-blind result artifact contains no reference answer",
       not {"bce_first", "bce_second", "ancient_year", "period"}
       .intersection(dated.columns))
    ok("astronomy-only claims obey the public contract",
       dated.claim.isin({"decline", "ranked", "committed"}).all()
       and dated.loc[dated.claim.eq("committed"), "strict_count"].eq(1).all()
       and dated.loc[dated.claim.eq("committed") & dated.basis.eq("independent_astronomy"),
                     "safety_count"].eq(1).all())
    ok("candidate lists and calendar spans are complete",
       dated.apply(lambda row: len(row.strict_years_astro) == row.strict_count
                   and len(row.safety_years_astro) == row.safety_count
                   and len(row.strict_spans) == row.strict_count
                   and len(row.safety_spans) == row.safety_count, axis=1).all())
    evidence_outputs_are_complete = dated.apply(
        lambda row: (
            len(row.top_evidence_group_years_astro)
            == row.top_evidence_group_size
            and len(row.top_three_evidence_groups_years_astro)
            == row.top_three_evidence_groups_size
            and len(json.loads(row.evidence_levels_json))
            == min(3, row.evidence_group_count)
            and sorted({
                year
                for level in json.loads(row.evidence_levels_json)
                for year in level["years_astro"]
            }) == list(row.top_three_evidence_groups_years_astro)
        ), axis=1)
    frozen_records = pd.read_parquet(HERE / "data" / "benchmark" / "benchmark_records.parquet")
    # The benchmark records whose current result keeps a decisive clause.
    active_ids = (set(frozen_records.loc[frozen_records.tool_benchmark_eligible, "tablet"])
                  & set(dated.loc[dated.n_decisive_statements.fillna(0).gt(0), "tablet"]))
    active_public = dated[dated.tablet.isin(active_ids)]
    ok("public results expose complete score levels and exact physical subgroups",
       bool(evidence_outputs_are_complete.all())
       and dated.method_version.eq("date-all-v7-reviewed-clauses").all()
       and dated.loc[dated.tablet.eq("X301492"),
                     "top_evidence_group_years_astro"].iloc[0]
       == [yr.bce_to_astro(150)])
    statements_are_public = dated.apply(
        lambda row: len(json.loads(row.statement_verdicts_json))
        == row.n_statements, axis=1)
    ok("every extracted statement has public candidate verdicts",
       bool(statements_are_public.all()))
    source_units_are_complete = True
    sirius_subjects_are_explicit = True
    for row in dated.itertuples(index=False):
        records = json.loads(row.statement_verdicts_json)
        groups = {}
        for record in records:
            if "scoring_unit" not in record:
                source_units_are_complete = False
                continue
            groups.setdefault(record["scoring_unit"], []).append(record)
            if record["kind"] == "sirius":
                sirius_subjects_are_explicit &= record["planet"] == "sirius"
        source_units_are_complete &= len(groups) == row.n_scoring_units
        decisive_units = sum(any(r["decisive"] for r in group) for group in groups.values())
        source_units_are_complete &= decisive_units == row.n_decisive_statements
        for group in groups.values():
            joint = group[0].get("joint_positions", [])
            if not joint:
                continue
            source_units_are_complete &= {(r["planet"], r["sign"]) for r in group} == {
                tuple(pair) for pair in joint}
            for record in group:
                source_units_are_complete &= record.get("joint_positions") == joint
                for tier in ("strict", "safety"):
                    source_units_are_complete &= record[tier + "_verdict_years_astro"] == group[0][tier + "_verdict_years_astro"]
    ok("coordinated atomic records share complete source-level scoring units",
       source_units_are_complete)
    ok("Sirius is labelled as Sirius in public records", sirius_subjects_are_explicit)
    level_flags_match = dated.apply(
        lambda row: row.top_evidence_group_predicate_exact
        == json.loads(row.evidence_levels_json)[0]["predicate_exact"]
        and row.top_three_evidence_groups_predicate_exact
        == all(level["predicate_exact"]
               for level in json.loads(row.evidence_levels_json))
        and all(level["predicate_exact"]
                == (len(level["physical_groups_years_astro"]) == 1)
                for level in json.loads(row.evidence_levels_json)), axis=1)
    ok("each exact-predicate flag says whether its level is one physical group",
       bool(level_flags_match.all()))
    ok("chronology-only lookup never becomes astronomy success",
       not dated[dated.chronology_relation.eq("lookup_only")]
       .claim.eq("committed").any())
    ok("chronology adjudication is separately labeled",
       dated[dated.combined_basis.eq("chronology_adjudicated")]
       .claim.eq("ranked").all())
    committed = dated[dated.claim == "committed"]
    ok("every commitment names exactly one year",
       committed.strict_count.eq(1).all())
    references = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "bce_first", "bce_second"])
    audited = committed.merge(references, on="tablet")
    audited = audited[audited.bce_first.notna()]
    # A catalogue date such as "150/149" names the Babylonian year that begins
    # in 150 BCE and ends in 149 BCE. In astronomical numbering, the year that
    # ends in bce_second BCE is -bce_second. A commitment may fall in any year
    # from the one that begins in bce_first to that one.
    commitment_agrees = audited.apply(
        lambda row: yr.bce_to_astro(int(row.bce_first))
        <= int(row.strict_years_astro[0])
        <= max(yr.bce_to_astro(int(row.bce_first)), -int(row.bce_second)),
        axis=1)
    ok("every testable commitment agrees with the independent reference",
       bool(commitment_agrees.all()),
       f"{int(commitment_agrees.sum())}/{len(commitment_agrees)} commitments")

    facts = pd.read_parquet(HERE / "data" / "clauses" / "parser_clauses.parquet")
    ok("every fact has exact source and parser provenance",
       facts[["tablet", "statement_index"]].duplicated().sum() == 0
       and facts.source_translation.fillna("").str.len().gt(0).all()
       and facts.parser_rule.fillna("").str.len().gt(0).all()
       and facts.rule_origin.ne("unclassified_rule_origin").all())
    # The verdict of each parser clause at its published year is computed
    # again from the text, and must match the stored verdict.
    text_records = pd.read_parquet(HERE / "data/corpus/tablets.parquet",
                                  columns=["tablet", "text", "bce_first"])
    forward_current = {}
    for tablet in text_records.itertuples(index=False):
        source = tablet.text if isinstance(tablet.text, str) else ""
        for index, st in enumerate(S.extract(source)):
            forward_current[(tablet.tablet, index)] = (
                "not_evaluated" if pd.isna(tablet.bce_first) else
                S.check(st, yr.bce_to_astro(int(tablet.bce_first)),
                        S.months_in(source)).result)
    ok("forward facts reproduce the current deterministic predicates",
       len(forward_current) == len(facts)
       and all(forward_current.get((f.tablet, f.statement_index)) == f.forward_verdict
               for f in facts.itertuples(index=False)))
    ok("Sirius clauses are labelled as Sirius in the parser clauses",
       facts.kind.eq("sirius").any()
       and facts.loc[facts.kind.eq("sirius"), "planet"].eq("sirius").all())
    population = pd.read_parquet(
        HERE / "data" / "benchmark" / "benchmark_records.parquet")
    ok("evaluation selection is outcome-independent and explicit",
       len(population) == 417
       and not population.selection_uses_backward_success.any()
       and population.cohort_source.eq("regex").all()
       and int(population.tool_benchmark_eligible.sum()) == 179
       and int(population.tool_year_bearing.sum()) == 112)
    ok("the frozen benchmark file holds the same 179 records and labels",
       ids_sha256(population.loc[population.tool_benchmark_eligible, "tablet"])
       == FROZEN_ELIGIBLE_IDS_SHA256
       and ids_sha256(population.loc[population.tool_year_bearing, "tablet"])
       == FROZEN_YEAR_BEARING_IDS_SHA256
       and not (population.tool_year_bearing
                & ~population.tool_benchmark_eligible).any())
    from scripts.audit.audit_sensitivity import CONFIGS as SENSITIVITY_CONFIGS
    sensitivity = pd.read_parquet(
        HERE / "data" / "benchmark" / "sensitivity_summary.parquet").set_index("config")
    backward = pd.read_parquet(HERE / "data" / "benchmark" / "benchmark_results.parquet")
    active_backward = backward[backward.decisive]
    ok("the benchmark's records with decisive clauses are those of the results",
       set(active_backward.tablet) == active_ids and len(active_ids) == 120)
    ok("sensitivity baseline matches the current backward run",
       int(sensitivity.loc["baseline", "decisive_records"]) == len(active_backward)
       and int(sensitivity.loc["baseline", "truth_strict"]) == int(active_backward.truth_in_strict.sum())
       and int(sensitivity.loc["baseline", "truth_safety"]) == int(active_backward.truth_in_safety.sum())
       and int(sensitivity.loc["baseline", "display_top1_correct"]) == int(active_backward.display_top1_correct.sum())
       and int(sensitivity.loc["baseline", "display_top3_correct"]) == int(active_backward.display_top3_correct.sum())
       and "independent_joint_atoms" in sensitivity.index
       and len(SENSITIVITY_CONFIGS) == 32
       and sorted(sensitivity.index) == sorted(c["config"] for c in SENSITIVITY_CONFIGS))
    ok("default backward runner writes the full evaluation population",
       len(backward) == 179 and int(backward.decisive.sum()) == len(active_ids)
       and set(backward.tablet) == set(population.loc[population.tool_benchmark_eligible, "tablet"])
       and backward.top_evidence_group_size.between(1, len(years)).all()
       and backward.top_three_evidence_groups_size.between(1, len(years)).all())
    ranking_candidates = pd.read_parquet(
        HERE / "data" / "benchmark" / "ranking_candidates.parquet")
    # A verdict signature is the hash of a year's verdicts on all clauses. The
    # years of one reported subgroup must share one signature, and different
    # subgroups must differ. A group of tied years may hold several subgroups.
    signatures_by_tablet = {
        tablet: group.set_index("candidate_year_astro").verdict_signature_sha256.to_dict()
        for tablet, group in ranking_candidates.groupby("tablet", sort=False)}
    public_subgroups_match = True
    for row in active_public.itertuples(index=False):
        lookup = signatures_by_tablet.get(row.tablet, {})
        for level in json.loads(row.evidence_levels_json):
            groups = level["physical_groups_years_astro"]
            public_subgroups_match &= sorted({y for group in groups for y in group}) == sorted(level["years_astro"])
            group_signatures = []
            for group in groups:
                sigs = {lookup.get(y) for y in group}
                public_subgroups_match &= len(sigs) == 1 and None not in sigs
                group_signatures.append(next(iter(sigs)) if sigs else None)
            public_subgroups_match &= len(set(group_signatures)) == len(group_signatures)
    ok("reported physical subgroups reproduce full candidate verdict signatures",
       public_subgroups_match)

    head("provenance")
    # Each SHA-256 that a result records, with the file it names. The
    # benchmark file records the hash of the script that fixed it, at that
    # time. That script has changed since, and its hash is left out here. The
    # two hashes of record identifiers above pin the membership instead.
    tool_hashes = {
        "date_all_sha256": "scripts/inference/date_all.py",
        "statements_sha256": "tools/statements.py",
        "skill_sha256": "skills/dating_protocol.md",
        "cal_sha256": "tools/cal.py",
        "sky_sha256": "tools/sky.py",
        "eclipse_tool_sha256": "tools/eclipses.py",
        "eras_sha256": "tools/eras.py",
        "calendar_sha256": "data/physical/calendar.parquet",
        "ephemeris_sha256": "data/physical/ephemeris.parquet",
        "eclipse_data_sha256": "data/physical/eclipses.parquet",
        "normal_stars_sha256": "data/physical/normal_stars.parquet",
    }
    # Where a reader puts the ADART IV scan, and the reading made from it.
    adart_iv_pdf = "data/source/adart_iv/scan.pdf"
    ocr_lines = "data/source/adart_iv/mathpix_lines.json"
    hashed_columns = {
        "data/results/tablet_dates.parquet": {
            **tool_hashes, "clauses_sha256": "data/clauses/reviewed_clauses.parquet"},
        "data/results/adart_iv_dates.parquet": {
            **tool_hashes,
            "clauses_sha256": "data/clauses/adart_iv_reviewed_clauses.parquet",
            "date_adart_iv_sha256": "scripts/inference/date_adart_iv.py",
            "corpus_sha256": "data/corpus/adart_iv_tablets.parquet"},
        "data/clauses/parser_clauses.parquet": {"extractor_sha256": "tools/statements.py"},
        "data/corpus/adart_iv_tablets.parquet": {
            "source_pdf_sha256": adart_iv_pdf, "ocr_lines_sha256": ocr_lines},
        "data/benchmark/adart_iv_corrigenda.parquet": {
            "source_pdf_sha256": adart_iv_pdf},
    }
    recorded = []
    for rel, named in hashed_columns.items():
        if rel in NOT_DISTRIBUTED and not (HERE / rel).is_file():
            continue
        frame = pd.read_parquet(HERE / rel, columns=list(named))
        for column, source in named.items():
            values = set(frame[column].astype(str))
            recorded.append((f"{Path(rel).name}:{column}",
                             values.pop() if len(values) == 1 else "several", source))

    def load(rel: str):
        return json.loads((HERE / rel).read_text(encoding="utf-8"))

    prospective = load("data/results/adart_iv_summary.json")
    corpus_summary = load("data/corpus/adart_iv_corpus_summary.json")
    corrigenda = load("data/benchmark/adart_iv_corrigenda_summary.json")
    mathpix = load("data/source/adart_iv/mathpix_manifest.json")
    held_summary = load("data/benchmark/held_out_summary.json")
    recorded += [("adart_iv_summary.json", value, source)
                 for source, value in {**prospective["inputs"],
                                       **prospective["outputs"]}.items()]
    recorded += [("adart_iv_corpus_summary.json", corpus_summary[f"{name}_sha256"],
                  corpus_summary[name]) for name in ("ocr_lines", "ocr_corrections")]
    recorded.append(("adart_iv_corpus_summary.json", corpus_summary["source_pdf_sha256"],
                     adart_iv_pdf))
    recorded += [("adart_iv_corpus_summary.json", value, source)
                 for source, value in corpus_summary["outputs"].items()]
    recorded += [("adart_iv_corrigenda_summary.json", corrigenda["output_sha256"],
                  "data/benchmark/adart_iv_corrigenda.parquet"),
                 ("adart_iv_corrigenda_summary.json", corrigenda["source_pdf_sha256"],
                  adart_iv_pdf)]
    # The Mathpix manifest records the OCR job, with the file paths of that
    # time. Of the files that it hashes, the scan and the line reading are
    # checked here. Neither is distributed (data/source/not_distributed.csv),
    # and the MMD exports are not kept.
    recorded += [("mathpix_manifest.json", mathpix["outputs"]["lines.json"]["sha256"],
                  ocr_lines),
                 ("mathpix_manifest.json", mathpix["source_sha256"], adart_iv_pdf)]
    recorded += [("held_out_summary.json", value,
                  "scripts/audit/audit_held_out.py" if name == "audit_held_out"
                  else "data/clauses/reviewed_clauses.parquet" if name == "reviewed_clauses"
                  else f"tools/{name}.py")
                 for name, value in held_summary["hashes"].items()]
    stale = [f"{where} ({source})" for where, value, source in recorded
             if value != digest(source)]
    ok("every recorded SHA-256 matches the file it names", not stale,
       "; ".join(stale[:4]) if stale else f"{len(recorded)} hashes")
    present = [rel for rel in NOT_DISTRIBUTED if (HERE / rel).is_file()]
    ok("each file that is not distributed matches the SHA-256 that data/source/not_distributed.csv records",
       adart_iv_pdf in NOT_DISTRIBUTED
       and all(digest(rel) == value for rel, value in NOT_DISTRIBUTED.items()),
       f"{len(present)} of {len(NOT_DISTRIBUTED)} present")

    head("one run, record by record")
    # The benchmark, the public results, the ranked candidates, the
    # sensitivity audit and the bits of each clause come from separate scripts.
    # They must agree record by record. Each field that uses the published
    # year is scored again here, from the candidate sets and the catalogue.
    from scripts.audit import audit_sensitivity, rank_candidates
    catalogue = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "project", "bce_first", "n_lines", "text"]).set_index("tablet")
    published = {tablet: yr.bce_to_astro(int(bce))
                 for tablet, bce in catalogue.bce_first.dropna().items()}

    def ints(values) -> list[int]:
        return sorted(int(value) for value in values)

    def first_few(names: list[str]) -> str:
        return ", ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else "")

    public = dated.set_index("tablet")
    bench = backward.set_index("tablet")
    runs = pd.read_parquet(HERE / "data" / "benchmark" / "sensitivity_runs.parquet")
    baseline = runs[runs.config.eq("baseline")].set_index("tablet")
    ranked_levels = {tablet: rank_candidates.evidence_levels(group)
                     for tablet, group in ranking_candidates.groupby("tablet", sort=False)}
    differ = {"public": [], "truth": [], "candidates": [], "baseline": [],
              "signatures": []}
    for tablet, row in bench.iterrows():
        truth = published.get(tablet)
        strict, safety = ints(row.strict_years_astro), ints(row.safety_years_astro)
        top = ints(row.top_evidence_group_years_astro)
        top3 = ints(row.top_three_evidence_groups_years_astro)
        display = [int(year) for year in row.display_top10_years_astro]
        mine = public.loc[tablet]
        if not (strict == ints(mine.strict_years_astro)
                and safety == ints(mine.safety_years_astro)
                and top == ints(mine.top_evidence_group_years_astro)
                and top3 == ints(mine.top_three_evidence_groups_years_astro)
                and row.evidence_group_count == mine.evidence_group_count
                and row.top_evidence_group_predicate_exact
                == mine.top_evidence_group_predicate_exact
                and row.top_three_evidence_groups_predicate_exact
                == mine.top_three_evidence_groups_predicate_exact
                and row.strict_count == len(strict) == mine.strict_count
                and row.safety_count == len(safety) == mine.safety_count
                and row.top_evidence_group_size == len(top) == mine.top_evidence_group_size
                and row.top_three_evidence_groups_size == len(top3)
                == mine.top_three_evidence_groups_size):
            differ["public"].append(tablet)
        group = ranked_levels.get(tablet)
        if group is None or truth not in set(group.candidate_year_astro):
            differ["candidates"].append(tablet)
            continue
        at_truth = group[group.candidate_year_astro.eq(truth)].iloc[0]
        level = int(at_truth.evidence_group_rank)
        outranks = (group.decisive_families > at_truth.decisive_families) | (
            group.decisive_families.eq(at_truth.decisive_families)
            & (group.primary_bits > at_truth.primary_bits + 1e-12))
        if not (row.truth_astro == truth
                and row.truth_in_strict == (truth in strict)
                and row.truth_in_safety == (truth in safety)
                and row["unique"] == (strict == [truth])
                and row.truth_evidence_group_rank == level
                and row.truth_in_top_evidence_group == (truth in top) == (level == 1)
                and (truth in top3) == (level <= 3)
                and row["rank"] == int(outranks.sum()) + 1
                and row.truth_predicate_equivalent_years == int(
                    group.verdict_signature_sha256.eq(
                        at_truth.verdict_signature_sha256).sum())
                and row.display_top1_correct == (display[0] == truth)
                and row.display_top3_correct == (truth in display[:3])
                and row.display_top1_abs_error == abs(truth - display[0])
                and row.display_top3_min_abs_error
                == min(abs(truth - year) for year in display[:3])):
            differ["truth"].append(tablet)
        # The public results give the verdict of each clause in each year of
        # the safety set. The verdict signature of each such year is computed
        # again from them, with agree as 1, contradict as -1 and untestable as
        # 0, in int8.
        records = json.loads(mine.statement_verdicts_json)
        position = {year: i for i, year in enumerate(safety)}
        outcomes = np.zeros((len(safety), len(records)), dtype=np.int8)
        for j, record in enumerate(records):
            verdicts = record["safety_verdict_years_astro"]
            for year in verdicts["agree"]:
                outcomes[position.get(year, 0), j] = 1
            for year in verdicts["contradict"]:
                outcomes[position.get(year, 0), j] = -1
        signatures = group.set_index("candidate_year_astro").verdict_signature_sha256
        if not all(hashlib.sha256(outcomes[i].tobytes()).hexdigest()
                   == signatures.get(year) for year, i in position.items()):
            differ["signatures"].append(tablet)
        in_order = group.sort_values("baseline_rank").candidate_year_astro
        if not (len(group) == len(years)
                and ints(group.loc[group.strict, "candidate_year_astro"]) == strict
                and ints(group.loc[group.safety, "candidate_year_astro"]) == safety
                and group.strict_count.eq(len(strict)).all()
                and group.safety_count.eq(len(safety)).all()
                and ints(group.loc[group.evidence_group_rank.eq(1),
                                   "candidate_year_astro"]) == top
                and ints(group.loc[group.evidence_group_rank.le(3),
                                   "candidate_year_astro"]) == top3
                and int(group.evidence_group_rank.max()) == row.evidence_group_count
                and [int(year) for year in in_order[:10]] == display
                and group.decisive.eq(row.decisive).all()
                and group.active_decisive.eq(row.active_decisive).all()
                and group.truth.eq(truth).all()
                and group.correct.eq(group.candidate_year_astro.eq(truth)).all()):
            differ["candidates"].append(tablet)
        if tablet not in baseline.index:
            differ["baseline"].append(tablet)
            continue
        base = baseline.loc[tablet]
        if not (ints(base.strict_years_astro) == strict
                and ints(base.safety_years_astro) == safety
                and [int(year) for year in base.display_top10_years_astro] == display
                and base.leading_group_size == len(top)
                and base.strict_count == len(strict) and base.safety_count == len(safety)
                and base.active_decisive == row.active_decisive
                and base.decisive == row.decisive
                # The sensitivity audit keeps four groups, and gives rank 99
                # to a year below them.
                and base.truth_group_rank == (row.truth_evidence_group_rank
                                              if row.truth_evidence_group_rank <= 4 else 99)
                and base.truth_strict == row.truth_in_strict
                and base.truth_safety == row.truth_in_safety
                and base.display_top1_correct == row.display_top1_correct
                and base.display_top3_correct == row.display_top3_correct):
            differ["baseline"].append(tablet)
    ok("each benchmark set is the public result for its record",
       not differ["public"], first_few(differ["public"]))
    ok("each benchmark score follows from its sets, its candidates and the catalogue",
       not differ["truth"], first_few(differ["truth"]))
    ok("the ranking candidates give every benchmark set, group, order and rank",
       not differ["candidates"] and set(ranked_levels) == set(bench.index),
       first_few(differ["candidates"]))
    ok("each verdict signature in the safety set follows from the public verdicts",
       not differ["signatures"], first_few(differ["signatures"]))
    ok("the sensitivity baseline is the benchmark run, record by record",
       not differ["baseline"] and set(baseline.index) == set(bench.index),
       first_few(differ["baseline"]))
    run_truth = runs.tablet.map(published)
    runs_scored = pd.Series([
        truth == r.truth_astro
        and r.truth_strict == (truth in set(r.strict_years_astro))
        and r.truth_safety == (truth in set(r.safety_years_astro))
        and r.strict_count == len(r.strict_years_astro)
        and r.safety_count == len(r.safety_years_astro)
        and r.display_top1_correct == (int(r.display_top10_years_astro[0]) == truth)
        and r.display_top3_correct == (truth in set(r.display_top10_years_astro[:3]))
        for truth, r in zip(run_truth, runs.itertuples(index=False))])
    per_config = runs.groupby("config").tablet.apply(
        lambda tablets: tablets.is_unique and set(tablets) == set(bench.index))
    ok("each sensitivity run scores every benchmark record against the catalogue",
       bool(runs_scored.all()) and bool(per_config.all())
       and sorted(per_config.index) == sorted(c["config"] for c in SENSITIVITY_CONFIGS),
       f"{int((~runs_scored).sum())} rows differ" if not runs_scored.all() else "")
    ok("the sensitivity summary follows from the sensitivity runs",
       not frames_equal(audit_sensitivity.summarize(runs), pd.read_parquet(
           HERE / "data" / "benchmark" / "sensitivity_summary.parquet")))
    information = pd.read_parquet(
        HERE / "data" / "benchmark" / "benchmark_clause_bits.parquet")
    # The public results mark a clause as decisive when it is of the decisive
    # tier and carries bits.
    clause_fields = {"index": "statement_index", "kind": "kind", "planet": "planet",
                     "parser_rule": "parser_rule", "restored": "restored",
                     "omitted": "omitted", "decisive": "active",
                     "agreeing_candidates": "agreeing_years",
                     "testable_candidates": "testable_years",
                     "weight_bits": "information_bits"}
    from_public = pd.DataFrame([
        {"tablet": tablet, **{column: record[field]
                              for field, column in clause_fields.items()}}
        for tablet in bench.index
        for record in json.loads(public.at[tablet, "statement_verdicts_json"])])
    clause_key = ["tablet", "statement_index"]
    stored = information.assign(
        active=information.decisive_tier & information.information_bits.gt(0))
    why = frames_equal(
        stored[from_public.columns].sort_values(clause_key).round(6),
        from_public.sort_values(clause_key).round(6))
    ok("per-clause bits of the benchmark match the public clause verdicts",
       not why, why)

    head("reviewed clauses and Tables S3 and S4")
    from scripts.audit import build_paper_counts as scope_builder
    fingerprint = hashlib.sha256(json.dumps(
        {rel: digest(rel) for rel in scope_builder.SCORE_DEPENDENCIES},
        sort_keys=True).encode()).hexdigest()
    # build_paper_counts.py copies its list of score dependencies from
    # scripts/agent/model_review.py, which is on the list itself. That script's
    # own list is read from its source, without importing it, and the two
    # lists must be the same.
    producer = next(rel for rel in scope_builder.SCORE_DEPENDENCIES
                    if rel.startswith("scripts/"))
    produced = [ast.literal_eval(statement.value)
                for function in ast.walk(ast.parse(
                    (HERE / producer).read_text(encoding="utf-8")))
                if isinstance(function, ast.FunctionDef)
                and function.name == "scoring_fingerprint"
                for statement in ast.walk(function)
                if isinstance(statement, ast.Assign)
                and any(getattr(target, "id", "") == "paths"
                        for target in statement.targets)]
    # The reviewed clauses are read where the builder reads them.
    reviewed = pd.read_parquet(HERE / scope_builder.FILES["information"])
    reviewed_forward = pd.read_parquet(HERE / scope_builder.FILES["reviewed_facts"])
    scope_dir = HERE / "data" / "paper_counts"
    scope = json.loads((scope_dir / "counts.json").read_text(encoding="utf-8"))
    ok("reviewed-clause scores were made with the current tools and calendar",
       produced == [list(scope_builder.SCORE_DEPENDENCIES)]
       and reviewed.scoring_fingerprint.eq(fingerprint).all()
       and reviewed.candidate_year_count.eq(len(years)).all()
       and reviewed.candidate_first_astro.eq(years[0]).all()
       and reviewed.candidate_last_astro.eq(years[-1]).all()
       and scope["table1_scoring_fingerprint"] == fingerprint
       and not reviewed.duplicated(clause_key).any()
       and set(map(tuple, reviewed[clause_key].to_numpy()))
       == set(map(tuple, reviewed_forward[clause_key].to_numpy())),
       fingerprint[:12])
    rebuilt = scratch / "paper_counts"
    problems = []
    try:
        scope_builder.build(HERE, rebuilt)
        patterns = ignore_patterns()
        for committed in sorted(scope_dir.iterdir()):
            if ignored(committed.relative_to(HERE).as_posix(), patterns):
                continue
            fresh = rebuilt / committed.name
            if not fresh.exists():
                problems.append(f"{committed.name} is not written by the builder")
                continue
            if committed.suffix == ".parquet":
                why = frames_equal(pd.read_parquet(fresh), pd.read_parquet(committed))
            elif committed.suffix == ".csv":
                why = frames_equal(pd.read_csv(fresh), pd.read_csv(committed))
            elif committed.suffix == ".json":
                a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (fresh, committed))
                a.pop("generated_utc", None)
                b.pop("generated_utc", None)
                why = first_difference(a, b)
            else:
                why = ("" if fresh.read_text(encoding="utf-8")
                       == committed.read_text(encoding="utf-8") else "text differs")
            if why:
                problems.append(f"{committed.name}: {why}")
        problems += [f"{p.name} is not committed" for p in rebuilt.iterdir()
                     if not ignored(f"data/paper_counts/{p.name}", patterns)
                     and not (scope_dir / p.name).exists()]
    except (scope_builder.ScopeError, OSError, ValueError, KeyError) as exc:
        problems = [str(exc)]
    ok("data/paper_counts/ is what build_paper_counts.py writes",
       not problems, "; ".join(problems[:3]))

    head("held-out test")
    from scripts.audit import audit_held_out as held
    held_runs = pd.read_parquet(HERE / "data" / "benchmark" / "held_out_runs.parquet")
    # Each row is scored again here, with the published years read from the
    # catalogue.
    rescored = held.evaluate(held_runs.drop(columns=[
        "truth_astro", "truth_in_strict", "truth_in_safety", "truth_group_rank"]))
    rounds = {entry["round"]: entry for entry in held_summary["rounds"]}
    by_round = {name: rescored[rescored["round"].eq(name)]
                for name in rescored["round"].unique()}
    volume_of = {project: name for name, project in held.VOLUMES.items()}
    eligible = population[population.tool_benchmark_eligible].set_index("tablet")
    members_ok = all(
        set(by_round.get(entry["round"], pd.DataFrame(columns=["tablet"])).tablet)
        == {tablet for tablet in eligible.index
            if volume_of.get(eligible.at[tablet, "project"]) in entry["test"]}
        for entry in held_summary["rounds"])
    ok("each round dates exactly the benchmark records of its test volumes",
       members_ok
       and rescored.groupby("round").tablet.apply(lambda s: s.is_unique).all()
       and rescored.decisive.eq(rescored.tablet.isin(active_ids)).all()
       and rescored.volume.eq(
           rescored.tablet.map(eligible.project).map(volume_of)).all())
    ok("the held-out rows score their sets against the catalogue",
       not frames_equal(
           rescored[["truth_astro", "truth_in_strict", "truth_in_safety",
                     "truth_group_rank"]],
           held_runs[["truth_astro", "truth_in_strict", "truth_in_safety",
                      "truth_group_rank"]]))
    pooled = held.table_row(
        "all three rounds", rescored[rescored["round"].isin(held.HELD_OUT)])
    reference_row = held.table_row("all volumes", by_round.get(
        "all volumes", rescored.iloc[:0]))
    reading = {
        measure: {
            "all_volumes_percent": reference_row[measure]["percent"],
            "held_out_percent": pooled[measure]["percent"],
            "drop_points": round(reference_row[measure]["percent"]
                                 - pooled[measure]["percent"], 1),
            "close": (reference_row[measure]["percent"]
                      - pooled[measure]["percent"]) <= held.CLOSE_POINTS,
        } for measure in ("first_group", "strict_set")}
    ok("the reading of the test is the pre-registered one, and is scored",
       held.CLOSE_POINTS == 5.0
       and held_summary["settings"]["close_points"] == held.CLOSE_POINTS
       and held_summary["held_out_scored"] is True
       and set(rounds) == {*held.HELD_OUT, "all volumes"}
       and all(entry["scored"] for entry in rounds.values()))
    ok("every row of Table S6 and the reading follow from the held-out runs",
       all(same_json(plain(held.table_row(name, by_round.get(name, rescored.iloc[:0]))),
                     entry["table_row"]) for name, entry in rounds.items())
       and same_json(plain(pooled), held_summary.get("all_three_rounds"))
       and same_json(plain(reading), held_summary.get("reading")))
    everything = by_round.get("all volumes", rescored.iloc[:0]).set_index("tablet")
    differ_all = [
        tablet for tablet, row in bench.iterrows()
        if tablet not in everything.index
        or not (ints(everything.at[tablet, "strict_years_astro"])
                == ints(row.strict_years_astro)
                and ints(everything.at[tablet, "safety_years_astro"])
                == ints(row.safety_years_astro)
                and everything.at[tablet, "first_group_size"] == row.top_evidence_group_size
                and everything.at[tablet, "evidence_group_count"] == row.evidence_group_count
                and everything.at[tablet, "active_decisive"] == row.active_decisive
                and ints(json.loads(everything.at[tablet, "level_years_json"])[0])
                == ints(row.top_evidence_group_years_astro)
                and ints(year for level in json.loads(
                    everything.at[tablet, "level_years_json"])[:3] for year in level)
                == ints(row.top_three_evidence_groups_years_astro))]
    check = rounds.get("all volumes", {}).get("repo_check", {})
    main_text = {
        "records": len(active_backward),
        "first_group": int(active_backward.truth_in_top_evidence_group.sum()),
        "first_three_groups": int(active_backward.truth_evidence_group_rank.le(3).sum()),
        "strict_set": int(active_backward.truth_in_strict.sum()),
        "median_strict_set": float(active_backward.strict_count.median())}
    ok("the all-volume round is the unmodified pipeline, record by record",
       not differ_all and check.get("identical") is True
       and check.get("records") == len(bench) and not check.get("differing_records")
       and same_json(plain(held.MAIN_TEXT), main_text)
       and same_json(plain(held_summary["main_text"]), main_text),
       first_few(differ_all))

    head("tables written again from the code")
    for outputs, process in running:
        _, error = process.communicate(timeout=600)
        for rel in outputs:
            fresh = scratch / Path(rel).name
            if process.returncode or not fresh.exists():
                ok(f"{rel} is what its script writes", False,
                   (error.strip().splitlines() or ["no output"])[-1][:160])
                continue
            if fresh.suffix == ".parquet":
                why = frames_equal(pd.read_parquet(fresh), pd.read_parquet(HERE / rel))
            elif fresh.suffix == ".csv":
                why = frames_equal(pd.read_csv(fresh), pd.read_csv(HERE / rel))
            elif fresh.suffix == ".json":
                why = first_difference(json.loads(fresh.read_text(encoding="utf-8")),
                                       json.loads((HERE / rel).read_text(encoding="utf-8")))
            else:
                why = ("" if fresh.read_text(encoding="utf-8")
                       == (HERE / rel).read_text(encoding="utf-8") else "text differs")
            ok(f"{rel} is what its script writes", not why, why)

    head("ADART IV source, OCR, corpus, and prospective inference")
    # The full text of ADART IV is not distributed. These checks run where a
    # reader has rebuilt it from the scan.
    text_files = [ocr_lines, "data/corpus/adart_iv_tablets.parquet",
                  "data/corpus/adart_iv_lines.parquet"]
    if all((HERE / rel).is_file() for rel in text_files):
        # The OCR reading must hold each page of the scan in order, and as many
        # lines of text as were laid over the searchable PDF.
        pages = mathpix["source_pages"]
        ocr_pages = json.loads((HERE / ocr_lines).read_text(encoding="utf-8"))["pages"]
        read_lines = sum(1 for page in ocr_pages for line in page.get("lines", [])
                         if line.get("conversion_output") and line.get("text", "").strip())
        ok("ADART IV OCR covers every facsimile page",
           [int(page["page"]) for page in ocr_pages] == list(range(1, pages + 1))
           and mathpix["last_status"]["num_pages_completed"] == pages
           and read_lines == mathpix["outputs"]["searchable_pdf"]["inserted_ocr_lines"],
           f"{pages} pages, {read_lines:,} lines")
        adart_iv_corpus = pd.read_parquet(
            HERE / "data" / "corpus" / "adart_iv_tablets.parquet")
        adart_iv_lines = pd.read_parquet(
            HERE / "data" / "corpus" / "adart_iv_lines.parquet")
        corpus_counts = {
            "entries": len(adart_iv_corpus),
            "entries_with_translation": int(adart_iv_corpus.n_translation_lines.gt(0).sum()),
            "entries_without_translation": adart_iv_corpus.loc[
                adart_iv_corpus.n_translation_lines.eq(0), "museum_no"].tolist(),
            "entries_with_internal_chronology": int(
                adart_iv_corpus.chronology_text.fillna("").str.strip().ne("").sum()),
            "translation_lines": int(adart_iv_lines.included_in_translation.sum()),
            "astronomy_input_lines": int(adart_iv_lines.included_in_astronomy.sum()),
            "editorial_lines_excluded": int(adart_iv_lines.role.isin(
                ["editorial_heading", "editorial_excluded"]).sum()),
        }
        differ_iv = [name for name, value in corpus_counts.items()
                     if not same_json(value, corpus_summary.get(name))]
        ok("ADART IV segmentation is complete, translation-only, and answer-hidden",
           not differ_iv
           and adart_iv_corpus.tablet.str.startswith("ADART_IV_D_").all()
           and adart_iv_corpus.tablet.is_unique
           and not adart_iv_lines.loc[
               adart_iv_lines.role.isin(["editorial_heading", "editorial_excluded"]),
               "included_in_astronomy"].any(),
           ", ".join(differ_iv))
    else:
        warns.append("the ADART IV text is not distributed, so its OCR and segmentation checks did not run")
    adart_iv_dates = pd.read_parquet(
        HERE / "data" / "results" / "adart_iv_dates.parquet")
    active_iv = adart_iv_dates[adart_iv_dates.n_decisive_statements.gt(0)]

    def counts(series) -> dict:
        return {str(key): int(value) for key, value
                in series.value_counts(dropna=False).sort_index().items()}

    def median(series):
        return float(series.median()) if len(series) else None

    prospective_counts = {
        "candidate_babylonian_years_astro": [years[0], years[-1]],
        "candidate_babylonian_year_labels": [yr.year_label(years[0]),
                                             yr.year_label(years[-1])],
        "candidate_year_count": len(years),
        "entries": len(adart_iv_dates),
        "entries_with_translation": int(adart_iv_dates.n_translation_lines.gt(0).sum()),
        "entries_with_recognized_astronomical_facts": int(
            adart_iv_dates.n_statements.gt(0).sum()),
        "entries_with_at_least_three_facts": int(adart_iv_dates.n_statements.ge(3).sum()),
        "entries_with_decisive_astronomy": len(active_iv),
        "recognized_facts": int(adart_iv_dates.n_statements.sum()),
        "active_decisive_facts": int(adart_iv_dates.n_decisive_statements.sum()),
        "astronomy_claims": counts(adart_iv_dates.claim),
        "astronomy_bases": counts(adart_iv_dates.basis),
        "unique_strict_year": int(adart_iv_dates.strict_count.eq(1).sum()),
        "active_singleton_first_physical_group": int(
            active_iv.top_evidence_group_size.eq(1).sum()),
        "all_rows_singleton_first_group_diagnostic": int(
            adart_iv_dates.top_evidence_group_size.eq(1).sum()),
        "combined_claims": counts(adart_iv_dates.combined_claim),
        "chronology_relations": counts(adart_iv_dates.chronology_relation),
        "active_median_strict_set_size": median(active_iv.strict_count),
        "active_median_safety_set_size": median(active_iv.safety_count),
        "active_median_first_group_size": median(active_iv.top_evidence_group_size),
        "active_median_first_three_group_union": median(
            active_iv.top_three_evidence_groups_size),
    }
    differ_iv = [name for name, value in prospective_counts.items()
                 if not same_json(plain(value), prospective.get(name))]
    ok("ADART IV inference is prospective and reports uncertainty rather than accuracy",
       not differ_iv
       and not adart_iv_dates.claim.eq("committed").any()
       and "no accuracy" in prospective["interpretation"].lower()
       and "652--627" in prospective["candidate_range_limit"],
       ", ".join(differ_iv))
    corrigenda_rows = pd.read_parquet(
        HERE / "data" / "benchmark" / "adart_iv_corrigenda.parquet")
    labels = corrigenda_rows.digital_status.eq("incorporated_reference_label")
    ok("every astronomy-consequential ADART I--III corrigendum is traced",
       corrigenda["corrections"] == len(corrigenda_rows)
       and corrigenda["status_counts"] == counts(corrigenda_rows.digital_status)
       and corrigenda["reference_labels_already_incorporated"] == int(labels.sum())
       and corrigenda["non_label_corrections_changing_current_scores"]
       == int((corrigenda_rows.current_metric_effect & ~labels).sum()))
    # No module that builds a candidate set may name a column that holds a
    # published date. Comments and docstrings are skipped. The published dates
    # enter only when the benchmark is scored, after the candidate sets are
    # final.
    answer_columns = re.compile(
        r"bce_first|bce_second|ancient_year|reference_year|\btruth\b")
    blind = sorted((HERE / "tools").glob("*.py"))
    blind += sorted((HERE / "scripts" / "inference").glob("*.py"))
    leaks = []
    for path in blind:
        body = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("#"))
        body = re.sub(r'"""(?:.|\n)*?"""', "", body)
        if answer_columns.search(body):
            leaks.append(path.name)
    ok("no module that builds a candidate set can read a reference date",
       not leaks, ", ".join(leaks))

    head("repository layout")
    files, dirs = tree()
    names = {Path(rel).name for rel in files}
    python_files = [rel for rel in files if rel.endswith(".py")]
    undeclared = [rel for rel in python_files
                  if not ast.get_docstring(ast.parse(
                      (HERE / rel).read_text(encoding="utf-8")))]
    ok("every Python file declares its purpose in a module docstring",
       not undeclared, ", ".join(undeclared[:4]))
    # The Makefile names each script that runs, in the rebuild, the figures,
    # the site and the lists of scripts outside the rebuild. A script that
    # another script imports counts as a library.
    makefile = "\n".join(line.split("#", 1)[0] for line in
                         (HERE / "Makefile").read_text(encoding="utf-8").splitlines())
    figures_line = re.search(r"^FIGURES\s*:=\s*(.+)$", makefile, re.M)
    make_figures = figures_line.group(1).split() if figures_line else []
    scheduled = {f"figures/{name}.py" for name in make_figures}
    scheduled |= {path[len("scripts/"):] for path in re.findall(r"scripts/[\w/]+\.py", makefile)}
    imported = set()
    for rel in python_files:
        if not rel.startswith("scripts/"):
            continue
        for node in ast.walk(ast.parse((HERE / rel).read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
                imported.add((node.module or "").split(".")[-1])
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[-1] for alias in node.names)
    unscheduled = [rel for rel in python_files if rel.startswith("scripts/")
                   and Path(rel).name != "__init__.py"
                   and rel[len("scripts/"):] not in scheduled
                   and Path(rel).stem not in imported]
    vanished = sorted(name for name in scheduled if not (HERE / "scripts" / name).is_file())
    ok("every script runs from the Makefile or is imported by another script",
       not unscheduled and not vanished,
       ", ".join(unscheduled + [f"{name} (in the Makefile, absent)" for name in vanished]))
    patterns = ignore_patterns()
    scans = {Path(rel).name for rel in NOT_DISTRIBUTED}
    ghosts = []
    for readme in ["README.md"] + [f"{folder}/README.md" for folder in FOLDERS]:
        path = HERE / readme
        if not path.is_file():
            ghosts.append(readme)
            continue
        for token in re.findall(r"`([^`\s]+)`", path.read_text(encoding="utf-8")):
            if (not FILE_TOKEN.fullmatch(token) or token.startswith((".", "/", "~"))
                    or not re.search(r"[A-Za-z]", Path(token).stem)):
                continue
            if ((HERE / token).exists() or (path.parent / token).exists()
                    or Path(token).name in names or Path(token).name in scans
                    or any(fnmatch.fnmatch(Path(token).name, Path(p.strip("/")).name)
                           for p in patterns if Path(p.strip("/")).name.strip("*?"))):
                continue
            ghosts.append(f"{readme}: {token}")
    ok("the READMEs name no file that is gone", not ghosts, ", ".join(ghosts[:4]))

    # Each folder README names each file in its folder. It names a script by its
    # path in scripts/, and a module or a test by its name. In data/, docs/ and
    # skills/ it names each entry at the top of the folder.
    unnamed = []
    for folder in FOLDERS:
        readme = HERE / folder / "README.md"
        text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        inside = [rel[len(folder) + 1:] for rel in files if rel.startswith(folder + "/")]
        if folder in ("scripts", "tools", "tests"):
            wanted = sorted(rel if folder == "scripts" else Path(rel).name for rel in inside
                            if rel.endswith(".py") and Path(rel).name != "__init__.py")
        else:
            wanted = sorted({rel.split("/")[0] for rel in inside} - {"README.md"})
        unnamed += [f"{folder}/{name}" for name in wanted if f"`{name}" not in text]
    ok("each folder README names each file in its folder", not unnamed,
       ", ".join(unnamed[:4]))

    # data/columns.csv describes each data file in a row with no column, and
    # each column of a table in a row of its own.
    described = set()
    dictionary = HERE / "data" / "columns.csv"
    if dictionary.exists():
        with dictionary.open(encoding="utf-8", newline="") as handle:
            described = {row["file"] for row in csv.DictReader(handle) if not row["column"]}
    undocumented = [rel for rel in files if rel.startswith("data/")
                    and rel != "data/README.md"
                    and rel[len("data/"):] not in described]
    markdown = [rel for rel in files if rel.endswith(".md")
                and rel != "README.md" and rel not in DOCUMENTS]
    ok("every data file is described in data/columns.csv, and the only Markdown "
       "files are the READMEs and the protocol",
       not undocumented and not markdown, ", ".join((undocumented + markdown)[:4]))
    top = {rel.split("/")[0] for rel in files + dirs}
    transient = [rel for rel in dirs if Path(rel).name in TRANSIENT_DIRS]
    ok("the top level holds only the documented files and directories, and no "
       "scratch or cache directory", top <= TOP_LEVEL and not transient,
       ", ".join(sorted(top - TOP_LEVEL) + transient))
    print(f"\n{'-' * 60}")
    for note in warns:
        print(f"optional absent: {note}")
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: {', '.join(fails)}")
        print("Do not trust the generated results until these pass.")
        return 1
    print(f"all checks passed in {time.time() - t0:.1f}s")
    print("The scientific pipeline and the committed results agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
