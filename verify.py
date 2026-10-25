#!/usr/bin/env python3
"""Prove this directory works, and, where the paper is present, that it follows from the data.

Run this first, before reading anything else and before changing anything. It
checks the environment and the data, runs each tool's own self-check, and dates
a tablet whose answer is known. Then it audits the tree.

- Every SHA-256 that a result records must match the file it names.
- Every summary must follow from the rows it summarises, and every saved table
  from the script that writes it.
- The benchmark, the public results, the ranked candidates, the sensitivity
  baseline and the held-out test must be one run, record by record.
- Where paper/ is present, every number the paper prints must follow from the
  data, and Tables S3, S4, S6 and S7 must be the tables the code writes.
- The repository holds only what its READMEs describe.

It does not run the tests. ``make check`` runs them, then this file. The book
scans and the full text read from ADART IV are not distributed. data/README.md
names where a reader puts each file, and records its SHA-256. A file that is
there is checked against that value. When it is absent, the checks use the
recorded value, and say so.

    python3 verify.py

Exit code 0 means the directory is sound.
Anything else means stop and read the failure before working.
"""

from __future__ import annotations

import ast
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REQUIRED = ["numpy", "pandas", "scipy", "pyarrow", "swisseph", "requests",
            "matplotlib"]
OPTIONAL = {"jplephem": "only needed to rebuild the ephemeris from a kernel"}
# The paper is kept outside the public repository. Checks that read it run
# only when paper/PNAS_Paper.tex is present.
PAPER = Path(__file__).resolve().parent / "paper" / "PNAS_Paper.tex"

# Row counts are kept only where they are a fixed property of the sources or
# of the design. Every other table is checked below against the code that
# writes it or the rows it summarises.
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

# The benchmark was frozen before any candidate set met a published date. Its
# membership is pinned here, so no rebuild can move a record in or out. The
# value is the SHA-256 of the sorted record identifiers, one per line.
FROZEN_ELIGIBLE_IDS_SHA256 = (
    "d2914bbc371a2faabc30a7a03320b8f112739161cbbc0a38c0ea63b2b4e04fae")
FROZEN_YEAR_BEARING_IDS_SHA256 = (
    "e39b6845256bb10c9597ec17635e831efc124bae7e1d2aaf77f5dfff7a8f76be")

# What may sit at the top of the tree, and in paper/, which goes to Overleaf
# as it is.
TOP_LEVEL = {"README.md", "LICENSE", "CITATION.cff", "Makefile", "requirements.txt", "verify.py",
             ".gitignore", "data", "docs", "paper", "scripts", "skills",
             "tests", "tools"}
PAPER_SUFFIXES = {".tex", ".bib", ".cls", ".bst", ".sty", ".ldf", ".pdf"}
PAPER_PDFS = {"PNAS_Paper.pdf", "PNAS_Logo.pdf"}
FIGURE_SUFFIXES = {".pdf", ".png"}
TRANSIENT_DIRS = {"tmp", "output", ".pytest_cache", ".mypy_cache",
                  ".ruff_cache", ".ipynb_checkpoints", ".cache"}
# Markdown files that are not a directory's README.
DOCUMENTS = {"skills/dating_protocol.md"}
FILE_TOKEN = re.compile(
    r"[\w./+-]+\.(?:py|parquet|jsonl?|csv|md|tex|pdf|png|jpg|ttf|txt|zip|bib"
    r"|mmd|sty|cls|bst|ldf)")
NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten"]

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

    These are the book scans and the full text read from ADART IV. Each is a
    row of the table "Not distributed" in data/README.md, which names the file
    by its path inside data/ in its first cell and gives its SHA-256 in its last.
    """
    readme = HERE / "data" / "README.md"
    files = {}
    for line in (readme.read_text(encoding="utf-8").splitlines()
                 if readme.exists() else []):
        row = re.match(r"^\|\s*`([^`]+\.(?:pdf|json|parquet))`\s*\|.*\|\s*`([0-9a-f]{64})`\s*\|\s*$", line)
        if row:
            files[f"data/{row.group(1)}"] = row.group(2)
    return files


NOT_DISTRIBUTED = not_distributed()
_digests: dict[str, str] = {}


def digest(rel: str) -> str:
    """SHA-256 of a file in the tree, by its path from the root.

    A file that is not distributed gives the value that
    data/README.md records for it, so every hash recorded elsewhere is
    still checked against one fixed value. Any other missing file gives
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


def pct(part, whole) -> str:
    """A percentage as the paper prints it, rounded half up to a whole number."""
    return str(int(100 * part / whole + 0.5))


def num(value) -> str:
    """A count or median as the paper prints it: 4,039, 17.5 or 101."""
    value = float(value)
    return f"{int(value):,}" if value.is_integer() else f"{value:g}"


def word(n: int) -> str:
    """A small count as the paper spells it, and a larger one as digits."""
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def wilson(part: int, whole: int, z: float = 1.959963984540054) -> tuple[int, int]:
    """The 95% Wilson score interval of a rate, in whole percent."""
    p, d = part / whole, 1 + z * z / whole
    centre = (p + z * z / (2 * whole)) / d
    half = z * ((p * (1 - p) / whole + z * z / (4 * whole * whole)) ** 0.5) / d
    return round(100 * (centre - half)), round(100 * (centre + half))


def everyone(part: int, whole: int) -> str:
    return f"all {whole}" if part == whole else f"{part} of the {whole}"


def strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())




def zodiac_guard_violations() -> list[str]:
    """Return source files that bypass or refit the published zodiac model."""
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
    # scratch directory while the other checks run, and compared at the end.
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
    # The table builder writes to a fixed path, so it runs with its output, and
    # the root it names in its message, moved to scratch. It needs the paper.
    if PAPER.is_file():
        rebuilds.append((["paper/tableS7_adart_iv.tex"],
                     ["-c", "import sys; from pathlib import Path; "
                            "from scripts.paper import build_adart_iv_table as table; "
                            "table.OUT = Path(sys.argv[1]); table.HERE = table.OUT.parent; "
                            "raise SystemExit(table.main())", "{0}"]))
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
    # A Babylonian month begins at first crescent, so every tabulated month
    # start should sit a short fixed distance after a computed conjunction.
    # The book and the JPL kernel know nothing about each other.
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

    # Month I after Month XII belongs to the next Babylonian year, and a day
    # heading from the old month must not leak across that boundary.
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
    # The benchmark records with decisive clauses: their result keeps at least one.
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
    # "150/149" is the one Babylonian year beginning in 150 BCE; bce_second
    # is the Julian year it ends in, so the years run to -bce_second.
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
    # Numerical outcomes changed with the new predicate. Validate against
    # freshly reconstructed source-grounded clauses, not the old totals.
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
    # Check each reported subgroup against the full-verdict signature of the
    # ranked candidates. A co-ranked level may hold more than one subgroup and
    # must not be rejected just for being co-ranked.
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
    # Every SHA-256 that a result records, with the file it names. The frozen
    # population's generator hash is old on purpose. Its membership is pinned
    # above instead, and it is not listed here.
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
    # The Mathpix manifest is the record of the job, with the paths of its
    # day. Of what it hashes, the tree keeps the line reading, and a reader
    # can supply the scan. The MMD exports and the searchable copy are not kept.
    recorded += [("mathpix_manifest.json", mathpix["outputs"]["lines.json"]["sha256"],
                  ocr_lines),
                 ("mathpix_manifest.json", mathpix["source_sha256"], adart_iv_pdf)]
    bibliography_record = HERE / "paper" / "bibliography.json"
    bibliography_sources = (json.loads(bibliography_record.read_text(encoding="utf-8"))
                            if bibliography_record.is_file() else {})
    recorded += [("bibliography.json", entry["file"]["sha256"], entry["file"]["path"])
                 for entry in bibliography_sources.values()
                 if isinstance(entry, dict) and isinstance(entry.get("file"), dict)]
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
    ok("each file that is not distributed matches the SHA-256 that data/README.md records",
       adart_iv_pdf in NOT_DISTRIBUTED
       and all(digest(rel) == value for rel, value in NOT_DISTRIBUTED.items()),
       f"{len(present)} of {len(NOT_DISTRIBUTED)} present")

    head("one run, record by record")
    # The benchmark, the public results, the ranked candidates, the
    # sensitivity audit and the per-clause bits come from separate scripts. They must
    # agree on every record. Each field that reads the published year is
    # scored again here, from the sets and the catalogue.
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
        # The public verdicts give every clause's verdict in each year of the
        # safety set, so the full-verdict signature of each of those years can
        # be computed again: agree 1, contradict -1, untestable 0, as int8.
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
                # The sensitivity audit keeps four levels and ranks a year
                # below them as 99.
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
    # A public verdict is decisive when its clause is of the decisive tier and
    # carries bits.
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
    from scripts.paper import build_paper_counts as scope_builder
    fingerprint = hashlib.sha256(json.dumps(
        {rel: digest(rel) for rel in scope_builder.SCORE_DEPENDENCIES},
        sort_keys=True).encode()).hexdigest()
    # The builder copies the dependency list of the review script, which is
    # one of the dependencies. Read that script's own list from its source,
    # without importing it.
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
    # The published years enter again from the catalogue, so each row is
    # scored here rather than read back.
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
        # The optical reading holds every page of the scan in order, and as many
        # lines with text as the searchable copy was given.
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
    # No module that produces a candidate set may so much as name a reference
    # date. The answer enters only in the scoring phase, after every candidate
    # set is frozen, and this is the check that keeps it that way.
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

    if PAPER.is_file():
        head("the paper against the data")
        tex_source = (HERE / "paper" / "PNAS_Paper.tex").read_text(encoding="utf-8")
        tex = re.sub(r"\s+", " ", tex_source)
        os.environ.setdefault("MPLBACKEND", "Agg")
        sys.path.insert(0, str(HERE / "scripts" / "paper" / "figures"))
        import fig4_benchmark
        dated_ids = set(population.tablet)
        in_dated = public.loc[sorted(dated_ids)]
        n_bench, n_act, n_dated, n_years = len(backward), len(active_backward), len(dated_ids), len(years)
        supporting = backward[~backward.decisive]
        first = int(active_backward.truth_in_top_evidence_group.sum())
        three = int(active_backward.truth_evidence_group_rank.le(3).sum())
        strict_in = int(active_backward.truth_in_strict.sum())
        safety_in = int(active_backward.truth_in_safety.sum())
        places = fig4_benchmark.positions(ranking_candidates[ranking_candidates.decisive])
        reach = {n: pct(int((places <= n).sum()), len(places)) for n in (1, 3, 10)}
        parser = facts[facts.tablet.isin(dated_ids)]
        span = scope["adart_i_iii"]
        reviewed_fw, parser_fw = scope["reviewed_forward"], scope["deterministic_forward"]
        s3 = pd.read_csv(scope_dir / "table_s3.csv")
        s4 = pd.read_csv(scope_dir / "table_s4.csv")
        station3 = s3[s3.kind.eq("station_appulse")].iloc[0]
        station4 = s4[s4.kind.eq("station_appulse")].iloc[0]
        normal3 = s3[s3.kind.eq("appulse")].iloc[0]
        scheme4 = s4[s4.kind.isin(["season", "sirius"])].set_index("kind")
        outside = in_dated.loc[sorted(dated_ids - set(backward.tablet))]
        outside_ranked = outside[outside.claim.eq("ranked")]
        committed_all = in_dated[in_dated.claim.eq("committed")]
        committed_match = sum(int(row.strict_years_astro[0]) == published.get(tablet)
                              for tablet, row in committed_all.iterrows())
        stated = in_dated[~in_dated.chronology_relation.eq("absent")]
        narrowed = stated[~stated.chronology_relation.eq("lookup_only")]
        bases = committed_all.basis.value_counts()
        volumes = population.project.value_counts()
        no_decisive = active_backward[active_backward.active_decisive.eq(0)]
        decisive_now = active_backward[active_backward.active_decisive.gt(0)]
        ranked_iv = adart_iv_dates[adart_iv_dates.claim.eq("ranked")]
        held_rows = {name: held.table_row(name, rows) for name, rows in by_round.items()}
        x301492 = information[information.tablet.eq("X301492") & information.decisive_tier]
        x301492_years = sorted(x301492.agreeing_years.astype(int), reverse=True)
        solar = parser[parser.kind.eq("eclipse") & parser.planet.eq("solar")]
        solar_usable = solar[~solar.restored & ~solar.omitted
                             & solar.month.notna() & solar.day.notna()]
        tolerances = {name: rounds[name]["choices"]["tolerance"]["planet"]
                      for name in held.HELD_OUT if name in rounds}

        # The partial checks that the reviews asked about, recounted from the committed tables.
        benchmark_now = pd.read_parquet(HERE / "data" / "benchmark" / "benchmark_results.parquet")
        decisive_ids = set(benchmark_now[benchmark_now.decisive.astype(bool)].tablet)
        parser_now = pd.read_parquet(HERE / "data" / "clauses" / "parser_clauses.parquet")
        latitude = parser_now[parser_now.tablet.isin(decisive_ids) & parser_now.semantic_tier.eq("decisive")
                              & parser_now.direction.isin(["above", "below"])]
        lat_decisive, lat_records = len(latitude), latitude.tablet.nunique()
        verdicts_now = pd.read_parquet(HERE / "data" / "clauses" / "model_review_gpt-6-luna-2026-09-22"
                                       / "reviewed_clause_verdicts.parquet")
        eclipse_rows = verdicts_now[verdicts_now.kind.eq("eclipse")]
        eclipse_untestable_reasons = {
            "unobserved": int(eclipse_rows.forward_detail.str.contains("omitted, unwatched, or unobserved").sum()),
            "no date": int(eclipse_rows.forward_detail.str.contains("no secure month and day").sum())}
        station_rows = verdicts_now[verdicts_now.kind.eq("station_appulse")
                                    | verdicts_now.parser_rule.astype(str).str.startswith("measured_station_normal_star")]
        station_contradictions = station_rows[station_rows.forward_verdict.eq("contradict")]
        station_window_misses = int(station_contradictions.forward_detail.str.contains("no matching station").sum())

        def config(name: str, column: str) -> int:
            return int(sensitivity.loc[name, column])

        def shift(name: str, column: str) -> int:
            """Change from the baseline, in whole percent of the records with decisive clauses."""
            return int(pct(config(name, column), n_act)) - int(pct(config("baseline", column), n_act))

        def within_point(name: str) -> bool:
            return all(abs(shift(name, column)) <= 1 for column in ("truth_in_leading_group", "truth_strict"))

        conventions = ("cubit_at_2_2_degrees", "britton_offset_minus_half", "britton_offset_plus_half",
                       "delta_t_applied", "tolerance_narrow", "tolerance_wide", "moon_night_window")
        alternatives = [name for name in sensitivity.index if name != "baseline"]
        base_strict = float(sensitivity.loc["baseline", "median_strict_count"])
        base_first = float(sensitivity.loc["baseline", "median_leading_group"])

        def years_word(value) -> str:
            return f"{num(value)} year" + ("" if float(value) == 1 else "s")

        def english(values: list[int]) -> str:
            return ", ".join(str(v) for v in values[:-1]) + f" and {values[-1]}"

        addenda = load("data/benchmark/addenda_results.json")["scores"]["not_linked_to_adart_i_iii"]
        addenda_hits = {addenda["published_year_in_first_group"]["count"], addenda["published_year_in_strict_set"]["count"]}
        claims = {
            "candidate calendar": f"in all {n_years} years from "
                                  f"{yr.year_label(years[0])} to {yr.year_label(years[-1])}",
            "benchmark (abstract)": f"On {n_act} dated tablets, the years tied at the top",
            "first group (abstract)": "the years tied at the top of the ranking include the "
                                      f"published year in {pct(first, n_act)}\\% of cases",
            "strict set (abstract)": f"In {everyone(strict_in, n_act)}, the reported candidate set "
                                     "contains it.",
            "frozen addenda test": (f"Of the {addenda['texts']} addenda that do not join (fit onto) or duplicate a record of ADART I to III, "
                                    f"{addenda['texts_with_a_decisive_clause']} have a decisive clause. For {min(addenda_hits)} of them, the first "
                                    "group and the strict set each hold the published year")
                                   if len(addenda_hits) == 1 else "addenda: first group and strict set differ",
            "ADART IV (abstract)": f"For {len(ranked_iv)} undated entries of ADART IV, the system "
                                   "gives ranked candidate years",
            "significance": f"On {n_act} dated tablets, its leading candidate years include the "
                            f"published year in {pct(first, n_act)}\\% of cases.",
            "parser clauses": f"it finds {num(len(parser))} clauses in "
                              f"{parser.tablet.nunique()} of the {n_dated} records with a published date",
            "reviewed clauses": f"It raises the {num(span['deterministic_facts'])} parser "
                                f"clauses to {num(span['reviewed_clauses'])} clauses from "
                                f"{span['reviewed_records']} records",
            "forward agreement": "the reviewed clauses agree in "
                                 f"{100 * reviewed_fw['agree'] / reviewed_fw['testable']:.1f}\\% "
                                 "of the cases that the tools can test and the parser clauses in "
                                 f"{100 * parser_fw['agree'] / parser_fw['testable']:.1f}\\%",
            "Normal Star clauses": f"Such clauses agree with a median of {num(normal3['Median years allowed'])} of the "
                                   f"{n_years} years, or {normal3.Bits:.1f} bits",
            "station against a star": "A station against a star carries a median of "
                                      f"{station3.Bits:.1f} bits. However, it agrees with the published "
                                      f"year in {pct(station4.Agree, station4.Testable)}\\% of "
                                      "testable cases",
            "three clocks": f"Record X301492, one of {int(backward.best_bodies.ge(3).sum())} "
                            "benchmark records with three or more clocks",
            "X301492 clauses": f"Five of its decisive clauses agree with "
                               f"{english(x301492_years)} years"
                               if len(x301492_years) == 5 else "X301492: not five clauses",
            "X301492 bits": "the decisive bits $I_D$ total "
                            f"{float(public.at['X301492', 'primary_bits']):.1f}",
            "benchmark size": f"we fixed a benchmark of {n_bench} dated records",
            "decisive records": f"Of these, {n_act} have at least one decisive clause",
            "supporting-only records": f"The other {n_bench - n_act} are supporting-only records",
            "first group": f"For the {n_act} records with decisive clauses, it contains the "
                           f"published year in {first} ({pct(first, n_act)}\\%",
            "first group size": f"For {int(active_backward.top_evidence_group_size.eq(1).sum())} of the {n_act}, it holds a single year, "
                                f"and in {int((active_backward.top_evidence_group_size.eq(1) & active_backward.truth_in_top_evidence_group.astype(bool)).sum())} this is the published year",
            "first group size (abstract)": f"For {int(active_backward.top_evidence_group_size.eq(1).sum())} of them, this top group is a single year, "
                                           f"the published year in {int((active_backward.top_evidence_group_size.eq(1) & active_backward.truth_in_top_evidence_group.astype(bool)).sum())}",
            "first two groups": (lambda rank: f"The first two groups contain it in {int(rank(active_backward).le(2).sum())} "
                                 f"({pct(int(rank(active_backward).le(2).sum()), n_act)}\\%), and the third and fourth add none"
                                 if all(int(rank(part).le(2).sum()) == int(rank(part).le(4).sum())
                                        for part in (active_backward, backward, supporting))
                                 else "Fig. 4b: the third or fourth group adds a record")(
                                     lambda part: part.truth_evidence_group_rank),
            "supporting-only first group": "The supporting-only records also reach "
                                           f"{pct(supporting.truth_in_top_evidence_group.sum(), len(supporting))}"
                                           "\\% for the first group",
            "ranked order": f"the first candidate year alone is the published year in {reach[1]}\\%",
            "ties": f"Ties explain the gap to the {pct(first, n_act)}\\% of the first group",
            "strict set": (lambda missed: f"It contains the published year in {everyone(strict_in, n_act)} records "
                           f"with decisive clauses ({pct(strict_in, n_act)}\\%), with a median size of "
                           f"{num(active_backward.strict_count.median())} years"
                           + (f". In the one miss, {missed.tablet.iloc[0]}, the published year stays in the safety set"
                              if len(missed) == 1 and bool(missed.truth_in_safety.iloc[0]) else ""))(
                active_backward[~active_backward.truth_in_strict.astype(bool)]),
            "alternative settings": f"most of the {len(SENSITIVITY_CONFIGS) - 1} alternative settings",
            "outside the benchmark": f"The same rules date the {len(outside)} dated records "
                                     "outside the benchmark, and "
                                     f"{int(outside.n_decisive_statements.gt(0).sum())} of them "
                                     "have a decisive clause",
            "outside inclusion": (lambda truth: "and "
                                  f"{sum(yr.bce_to_astro(int(truth[t])) in {int(y) for y in years_in} for t, years_in in zip(outside_ranked.index, outside_ranked.strict_years_astro))}"
                                  " of them hold the published year")(
                pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet",
                                columns=["tablet", "bce_first"]).set_index("tablet").bce_first),
            "outside median": "Their strict sets hold a median of "
                              f"{num(outside_ranked.strict_count.median())} years, against "
                              f"{num(active_backward.strict_count.median())} in the benchmark",
            "commitments": f"In the whole corpus of {n_dated} dated records, the commitment rule "
                           f"of Eq.~\\ref{{eq:commit}} commits to one year for {len(committed_all)}. "
                           f"All {committed_match} match the published year, and all "
                           f"{int(committed_all.index.isin(backward.tablet).sum())} lie in the benchmark"
                           if committed_match == len(committed_all)
                           == int(committed_all.index.isin(backward.tablet).sum())
                           else "commitments: not all match or lie in the benchmark",
            "stated years": f"For {len(stated)} dated records, the system extracts a year",
            "stated years in the strict set": f"In all {len(narrowed)} records where the astronomy "
                                              "narrows the year, the stated year lies in the strict set"
                                              if narrowed.chronology_relation.isin(
                                                  ["selects_strict_candidate",
                                                   "confirms_astronomy"]).all()
                                              else "stated years: one lies outside the strict set",
            "Fig. 4": f"Inclusion of the published year in the benchmark of {n_bench} dated records",
            "ADART IV coverage": "The parser and the search find astronomical clauses in "
                                 f"{int(adart_iv_dates.n_statements.gt(0).sum())} entries, and "
                                 f"{len(ranked_iv)} of them have a decisive clause",
            "ADART IV median": f"cut the {n_years} candidate years to a median of "
                               f"{num(ranked_iv.strict_count.median())}",
            "ADART IV heading": f"Ranked candidate years for {len(ranked_iv)} undated entries",
            "ADART IV entries": f"which gives {corpus_summary['entries']} entries. Of these, "
                                f"{corpus_summary['entries_with_translation']} have an English translation",
            "ADART IV first group": f"Of the {len(ranked_iv)} ranked entries, "
                                    f"{int(ranked_iv.n_decisive_statements.eq(1).sum())} hold a single decisive "
                                    "clause. One clock leaves many years, and the first groups of all "
                                    f"{len(ranked_iv)} have a median of {num(ranked_iv.top_evidence_group_size.median())} years",
            "ADART IV passages": f"The other {int(ranked_iv.n_decisive_statements.sum())} "
                                 "passages agree with the page",
            "Fig. S2": f"Strict candidate sets of the {len(ranked_iv)} ADART IV entries",
            "Table S7": f"Ranked candidate years of the {len(ranked_iv)} ADART IV entries",
            "corpus": f"We use its {n_dated} records of ADART I to III with a published date. Of these, "
                      f"{span['records_with_edition_text']} have edition text and "
                      f"{int(catalogue.loc[sorted(dated_ids), 'text'].fillna('').str.strip().ne('').sum())}"
                      " an English translation",
            "corrigenda": f"ADART IV makes {len(corrigenda_rows)} corrections",
            "calendar corrections": f"We corrected the {len(pd.read_csv(HERE / 'data' / 'physical' / 'calendar_ocr_corrections.csv'))} "
                                    "transcription errors that this comparison found",
            "solar eclipses": "Solar eclipses stay supporting, because the dated corpus has only "
                              f"{word(len(solar_usable))} unrestored, observed report",
            "Table S3 caption": f"for {num(span['reviewed_clauses'])} clauses from "
                                f"{span['reviewed_records']} records of ADART I to III",
            "Table S4 caption": f"of the {num(span['reviewed_clauses'])} reviewed clauses from "
                                f"{span['reviewed_records']} dated records of ADART I to III",
            "Table S4 parser note": f"For the parser clauses alone, from {parser_fw['records']} "
                                    f"records, {num(parser_fw['agree'])} of "
                                    f"{num(parser_fw['testable'])} testable clauses agree "
                                    f"({100 * parser_fw['agree'] / parser_fw['testable']:.1f}\\%)",
            "station agreement": ("The station against a star of the main text agrees in about half of its testable cases"
                                  if 0.4 <= station4.Agree / station4.Testable <= 0.6 else "station: not about half"),
            "scheme types": ("The solstice, equinox and Sirius types agree in nearly all of theirs"
                             if min(scheme4.at[k, "Agree"] / scheme4.at[k, "Testable"] for k in ("season", "sirius")) >= 0.9
                             else "scheme types: not nearly all"),
            "without the solar year": ("Without this clock, the benchmark values stay as in the baseline"
                                       if all(shift("no_sun_family", column) == 0 for column in ("truth_in_leading_group", "truth_strict"))
                                       and sensitivity.loc["no_sun_family", "median_leading_group"] == sensitivity.loc["baseline", "median_leading_group"]
                                       else "without the solar year: a value changes"),
            "commitment bases": f"{word(int(bases.get('independent_astronomy', 0))).capitalize()} rest on two "
                                f"or more clocks and {word(int(bases.get('secure_lunar_eclipse', 0)))} on a lunar eclipse",
            "approximation runs": (lambda most: "Making the distances tested only in longitude supporting moves each rate by at most "
                                   f"{word(most('longitude_only_distances_supporting'))} point"
                                   + ("" if most("longitude_only_distances_supporting") == 1 else "s")
                                   + ", and a window of one night for the Moon by at most "
                                   f"{word(most('moon_night_window'))}")(
                lambda name: max(abs(shift(name, column)) for column in ("truth_in_leading_group", "truth_strict"))),
            "strict set protected": ("The strict set then shrinks to about one year, which is more often wrong"
                                     if all(float(sensitivity.loc[name, "median_strict_count"]) <= 1.5
                                            and shift(name, "truth_strict") < 0
                                            for name in ("contradictions_bound_the_leading_set", "information_weighting"))
                                     else "strict-set runs: the strict set does not shrink as stated"),
            "ordering runs": ("and they give the same rates as the baseline"
                              if all(shift(name, column) == 0
                                     for name in ("contradictions_before_information", "equal_weighting_with_tiers")
                                     for column in ("truth_in_leading_group", "truth_strict"))
                              else "ordering runs: a rate changes"),
            "rates change little": ("Across the 31 settings, the rates change little"
                                    if len(alternatives) == 31
                                    and all(abs(shift(name, column)) <= 8 for name in alternatives
                                            for column in ("truth_in_leading_group", "truth_strict"))
                                    else "sensitivity: a rate changes by more than eight points"),
            "clocks widen": (lambda ratios: "Removing one planet "
                             + ("more than triples" if all(3 <= r <= 4 for r in ratios)
                                else "about triples" if all(2.5 <= r <= 3.5 for r in ratios)
                                else "changes otherwise") + " the median strict set")(
                [float(sensitivity.loc[name, "median_strict_count"]) / base_strict
                 for name in ("no_mars_family", "no_jupiter_family", "no_saturn_family")]),
            "month context ranks": ("The month context sets the width of the sets and the ranking"
                                    if float(sensitivity.loc["no_month_context", "median_strict_count"])
                                    > max(float(sensitivity.loc[name, "median_strict_count"])
                                          for name in ("no_mars_family", "no_jupiter_family", "no_saturn_family"))
                                    and float(sensitivity.loc["no_month_context", "median_leading_group"]) >= 10 * base_first
                                    else "month context: the sets do not widen most"),
            "conventions matter least": (lambda most: f"Each moves a rate by at most {word(most)} "
                                                      + ("point" if most == 1 else "points"))(
                max(abs(shift(name, column)) for name in conventions
                    for column in ("truth_in_leading_group", "truth_strict"))),
            "no agent rules": (lambda growth: (
                "Without the agent rules, each rate falls by at most two points, and the median strict set "
                + ("about doubles" if 1.7 <= growth <= 2.5 else "grows by about half" if 1.4 <= growth < 1.7
                   else "grows by about a third" if 1.2 <= growth < 1.4 else "changes otherwise"))
                if all(-2 <= shift("no_agent_suggested_rules", column) <= 0
                       for column in ("truth_in_leading_group", "truth_strict"))
                else "agent rules: a rate falls by more than two points")(
                float(sensitivity.loc["no_agent_suggested_rules", "median_strict_count"]) / base_strict),
            "eclipses untestable": f"Of {len(eclipse_rows)} eclipse clauses, the scribes mark "
                                   f"{int(eclipse_untestable_reasons.get('unobserved', 0))} as unobserved, and "
                                   f"{int(eclipse_untestable_reasons.get('no date', 0))} lack a secure month and day",
            "station contradictions": ("Most of its contradictions fall outside the window of the clause"
                                       if station_window_misses > len(station_contradictions) / 2
                                       else "station: most contradictions lie inside the window"),
            "calendar lag": f"fall a median of {med:.1f} days after the computed new moons",
            "sensitivity configurations": f"The sensitivity audit tests {len(SENSITIVITY_CONFIGS)} "
                                          "fixed configurations, including the baseline, on the "
                                          f"same {n_bench} records",
            "held-out tolerances": (f"Round I chose a planet tolerance of ${num(tolerances.get('ADART I', 0))}^\\circ$."
                                    if tolerances.get("ADART II") == tolerances.get("ADART III") == S.TOL_PLANET
                                    else "held-out: rounds II or III change the planet tolerance"),
            "held-out types": ("the distances of the outer planets to Normal Stars in compound "
                               "units supporting, and Round III also made the sign entries of the outer planets supporting"
                               if {name: sorted(rounds[name]["choices"]["demoted_types"]) for name in held.HELD_OUT}
                               == {"ADART I": [], "ADART II": ["distance to Normal Star|outer|compound"],
                                   "ADART III": ["distance to Normal Star|outer|compound", "sign entry|outer|-"]}
                               and not any(rounds[name]["choices"]["promoted_types"] for name in held.HELD_OUT)
                               else "held-out: the rounds change other decisive types"),
            "outer pairs and eclipses": (lambda types: (
                lambda pairs, lunar: f"Distances between two outer planets, for example, agree in only {pairs[0]} of "
                                     f"{pairs[1]} testable cases. Lunar eclipses agree in all {lunar[1]} of theirs"
                                     if lunar[0] == lunar[1] else "lunar eclipses: a testable case disagrees")(
                    [sum(row[k] for row in types if row["type"].startswith("planet distance to planet|outer|"))
                     for k in ("agree", "testable")],
                    [sum(row[k] for row in types if row["type"] == "eclipse|lunar|-") for k in ("agree", "testable")]))(
                rounds["all volumes"]["choices"]["types"]),
            "held-out reading": "passes if it lies at most "
                                f"{word(int(held.CLOSE_POINTS))} percentage points",
            "one record of ADART II": "because single volumes hold few records, such as "
                                      f"{held_rows.get('ADART II', {}).get('records')} in ADART II",
            "Wilson intervals": ((lambda ci: f"With {n_act} records, the 95\\% Wilson interval of the first-group "
                                  f"rate is {ci[0]}--{ci[1]}\\%")(wilson(first, n_act))
                                 if pooled["first_group"]["n"] == first
                                 else "held-out: the pooled first-group rate differs from the benchmark"),
            "addenda interval": (lambda ci: f"the 95\\% interval of their first-group rate, {ci[0]}--{ci[1]}\\%, is wide, "
                                 "and it holds the benchmark rate"
                                 if ci[0] <= int(pct(first, n_act)) <= ci[1] else "addenda: the interval misses the benchmark rate")(
                                     wilson(addenda["published_year_in_first_group"]["count"],
                                            addenda["texts_with_a_decisive_clause"])),
            "held-out result": ("With settings chosen on the other volumes, both pooled rates equal those of the "
                                f"full benchmark, well within the {word(int(held.CLOSE_POINTS))} points"
                                if all(r["close"] for r in reading.values())
                                and (pooled["first_group"]["n"], pooled["strict_set"]["n"]) == (first, strict_in)
                                else "held-out: a pooled rate differs from the benchmark"),
            "held-out (main text)": "Over the three rounds, the first group holds the published year in "
                                    f"{pct(pooled['first_group']['n'], pooled['records'])}\\% and the "
                                    f"strict set in {pct(pooled['strict_set']['n'], pooled['records'])}\\%, "
                                    "as in the full benchmark"
                                    if (pooled["first_group"]["n"], pooled["strict_set"]["n"]) == (first, strict_in)
                                    else "held-out: a pooled rate differs from the benchmark",
            "Britton": "\\lambda_{\\mathrm{Bab}}=\\lambda_{\\mathrm{date}}"
                       f"+{sky.BABYLONIAN_OFFSET_YEAR0_DEG:.2f}^\\circ-"
                       f"{sky.BABYLONIAN_PRECESSION_DEG_PER_CENTURY:.4f}^\\circ\\,Y",
            "tolerances": f"tolerances of ${num(S.TOL_MOON)}^\\circ$ for the Moon and "
                          f"${num(S.TOL_PLANET)}^\\circ$ for a planet",
        }
        missing = [label for label, phrase in claims.items() if phrase not in tex]
        ok("every number the paper prints follows from the data",
           not missing, "; ".join(f"{m}: '{claims[m][:90]}'" for m in missing[:3])
           + (f" (+{len(missing) - 3})" if len(missing) > 3 else ""))
        ok("the method constants are the ones the paper states",
           S.TOL_PLANET_PAIR == S.TOL_PLANET and not S.ORDER_CONTRADICTIONS_FIRST)

        def tex_rows(label: str) -> list[str]:
            """The rows of a TeX table, one string per row, as the paper prints them."""
            body = tex.split(f"\\label{{{label}}}", 1)[-1].split("\\midrule", 1)[-1]
            body = body.split("\\bottomrule", 1)[0].replace("\\addlinespace", "")
            return [" ".join(row.split()) for row in body.split("\\\\") if "&" in row]

        def row(*cells) -> str:
            return " & ".join(str(cell) for cell in cells)

        def paper_label(kind: str) -> str:
            """The paper names each clause kind as Table S3 does, without "measured"."""
            return scope_builder.LABELS[kind].replace(" measured", "")

        def cell(value, places: int = 0) -> str:
            return "\\textemdash" if pd.isna(value) else f"{float(value):.{places}f}"

        # Tables S3 and S4 list the same kinds. Their order is presentation, so
        # the rows are compared as sorted lists.
        want_s3 = sorted(row(paper_label(r.kind), cell(r.Clauses),
                             cell(r["Testable across candidate years"]),
                             cell(r["Median years allowed"]), cell(r.Bits, 1),
                             cell(r["Decisive %"])) for _, r in s3.iterrows())
        want_s4 = sorted(row(paper_label(r.kind), cell(r.Clauses), cell(r.Testable),
                             cell(r.Agree), cell(r.Contradict), cell(r.Untestable),
                             cell(r["Agreement %"], 1)) for _, r in s4.iterrows())
        want_s4.append(row("All types", reviewed_fw["clauses"], reviewed_fw["testable"],
                           reviewed_fw["agree"], reviewed_fw["contradict"],
                           reviewed_fw["untestable"],
                           f"{100 * reviewed_fw['agree'] / reviewed_fw['testable']:.1f}"))

        def s6(label: str, table: dict) -> str:
            n = table["records"]
            return row(label, n, *(f"{table[part]['n']} ({pct(table[part]['n'], n)}\\%)"
                                   for part in ("first_group", "first_three_groups", "strict_set")),
                       f"{num(table['median_strict_set'])} years")

        want_s6 = [s6(name, held_rows[name]) for name in held.HELD_OUT if name in held_rows]
        want_s6.append(s6("All three rounds", pooled))
        want_s6.append(s6("Rules from all volumes", held_rows.get("all volumes", pooled)))
        # The last row is the frozen test on the dated addenda of ADART IV that join no record of ADART I to III.
        addenda_json = json.loads((HERE / "data" / "benchmark" / "addenda_results.json").read_text(encoding="utf-8"))
        addenda_dates = pd.read_parquet(HERE / "data" / "results" / "adart_iv_addenda_dates.parquet").set_index("tablet")
        addenda_new = [t for t in addenda_json["texts_with_a_decisive_clause"] if not t["linked_to_adart_i_iii"]]

        def addenda_in(t, column):
            held = addenda_dates.at[t["tablet"], column]
            held = json.loads(held) if isinstance(held, str) else [int(y) for y in held]
            return any(y in held for y in t["published_years_astro"])

        addenda_row = {"records": len(addenda_new),
                       "first_group": {"n": sum(t["published_year_in_first_group"] for t in addenda_new)},
                       "first_three_groups": {"n": sum(addenda_in(t, "top_three_evidence_groups_years_astro") for t in addenda_new)},
                       "strict_set": {"n": sum(t["published_year_in_strict_set"] for t in addenda_new)},
                       "median_strict_set": addenda_json["scores"]["not_linked_to_adart_i_iii"]["median_strict_set_size"]}
        want_s6.append(s6("Frozen rules, ADART IV addenda", addenda_row))
        for name, label, want in (("S3", "tab:information", want_s3),
                                  ("S4", "tab:forward", want_s4),
                                  ("S6", "tab:heldout", want_s6)):
            have = tex_rows(label) if name == "S6" else sorted(tex_rows(label))
            if name == "S4":
                want = sorted(want)
            wrong = [f"'{h}' should be '{w}'" for h, w in zip(have, want) if h != w]
            ok(f"SI Table {name} prints the rows the data give",
               have == want, "; ".join(wrong[:2]) or f"{len(have)} rows, {len(want)} expected")

    else:
        warns.append("paper/PNAS_Paper.tex is kept outside the repository, so the paper checks did not run")

    head("repository layout")
    files, dirs = tree()
    names = {Path(rel).name for rel in files}
    readmes = [rel for rel in files if Path(rel).name == "README.md"]
    scripts_readme = (HERE / "scripts" / "README.md").read_text(encoding="utf-8")
    packages = sorted(p.parent.name for p in (HERE / "scripts").glob("*/__init__.py"))
    python_files = [rel for rel in files if rel.endswith(".py")]
    undeclared = [rel for rel in python_files
                  if not ast.get_docstring(ast.parse(
                      (HERE / rel).read_text(encoding="utf-8")))]
    ok("every Python file declares its purpose in a module docstring",
       not undeclared, ", ".join(undeclared[:4]))
    code_dirs = ["tools", "tests", *(f"scripts/{p}" for p in packages)]
    unlisted = []
    for folder in code_dirs:
        readme = HERE / folder / "README.md"
        listed = readme.read_text(encoding="utf-8") if readme.exists() else ""
        unlisted += [rel for rel in python_files
                     if rel.startswith(folder + "/") and Path(rel).name != "__init__.py"
                     and f"`{Path(rel).name}`" not in listed]
    ok("every script, tool and test is listed in its directory's README",
       not unlisted, ", ".join(unlisted[:4]))
    # The run order is the text block of scripts/README.md and the Makefile.
    # A module that another script imports is a library. Anything else is
    # named in scripts/README.md as an exception, by its path under scripts/.
    block = re.search(r"```text\n(.*?)```", scripts_readme, re.S)
    scheduled = set()
    for line in (block.group(1).splitlines() if block else []):
        step = re.match(r"^(\w+)/\s+(.*)$", line)
        if step:
            scheduled.update(f"{step.group(1)}/{name}.py" for name in step.group(2).split()
                             if not name.startswith("-"))
    makefile = "\n".join(line.split("#", 1)[0] for line in
                         (HERE / "Makefile").read_text(encoding="utf-8").splitlines())
    figures_line = re.search(r"^FIGURES\s*:=\s*(.+)$", makefile, re.M)
    make_figures = figures_line.group(1).split() if figures_line else []
    scheduled |= {f"paper/figures/{name}.py" for name in make_figures}
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
                   and Path(rel).stem not in imported
                   and f"`{rel[len('scripts/'):]}`" not in scripts_readme]
    vanished = sorted(name for name in scheduled if not (HERE / "scripts" / name).is_file())
    ok("every script is in the run order or named in scripts/README.md as an exception",
       bool(block) and not unscheduled and not vanished,
       ", ".join(unscheduled + [f"{name} (scheduled, absent)" for name in vanished]))
    patterns = ignore_patterns()
    scans = {Path(rel).name for rel in NOT_DISTRIBUTED}
    # Files in paper/ belong to the paper, which is kept outside the repository.
    ghosts = []
    for rel in readmes:
        folder = Path(rel).parent
        text = (HERE / rel).read_text(encoding="utf-8")
        for token in re.findall(r"`([^`\s]+)`", text):
            if (not FILE_TOKEN.fullmatch(token) or token.startswith((".", "/", "~"))
                    or not re.search(r"[A-Za-z]", Path(token).stem)):
                continue
            if (any((HERE / base / token).exists()
                    for base in (Path("."), folder, Path("scripts"), Path("data")))
                    or Path(token).name in names or Path(token).name in scans
                    or token.startswith("paper/")
                    or any(fnmatch.fnmatch(Path(token).name, Path(p.strip("/")).name)
                           for p in patterns)):
                continue
            ghosts.append(f"{rel}: {token}")
    ok("no README names a file that is gone", not ghosts, ", ".join(ghosts[:4]))

    # data/README.md has a section per folder, headed "## <folder>/", whose
    # tables name each file by its path inside that folder. A file of a run
    # directory may be named without the directory, and a placeholder such
    # as model_review_<model> stands for any name.
    named: dict[str, set[str]] = {}
    folder = ""
    readme = HERE / "data" / "README.md"
    for line in (readme.read_text(encoding="utf-8").splitlines()
                 if readme.exists() else []):
        heading = re.match(r"^## (\S+)/\s*$", line)
        if heading:
            folder = heading.group(1)
            continue
        named.setdefault(folder, set()).update(
            re.sub(r"<[^>]+>", "*", token)
            for token in re.findall(r"`([^`\s]+)`", line) if not token.endswith("/"))
    undocumented = []
    for rel in files:
        parts = rel.split("/")
        if parts[0] != "data" or rel == "data/README.md":
            continue
        section, inside = (parts[1], "/".join(parts[2:])) if len(parts) > 2 else ("", parts[1])
        if not any(fnmatch.fnmatchcase(inside, token)
                   or fnmatch.fnmatchcase(inside, "*/" + token)
                   for token in named.get(section, ())):
            undocumented.append(rel)
    markdown = [rel for rel in files if rel.endswith(".md")
                and Path(rel).name != "README.md" and rel not in DOCUMENTS]
    undocumented_dirs = [folder for folder in ("data", "scripts", "skills",
                                               *code_dirs)
                         if not (HERE / folder / "README.md").is_file()]
    ok("every data file is described in data/README.md",
       not undocumented and not markdown and not undocumented_dirs,
       ", ".join((undocumented + markdown + undocumented_dirs)[:4]))
    top = {rel.split("/")[0] for rel in files + dirs}
    transient = [rel for rel in dirs if Path(rel).name in TRANSIENT_DIRS]
    ok("the top level holds only the documented files and directories, and no "
       "scratch or cache directory", top <= TOP_LEVEL and not transient,
       ", ".join(sorted(top - TOP_LEVEL) + transient))
    if PAPER.is_file():
        # The preamble names the PNAS logo, which sits in paper/ itself, to remove it from the arXiv version.
        included = [name for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                                                strip_tex_comments(tex_source))
                    if name not in PAPER_PDFS]
        inputs = re.findall(r"\\input\{([^}]+)\}", strip_tex_comments(tex_source))
        stray = []
        for rel in files:
            parts = rel.split("/")
            if parts[0] != "paper":
                continue
            name, suffix = parts[-1], Path(rel).suffix
            if len(parts) == 2 and suffix in PAPER_SUFFIXES and (
                    suffix not in {".pdf", ".tex"} or name in PAPER_PDFS
                    or name == "PNAS_Paper.tex" or Path(name).stem in inputs):
                continue
            if len(parts) == 3 and parts[1] == "Figures" and suffix in FIGURE_SUFFIXES \
                    and name in included:
                continue
            stray.append(rel)
        missing_inputs = ([f"paper/Figures/{name}" for name in included
                           if not (HERE / "paper" / "Figures" / name).is_file()]
                          + [f"paper/{name}.tex" for name in inputs
                             if not (HERE / "paper" / f"{name}.tex").is_file()])
        figure_modules = {Path(rel).stem for rel in python_files
                          if rel.startswith("scripts/paper/figures/")
                          and re.match(r"figS?\d", Path(rel).stem)}
        ok("paper/ holds only what LaTeX needs, and every figure and table it includes",
           not stray and not missing_inputs
           and {f"{name}.pdf" for name in make_figures} <= set(included)
           and figure_modules == set(make_figures)
           and (HERE / "paper" / "PNAS_Paper.pdf").is_file()
           and (HERE / "paper" / "PNAS_Paper.pdf").stat().st_size > 0,
           ", ".join((stray + [f"{rel} missing" for rel in missing_inputs])[:4]))
        cited_source = strip_tex_comments(tex_source)
        bibliography = (HERE / "paper" / "Bibliography.bib").read_text(encoding="utf-8")
        entries = set(re.findall(r"@(?!comment|string|preamble)\w+\s*\{\s*([^,\s]+)\s*,",
                                 bibliography, re.I))
        cited = {key.strip() for keys in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]*)\}",
                                                    cited_source)
                 for key in keys.split(",") if key.strip()}
        ok("every citation has a bibliography entry, and every entry is cited",
           cited == entries,
           ", ".join([f"unresolved {k}" for k in sorted(cited - entries)]
                     + [f"uncited {k}" for k in sorted(entries - cited)][:4]))
        acceptance = HERE / "tests" / "test_paper_numbers.py"
        recipe = re.search(r"^check:.*?(?=^\S|\Z)", makefile, re.M | re.S)
        ok("the paper's acceptance test is present and make check runs it, then this file",
           acceptance.is_file()
           and "unittest.TestCase" in acceptance.read_text(encoding="utf-8")
           and "PNAS_Paper.tex" in acceptance.read_text(encoding="utf-8")
           and bool(recipe) and "tests/test_*.py" in recipe.group(0)
           and "verify.py" in recipe.group(0)
           and recipe.group(0).index("tests/test_*.py") < recipe.group(0).index("verify.py"))

    print(f"\n{'-' * 60}")
    for note in warns:
        print(f"optional absent: {note}")
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED: {', '.join(fails)}")
        print("Do not trust the generated results until these pass.")
        return 1
    print(f"all checks passed in {time.time() - t0:.1f}s")
    print("The scientific pipeline and the committed results agree"
          + (", and so does the paper." if PAPER.is_file() else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
