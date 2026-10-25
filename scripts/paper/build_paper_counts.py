#!/usr/bin/env python3
"""Write the counts and SI Tables S3 and S4 that the paper quotes for ADART I to III.

It reads saved outputs only: the corpus, the calendar, the parser clauses, the
reviewed clauses of the committed model review (RUN_DIR), the results and the
benchmark. It runs no model, extraction or astronomy. It refuses to run if a
saved score no longer matches the current tools and physical tables, or if an
input changes while it runs.

Outputs, in data/paper_counts/:

    counts.json            the counts the paper quotes
    table_s3.csv           SI Table S3, bits by observation type
    table_s4.csv           SI Table S4, agreement at the published year
    source_manifest.json   the SHA-256 of every input

Run from the repository root:

    python scripts/paper/build_paper_counts.py

With --clause-dir data/clauses/model_review_<model>, it reads the reviewed
clauses of another run, as scripts/agent/rescore_model_review.py writes them,
and writes to <clause-dir>/paper_counts/, so that the committed tables are
never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCOPE = ("adsd/adart1", "adsd/adart2", "adsd/adart3")
KEY = ["tablet", "statement_index"]
LABELS = {
    "appulse": "Moon or planet measured against a Normal Star",
    "in_sign": "Planet in a named sign",
    "last_app": "Last appearance in a sign",
    "eclipse": "Eclipse of a stated type",
    "entry": "Planet enters a sign",
    "planet_appulse": "Moon measured against a planet",
    "station_appulse": "Station measured against a Normal Star",
    "planet_planet_appulse": "Planet measured against a planet",
    "station": "Planet stations in a sign",
    "first_app": "First appearance in a sign",
    "season": "Solstice or equinox on a day",
    "sirius": "Sirius date on a day",
    "acronychal": "Acronychal rising on a day",
}
FACT_LABELS = {
    "appulse": "Normal-Star relation",
    "planet_appulse": "Moon-planet relation",
    "station_appulse": "Station-star relation",
    "planet_planet_appulse": "Planet-planet relation",
    "in_sign": "Planet in sign",
    "last_app": "Last appearance sign",
    "entry": "Sign entry",
    "eclipse": "Typed eclipse",
    "season": "Solstice or equinox",
    "sirius": "Sirius scheme date",
    "acronychal": "Acronychal rising",
    "first_app": "First appearance sign",
    "station": "Station in sign",
}
# The committed run of the model review, which the paper reports.
RUN_DIR = "data/clauses/model_review_gpt-6-luna-2026-09-22"
FILES = {
    "tablets": "data/corpus/tablets.parquet",
    "lines": "data/corpus/lines.parquet",
    "calendar": "data/physical/calendar.parquet",
    "information": f"{RUN_DIR}/reviewed_clause_bits.parquet",
    "reviewed_facts": f"{RUN_DIR}/reviewed_clause_verdicts.parquet",
    "deterministic_facts": "data/clauses/parser_clauses.parquet",
    "production": "data/results/tablet_dates.parquet",
    "frozen": "data/benchmark/benchmark_records.parquet",
    "benchmark": "data/benchmark/benchmark_results.parquet",
}
# This is the exact dependency list used by the current table producer's
# scoring_fingerprint(). No producer or model module is imported.
SCORE_DEPENDENCIES = (
    "tools/statements.py", "tools/sky.py", "tools/cal.py", "tools/eclipses.py",
    "scripts/agent/model_review.py",
    "data/physical/ephemeris.parquet", "data/physical/calendar.parquet",
    "data/physical/eclipses.parquet", "data/physical/normal_stars.parquet",
)
PRODUCTION_DEPENDENCIES = {
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


class ScopeError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ScopeError(message)


def columns(frame, names, description):
    missing = set(names) - set(frame.columns)
    require(not missing, f"{description}: missing columns {sorted(missing)}.")


def unique(frame, keys, description):
    columns(frame, keys, description)
    require(frame[keys].notna().all().all(), f"{description}: empty identifiers.")
    require(not frame.duplicated(keys).any(), f"{description}: duplicate {keys}.")


def bools(values, description):
    def convert(value):
        if pd.isna(value):
            raise ScopeError(f"{description}: missing Boolean value.")
        if isinstance(value, str):
            require(value.strip().lower() in {"true", "false", "0", "1"},
                    f"{description}: invalid Boolean {value!r}.")
            return value.strip().lower() in {"true", "1"}
        require(value in (True, False, 0, 1),
                f"{description}: invalid Boolean {value!r}.")
        return bool(value)
    return values.map(convert).astype(bool)


def sha256(path):
    require(path.is_file(), f"Missing required file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def attach_scope(frame, catalogue, description):
    unique(catalogue, ["tablet"], "catalogue")
    columns(catalogue, ["project"], "catalogue")
    columns(frame, ["tablet"], description)
    result = frame.copy()
    result["tablet"] = result.tablet.astype(str)
    owner = catalogue.set_index("tablet").project
    result["_project"] = result.tablet.map(owner)
    require(result._project.notna().all(),
            f"{description}: some tablet IDs are absent from the catalogue.")
    if "project" in result:
        require(result.project.eq(result._project).all(),
                f"{description}: volume labels disagree with the catalogue.")
    result["_in_scope"] = result._project.isin(SCOPE)
    return result


def validate_alignment(information, facts):
    unique(information, KEY, "per-clause information")
    unique(facts, KEY, "reviewed forward ledger")
    left = set(information[KEY].itertuples(index=False, name=None))
    right = set(facts[KEY].itertuples(index=False, name=None))
    require(left == right,
            "The information and forward files contain different clauses. "
            "Regenerate both with scripts/agent/rescore_model_review.py.")
    fields = ["kind", "planet", "parser_rule", "restored", "omitted"]
    columns(information, fields, "per-clause information")
    columns(facts, fields, "reviewed forward ledger")
    paired = information[KEY + fields].merge(
        facts[KEY + fields], on=KEY, validate="one_to_one",
        suffixes=("_information", "_forward"))
    for field in fields:
        a, b = paired[field + "_information"], paired[field + "_forward"]
        if field in {"restored", "omitted"}:
            a, b = bools(a, field), bools(b, field)
        else:
            a, b = a.fillna("").astype(str), b.fillna("").astype(str)
        require(a.eq(b).all(), f"Clause semantics differ between files: {field}.")


def information_table(information, candidate_count):
    columns(information, ["kind", "agreeing_years", "testable_years",
                          "decisive_tier"], "per-clause information")
    require(set(information.kind).issubset(LABELS),
            f"Unsupported clause kinds: {sorted(set(information.kind) - LABELS.keys())}")
    agreeing = pd.to_numeric(information.agreeing_years, errors="coerce")
    testable = pd.to_numeric(information.testable_years, errors="coerce")
    require(agreeing.notna().all() and testable.notna().all(),
            "Per-clause candidate counts are missing or nonnumeric.")
    require(((agreeing >= 0) & (agreeing <= testable) &
             (testable <= candidate_count) &
             agreeing.eq(agreeing.round()) &
             testable.eq(testable.round())).all(),
            "Invalid agreeing/testable counts for the candidate calendar.")
    work = information.copy()
    work["agreeing_years"], work["testable_years"] = agreeing, testable
    work["decisive_tier"] = bools(work.decisive_tier, "decisive_tier")
    rows = []
    for kind, group in work.groupby("kind"):
        usable = group[group.testable_years > 0]
        # Preserve the producer's minimum of five usable examples for
        # reporting selectivity. Keep every kind's clause count visible.
        enough = len(usable) >= 5
        median = float(usable.agreeing_years.median()) if enough else None
        bits = (math.log2(candidate_count / median) if median > 0 else 0.0
                ) if enough else None
        rows.append({
            "Observation type": LABELS[kind], "Clauses": len(group),
            "Median years allowed": int(pd.Series([median]).round(0).iloc[0])
            if enough else None,
            "Bits": float(pd.Series([bits]).round(1).iloc[0]) if enough else None,
            "Decisive %": int(pd.Series([
                100.0 * usable.decisive_tier.mean()]).round(0).iloc[0])
            if enough else None,
            "Testable across candidate years": len(usable),
            "Median years allowed unrounded": median,
            "Decisive % unrounded":
                100.0 * float(usable.decisive_tier.mean()) if enough else None,
            "kind": kind,
        })
    return pd.DataFrame(rows).sort_values(
        ["Clauses", "kind"], ascending=[False, True]).reset_index(drop=True)


def forward_counts(facts):
    columns(facts, ["forward_verdict"], "forward ledger")
    allowed = {"agree", "contradict", "untestable", "not_evaluated"}
    require(facts.forward_verdict.isin(allowed).all(),
            "The forward ledger contains an unknown or missing verdict.")
    count = facts.forward_verdict.value_counts()
    output = {key: int(count.get(key, 0)) for key in sorted(allowed)}
    output["clauses"] = len(facts)
    output["records"] = int(facts.tablet.nunique())
    output["testable"] = output["agree"] + output["contradict"]
    output["agreement_percent"] = (
        100.0 * output["agree"] / output["testable"]
        if output["testable"] else None)
    require(sum(output[k] for k in allowed) == len(facts),
            "Forward verdict accounting failed.")
    return output


def forward_table(facts):
    rows = []
    for kind in FACT_LABELS:
        group = facts[facts.kind.eq(kind)]
        if group.empty:
            continue
        n = forward_counts(group)
        rows.append({
            "Observation type": FACT_LABELS[kind], "Clauses": len(group),
            "Testable": n["testable"], "Agree": n["agree"],
            "Contradict": n["contradict"], "Untestable": n["untestable"],
            "Not evaluated": n["not_evaluated"],
            "Agreement %": float(pd.Series([n["agreement_percent"]]).round(1).iloc[0])
            if n["testable"] else None, "kind": kind,
        })
    result = pd.DataFrame(rows)
    require(int(result.Clauses.sum()) == len(facts),
            "A clause kind was omitted from the forward table.")
    return result


def values(frame, column):
    return {str(k): int(v) for k, v in frame[column].value_counts().items()}


def production_summary(production, catalogue):
    columns(production, ["claim", "basis", "chronology_relation",
                         "strict_years_astro", "strict_count"], "tablet results")
    unique(production, ["tablet"], "tablet results")
    require(production.claim.isin(["committed", "ranked", "decline"]).all(),
            "Unexpected astronomy-only claim in tablet results.")
    committed = production[production.claim.eq("committed")].copy()
    committed = committed.merge(
        catalogue[["tablet", "bce_first", "bce_second"]],
        on="tablet", validate="one_to_one")
    rows = []
    for row in committed.itertuples(index=False):
        years = [int(y) for y in row.strict_years_astro]
        require(len(years) == 1 and int(row.strict_count) == 1,
                f"Commitment for {row.tablet} is not a singleton.")
        # A "150/149" reference is the one Babylonian year beginning in
        # 150 BCE; bce_second is the Julian year it ends in.
        references = set()
        if pd.notna(row.bce_first):
            first = 1 - int(row.bce_first)
            last = (-int(row.bce_second) if pd.notna(row.bce_second)
                    else first)
            references = set(range(first, max(first, last) + 1))
        matches = years[0] in references if references else None
        rows.append({
            "tablet": row.tablet, "basis": row.basis,
            "year_astro": years[0],
            "year_label": year_label(years[0]),
            "reference_bce_first": row.bce_first,
            "reference_bce_second": row.bce_second,
            "agrees_with_reference": matches,
        })
    verified = [r for r in rows if r["agrees_with_reference"] is not None]
    require(all(r["agrees_with_reference"] for r in verified),
            "A saved I--III commitment conflicts with the reference. "
            "Inspect the latest verification before publishing this summary.")
    summary = {
        "records": len(production),
        "astronomy_claims": values(production, "claim"),
        "astronomy_committed_bases": values(
            production[production.claim.eq("committed")], "basis"),
        "astronomy_decline_bases": values(
            production[production.claim.eq("decline")], "basis"),
        "chronology_relations": values(production, "chronology_relation"),
        "reference_checked_commitments": len(verified),
        "reference_correct_commitments":
            sum(r["agrees_with_reference"] for r in verified),
        "method_versions": sorted(
            production.method_version.astype(str).unique())
            if "method_version" in production else [],
    }
    if "combined_claim" in production:
        summary["combined_claims"] = values(production, "combined_claim")
    if "combined_basis" in production:
        summary["combined_bases"] = values(production, "combined_basis")
    return summary, pd.DataFrame(rows, columns=[
        "tablet", "basis", "year_astro", "year_label",
        "reference_bce_first", "reference_bce_second", "agrees_with_reference"])


def year_label(year):
    return f"AD {year}" if year > 0 else f"{1 - year} BCE"


def benchmark_summary(benchmark, frozen):
    unique(benchmark, ["tablet"], "saved benchmark")
    unique(frozen, ["tablet"], "frozen population")
    columns(frozen, ["tool_benchmark_eligible", "tool_year_bearing"],
            "frozen population")
    columns(benchmark, ["frozen_active"], "saved benchmark")
    eligible = frozen[bools(frozen.tool_benchmark_eligible, "benchmark eligibility")]
    require(set(benchmark.tablet) == set(eligible.tablet),
            "Saved benchmark IDs differ from the complete frozen population.")
    require(benchmark._in_scope.all() and eligible._in_scope.all(),
            "The frozen dating benchmark contains a record outside ADART I--III.")
    active = bools(benchmark.frozen_active, "frozen_active")
    expected = eligible.set_index("tablet").tool_year_bearing
    require(active.eq(bools(benchmark.tablet.map(expected),
                           "frozen year-bearing membership")).all(),
            "Saved benchmark membership differs from the frozen ledger.")
    report = {}
    for name, part in (("all_eligible", benchmark),
                       ("frozen_year_bearing", benchmark[active]),
                       ("frozen_supporting_only", benchmark[~active])):
        record = {"records": len(part)}
        for col in ("truth_in_strict", "truth_in_safety", "unique",
                    "truth_in_top_evidence_group", "display_top1_correct",
                    "display_top3_correct"):
            if col in part:
                count = int(bools(part[col], col).sum())
                record[col] = {"count": count,
                               "percent": 100.0 * count / len(part)
                               if len(part) else None}
        if "truth_evidence_group_rank" in part:
            count = int(part.truth_evidence_group_rank.le(3).sum())
            record["truth_in_first_three_evidence_levels"] = {
                "count": count,
                "percent": 100.0 * count / len(part) if len(part) else None}
        for col in ("strict_count", "safety_count", "top_evidence_group_size",
                    "top_three_evidence_groups_size"):
            if col in part:
                record["median_" + col] = (
                    float(part[col].median()) if len(part) else None)
        report[name] = record
    return report


def validate_freshness(root, frames, years, digest):
    info = frames["information"]
    required = ["candidate_first_astro", "candidate_last_astro",
                "candidate_year_count", "scoring_fingerprint"]
    missing = set(required) - set(info.columns)
    require(not missing,
            "Saved per-clause scores predate the current provenance format "
            f"(missing {sorted(missing)}). Regenerate them with the current "
            "scripts/agent/rescore_model_review.py; it reuses the model caches.")
    require(info[required].notna().all().all(),
            "Per-clause scores have incomplete calendar/provenance metadata.")
    for col, expected in (("candidate_first_astro", years[0]),
                          ("candidate_last_astro", years[-1]),
                          ("candidate_year_count", len(years))):
        require(info[col].eq(expected).all(),
                f"Saved per-clause scores use a different calendar: {col}.")
    hashes = {p: digest(p) for p in SCORE_DEPENDENCIES}
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    require(info.scoring_fingerprint.eq(fingerprint).all(),
            "Saved per-clause scores do not match the current astronomy/tools. "
            "Run scripts/agent/rescore_model_review.py first; it makes no new "
            "model calls. This scope script cannot repair stale scores.")
    facts = frames["deterministic_facts"]
    columns(facts, ["extractor_sha256"], "deterministic ledger")
    require(facts.extractor_sha256.eq(digest("tools/statements.py")).all(),
            "The deterministic ledger is stale relative to tools/statements.py.")
    production = frames["production"]
    for col, path in PRODUCTION_DEPENDENCIES.items():
        columns(production, [col], "tablet results")
        require(production[col].notna().all() and
                production[col].eq(digest(path)).all(),
                f"Saved tablet results are stale relative to {path}. "
                "Use your latest verified tablet_dates.parquet.")
    return fingerprint



CLAUSE_FILES = {
    "information": "reviewed_clause_bits.parquet",
    "reviewed_facts": "reviewed_clause_verdicts.parquet",
}


def input_files(root, clause_dir=None):
    """FILES, with the two per-clause files taken from a run directory if given."""
    files = dict(FILES)
    if clause_dir is not None:
        try:
            relative = clause_dir.resolve().relative_to(root)
        except ValueError:
            raise ScopeError(f"--clause-dir must lie inside the repository: {clause_dir}")
        for name, filename in CLAUSE_FILES.items():
            files[name] = str(relative / filename)
    return files


def build(root, output, files=FILES):
    paths = {name: root / relative for name, relative in files.items()}
    for path in paths.values():
        require(path.is_file(), f"Missing saved input: {path}")
    states = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths.values()}
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    catalogue = frames["tablets"]
    columns(catalogue, ["tablet", "project", "n_lines", "bce_first",
                        "bce_second"], "catalogue")
    catalogue["tablet"] = catalogue.tablet.astype(str)
    catalogue["project"] = catalogue.project.astype(str).str.strip().str.lower()
    unique(catalogue, ["tablet"], "catalogue")
    line_numbers = pd.to_numeric(catalogue.n_lines, errors="coerce")
    require(line_numbers.notna().all() and
            ((line_numbers >= 0) & line_numbers.eq(line_numbers.round())).all(),
            "The catalogue contains invalid edition-line counts.")
    require(set(SCOPE).issubset(set(catalogue.project)),
            "The catalogue does not contain all three required ADART volumes.")
    columns(frames["calendar"], ["bab_year_astro"], "calendar")
    years = sorted(int(y) for y in frames["calendar"].bab_year_astro.unique())
    require(years and years == list(range(years[0], years[-1] + 1)),
            "The candidate calendar has missing years.")
    hashes = {}

    def digest(relative):
        if relative not in hashes:
            hashes[relative] = sha256(root / relative)
        return hashes[relative]

    fingerprint = validate_freshness(root, frames, years, digest)
    scoped = {}
    for name in ("lines", "information", "reviewed_facts",
                 "deterministic_facts", "production", "frozen", "benchmark"):
        frames[name] = attach_scope(frames[name], catalogue, name)
        scoped[name] = frames[name][frames[name]._in_scope].copy()
    for name in ("information", "reviewed_facts", "deterministic_facts"):
        unique(frames[name], KEY, name)
    validate_alignment(frames["information"], frames["reviewed_facts"])
    unique(frames["production"], ["tablet"], "tablet results")
    require(set(frames["production"].tablet) == set(catalogue.tablet),
            "Saved production results do not cover the complete catalogue.")
    line_counts = frames["lines"].groupby("tablet").size()
    require(catalogue.tablet.map(line_counts).fillna(0).astype(int).eq(
        catalogue.n_lines.astype(int)).all(),
        "Catalogue n_lines does not match the saved edition-line table.")
    metadata = catalogue[catalogue.project.isin(SCOPE)].copy()
    require(metadata.bce_first.notna().all(),
            "An I--III record lacks the expected catalogue reference year.")
    columns(frames["reviewed_facts"], ["kind", "forward_verdict"],
            "reviewed forward ledger")
    scope_forward = forward_counts(scoped["reviewed_facts"])
    full_forward = forward_counts(frames["reviewed_facts"])
    require(scope_forward["not_evaluated"] == 0,
            "Some I--III clauses are marked not_evaluated despite reference dates.")
    for verdict in ("agree", "contradict", "untestable"):
        require(scope_forward[verdict] == full_forward[verdict],
                "Forward-evaluated clauses outside I--III were found. "
                "Inspect the volume breakdown before updating the paper.")
    table1 = information_table(scoped["information"], len(years))
    table3 = forward_table(scoped["reviewed_facts"])
    production, _ = production_summary(scoped["production"], catalogue)
    benchmark = benchmark_summary(frames["benchmark"], frames["frozen"])
    deterministic = forward_counts(scoped["deterministic_facts"])
    scope_counts = {
        "catalogue_records": len(metadata),
        "records_with_edition_text": int(metadata.n_lines.gt(0).sum()),
        "edition_lines": len(scoped["lines"]),
        "reviewed_clauses": len(scoped["reviewed_facts"]),
        "reviewed_records": int(scoped["reviewed_facts"].tablet.nunique()),
        "deterministic_facts": len(scoped["deterministic_facts"]),
        "deterministic_records": int(scoped["deterministic_facts"].tablet.nunique()),
        "net_clause_inventory_increase":
            len(scoped["reviewed_facts"]) - len(scoped["deterministic_facts"]),
        "net_record_coverage_increase":
            int(scoped["reviewed_facts"].tablet.nunique()) -
            int(scoped["deterministic_facts"].tablet.nunique()),
    }
    summary = {
        "schema": "adart-i-iii-manuscript-scope-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "filter_and_aggregate_current_saved_outputs",
        "included_projects": list(SCOPE),
        "excluded_projects": sorted(set(catalogue.project) - set(SCOPE)),
        "candidate_calendar": {"years": len(years),
                               "first_astro": years[0], "last_astro": years[-1],
                               "first_label": year_label(years[0]),
                               "last_label": year_label(years[-1])},
        "original_inventory": {
            "catalogue_records": len(catalogue),
            "reviewed_clauses": len(frames["reviewed_facts"]),
            "reviewed_records": int(frames["reviewed_facts"].tablet.nunique()),
            "deterministic_facts": len(frames["deterministic_facts"]),
            "deterministic_records":
                int(frames["deterministic_facts"].tablet.nunique())},
        "adart_i_iii": scope_counts, "reviewed_forward": scope_forward,
        "deterministic_forward": deterministic, "production": production,
        "existing_benchmark": benchmark,
        "scope_checks": {
            "all_scoped_clause_and_production_rows_in_i_iii": True,
            "reviewed_clause_files_have_identical_keys_and_fields": True,
            "forward_agreement_counts_unchanged_by_volume_filter": True,
            "frozen_benchmark_has_no_v_vi_records": True,
            "candidate_scores_match_current_tools_and_calendar": True,
            "production_and_deterministic_provenance_match_current_tools": True},
        "table1_scoring_fingerprint": fingerprint,
    }
    for relative in files.values():
        digest(relative)
    require(all((p.stat().st_size, p.stat().st_mtime_ns) == state
                for p, state in states.items()),
            "An input changed while reporting was running. Run after other jobs finish.")
    manifest = {
        "schema": "adart-scope-input-manifest-v1",
        "script_sha256": sha256(Path(__file__)),
        "inputs": {p: {"sha256": h, "bytes": (root / p).stat().st_size}
                   for p, h in sorted(hashes.items())},
        "note": "Hashes identify the saved inputs. No model responses or API keys are included."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="adart-scope-", dir=output.parent) as temp:
        stage = Path(temp)
        table1.to_csv(stage / "table_s3.csv", index=False)
        table3.to_csv(stage / "table_s4.csv", index=False)
        write_json(stage / "counts.json", summary)
        write_json(stage / "source_manifest.json", manifest)
        output.mkdir(parents=True, exist_ok=True)
        for path in sorted(stage.iterdir()):
            shutil.copyfile(path, output / path.name)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        help="Repository root; normally detected automatically.")
    parser.add_argument("--clause-dir", type=Path,
                        help="Run directory holding the per-clause files of one "
                             "model run; default RUN_DIR, the committed run.")
    parser.add_argument("--out-dir", type=Path,
                        help="Output directory; default data/paper_counts, "
                             "or <clause-dir>/paper_counts with --clause-dir.")
    args = parser.parse_args(argv)
    if args.root:
        root = args.root.expanduser().resolve()
    else:
        candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
        root = next((p for p in candidates if
                     (p / FILES["tablets"]).is_file()), candidates[0]).resolve()
    clause_dir = args.clause_dir.expanduser().resolve() if args.clause_dir else None
    output = (args.out_dir.expanduser().resolve() if args.out_dir else
              clause_dir / "paper_counts" if clause_dir else
              root / "data/paper_counts")
    try:
        result = build(root, output, input_files(root, clause_dir))
    except (ScopeError, ImportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    count = result["adart_i_iii"]
    forward = result["reviewed_forward"]
    print("ADART I--III scope reporting completed")
    print(f"  catalogue: {count['catalogue_records']} records; "
          f"{count['records_with_edition_text']} with edition text; "
          f"{count['edition_lines']} lines")
    print(f"  reviewed: {count['reviewed_clauses']} clauses / "
          f"{count['reviewed_records']} records")
    print(f"  deterministic: {count['deterministic_facts']} facts / "
          f"{count['deterministic_records']} records")
    print(f"  forward: agree {forward['agree']}; contradict "
          f"{forward['contradict']}; untestable {forward['untestable']}; "
          f"not evaluated {forward['not_evaluated']}")
    print("  astronomy-only claims:",
          json.dumps(result["production"]["astronomy_claims"], sort_keys=True))
    print("  scope and provenance checks: passed")
    print(f"Outputs: {output}")
    print(f"Upload this file: {output / 'manuscript_scope_update.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
