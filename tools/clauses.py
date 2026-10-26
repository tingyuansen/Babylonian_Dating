"""Read the reviewed clauses, which the dates use, from the committed clause tables.

The parser reads each translation into clauses. The language model then reviews
each parser clause and adds the observations that the parser missed
(scripts/agent/model_review.py, SI Appendix section S2).
scripts/agent/rescore_model_review.py rebuilds these final clauses from the
review caches and writes them into one table per corpus, one row per clause.
This module turns the rows back into Statements in the order of the table. The
tables hold no published date.

Three columns record where each clause comes from:

    origin         "parser" for a reviewed parser clause, "search" for an
                   observation that the search added
    parser_index   the clause's index in tools.statements.extract(text), or -1
    review         "keep", "modify" or "add"
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from tools import statements as S

HERE = Path(__file__).resolve().parents[1]
CLAUSES = HERE / "data" / "clauses"
# The reviewed clauses of ADART I to III, and those of the undated ADART IV entries.
REVIEWED = CLAUSES / "reviewed_clauses.parquet"
ADART_IV_REVIEWED = CLAUSES / "adart_iv_reviewed_clauses.parquet"

_FIELDS = [field.name for field in fields(S.Statement)]


def to_rows(tablet: str, statements: list[S.Statement],
            provenance: list[tuple[str, int, str]]) -> list[dict]:
    """One table row per clause, in the order of the record's clause list."""
    rows = []
    for index, (st, (origin, parser_index, review)) in enumerate(
            zip(statements, provenance, strict=True)):
        row = {"tablet": tablet, "clause_index": index, **asdict(st)}
        row["joint_positions"] = json.dumps([list(pair) for pair in st.joint_positions])
        row.update(origin=origin, parser_index=int(parser_index), review=review)
        rows.append(row)
    return rows


def _value(value):
    """A table value as the Statement field holds it."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def from_row(row: dict) -> S.Statement:
    """The Statement of one table row, with the field types that the parser gives."""
    values = {name: _value(row[name]) for name in _FIELDS}
    values["joint_positions"] = tuple(
        tuple(pair) for pair in json.loads(row["joint_positions"]))
    for name in ("day", "joint_day"):
        if values[name] is not None:
            values[name] = int(values[name])
    for name in ("year_offset", "source_start", "source_end",
                 "joint_source_start", "joint_source_end"):
        values[name] = int(values[name])
    for name in ("latitudinal", "omitted", "restored", "joint_restored", "joint_omitted"):
        values[name] = bool(values[name])
    return S.Statement(**values)


def load(path: Path = REVIEWED, provenance: bool = False):
    """The clauses of each record, as {tablet: [Statement, ...]}.

    With provenance=True, each value is a list of (Statement, origin,
    parser_index, review) tuples.
    """
    table = pd.read_parquet(path).sort_values(["tablet", "clause_index"])
    out: dict[str, list] = {}
    for row in table.to_dict("records"):
        st = from_row(row)
        item = (st, row["origin"], int(row["parser_index"]), row["review"]) if provenance else st
        out.setdefault(row["tablet"], []).append(item)
    return out
