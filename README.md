# Babylonian Dating

This repository dates the Babylonian Astronomical Diaries from the astronomy that the tablets record. A rule-based parser reads the translation into clauses, a language model checks them against the text and adds the observations that the parser missed, and astronomical tools test each clause in each of 701 candidate years, 626 BCE to AD 75. Fixed rules then give the candidate years of each record. The published dates enter only after the candidate sets are final.

The interactive page at https://tingyuansen.github.io/Babylonian_Dating/ follows one record at a time from its text to its candidate years. Its source is in `docs/`. The method and the results are described in the paper cited below.

## Quick start

```bash
python3 -m pip install -r requirements.txt      # Python 3.10 or later
make check                                      # every test, then verify.py
python3 scripts/audit/date_tablet.py X301492   # trace one record from text to years
```

The committed results were made with Python 3.13, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, pyarrow 21.0.0, jplephem 2.24, pyswisseph 2.10.3.2 and matplotlib 3.10.6.

## What the data hold

| Question | File |
|---|---|
| What does each diary say? | `data/corpus/tablets.parquet` (one row per record), `data/corpus/lines.parquet` (one row per line) |
| Which clauses do the dates use? | `data/clauses/reviewed_clauses.parquet`, and `data/clauses/adart_iv_reviewed_clauses.parquet` for ADART IV |
| Which clauses did the parser read, from which words, and do they agree with the published year? | `data/clauses/parser_clauses.parquet` |
| Which years remain for each record, and why? | `data/results/tablet_dates.parquet` |
| How does each benchmark record score against its published year? | `data/benchmark/benchmark_results.parquet`, and the frozen benchmark in `data/benchmark/benchmark_records.parquet` |
| Where does the published year fall among all 701 candidates? | `data/benchmark/ranking_candidates.parquet` |
| What does the language model change or add? | `data/clauses/model_review_gpt-6-luna-2026-09-22/` |
| What do the undated entries of ADART IV give? | `data/results/adart_iv_dates.parquet` |
| Which calendar, positions, stars and eclipses do the tools use? | `data/physical/` |
| Which counts and tables does the paper quote? | `data/paper_counts/` |

`data/README.md` describes every file, and `data/columns.csv` gives the type, unit and meaning of every column. Years in the data are astronomical years, where 1 BCE is year 0 and 150 BCE is year -149. Columns that end in `_json` hold JSON text. The files are Parquet, CSV or JSON, and `pandas.read_parquet` reads the Parquet files.

The tools also answer questions about any record or year directly. `scripts/agent/mcp_server.py` serves them to an MCP client, and `scripts/audit/date_tablet.py` prints the full trace of one record.

## Rebuilding

The committed data can be rebuilt from `data/source/` in the order of `scripts/README.md`, in about 15 minutes on a laptop. The rebuild needs no network and no paid service. The steps that need one of these, or a book, keep their outputs in the repository:

- `scripts/source/fetch_adsd.py` and `fetch_texts.py` download the digital edition.
- `scripts/source/ocr_adart_iv_mathpix.py` and `scripts/physical/ocr_parker_dubberstein.py` read the book scans. The scans, and the full text read from ADART IV, are under copyright and stay out of the repository. `data/README.md` gives their SHA-256, and the ADART IV steps of `scripts/README.md` need that text.
- `scripts/agent/model_review.py` runs the paid language-model review. `scripts/agent/rescore_model_review.py` rebuilds its final clauses, which the dates use, from the committed caches. The caches of the review of ADART IV quote the volume, so they stay local, and the repository keeps its final clauses.
- `scripts/physical/build_ephemeris.py` needs the JPL DE441 kernel `de441_part-1.bsp` (1.65 GB) from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/, which is too large to keep.

The results record the SHA-256 of the code and tables that made them. An edit to the hashed modules of `tools/`, the clause tables, `scripts/inference/date_all.py`, `scripts/agent/model_review.py`, the protocol in `skills/` or the physical tables invalidates the committed results until the rebuild runs again.

## Checking

- `make check` runs the tests in `tests/`, then `verify.py`. The tests check the parser, the language-model checks, the figures and the MCP server. `verify.py` checks every recorded hash, that the results and audits come from one run, and that the tree holds only what its READMEs describe.
- `scripts/audit/audit_held_out.py --run-held-out` repeats the held-out test.
- `scripts/audit/diagnose_candidate.py <record>` explains why a candidate year ranks above the published year.

## Layout

```text
data/         sources, corpus, physical tables, clauses, results, benchmark and the counts the paper quotes
scripts/      the pipeline, the figures and the site data, in seven packages: source, physical, inference, audit, agent, paper, site
tools/        the parser and the astronomical tools that the pipeline is built from
skills/       the written protocol given unchanged to agents and human auditors
tests/        offline tests
docs/         the interactive page, for GitHub Pages
verify.py     the check of the whole tree
LICENSE       the MIT License of the code
CITATION.cff  how to cite this work
```

Each directory has a `README.md` that lists its files.

## How to cite

L. Carrit Delgado Pinheiro, Y.-S. Ting, J. Clindaniel, L. Li, Y. Liang and N. Shroff, Automated, auditable dating of the Babylonian Astronomical Diaries with language models and astronomical tools (2026). `CITATION.cff` gives the same reference in a form that GitHub and reference managers read.

## Licences

- The code is under the MIT License, in `LICENSE`.
- The data that this project made, in `data/`, are under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
- The sources keep their own terms. The diaries of ADART I to III come from Astronomical Diaries Digital (ADsD), released under CC0. The ORACC Sign List in `data/source/osl/` is released under CC0. The eclipse catalogs in `data/source/nasa_eclipses/` are the NASA Five Millennium Canons of Espenak and Meeus, from the NASA Goddard Space Flight Center. The tables in `data/source/adart_i/` and `parker_dubberstein/` were read from books whose copyright stays with their publishers, and they are included for scholarly checking only. For ADART IV, the repository keeps only the extracted statements, with page references.
- The photograph of BM 45962 (`data/source/photos/alexander_diary_BM45962.jpg`) is by Shaunnol, Wikimedia Commons, in the public domain. The illustrations of the site in `docs/img/` were made with Gemini 3 Pro Image, as `docs/README.md` states.
- Local eclipse circumstances use the Swiss Ephemeris through `pyswisseph`, which is licensed under the AGPL. Anyone who redistributes the code together with it should check that licence.
