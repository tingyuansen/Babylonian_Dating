# Babylonian Dating

The Babylonian Astronomical Diaries record the night sky over nearly six centuries, and many of their tablets have lost their date. This repository dates them from the astronomy that they record. A rule-based parser reads each English translation into clauses, short statements with fixed fields. A language model checks the clauses against the text and adds the observations that the parser missed. Astronomical tools then test each clause in each of the 701 candidate years from 626 BCE to AD 75. Fixed rules give the candidate years of each record. The published dates enter only after the candidate sets are final.

The paper cited at the end describes the method and the results. The interactive page at https://tingyuansen.github.io/Babylonian_Dating/ follows one tablet from its text to its candidate years.

## Try it

```bash
python3 -m pip install -r requirements.txt     # Python 3.10 or later
make check                                     # the tests, then tests/verify.py
python3 scripts/audit/date_tablet.py X301492  # one record, from its text to its years
```

`make check` takes less than a minute and needs no network. The committed results were made with Python 3.13, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, pyarrow 21.0.0, jplephem 2.24, pyswisseph 2.10.3.2 and matplotlib 3.10.6.

## What is here

```text
data/      the inputs as obtained (data/source/), and every table that the pipeline writes
scripts/   the pipeline, in seven steps: source, physical, inference, audit, agent, paper, site
tools/     the parser and the astronomical tools that the pipeline uses
skills/    the written protocol of the rules, given unchanged to agents and human auditors
tests/     offline tests, and verify.py, which checks the whole tree
docs/      the interactive page, served by GitHub Pages
```

Each script says what it does in its first lines. `data/columns.csv` describes every data file and every column of every table.

## The data

| Question | File |
|---|---|
| What does each diary say? | `data/corpus/tablets.parquet` (one row per record), `data/corpus/lines.parquet` (one row per line) |
| Which clauses do the dates use? | `data/clauses/reviewed_clauses.parquet`, and `data/clauses/adart_iv_reviewed_clauses.parquet` for ADART IV |
| What did the parser read, and what did the language model change or add? | `data/clauses/parser_clauses.parquet`, and the review in `data/clauses/model_review_gpt-6-luna-2026-09-22/` |
| Which years remain for each record, and why? | `data/results/tablet_dates.parquet`, and `data/results/adart_iv_dates.parquet` for the undated entries of ADART IV |
| How does each benchmark record score against its published year? | `data/benchmark/benchmark_results.parquet` |
| Which calendar, positions, stars and eclipses do the tools use? | `data/physical/` |
| Which counts and tables does the paper quote? | `data/paper_counts/` |

Years are astronomical years, so 1 BCE is year 0 and 150 BCE is year -149. Columns whose names end in `_json` hold JSON text. `pandas.read_parquet` reads the Parquet files.

## Rebuild

`make rebuild` runs the pipeline in order, from `data/source/` and the committed responses of the language model. It needs no network and no paid service, and it takes about 25 minutes. Run `make check` after it. `make site` then writes the data of the interactive page.

A few scripts stay outside the rebuild, and the Makefile lists them. They need the network, the paid language model or a book scan. The test on the dated addenda of ADART IV is also outside it, because it ran once, as `data/benchmark/addenda_registration.json` registers. Their outputs are committed. The ephemeris needs the JPL DE441 kernel `de441_part-1.bsp` (1.65 GB) from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/, which is too large to keep.

The book scans and the full text read from ADART IV are under copyright, so they stay out of the repository. `data/source/not_distributed.csv` gives the path and the SHA-256 of each such file, so a reader with the books can repeat a reading and check it. The two steps of `make rebuild` that read the ADART IV text run only where that text is present.

The results record the SHA-256 of the code and tables that made them. Editing `tools/`, the clause tables, `scripts/inference/date_all.py`, `scripts/agent/model_review.py`, the protocol in `skills/` or the physical tables invalidates the committed results until a rebuild. `make check` reports such an edit.

## Check a result

- `make check` runs the tests, then `tests/verify.py`. It checks every recorded SHA-256 and that the results and the audits come from one run. It also checks that the tree holds only what this README, the Makefile and `data/columns.csv` describe.
- `python3 scripts/audit/audit_held_out.py --run-held-out` repeats the held-out test.
- `python3 scripts/audit/diagnose_candidate.py X301492` explains why a candidate year ranks above the published year.
- `python3 scripts/agent/mcp_server.py` gives the read-only tools to an MCP client.

## The interactive page

`docs/` is a static page. To preview it, run `python3 -m http.server 8765 --directory docs` and open http://localhost:8765. The page loads D3 and its fonts (IM Fell English, IM Fell DW Pica, Roboto and Noto Sans Cuneiform) from the network.

## Licences

- The code is under the MIT License, in `LICENSE`.
- The data that this project made, in `data/`, and the figures are under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
- The sources keep their own terms. The diaries of ADART I to III come from Astronomical Diaries Digital (ADsD), released under CC0. The ORACC Sign List in `data/source/osl/` is released under CC0. The eclipse catalogs in `data/source/nasa_eclipses/` are the NASA Five Millennium Canons of Espenak and Meeus. The tables in `data/source/adart_i/` and `data/source/parker_dubberstein/` were read from books whose copyright stays with their publishers, and they are included for scholarly checking only. For ADART IV, the repository keeps only the extracted clauses, with page references.
- The photograph of BM 45962 is by Shaunnol, Wikimedia Commons, in the public domain. Gemini 3 Pro Image made the illustrations of the page in `docs/img/`, which show no real object.
- Local eclipse circumstances use the Swiss Ephemeris through `pyswisseph`, under the AGPL. Anyone who redistributes the code together with it should check that licence.

## How to cite

L. Carrit Delgado Pinheiro, Y.-S. Ting, J. Clindaniel, L. Li, Y. Liang and N. Shroff, Automated, auditable dating of the Babylonian Astronomical Diaries with language models and astronomical tools (2026). `CITATION.cff` gives the same reference for GitHub and reference managers.
