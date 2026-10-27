# Babylonian Dating

The Babylonian Astronomical Diaries record the night sky over nearly six centuries, and many of their tablets have lost their date. This repository holds the code, the data and the results of a system that dates these tablets from the astronomy that they record. A rule-based parser reads each English translation into clauses, short statements with fixed fields. A language model checks the clauses against the text and adds the observations that the parser missed. Astronomical tools then test each clause in each of the 701 candidate years from 626 BCE to AD 75. Fixed rules give the candidate years of each record.

The paper in [How to cite](#how-to-cite) describes the method and the results. The interactive page at https://tingyuansen.github.io/Babylonian_Dating/ follows one tablet from its text to its candidate years.

## Quick start

```bash
python3 -m pip install -r requirements.txt     # Python 3.10 or later
make check                                     # check the data, the code and the results together
python3 scripts/audit/date_tablet.py X301492   # follow one record from its text to its years
```

`make check` takes less than a minute and needs no network. The results in this repository were made with Python 3.13, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, pyarrow 21.0.0, jplephem 2.24, pyswisseph 2.10.3.2 and matplotlib 3.10.6.

## Contents

```text
data/      the inputs, every table that the pipeline writes, and the counts that the paper quotes
scripts/   the pipeline, in one folder per stage
tools/     the parser and the astronomical tools that the pipeline uses
skills/    the written protocol of the rules, for agents and human auditors
tests/     the offline tests, and the check of the whole repository
docs/      the interactive page
```

`data/columns.csv` describes each data file and each column of each table.

## From the paper to the files

| Part of the paper | Code | Results |
|---|---|---|
| Step 1, reading the text into clauses (SI Appendix, section S2) | `tools/statements.py`, `scripts/agent/` | `data/clauses/` |
| Step 2, checking the clauses against the sky (sections S3 and S4) | `tools/`, `scripts/physical/` | `data/physical/` |
| Step 3, deciding with several clocks (sections S5 and S6) | `scripts/inference/date_all.py` | `data/results/tablet_dates.parquet` |
| Benchmark of 179 dated records (Fig. 4) | `scripts/audit/build_benchmark.py`, `scripts/audit/score_benchmark.py` | `data/benchmark/benchmark_results.parquet` |
| Sensitivity (section S7, Fig. S1) | `scripts/audit/audit_sensitivity.py` | `data/benchmark/sensitivity_summary.parquet` |
| Held-out test (section S8, Table S6) | `scripts/audit/audit_held_out.py` | `data/benchmark/held_out_summary.json` |
| Undated entries of ADART IV (section S9, Fig. S2, Table S7) | `scripts/inference/date_adart_iv.py` | `data/results/adart_iv_dates.parquet` |
| Counts in the text, Tables S3 and S4 | `scripts/audit/build_paper_counts.py` | `data/paper_counts/` |
| Figs. 1 to 4, S1 and S2 | `scripts/figures/` | `figures/`, from `make figures` |

## Check a result

- `make check` runs the tests, then `tests/verify.py`. It checks each SHA-256 that the results record. It also checks that the benchmark, the ranking, the sensitivity audit and the held-out test come from one run, record by record.
- `python3 scripts/audit/date_tablet.py RECORD` shows the clauses, the verdicts and the candidate years of one record. With `--year N` it tests each clause in one year instead.
- `python3 scripts/audit/audit_held_out.py --run-held-out` repeats the held-out test.
- `python3 scripts/agent/mcp_server.py` gives the read-only astronomical tools to an MCP client, so that an agent can audit a record.

Years are astronomical years, so 1 BCE is year 0 and 150 BCE is year -149. Columns whose names end in `_json` hold JSON text. `pandas.read_parquet` reads the Parquet files.

## Rebuild

`make rebuild` runs the pipeline in order, from `data/source/` and the saved responses of the language model. It needs no network and no paid service, and it takes about 25 minutes. Run `make check` after it. `make site` writes the data of the interactive page, and `make figures` draws the figures of the paper into `figures/`.

Some scripts are not part of the rebuild, and the Makefile lists them. They download the sources, read a book scan, call the paid language model, or compute the ephemeris. The ephemeris needs the JPL DE441 kernel `de441_part-1.bsp` (1.65 GB) from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/. The test on the dated addenda of ADART IV is also outside the rebuild. Its rules were fixed in `data/benchmark/addenda_registration.json` before it ran, and it ran once. The outputs of all these scripts are in the repository.

The results record the SHA-256 of the code and the tables that made them. After an edit to one of these files, `make check` names the results that no longer match.

## Files not included

The book scans and the full text read from ADART IV are under copyright, so they are not in this repository. `data/source/not_distributed.csv` gives the path and the SHA-256 of each such file, so a reader with the books can repeat a reading and check it. Without these files, `make check` uses the recorded SHA-256 values, and `make rebuild` skips the two steps that read the ADART IV text. For ADART IV, the repository keeps the clauses, each with its short passage, and the pages of each entry.

## Licences

- The code is under the MIT License, in `LICENSE`.
- The data that this project made, in `data/`, and the figures are under the Creative Commons Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
- The sources keep their own terms. The diaries of ADART I to III come from Astronomical Diaries Digital (ADsD), released under CC0. The ORACC Sign List in `data/source/osl/` is released under CC0. The eclipse catalogs in `data/source/nasa_eclipses/` are the NASA Five Millennium Canons of Espenak and Meeus. The tables in `data/source/adart_i/` and `data/source/parker_dubberstein/` were read from books whose copyright stays with their publishers, and they are included for scholarly checking only.
- The photograph of BM 45962 is by Shaunnol, Wikimedia Commons, in the public domain. Gemini 3 Pro Image made the illustrations of the page in `docs/img/`, which show no real object.
- Local eclipse circumstances use the Swiss Ephemeris through `pyswisseph`, under the AGPL. Anyone who redistributes the code together with it should check that licence.

## How to cite

L. Carrit Delgado Pinheiro, Y.-S. Ting, J. Clindaniel, L. Li, Y. Liang and N. Shroff, Automated, auditable dating of the Babylonian Astronomical Diaries with language models and astronomical tools, [arXiv:2610.34297](https://arxiv.org/abs/2610.34297) (2026).

```bibtex
@article{CarritDelgadoPinheiro2026,
  author        = {Carrit Delgado Pinheiro, Lucas and Ting, Yuan-Sen and Clindaniel, Jon and Li, Lu and Liang, Yingbin and Shroff, Ness},
  title         = {Automated, auditable dating of the {Babylonian Astronomical Diaries} with language models and astronomical tools},
  journal       = {arXiv e-prints},
  year          = {2026},
  eprint        = {2610.34297},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2610.34297},
  url           = {https://arxiv.org/abs/2610.34297}
}
```
