# Data

The inputs of the pipeline, and each table that it writes. `columns.csv` describes each file and each column of each table.

| Folder | What it holds |
|---|---|
| `source/` | The inputs as obtained. The ADsD catalogue and corpus with the texts of ADART I to III (CC0), the NASA eclipse catalogs, the ORACC Sign List, the readings of the calendar of Parker and Dubberstein and of the Normal Star table of ADART I, and a photograph of a diary. `source/not_distributed.csv` lists the files under copyright that are not here. |
| `corpus/` | The diaries of ADART I to III, one row per record in `tablets.parquet` and one row per line in `lines.parquet`, with the evidence and the group of each record. Summaries of the ADART IV text, whose tables are not distributed. |
| `physical/` | The calendar, the ephemeris, the Normal Stars and the eclipses that the tools use, and the checks of them. |
| `clauses/` | The clauses. `parser_clauses.parquet` holds the clauses of the parser, and `reviewed_clauses.parquet` the clauses after the language-model review, which the dates use. The `model_review_*` folders hold the saved responses of the model and the changes that it made. |
| `results/` | The candidate years and the decision of each record of ADART I to III in `tablet_dates.parquet`, and of each undated entry of ADART IV in `adart_iv_dates.parquet`. |
| `benchmark/` | The benchmark of 179 dated records and its scores, the ranked candidate years, the sensitivity audit, the held-out test, and the registered test on the dated addenda of ADART IV. |
| `paper_counts/` | The counts that the paper quotes in `counts.json`, Tables S3 and S4 of the SI Appendix, and the SHA-256 of each input in `source_manifest.json`. |

## Conventions

- Years are astronomical years, so 1 BCE is year 0 and 150 BCE is year -149. The 701 candidate years run from -625 (626 BCE) to 75 (AD 75).
- A record is one text in ADsD, with an identifier such as `X301492`. ADART IV entries have identifiers that start with `ADART_IV_`.
- Columns whose names end in `_json` hold JSON text. `pandas.read_parquet` reads the Parquet files.
- A column whose name ends in `_sha256` holds the SHA-256 of a file or a text that the row comes from. `make check` compares these values with the files in this repository.
