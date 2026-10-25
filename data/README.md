# data

The data are grouped by what they are. Each folder below is rebuilt by the scripts named in its table, under `scripts/`, except `source/`, which holds the inputs as obtained.

| Folder | What it holds |
|---|---|
| `source/` | The inputs as obtained: the digital edition, the text read from the books, the corrections checked by eye, the cuneiform sign list, the photograph in Fig. 2 and the reference identifiers. Never edited in place |
| `corpus/` | The text, one row per record, of the 417 dated diaries of ADART I to III, and the counts and hashes of the ADART IV text, which stays local |
| `physical/` | The calendar, ephemeris, eclipses and Normal Stars that the tools use, and the measurements that check them |
| `clauses/` | The reviewed clauses that the dates use, the parser clauses, and the language-model review of step 1 |
| `results/` | The candidate years of every record. These files hold no published year |
| `benchmark/` | Everything that compares the results with the published years: the frozen benchmark, its scores, the ranking, the sensitivity, held-out and corrigenda audits, and the frozen test on the dated addenda of ADART IV |
| `paper_counts/` | The counts and SI tables that the paper quotes |
| `columns.csv` | The type, unit and meaning of every column of every table in these folders, with one more row for each key of a JSON column. Written from the scripts and the paper |

Years are astronomical years, where 1 BCE is year 0. A record is named by its ADsD identifier, such as `X301492`, in the column `tablet`. Columns that end in `_json` hold JSON text. The data that this project made are under CC BY 4.0, and the sources keep their own terms, as the root `README.md` says.

## source/

| File | What it holds | Made by |
|---|---|---|
| `adsd/adsd.zip`, `adsd/adsd-adart1.zip` | The ORACC catalogue, and the ADART I project, whose catalogue lists all 417 diaries | `source/fetch_adsd.py` |
| `adsd/manifest.json` | Byte counts and SHA-256 of the two archives | `source/fetch_adsd.py` |
| `adsd/texts.jsonl` | Transliteration and translation of every ADART I to III record that has text | `source/fetch_texts.py` |
| `adart_iv/mathpix_manifest.json` | The Mathpix reading of the ADART IV scan: the job, the scan's SHA-256 and the hashes of the outputs | `source/ocr_adart_iv_mathpix.py` (paid) |
| `adart_iv/ocr_corrections.csv` | Each line of that reading that was checked against the page and corrected | Checked by eye |
| `parker_dubberstein/calendar_ocr.txt` | The Tesseract reading of pages 43 to 63 of Parker and Dubberstein (1956) | `physical/ocr_parker_dubberstein.py` |
| `parker_dubberstein/intercalary_slots.csv` | For each of the 258 thirteen-month years, whether the table prints the extra month as VI2 or XII2 | `physical/ocr_parker_dubberstein.py --slots` |
| `adart_i/normal_stars.txt` | The 32 rows of the Normal Star table of the ADART I introduction | Read once from the scan with `pdftotext -layout` |
| `nasa_eclipses/lunar_catalog.txt`, `nasa_eclipses/solar_catalog.txt` | The NASA Five Millennium Canons of lunar and solar eclipses, -1999 to +3000, by Espenak and Meeus | NASA Goddard eclipse site |
| `osl/osl.asl` | The ORACC Sign List (OSL), CC0: each cuneiform sign with its readings and its Unicode glyph. The site in `docs/` uses it | Fetched on 24 October 2026 from https://raw.githubusercontent.com/oracc/osl/master/00lib/osl.asl |
| `photos/alexander_diary_BM45962.jpg` | The photograph of BM 45962 by Shaunnol, in the public domain. Fig. 2 shows it | Wikimedia Commons |

### Not distributed

The book scans, and the full text that the project read from ADART IV, are under copyright, so they stay out of the repository. The model review of ADART IV quotes that text, so its caches stay out too. For ADART IV, the repository keeps the counts and hashes of that text, the clauses extracted from it, the run summary of its review, and the candidate years, in `corpus/`, `clauses/` and `results/`. A reader with the books can repeat a reading, check each file against these SHA-256 values with `shasum -a 256`, and put it at the path given, which git ignores.

| File | What it is | SHA-256 |
|---|---|---|
| `source/adart_i/scan.pdf` | The introduction to ADART I (Sachs and Hunger 1988), with the Normal Star table | `d26df90c2937affed7e36e1e35153c4cdf39732bb7144ac6182b0766273335cc` |
| `source/adart_iv/scan.pdf` | ADART IV (Hunger 2022), with the corrections to Volumes I to III and the 867 undated entries | `5c2ca092f1f9a0c4a9a0394b9cc610615a7a53a8220a3b5b0fbaf289512f965c` |
| `source/parker_dubberstein/scan.pdf` | Parker and Dubberstein, *Babylonian Chronology 626 B.C.–A.D. 75* (Brown University Press, 1956) | `787be98b6ac7d8b43b22d8665033a528d038d4583b2e35d299500e09bff0073f` |
| `source/adart_iv/mathpix_lines.json` | The Mathpix reading of the ADART IV scan, line by line, from `source/ocr_adart_iv_mathpix.py` | `4b60c8a9876566996d23ce18e9449177e1f48a114447cfc9e1552a4537a6bc31` |
| `corpus/adart_iv_tablets.parquet` | One row per undated ADART IV entry, with its translation, from `source/build_adart_iv_corpus.py` | `dff212f9047e07fc9408ecdb1a1eada22c3630b19bde5e445bb4d777849b9782` |
| `corpus/adart_iv_lines.parquet` | One row per line of the translation column of ADART IV, from `source/build_adart_iv_corpus.py` | `36b48671042c31413c83f76677725438f7a38de9a26183a267336880bcadcffa` |
| `corpus/adart_iv_addenda_tablets.parquet` | One row per text of the dated addenda, with its blind translation, from `source/build_adart_iv_addenda.py` | `aa183f991a2c2b59a8f73c4fd3ea7880383bf59fe9c6865765a34c3c42dd556d` |
| `corpus/adart_iv_addenda_lines.parquet` | One row per line of the addenda pages, from `source/build_adart_iv_addenda.py` | `564b03836886f49fa50c388c1e1fe85ab592d4583f943a6da74335a8a8a43e7d` |
| `clauses/model_review_gpt-6-luna-2026-09-22_adart_iv/review_cache.jsonl` | Every model response of the review of the ADART IV parser clauses, from `agent/model_review.py --scope adart-iv` | `90f5dade68b736586d936dc367034cf2a1fabed44f520e99e31123e170864f99` |
| `clauses/model_review_gpt-6-luna-2026-09-22_adart_iv/recall_cache.jsonl` | Every model response of the search of ADART IV for missed clauses | `b509859436f22098ed8897dadf42f6089e2f85006a9869ee58acb7296054436c` |
| `clauses/model_review_gpt-6-luna-2026-09-22_adart_iv/review_changes.csv` | Every change the model proposed to an ADART IV parser clause, and its checks | `379ddf0b6390895ea6812a1ba5983d5d46238e1b72a86f3416240fddae1bf754` |
| `clauses/model_review_gpt-6-luna-2026-09-22_adart_iv/search_proposals.csv` | Every observation the search of ADART IV proposed, and what became of it | `9d50017a53bfa90c3d584848e0ae04e867c7fbfbef08c0b12150d0c183292d22` |

## corpus/

| File | What it holds | Made by |
|---|---|---|
| `tablets.parquet` | One row per record of ADART I to III: catalogue fields, published year and full text | `source/build_corpus.py` |
| `lines.parquet` | One row per line: label, transliteration and translation | `source/build_corpus.py` |
| `evidence.parquet` | The chronological evidence that each record preserves | `source/classify_evidence.py` |
| `cohorts.parquet` | Coarse text-property cohorts of the records | `source/build_cohort.py` |
| `adart_iv_corpus_summary.json` | Counts and hashes of the ADART IV corpus | `source/build_adart_iv_corpus.py` |
| `adart_iv_addenda_summary.json` | Counts and hashes of the addenda corpus | `source/build_adart_iv_addenda.py` |

## physical/

| File | What it holds | Made by |
|---|---|---|
| `calendar.parquet` | Month starts for the 701 Babylonian years 626 BCE to AD 75 | `physical/build_calendar.py` |
| `calendar_ocr_corrections.csv` | Each correction that the calendar build applied to the OCR reading | `physical/build_calendar.py` |
| `ephemeris.parquet` | Daily positions of the Sun, the Moon and the planets, 700 BCE to AD 76 | `physical/build_ephemeris.py` (needs the DE441 kernel) |
| `eclipses.parquet` | The NASA eclipse catalogs, with the circumstances at Babylon for three values of Delta T | `physical/build_eclipses.py` |
| `normal_stars.parquet` | The 32 Normal Stars in the J2000 ecliptic | `physical/build_normal_stars.py` |
| `appulse_residuals.parquet` | The difference between each stated and computed distance at the published year | `physical/appulse_residuals.py` |
| `visibility_calibration.parquet` | First and last appearance thresholds and their held-out check | `physical/calibrate_visibility.py` |

## clauses/

| File | What it holds | Made by |
|---|---|---|
| `reviewed_clauses.parquet` | The clauses that the dates of ADART I to III use, one row per clause: the parser clauses as the model review leaves them, and the clauses that its search added. Each row gives the clause's fields, its origin (parser or search), its index in the parser's list and the review's action. `tools/clauses.py` reads it | `agent/rescore_model_review.py`, from the caches |
| `adart_iv_reviewed_clauses.parquet` | The same for the undated entries of ADART IV, whose clauses keep only their extracted statement | `agent/rescore_model_review.py --scope adart-iv`, from the local caches |
| `parser_clauses.parquet` | Every parser clause: its fields, its source passage and line, and its verdict at the published year | `audit/build_parser_clauses.py` |
| `model_review_gpt-6-luna-2026-09-22/` | The language-model review of step 1, listed below | |
| `model_review_gpt-6-luna-2026-09-22_adart_iv/run_summary.json` | The calls, tokens, cost and outcomes of the review of ADART IV. The rest of this run quotes the volume and stays local | `agent/model_review.py --scope adart-iv` (paid) |
| `model_review_pilot/` | The comparison of three models on ten records: for each model, `<model>_summary.json` (calls, cost and outcomes), `<model>_changes.csv` (its changes to parser clauses) and `<model>_additions.csv` (its search proposals), and `comparison.json`, their verdicts at the published year | `agent/model_review.py` (paid); `comparison.json` by `agent/compare_models.py` |

`model_review_gpt-6-luna-2026-09-22/` is the committed run of the review. A run with another model writes to its own `model_review_<model>/`.

| File | What it holds | Made by |
|---|---|---|
| `review_cache.jsonl`, `recall_cache.jsonl` | Every model response of the review and of the search for missed clauses, with the model, tokens and cost of each call | `agent/model_review.py` (paid) |
| `review_changes.csv` | Every change the model proposed to a parser clause, and its checks | `agent/model_review.py` |
| `search_proposals.csv` | Every observation the search proposed, and what became of it | `agent/model_review.py` |
| `fallbacks.jsonl`, `run_summary.json` | The requests that the endpoint rejected, and the calls, tokens, cost and outcomes of the run | `agent/model_review.py` |
| `reviewed_clause_bits.parquet`, `reviewed_clause_verdicts.parquet` | Per reviewed clause, its bits and tier, and its verdict at the published year | `agent/rescore_model_review.py`, from the caches |

## results/

| File | What it holds | Made by |
|---|---|---|
| `tablet_dates.parquet` | One row per ADART I to III record: its decision, strict and safety sets, groups of tied years, and the verdicts of its clauses in those years | `inference/date_all.py` |
| `adart_iv_dates.parquet` | The same for the 867 ADART IV entries. Each clause keeps its extracted statement | `inference/date_adart_iv.py` |
| `adart_iv_summary.json` | Counts and hashes of the ADART IV run | `inference/date_adart_iv.py` |
| `adart_iv_addenda_dates.parquet` | The same for the 52 texts of the dated addenda of ADART IV, dated with the parser clauses, as the test registered | `inference/date_adart_iv_addenda.py` |
| `adart_iv_addenda_summary.json` | Counts and hashes of the addenda run | `inference/date_adart_iv_addenda.py` |

## benchmark/

| File | What it holds | Made by |
|---|---|---|
| `benchmark_records.parquet` | The frozen benchmark: the 417 records, of which 179 are in the benchmark | `audit/build_benchmark.py --rebuild-population` only |
| `benchmark_results.parquet` | The 179 benchmark records, scored against the published year | `audit/score_benchmark.py` |
| `benchmark_clause_bits.parquet` | Bits and tier of each clause of the 179 benchmark records | `audit/score_benchmark.py` |
| `ranking_candidates.parquet` | Every candidate year of every benchmark record, with its scores and rank | `audit/rank_candidates.py` |
| `sensitivity_runs.parquet`, `sensitivity_summary.parquet` | Every benchmark record under each of the 32 configurations, and the summary per configuration | `audit/audit_sensitivity.py` |
| `held_out_runs.parquet`, `held_out_summary.json` | The choices and the records of each held-out round | `audit/audit_held_out.py` |
| `addenda_registration.json` | The registration of the test on the dated addenda of ADART IV: the frozen commit, the input, the blinding and the scores, fixed before any addendum was read | Written by hand before the test |
| `addenda_published_years.parquet` | The published year of each text of the addenda, from its diary number, and whether ADART IV links the text to a text of ADART I to III. The public copy leaves out the quoted notes of the volume. Only the scorer reads it | `source/build_adart_iv_addenda.py` |
| `addenda_results.json` | The scores of the addenda test as registered, for the texts not linked to ADART I to III and, in a separate row, for the linked texts, with the benchmark rates at registration beside them, every text with a decisive clause, and the hashes of the inputs and the code. The public copy leaves out the quoted notes of the volume | `audit/score_adart_iv_addenda.py` |
| `adart_iv_corrigenda.parquet`, `adart_iv_corrigenda_summary.json` | The corrections that ADART IV prints to Volumes I to III, traced into the digital edition | `audit/audit_adart_iv_corrigenda.py` |
| `era_readings.parquet` | The era and regnal years stated in the texts, checked at the published years | `audit/audit_era_readings.py` |

## paper_counts/

| File | What it holds | Made by |
|---|---|---|
| `counts.json` | The counts the paper quotes | `paper/build_paper_counts.py` |
| `table_s3.csv`, `table_s4.csv` | SI Tables S3 and S4 | `paper/build_paper_counts.py` |
| `source_manifest.json` | SHA-256 and size of every input of the counts | `paper/build_paper_counts.py` |
