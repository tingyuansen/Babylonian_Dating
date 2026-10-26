# Scripts

The pipeline, in one folder per stage. `make rebuild` runs the stages in the order that the `Makefile` gives. Run each script from the repository root, for example `python3 scripts/audit/date_tablet.py X301492`. In the tables below, the paths of the scripts are relative to `scripts/`, and the paths of the outputs are relative to `data/`, except `docs/data/`.

## source

The texts of the diaries, and the blind text of ADART IV.

| Script | What it does | Writes |
|---|---|---|
| `source/fetch_adsd.py` | Downloads the ADsD catalogue and corpus from ORACC. Needs the network. | `source/adsd/adsd.zip`, `source/adsd/adsd-adart1.zip`, `source/adsd/manifest.json` |
| `source/fetch_texts.py` | Downloads the transliteration and the translation of each diary of ADART I to III. Needs the network. | `source/adsd/texts.jsonl` |
| `source/build_corpus.py` | Joins the catalogue to the texts, one row per record and one row per line. | `corpus/tablets.parquet`, `corpus/lines.parquet` |
| `source/classify_evidence.py` | Records which kinds of chronological evidence each record keeps. | `corpus/evidence.parquet` |
| `source/build_cohort.py` | Puts each dated record into a group by the properties of its text. | `corpus/cohorts.parquet` |
| `source/ocr_adart_iv_mathpix.py` | Reads the scan of ADART IV with the paid Mathpix service. Needs the scan. | `source/adart_iv/mathpix_manifest.json`, and the reading itself, which is not distributed |
| `source/build_adart_iv_corpus.py` | Splits that reading into the 867 undated entries, and keeps their translations apart from their dates and comments. | `corpus/adart_iv_corpus_summary.json`, and the entry tables, which are not distributed |
| `source/build_adart_iv_addenda.py` | Builds the blind input of the test on the dated addenda of ADART IV, and their published years. | `benchmark/addenda_published_years.parquet`, `corpus/adart_iv_addenda_summary.json`, and the text tables, which are not distributed |

## physical

The calendar, the positions of the Sun, the Moon and the planets, the Normal Stars and the eclipses, and checks of them. Step 2 of the paper uses these tables (SI Appendix, sections S3 and S4).

| Script | What it does | Writes |
|---|---|---|
| `physical/ocr_parker_dubberstein.py` | Reads the calendar tables of Parker and Dubberstein (1956) from the scan with Tesseract. Needs the scan. | `source/parker_dubberstein/calendar_ocr.txt`, `source/parker_dubberstein/intercalary_slots.csv` |
| `physical/build_calendar.py` | Builds the Babylonian calendar of the 701 candidate years from that reading, with each correction listed. | `physical/calendar.parquet`, `physical/calendar_ocr_corrections.csv` |
| `physical/check_calendar_against_sky.py` | Compares each month start of the calendar with the computed new Moon, and prints the result. | nothing |
| `physical/build_ephemeris.py` | Computes the positions from the JPL DE441 ephemeris. Needs the 1.65 GB kernel. | `physical/ephemeris.parquet` |
| `physical/build_normal_stars.py` | Builds the table of the Normal Stars, the reference stars of the diaries, from the introduction to ADART I. | `physical/normal_stars.parquet` |
| `physical/eclipse_catalog.py` | Reads the two NASA eclipse catalogs. `build_eclipses.py` uses it. | nothing |
| `physical/build_eclipses.py` | Lists the eclipses, and computes whether each one was visible at Babylon. | `physical/eclipses.parquet` |
| `physical/appulse_residuals.py` | Compares each recorded distance between the Moon or a planet and a star with the computed distance, at the published years. | `physical/appulse_residuals.parquet` |
| `physical/calibrate_visibility.py` | Fits a rule for the first and last appearances of the planets on some records, and tests it on others. | `physical/visibility_calibration.parquet` |

## agent

Step 1 of the paper, the language-model review of the clauses (SI Appendix, section S2), and the tools that an agent can call.

| Script | What it does | Writes |
|---|---|---|
| `agent/model_review.py` | The language model checks each parser clause against the text and searches for the observations that the parser missed. Needs the network and the paid model. | `clauses/model_review_gpt-6-luna-2026-09-22/` and the other run folders |
| `agent/rescore_model_review.py` | Rebuilds the reviewed clauses from the saved responses of the model, with no model call. | `clauses/reviewed_clauses.parquet`, `clauses/adart_iv_reviewed_clauses.parquet`, and the clause verdicts and bits in the run folder |
| `agent/compare_models.py` | Compares the three models of the pilot on ten records, with the current rules. | `clauses/model_review_pilot/comparison.json` |
| `agent/blind_case_tools.py` | The read-only tools for one record. They see the text and the physical tables, and no published date. | nothing |
| `agent/mcp_server.py` | Gives the read-only tools to an MCP client. | nothing |

## inference

Step 3 of the paper, the decision for each record (SI Appendix, sections S5 and S6).

| Script | What it does | Writes |
|---|---|---|
| `inference/date_all.py` | Gives each record of ADART I to III its strict set, its safety set and its decision, committed, ranked or decline. | `results/tablet_dates.parquet` |
| `inference/date_adart_iv.py` | Dates the undated entries of ADART IV in the same way. Needs the ADART IV text. | `results/adart_iv_dates.parquet`, `results/adart_iv_summary.json` |
| `inference/date_adart_iv_addenda.py` | Dates the dated addenda of ADART IV for the registered test, with the same code. | `results/adart_iv_addenda_dates.parquet`, `results/adart_iv_addenda_summary.json` |

## audit

The benchmark of 179 dated records, the tests of the method, and the numbers of the paper.

| Script | What it does | Writes |
|---|---|---|
| `audit/build_parser_clauses.py` | Runs the parser on each record, and records the source line, the rule and the verdict of each clause. | `clauses/parser_clauses.parquet` |
| `audit/build_benchmark.py` | Fixes the benchmark records, and checks that the dating code reads no published date. | `benchmark/benchmark_records.parquet` |
| `audit/score_benchmark.py` | Dates the benchmark records and compares each candidate set with the published year (Fig. 4). | `benchmark/benchmark_results.parquet`, `benchmark/benchmark_clause_bits.parquet` |
| `audit/rank_candidates.py` | Ranks the 701 candidate years of each benchmark record, and only then finds the rank of the published year. | `benchmark/ranking_candidates.parquet` |
| `audit/audit_sensitivity.py` | Repeats the benchmark with alternative settings of the method (SI Appendix, section S7, Fig. S1). | `benchmark/sensitivity_runs.parquet`, `benchmark/sensitivity_summary.parquet` |
| `audit/audit_held_out.py` | Dates the records of each volume with rules set on the other two (section S8, Table S6). | `benchmark/held_out_runs.parquet`, `benchmark/held_out_summary.json` |
| `audit/audit_era_readings.py` | Reads the era years that the texts state, and checks them at the published years. | `benchmark/era_readings.parquet` |
| `audit/audit_adart_iv_corrigenda.py` | Finds the corrections that ADART IV makes to Volumes I to III in the digital corpus. | `benchmark/adart_iv_corrigenda.parquet`, `benchmark/adart_iv_corrigenda_summary.json` |
| `audit/score_adart_iv_addenda.py` | Scores the registered test on the dated addenda of ADART IV (section S9). | `benchmark/addenda_results.json` |
| `audit/build_paper_counts.py` | Writes the counts that the paper quotes, and Tables S3 and S4. | `paper_counts/` |
| `audit/date_tablet.py` | Shows the clauses, the verdicts and the candidate years of one record. | nothing |

## site

| Script | What it does | Writes |
|---|---|---|
| `site/build_site_data.py` | Writes the data of the interactive page from the results, and checks that they give the same candidate sets. | `docs/data/` |
| `site/cuneiform.py` | Turns a transliteration into cuneiform signs for the page, with the ORACC Sign List. | nothing |

## figures

`make figures` draws each figure of the paper into `figures/`, from the results in `data/`.

| Script | Figure |
|---|---|
| `figures/figure_style.py` | The fonts, sizes and colours that all figures share |
| `figures/fig1_workflow.py` | Fig. 1, the workflow in three steps |
| `figures/fig2_diary.py` | Fig. 2, a diary, and the candidate years that each line of one record leaves |
| `figures/fig3_x301492.py` | Fig. 3, record X301492 from its clauses to one year |
| `figures/fig4_benchmark.py` | Fig. 4, the results on the benchmark |
| `figures/figS1_sensitivity.py` | Fig. S1, the sensitivity of the results to the settings of the method |
| `figures/figS2_adart_iv.py` | Fig. S2, the candidate years of the ranked ADART IV entries |
