# audit

These scripts compare the method with the published years, which `inference/` never reads. They write the parser clauses into `data/clauses/` and the benchmark, sensitivity, held-out, corrigenda and addenda results into `data/benchmark/`.

| Script | What it does |
|---|---|
| `build_parser_clauses.py` | Write every parser clause, with its source line and its verdict at the published year, into `data/clauses/parser_clauses.parquet` |
| `build_benchmark.py` | Check that the dating code reads no published date or catalogue field. The frozen benchmark, `benchmark_records.parquet`, is rewritten only with `--rebuild-population` |
| `score_benchmark.py` | Score each benchmark record, with its reviewed clauses, against its published year into `data/benchmark/benchmark_results.parquet`, and the bits of its clauses into `data/benchmark/benchmark_clause_bits.parquet` |
| `rank_candidates.py` | Write every candidate year of each benchmark record, with its scores and the rank of the published year, into `data/benchmark/ranking_candidates.parquet`. Fig. 4 reads it |
| `audit_sensitivity.py` | Rerun the benchmark under each of 32 configurations, each with one convention, rule or clock changed. The slow step |
| `audit_adart_iv_corrigenda.py` | Trace the corrections that ADART IV prints to Volumes I to III into the digital edition |
| `audit_era_readings.py` | Read the era and regnal years that the texts state, and check them at the published years |
| `audit_held_out.py` | The held-out test of SI section S8. It sets the tolerances, parser rules and decisive types on two volumes and dates the third. Without `--run-held-out`, it scores only the round on all volumes, which must equal the main pipeline |
| `score_adart_iv_addenda.py` | Score the frozen test on the dated addenda of ADART IV, as `data/benchmark/addenda_registration.json` registers it, into `data/benchmark/addenda_results.json`. It reads the published years only after the candidate sets, and it reports the texts that ADART IV links to ADART I to III in a separate row. The committed result came from the frozen commit. Later commits change `tools/`, so a rerun now reports that the tools differ from that commit. With the parser clauses, the dates of the addenda stay as they were |
| `date_tablet.py` | Print the trace of one record from text to candidate years, with its reviewed clauses. `--year` tests its clauses in a given year, `--show-reference` marks the published year, and `--parser` uses the parser clauses alone. Interactive |
| `diagnose_candidate.py` | Explain why a leading candidate year outranks the published year of one record. Interactive |

`__init__.py` only marks the package.
