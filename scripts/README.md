# scripts

The seven packages are listed in the order the pipeline runs. Each script states its purpose in its module docstring, and each package's `README.md` lists its scripts.

| Package | Role |
|---|---|
| `source/` | Fetch the digital edition, join the catalogue to the text, and read the ADART IV volume |
| `physical/` | Build and check the calendar, ephemeris, star and eclipse tables |
| `inference/` | Turn text into candidate years, with no published date in scope |
| `audit/` | Test the method at the published years, and write the benchmark, sensitivity and held-out results |
| `agent/` | The language-model review, the MCP server and the model client |
| `paper/` | Write the counts, tables and figures that the paper quotes |
| `site/` | Write the data of the interactive site in `docs/` from the committed results |

Run each script from the repository root as `python scripts/<package>/<name>.py`. The full rebuild runs these lines in order, each from left to right.

```text
source/     build_corpus  classify_evidence  build_cohort  build_adart_iv_corpus
physical/   build_calendar  build_normal_stars  build_eclipses
audit/      build_parser_clauses  build_benchmark
agent/      rescore_model_review  compare_models
inference/  date_all  date_adart_iv
audit/      score_benchmark  rank_candidates  compare_clause_sets  audit_sensitivity  audit_adart_iv_corrigenda  audit_era_readings
physical/   appulse_residuals  calibrate_visibility  validate_zodiac  check_calendar_against_sky
paper/      build_paper_counts  build_adart_iv_table
audit/      audit_held_out --run-held-out
source/     build_adart_iv_addenda
inference/  date_adart_iv_addenda
audit/      score_adart_iv_addenda
```

The last three lines are the frozen test on the dated addenda of ADART IV that `data/benchmark/addenda_registration.json` registers. The scorer reads the benchmark rates from `data/paper_counts/counts.json`, so it runs after `build_paper_counts`. Then `make check`. Where the paper is present, `make figures paper` builds it. The sensitivity audit is the slow step, and `--workers 4` runs it in about 10 minutes.

Scripts outside that order:

- **Need the network, a paid service or a book scan.** Their outputs are committed in `data/source/` and `data/clauses/model_review_*/`.
  - `source/fetch_adsd.py` and `source/fetch_texts.py` download the digital edition.
  - `source/ocr_adart_iv_mathpix.py` reads the ADART IV scan with Mathpix.
  - `physical/ocr_parker_dubberstein.py` reads the Parker and Dubberstein scan with Tesseract.
  - `physical/build_ephemeris.py` needs the JPL DE441 kernel.
  - `agent/model_review.py` runs the paid language-model review, of ADART I to III and, with `--scope adart-iv`, of ADART IV.
  - `paper/build_bibliography.py` writes `paper/Bibliography.bib` from bibliographic APIs.
- **Need the ADART IV text.** `source/build_adart_iv_corpus.py`, `source/build_adart_iv_addenda.py`, `agent/rescore_model_review.py --scope adart-iv`, `inference/date_adart_iv.py`, `inference/date_adart_iv_addenda.py` and `audit/score_adart_iv_addenda.py` read the full ADART IV text or the caches of its review, which stay local (`data/README.md`). Their outputs are committed.
- **Optional, after the rebuild.** `site/build_site_data.py` writes the data of the site in `docs/`. The paper does not need it.
- **Interactive.** `audit/date_tablet.py` and `audit/diagnose_candidate.py` trace one record. `agent/mcp_server.py` serves the read-only tools to any MCP client.
- **Imported, not run.** `physical/eclipse_catalog.py`, `agent/blind_case_tools.py`, `agent/llm.py` and `site/cuneiform.py`.
- **Figures.** The modules in `paper/figures/`, which `make figures` runs.

Nothing in `inference/` may import anything that knows a published date, and nothing in `agent/` may write a candidate score.
