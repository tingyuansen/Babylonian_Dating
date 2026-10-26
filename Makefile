# Check the repository, rebuild the results, and draw the figures of the paper.
# Run make from the repository root. To use another Python, run make PYTHON=...
PYTHON ?= python3
# One script per figure in scripts/figures/. Each one writes figures/<name>.pdf.
FIGURES := fig1_workflow fig2_diary fig3_x301492 fig4_benchmark figS1_sensitivity figS2_adart_iv

.PHONY: check rebuild figures site

figures:
	for f in $(FIGURES); do $(PYTHON) scripts/figures/$$f.py || exit 1; done

# The pipeline, in the order in which the steps depend on each other. It starts
# from data/source/ and the saved responses of the language-model review. It
# needs no network and no paid service, and it takes about 25 minutes. Two steps
# read the ADART IV text, which is not distributed (data/source/not_distributed.csv),
# so they run only when that text is present. Run make check after the rebuild.
ADART_IV_OCR := data/source/adart_iv/mathpix_lines.json
ADART_IV_TEXT := data/corpus/adart_iv_tablets.parquet
rebuild:
	$(PYTHON) scripts/source/build_corpus.py
	$(PYTHON) scripts/source/classify_evidence.py
	$(PYTHON) scripts/source/build_cohort.py
	if [ -f $(ADART_IV_OCR) ]; then $(PYTHON) scripts/source/build_adart_iv_corpus.py; fi
	$(PYTHON) scripts/physical/build_calendar.py
	$(PYTHON) scripts/physical/build_normal_stars.py
	$(PYTHON) scripts/physical/build_eclipses.py
	$(PYTHON) scripts/audit/build_parser_clauses.py
	$(PYTHON) scripts/audit/build_benchmark.py
	$(PYTHON) scripts/agent/rescore_model_review.py
	$(PYTHON) scripts/agent/compare_models.py
	$(PYTHON) scripts/inference/date_all.py
	if [ -f $(ADART_IV_TEXT) ]; then $(PYTHON) scripts/inference/date_adart_iv.py; fi
	$(PYTHON) scripts/audit/score_benchmark.py
	$(PYTHON) scripts/audit/rank_candidates.py
	$(PYTHON) scripts/audit/audit_sensitivity.py --workers 4
	$(PYTHON) scripts/audit/audit_adart_iv_corrigenda.py
	$(PYTHON) scripts/audit/audit_era_readings.py
	$(PYTHON) scripts/physical/appulse_residuals.py
	$(PYTHON) scripts/physical/calibrate_visibility.py
	$(PYTHON) scripts/physical/check_calendar_against_sky.py
	$(PYTHON) scripts/audit/build_paper_counts.py
	$(PYTHON) scripts/audit/audit_held_out.py --run-held-out

# The data of the interactive page in docs/, from the results in data/.
site:
	$(PYTHON) scripts/site/build_site_data.py

# Scripts outside the rebuild. Their outputs are in the repository.
# These need the network, a paid service, a book scan or the JPL kernel.
NEEDS_SOURCE := scripts/source/fetch_adsd.py scripts/source/fetch_texts.py \
	scripts/source/ocr_adart_iv_mathpix.py scripts/physical/ocr_parker_dubberstein.py \
	scripts/physical/build_ephemeris.py scripts/agent/model_review.py
# These ran once, for the test on the dated addenda of ADART IV that
# data/benchmark/addenda_registration.json registers. They read the ADART IV text.
REGISTERED_TEST := scripts/source/build_adart_iv_addenda.py \
	scripts/inference/date_adart_iv_addenda.py scripts/audit/score_adart_iv_addenda.py
# These show one record, or give the tools to an MCP client.
EXPLORE := scripts/audit/date_tablet.py scripts/agent/mcp_server.py

# The offline tests, then the check of the data, the code and the results.
check:
	for t in tests/test_*.py; do $(PYTHON) $$t || exit 1; done
	$(PYTHON) tests/verify.py
