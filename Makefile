# Check the repository, and build the paper's figures, SI Table S7 and the
# PDF, from the repository root. paper/ holds only what LaTeX needs, so it
# uploads to Overleaf as it is.
# ``python3`` is the project interpreter; override with ``make PYTHON=...``.
PYTHON ?= python3
# Keep disposable LaTeX intermediates outside the repository.
BUILD ?= /tmp/Babylonian_Dating_Paper_build
# One module per figure, each writing paper/Figures/<name>.pdf. Fig. 1 is the
# workflow schematic, and the others are drawn from committed results.
FIGURES := fig1_workflow fig2_diary fig3_x301492 fig4_benchmark figS1_sensitivity figS2_adart_iv

.PHONY: paper figures check clean arxiv rebuild site

paper: paper/PNAS_Paper.pdf

figures:
	for f in $(FIGURES); do $(PYTHON) scripts/paper/figures/$$f.py || exit 1; done

# SI Table S7 is written from the ADART IV results.
paper/tableS7_adart_iv.tex: data/results/adart_iv_dates.parquet scripts/paper/build_adart_iv_table.py
	$(PYTHON) scripts/paper/build_adart_iv_table.py

# The SI keeps its own reference list (multibib), hence the second bibtex.
PAPER_INPUTS := paper/PNAS_Paper.tex paper/Bibliography.bib paper/tableS7_adart_iv.tex $(FIGURES:%=paper/Figures/%.pdf)
define LATEX
mkdir -p $(BUILD)
cd paper && TEXMFOUTPUT=$(BUILD) pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) PNAS_Paper
cd paper && TEXMFOUTPUT=$(BUILD) bibtex $(BUILD)/PNAS_Paper
cd paper && TEXMFOUTPUT=$(BUILD) bibtex $(BUILD)/S
cd paper && TEXMFOUTPUT=$(BUILD) pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) PNAS_Paper
cd paper && TEXMFOUTPUT=$(BUILD) pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) PNAS_Paper
cp $(BUILD)/PNAS_Paper.pdf paper/PNAS_Paper.pdf
endef

paper/PNAS_Paper.pdf: $(PAPER_INPUTS)
	$(LATEX)

# The arXiv upload: the LaTeX inputs of paper/ with both reference lists as
# .bbl files, since arXiv does not run bibtex. It is checked by a build with
# pdflatex alone, and written to $(BUILD)/arxiv.tar.gz. LaTeX always runs
# first, because a fresh checkout has the committed PDF but not the .bbl files.
# The arXiv version prints no PNAS logo, and arXiv rejects a source that names a
# file it lacks, so its copies of the class and the paper drop the logo command.
# arXiv also deletes the files that the main file does not name: the packages
# that the class loads, their language files and the SI reference list. The
# arXiv main file therefore carries them in filecontents environments.
# arXiv prints its stamp in the left margin of the first page, which the narrow
# PNAS margin cannot hold, so the arXiv pages are 36pt wider and the text block
# sits 18pt further right. Line and page breaks stay the same.
ARXIV_EMBED = pnasresearcharticle.sty jabbrv.sty jabbrv-ltwa-all.ldf jabbrv-ltwa-en.ldf S.bbl
arxiv: $(PAPER_INPUTS)
	$(LATEX)
	rm -rf $(BUILD)/arxiv && mkdir -p $(BUILD)/arxiv/Figures
	cp paper/PNAS_Paper.tex paper/tableS7_adart_iv.tex paper/*.cls paper/*.sty paper/*.bst paper/*.ldf $(BUILD)/arxiv/
	cp paper/Figures/*.pdf $(BUILD)/arxiv/Figures/
	cp $(BUILD)/PNAS_Paper.bbl $(BUILD)/S.bbl $(BUILD)/arxiv/
	cd $(BUILD)/arxiv && perl -pi -e 's/\\includegraphics\[height=24\.5pt,width=66\.4pt\]\{PNAS_Logo\.pdf\}//' pnas-new.cls PNAS_Paper.tex
	cd $(BUILD)/arxiv && ! grep -rq PNAS_Logo .
	cd $(BUILD)/arxiv && perl -pi -e 's/^(\\templatetype\{pnasresearcharticle\})$$/$$1\n\\addtolength{\\paperwidth}{36pt}\\addtolength{\\oddsidemargin}{18pt}\\addtolength{\\evensidemargin}{18pt}/' PNAS_Paper.tex
	cd $(BUILD)/arxiv && grep -q '^\\addtolength{\\paperwidth}{36pt}' PNAS_Paper.tex
	cd $(BUILD)/arxiv && { for f in $(ARXIV_EMBED); do printf '\\begin{filecontents*}[overwrite]{%s}\n' $$f; cat $$f; printf '\n\\end{filecontents*}\n'; done; cat PNAS_Paper.tex; } > main.tmp && mv main.tmp PNAS_Paper.tex && rm $(ARXIV_EMBED)
	cd $(BUILD)/arxiv && pdflatex -interaction=nonstopmode -halt-on-error PNAS_Paper > /dev/null && pdflatex -interaction=nonstopmode -halt-on-error PNAS_Paper > /dev/null
	cd $(BUILD)/arxiv && ! grep -E -q "(Citation|Reference) .* undefined|There were undefined" PNAS_Paper.log
	cd $(BUILD)/arxiv && rm -f *.aux *.log *.out PNAS_Paper.pdf $(ARXIV_EMBED) && COPYFILE_DISABLE=1 tar --no-xattrs -czf ../arxiv.tar.gz .
	@echo "$(BUILD)/arxiv.tar.gz"

# The pipeline, in the order the steps depend on each other. It starts from
# data/source/ and the committed caches of the language-model review, and it
# needs no network and no paid service. Two steps read the ADART IV text, which
# is not distributed (data/source/not_distributed.csv), so they run only where
# their input is present. The whole rebuild takes about 25 minutes; the
# sensitivity audit is the slowest step. Run make check after it.
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
	$(PYTHON) scripts/paper/build_paper_counts.py
	$(PYTHON) scripts/audit/audit_held_out.py --run-held-out

# The data of the interactive page in docs/, from the committed results.
site:
	$(PYTHON) scripts/site/build_site_data.py

# Scripts outside the rebuild, whose outputs are committed. These need the
# network, a paid model or a book scan:
NEEDS_SOURCE := scripts/source/fetch_adsd.py scripts/source/fetch_texts.py \
	scripts/source/ocr_adart_iv_mathpix.py scripts/physical/ocr_parker_dubberstein.py \
	scripts/physical/build_ephemeris.py scripts/agent/model_review.py \
	scripts/paper/build_bibliography.py
# These ran once, as data/benchmark/addenda_registration.json registers the
# test on the dated addenda of ADART IV, and they read the ADART IV text:
REGISTERED_TEST := scripts/source/build_adart_iv_addenda.py \
	scripts/inference/date_adart_iv_addenda.py scripts/audit/score_adart_iv_addenda.py
# These trace one record, or give the tools to an MCP client:
EXPLORE := scripts/audit/date_tablet.py scripts/agent/mcp_server.py

# Every offline test, then the cold check of data, provenance and results.
check:
	for t in tests/test_*.py; do $(PYTHON) $$t || exit 1; done
	$(PYTHON) tests/verify.py

clean:
	rm -rf $(BUILD)
