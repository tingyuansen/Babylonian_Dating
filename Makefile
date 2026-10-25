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

.PHONY: paper figures check clean arxiv

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
ARXIV_EMBED = pnasresearcharticle.sty jabbrv.sty jabbrv-ltwa-all.ldf jabbrv-ltwa-en.ldf S.bbl
arxiv: $(PAPER_INPUTS)
	$(LATEX)
	rm -rf $(BUILD)/arxiv && mkdir -p $(BUILD)/arxiv/Figures
	cp paper/PNAS_Paper.tex paper/tableS7_adart_iv.tex paper/*.cls paper/*.sty paper/*.bst paper/*.ldf $(BUILD)/arxiv/
	cp paper/Figures/*.pdf $(BUILD)/arxiv/Figures/
	cp $(BUILD)/PNAS_Paper.bbl $(BUILD)/S.bbl $(BUILD)/arxiv/
	cd $(BUILD)/arxiv && perl -pi -e 's/\\includegraphics\[height=24\.5pt,width=66\.4pt\]\{PNAS_Logo\.pdf\}//' pnas-new.cls PNAS_Paper.tex
	cd $(BUILD)/arxiv && ! grep -rq PNAS_Logo .
	cd $(BUILD)/arxiv && { for f in $(ARXIV_EMBED); do printf '\\begin{filecontents*}[overwrite]{%s}\n' $$f; cat $$f; printf '\n\\end{filecontents*}\n'; done; cat PNAS_Paper.tex; } > main.tmp && mv main.tmp PNAS_Paper.tex && rm $(ARXIV_EMBED)
	cd $(BUILD)/arxiv && pdflatex -interaction=nonstopmode -halt-on-error PNAS_Paper > /dev/null && pdflatex -interaction=nonstopmode -halt-on-error PNAS_Paper > /dev/null
	cd $(BUILD)/arxiv && ! grep -E -q "(Citation|Reference) .* undefined|There were undefined" PNAS_Paper.log
	cd $(BUILD)/arxiv && rm -f *.aux *.log *.out PNAS_Paper.pdf $(ARXIV_EMBED) && COPYFILE_DISABLE=1 tar --no-xattrs -czf ../arxiv.tar.gz .
	@echo "$(BUILD)/arxiv.tar.gz"

# Every offline test, then the cold check of data, provenance and results.
check:
	for t in tests/test_*.py; do $(PYTHON) $$t || exit 1; done
	$(PYTHON) verify.py

clean:
	rm -rf $(BUILD)
