# paper

These scripts write the counts, tables and figures of the paper. The paper itself is kept outside the public repository, in `paper/`, which git ignores. Where it is present, build from the repository root.

```bash
make figures   # redraw Fig. 1 and the five data figures
make paper     # write SI Table S7 and build paper/PNAS_Paper.pdf
make arxiv     # pack paper/ with its .bbl files for arXiv
```

`make` keeps the LaTeX intermediates in `/tmp/Babylonian_Dating_Paper_build` and copies only the PDF back. The SI keeps its own reference list through `multibib`, so the build runs `bibtex` twice. arXiv does not run `bibtex`, so `make arxiv` packs both `.bbl` files with the LaTeX inputs into `/tmp/Babylonian_Dating_Paper_build/arxiv.tar.gz`. It first checks that the pack builds with `pdflatex` alone.

| Script | What it does |
|---|---|
| `build_paper_counts.py` | Filter the saved outputs to ADART I to III and write the counts and SI tables the paper quotes into `data/paper_counts/`. It runs no model, extraction or inference |
| `build_adart_iv_table.py` | Write the 52 ranked ADART IV entries into `data/results/adart_iv_ranked.csv`, and, where the paper is present, the rows of SI Table S7 into `paper/tableS7_adart_iv.tex`. It reads only the committed results |
| `build_bibliography.py` | Write `paper/Bibliography.bib` from Crossref, NASA ADS, arXiv and other APIs, using the identifiers in `paper/bibliography.json`. Only cited keys are written. `--check` fails if the APIs no longer give the same entries. Outside the rebuild order |

`build_paper_counts.py` reads the reviewed clauses written by `agent/rescore_model_review.py`, the parser clauses, the committed results, the benchmark and the calendar. It refuses to run if a saved score no longer matches the current tools and physical tables, or if an input changes while it runs. It writes four files.

- `counts.json` holds the counts the paper quotes, such as the numbers of reviewed and parser clauses and their agreement at the published year.
- `table_s3.csv` is SI Table S3, bits by observation type.
- `table_s4.csv` is SI Table S4, agreement at the published year.
- `source_manifest.json` holds the SHA-256 of every input.

## figures/

Each figure has one module. It writes the PDF of the same name into `paper/Figures/`. The data figures read committed results. Labels use the paper's terms.

| File | What it draws |
|---|---|
| `figure_style.py` | Shared typography, palette and widths |
| `fig1_workflow.py` | Fig. 1. The workflow in three steps, drawn in code as vector graphics. It reads no data. The module computes the bits, the clocks and the strict set of its small example |
| `fig2_diary.py` | Fig. 2. A photograph of a diary, and one tablet's lines against the candidate years each leaves standing. The public-domain photograph is `data/source/photos/alexander_diary_BM45962.jpg` |
| `fig3_x301492.py` | Fig. 3. Record X301492 carried from its clauses to a single year, and its decisive bits over the candidate calendar |
| `fig4_benchmark.py` | Fig. 4. How often the published year is included, and how large a set that costs |
| `figS1_sensitivity.py` | Fig. S1. The benchmark under each alternative setting of the sensitivity audit |
| `figS2_adart_iv.py` | Fig. S2. The strict sets of the ranked ADART IV entries |

`__init__.py` only marks the package.
