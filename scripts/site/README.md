# site

These scripts write the data of the interactive site in `docs/`. They read the committed results and change no number of the paper. Run them after the full rebuild. The paper does not need them.

| Script | What it does |
|---|---|
| `build_site_data.py` | Write `docs/data/meta.json`, `docs/data/sky.json` and one `docs/data/records/<id>.json` per ADART I to III record. It reads the corpus, `data/results/tablet_dates.parquet` and the physical tables. It recomputes the score of each year with `tools/statements.py`, and stops if a strict set differs from the committed one |
| `cuneiform.py` | Turn the ADsD transliteration into Unicode cuneiform signs with the ORACC Sign List, `data/source/osl/osl.asl`. A reading that the list does not hold stays in Latin letters. It runs only as an import of `build_site_data.py` |

`__init__.py` only marks the package.
