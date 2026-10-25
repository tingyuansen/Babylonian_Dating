# inference

These scripts turn text into candidate years. They read a record's identifier, its surviving text and its reviewed clauses from `data/clauses/` (for the frozen addenda test, its parser clauses), and nothing else: no published date, catalogue position, volume, period or join.

| Script | What it does |
|---|---|
| `date_all.py` | Date every ADART I to III record into `data/results/tablet_dates.parquet`: its decision, candidate sets, groups of tied years and clause verdicts |
| `date_adart_iv.py` | Date the 867 undated ADART IV entries into `data/results/adart_iv_dates.parquet` |
| `date_adart_iv_addenda.py` | Date the 52 texts of the dated addenda of ADART IV into `data/results/adart_iv_addenda_dates.parquet`. It runs the dating of `date_adart_iv.py` with the paths of the addenda and, as the frozen test registered, with the parser clauses |

Each records the SHA-256 of `date_all.py`, the inference modules in `tools/`, the protocol `skills/dating_protocol.md` and the physical tables. `date_all.py` and `date_adart_iv.py` also record the clause table that they read, and the two ADART IV runs record `date_adart_iv.py` and the corpus that they read. An edit to any of these files invalidates the committed results until the rebuild runs again. `__init__.py` only marks the package.
