# source

These scripts turn a published edition into text that this repository can read. Nothing here computes astronomy.

| Script | What it does |
|---|---|
| `fetch_adsd.py` | Download the ORACC catalogue and the ADART I project into `data/source/adsd/`, and record byte counts and hashes in its `manifest.json` |
| `fetch_texts.py` | Fetch transliteration and translation for every ADART I to III record into `data/source/adsd/texts.jsonl` |
| `build_corpus.py` | Join the catalogue to the text, one row per ADART I to III record, into `data/corpus/tablets.parquet` and `lines.parquet` |
| `classify_evidence.py` | Record what chronological evidence each ADART I to III record preserves, by rule and, with `--llm`, by an independent model pass |
| `build_cohort.py` | Assign reproducible coarse cohorts from text properties alone |
| `ocr_adart_iv_mathpix.py` | Submit and retrieve the optical reading of the ADART IV scan, using the `MATHPIX` key in `~/.env`. It keeps the line JSON only. Paid, and it needs the scan at `data/source/adart_iv/scan.pdf`. Run only when a new reading is intentionally authorised |
| `build_adart_iv_corpus.py` | Segment the undated section into entries, and stop an entry at any catalogue heading belonging to another tablet |
| `build_adart_iv_addenda.py` | Segment the dated addenda of ADART IV into texts for the frozen test of `data/benchmark/addenda_registration.json`, with the filters of `build_adart_iv_corpus.py`. Write the published year of each text, and its links to ADART I to III, to a separate file that only the scorer reads |

`__init__.py` only marks the package.
