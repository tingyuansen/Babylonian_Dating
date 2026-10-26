# Tools

The parser and the astronomical tools. The scripts in `scripts/` import them as the package `tools`. The tools read the text that they receive and the tables in `data/physical/`. They read no published date.

| Module | What it does |
|---|---|
| `statements.py` | The parser and the rules of step 2. It reads a translation into clauses, and gives each clause a verdict in a candidate year, agree, contradict or untestable (SI Appendix, sections S2 and S4, Table S2). |
| `cal.py` | The Babylonian calendar. It converts a Babylonian year, month and day to a Julian date, from `data/physical/calendar.parquet`. |
| `sky.py` | The positions of the Sun, the Moon, the planets and the Normal Stars, from `data/physical/ephemeris.parquet`, and the events that follow from them, such as stations, sign entries, solstices and elongations. |
| `eclipses.py` | The eclipses of `data/physical/eclipses.parquet` on a Babylonian day, and whether each one was visible at Babylon. |
| `eras.py` | Reads an era year or a regnal year that a text states, and converts it to an astronomical year. |
| `years.py` | The numbering of years, and the 701 candidate years from 626 BCE to AD 75. |
| `clauses.py` | Reads the reviewed clauses from the clause tables in `data/clauses/`, in the form that `statements.py` checks. |

`cal.py`, `sky.py`, `eclipses.py` and `eras.py` each have a `check()` function that tests the module on known cases, and `tests/verify.py` runs these functions. The results record the SHA-256 of each module except `clauses.py`, so `make check` reports any edit to them.
