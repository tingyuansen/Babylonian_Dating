# tools

These modules are the primitives that inference is built from. Each answers one question. `cal`, `sky`, `eclipses` and `eras` are built from a primary source and check themselves against something the source does not state. `years` fixes the year convention they all share.

| Module | Question it answers |
|---|---|
| `years.py` | How is a year numbered? Years use astronomical numbering throughout, so 626 BCE is −625, 1 BCE is 0 and AD 1 is 1. The candidate years run from −625 to 75, 701 in all. BCE numbers are converted only where data enters or leaves |
| `cal.py` | Which Julian days does this Babylonian month and day cover, which month did this year double, and when must an era conversion refuse? |
| `sky.py` | Where were the Sun, Moon and planets, and where did a planet station or cross a sign boundary? When did the Sun reach a solstice or equinox, and when did it set and rise at Babylon? Each sign covers 30 degrees of Babylonian longitude, converted from the tropical longitude of date by Britton's published relation. Nothing is fitted |
| `eclipses.py` | Was there an eclipse of this type on this Babylonian day, and which phase was visible at Babylon? Does that hold over the uncertainty in Delta T? |
| `eras.py` | Does an unrestored era formula in the text name an absolute year? |
| `statements.py` | What claim does this clause make, and which candidate years agree with it? |
| `clauses.py` | Which clauses does a record's date use? It reads the reviewed clauses of `data/clauses/` back into the clauses of `statements.py` |
| `__init__.py` | Marks the package |

`statements.py` holds the parser, the tier rule, the scoring and the candidate sets, and is the only place a candidate year is decided.

Every module except `statements.py` has a `check()` that runs when the module is executed, for example `python -m tools.cal`. `verify.py` runs the checks of `cal`, `eras`, `sky` and `eclipses`. `statements.py` is exercised by the worked cases in `verify.py` and by `tests/test_joint_positions.py`.

Committed results record SHA-256 hashes of these files. `date_all.py` and `date_adart_iv.py` hash `cal`, `sky`, `eclipses`, `eras` and `statements`, and `verify.py` fails when the hashes in `data/results/tablet_dates.parquet` no longer match. The reviewed-clause tables carry a fingerprint of `statements`, `sky`, `cal` and `eclipses`. The same fingerprint covers `scripts/agent/model_review.py` and the four physical tables `ephemeris.parquet`, `calendar.parquet`, `eclipses.parquet` and `normal_stars.parquet`. Editing any of them, even a comment, invalidates the committed results until the pipeline is rerun. `years.py` is not hashed.
