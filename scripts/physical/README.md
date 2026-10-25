# physical

These scripts build the tables that the astronomy is computed from. Each table comes from a primary source and is then checked against something the source does not state.

| Script | What it does |
|---|---|
| `build_calendar.py` | Parse the OCR text of Parker and Dubberstein (1956) into month starts for 701 years, 626 BCE to AD 75. Apply the visually verified OCR corrections and log them to `data/physical/calendar_ocr_corrections.csv`. Take the month that each thirteen-month year doubled from the printed column in `data/source/parker_dubberstein/intercalary_slots.csv`. The four 28- and 31-day intervals that the tables print are kept. Any other month-start interval outside 29 to 30 days, within a year or across a year boundary, fails the build |
| `build_normal_stars.py` | Extract the Normal Star catalogue from the ADART I introduction and reduce it to the fixed J2000 ecliptic with the precession `build_ephemeris.py` uses |
| `build_ephemeris.py` | Generate the daily grid from the JPL DE441 kernel. It uses fixed J2000 coordinates for stellar geometry, and tropical ecliptic coordinates of date for the seasons and the zodiac conversion. The kernel is not retained, so this runs only when one is supplied |
| `build_eclipses.py` | Build the NASA eclipse catalog, and compute local circumstances at Babylon, phase by phase, for the nominal Delta T and one standard error either side |
| `eclipse_catalog.py` | Shared reader for the two NASA catalogue formats, including catalogue Delta T and magnitudes. It runs only as an import of `build_eclipses.py` and `build_ephemeris.py` |
| `check_calendar_against_sky.py` | Check month starts against computed conjunctions, independently of the parser. It fails if a month start lies less than 0.3 or more than 3.4 days after conjunction. Rows listed as checked against the print are exempt. It repairs nothing, because repairs belong to `build_calendar.py` |
| `appulse_residuals.py` | Measure the difference between each stated and computed distance, at the published years |
| `validate_zodiac.py` | Check dated planet-in-sign clauses against ideal Babylonian signs of 30 degrees, from Britton's published relation, λ_Bab = λ_date + 3.20° − 1.3828° Y. It reports agreement and fits no boundary |
| `calibrate_visibility.py` | Fit first and last appearance thresholds, then check them on held-out cases. The fit failed this check, so the tools leave it out |
| `ocr_parker_dubberstein.py` | Read PDF pages 43 to 63 of the 1956 Parker and Dubberstein scan with Tesseract into `data/source/parker_dubberstein/calendar_ocr.txt`. With `--slots`, read from Tesseract's word positions whether the extra month of each thirteen-month year is printed under U II or A II. Write this into `data/source/parker_dubberstein/intercalary_slots.csv`. The script lists the rows checked by eye, and the reading must agree with them. Needs the scan at `data/source/parker_dubberstein/scan.pdf`, and `pdftoppm` and `tesseract` on the path. Run once, with the output retained |

`__init__.py` only marks the package.
