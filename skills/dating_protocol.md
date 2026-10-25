# Dating Skill, version 1

This is the result-free operating protocol supplied to human and agent auditors. Deterministic tools supply calendar arithmetic and astronomy; the auditor interprets surviving text, checks tool coverage, and returns a committed year, complete ranked set, or decline.

## 1. Output contract

- **Committed:** one year is fixed by a secure dated lunar eclipse or at least two independent decisive astronomy families that leave the same single candidate both strictly and after removal of any one decisive statement.
- **Ranked:** more than one candidate survives; return the complete strict and safety sets, their sizes, and the missing observation that would distinguish them.
- **Decline:** no active decisive year-bearing evidence survives; state the precise text, context, or tool reason and retain the full candidate set.

Never turn a ranking into an astronomy commitment with an era or regnal formula, catalogue order, period, prices, river levels, preservation, volume, museum order, or join. Record secure chronology separately; it may adjudicate an explicit astronomical set but is not astronomy-only success. Published dates and catalogue metadata are evaluation data, not auditor inputs.

## 2. Read what the text asserts

- “Became stationary to the east” begins retrograde motion; “to the west” ends it. A station is an event, not merely a position.
- “Last appearance in the east” is a visibility event near the Sun. The present elongation-only model failed its holdout gate, so its sign may support a ranking but its date may not eliminate a year.
- “Year A, which is Year B” can identify one year in two eras only when the epoch offset is valid and neither number is restored. An unrestored diary heading or edge colophon “Year N, king Seleucus/Antiochus/Demetrius” may be read as Seleucid Era under the Parker--Dubberstein chronology; the same words inside a historical report are not a date formula. Bare years, restored names or numbers, generic Arsaces, and event narratives do not identify an era.
- A tablet can span consecutive Babylonian years. Month rollover advances the candidate year; it is not a contradiction.
- A measured relation preserves body, target, amount, unit, direction, month, day, and restoration. One cubit is two degrees and 24 fingers make one cubit; compound cubit-plus-finger facts remain supporting until a decisive-tier forward gate exists.
- A measured station-to-star relation inherits its planet only from a named `when <planet> became stationary` antecedent in the same intact clause. Anaphoric, approximate-day, damaged, restored, or unwatched forms never become decisive; the admitted grammar remains supporting until a separate forward residual gate is frozen.
- A non-lunar planet-to-planet measurement requires explicit named subject, amount, unit, direction, and named target in one clause. It remains supporting pending a separate residual gate, and one outer-planet pair can never manufacture two independent decisive families.
- “Above” and “below” constrain latitude and longitude proximity. “In front of” is westward and “behind” eastward.

Do not move a sign, star, amount, direction, day, or visibility condition across a clause boundary or from one body to another. Explicitly omitted, excluded, ideal, computed, clouded-out, or unwatched values are not observations. Solstice, equinox, and Sirius dates are the exception: the scribes computed them with an arithmetical scheme, the tools test that scheme, and "I did not watch" after such a date does not change its tier.

## 3. Context and restoration

- A new month resets the day. Month I after Month XII advances one Babylonian year. Explicit consecutive year headings override rollover inference.
- In “the 1st, which followed the 30th of the preceding month,” day 30 belongs to the old month and the following observation is day 1.
- Recombine split Roman month numerals such as `Month V [ II ]` to VII and `Month I [ II ]` to III.
- A day does not govern a measured relation or eclipse across a lacuna or completed sentence. If local context is absent, retain the fact as unchecked.
- A summary directly before the next surviving month may inherit only an unambiguous immediately preceding month. Never interpolate two or more lost month headings from a favored year.
- Before Month VII, retain VI and VI₂ as an explicit union unless intercalation survives. A large day-number reset without a month heading marks lost context rather than reusing a stale month.
- Coordinated summaries apply the stated sign to every named planet, including three-body lists and beginning-to-end monthly paths.
- An explicitly coordinated two- or three-planet sign clause retains its atomic source records but is tested as one simultaneous configuration. There must be a nonzero interval in the governing month or secure day when all named planets occupy the stated sign. Separate clauses are not paired merely because they are nearby. Missing month context leaves the joint configuration unchecked.
- Bracketed, restored, or explicitly uncertain evidence may support a ranking but may not carry decisive weight on its own.

## 4. Evidence and scoring

The deterministic scorer checks every candidate year on the same frozen statements. Each predicate receives Shannon information `log2(checkable/admitted)` and clauses with identical candidate masks share one unit of information.

Decisive families are securely contextualized Mars, Jupiter, Saturn, measured Moon relations to an outer planet, an unrestored observed locally verified umbral lunar eclipse on a secure day, and the shared seasonal/Sirius scheme family. Sirius is labelled as Sirius, never as the Sun or a solar eclipse. Its scheme date shares the seasonal clock and does not introduce an additional independent family. Moon-to-star relations, Venus, Mercury, solar eclipses, last appearance, restored clauses, unresolved visibility, and compound-unit grammar remain supporting.

An unrestored simultaneous configuration of outer planets receives one total information weight and may identify its distinct outer-planet families. The whole coordinated source is one removal unit in the safety test; its atomic records are not additional independent votes. A configuration containing Venus or Mercury remains supporting as a whole, so an inner-planet conjunct cannot indirectly exclude years through an outer planet.

Count independent decisive families before information within a family. The strict result is the Pareto frontier over family count and information. Also return the union of the baseline frontier and each leave-one-decisive-statement-out frontier as a safety diagnostic. Never describe the broad safety set as dating accuracy.

For inspection, list strict candidates first, safety-only candidates second, and all other years last; within a tier order by decisive families, decisive information, supporting information, fewer contradictions, and a fixed year tie-break. This deterministic order is not a calibrated probability. Report top-k accuracy only against answers loaded after the order is frozen.

## 5. Commitment rules

- A secure dated lunar eclipse may commit on its own. A chronology formula is a separately labeled lookup or adjudicator.
- At least two decisive astronomy families may commit only when strict and safety sets are the same single candidate.
- One outer-planet family with repeated synodic recurrences is a ranking even when one candidate has the smallest residual.
- Never eliminate on Venus, Mercury, restored context, unresolved visibility, or supporting compound-unit grammar alone.
- Never break an astronomical tie with chronology the supplied tools did not validate.
- Work in astronomical year numbering internally and convert to BCE only in the report.

## 6. Mismatch audit

Before attributing a mismatch to a tablet, editor, scribe, or catalogue, check in order: candidate range and BCE conversion; Babylonian year rollover and month assignment; day assignment and lacuna leakage; split or restored month numerals; clause attachment, body, star, sign, amount, unit, and direction; observed versus omitted, predicted, restored, or unwatched status; calendar and event lookup; geocentric/topocentric and visibility limits; candidate weighting and evidence tier; Skill, prompt, permissions, runtime, and model behavior; translation against transliteration and edition witnesses.

If these checks do not settle the cause, report `unresolved method/evidence attribution`. Edition or scribal error is a last-resort claim requiring positive textual evidence.

## 7. Workflow

1. Read the whole surviving tablet without forming a date hypothesis.
2. Inventory year, month, day, eclipse, station, measured relation, and planet-in-sign clauses, including summaries and restoration.
3. Run deterministic `summary`, `statements`, and `rank` tools.
4. Inspect the context of every decisive clause and every contradiction at a leading candidate.
5. Compare strict candidates and ablate every decisive statement.
6. Search the source for unextracted outer planets, stations, measured relations, qualified signs, monthly paths, and typed eclipses.
7. Commit, rank, or decline under the output contract.
8. Record any tool or Skill gap separately from the tablet result; never patch one tablet.

Decline for absence of decisive evidence, not for difficulty. Fragmentary text and a large candidate set require an honest ranking or precise decline, not a guessed year.

## 8. Fixed conventions and limits

- One cubit is two degrees; 24 fingers make one cubit. The Babylonian day begins at sunset, a month begins after first-crescent visibility, and intercalary months are VI₂ and XII₂.
- Formal zodiacal signs use Babylonian sidereal longitude. Convert tropical longitude of date with Britton (2010), Eq. (5), $\Delta\lambda^*=3.20^\circ-1.3828^\circ Y$, where $Y$ is measured in Julian centuries from astronomical year 0, then use ideal 30-degree bins: Aries 0--30 degrees through Pisces 330--360 degrees. No conversion parameter is fitted to this corpus. Actual constellation or star-region statements remain separate evidence and are not inferred from these sign bins.
- The planetary ephemeris is geocentric. Local eclipse circumstances are calculated separately at Babylon across fixed Delta T trials. Robustly local solar events remain supporting; no solar-magnitude admission threshold is used. Atmospheric extinction and fitted planetary visibility dates remain outside the implemented sign predicates.
