#!/usr/bin/env python3
"""Review the parser's clauses with a language model, and search for missed ones.

This is step 1 of the paper's pipeline after the rule-based parser:

    tablet translation
        -> deterministic tools.statements.extract()
        -> the model reviews EACH parser clause: KEEP, MODIFY or DROP
        -> a MODIFY is split into single-field changes; a second call confirms
           each, and code accepts it only if its quote lies in the clause's own
           sentence and contains the new value
        -> a DROP needs a second call that names the contradicted field and the
           value the text gives instead; code accepts it only if that quote lies
           in the clause's own sentence and contains that value
        -> the model searches the text for clauses the parser missed; code finds
           every cited passage word for word and checks every field against it
        -> deterministic astronomy scores the combined clause list

Answer blinding. The model never receives a published year, a candidate year,
an astronomical verdict, or the record identifier. Each payload carries an
opaque token instead of the ADsD id, and year numbers and ruler names in the
translation are replaced by {year} and {ruler} before any text reaches the
model. Offsets of every cited passage are mapped back to the original text.

Unobserved events. Only the parser's rule, tools.statements.unwatched, marks a
clause as unobserved. The model cannot change the mark in either direction.

The model is called through an OpenAI-compatible chat-completions endpoint:
the course LiteLLM proxy (--route litellm, key LITELLM_API_KEY) or, as a
fallback, the Gemini API (--route gemini, key GOOGLE_API_KEY). Keys are read
from the environment, a .env at the repository root, or ~/.env. Every call
records the model the endpoint reports, its tokens and, where the proxy
reports it, its cost. Requests run in parallel under an adaptive limit that
backs off on HTTP 429.

All outputs go to one directory per model, by default data/clauses/model_review_<model>:

    review_cache.jsonl, recall_cache.jsonl   cached model work (paid)
    review_changes.csv                every proposed change and its checks
    search_proposals.csv                 every search proposal and its fate
    fallbacks.jsonl                          candidate-local fallbacks taken
    run_summary.json                         calls, tokens, cost, time, failures

Examples:
    # Count the calls a run would make; no API calls.
    python scripts/agent/model_review.py --dry-run

    # Pilot: review and search a few records, write the audits and a summary.
    python scripts/agent/model_review.py --model gpt-6-luna-2026-09-22 \\
        --tablet X102732 --tablet X102890

    # Full run, then the cache-only rebuilds and the ADART I-III scope tables.
    python scripts/agent/model_review.py --model gpt-6-luna-2026-09-22
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from tools import sky, statements as S  # noqa: E402
from tools import years as yr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The versions enter every cache key together with the model name, so a change
# to a prompt, a payload or the blinding makes earlier responses cache misses.
# "step1" is the old name of this review. It stays in these strings, because a
# new string would make every committed response a cache miss.
PROMPT_VERSION = "step1-review-v7-blind"
RECALL_DISCOVERY_PROMPT_VERSION = "step1-recall-discovery-v6-blind"
RECALL_NORMALIZE_PROMPT_VERSION = "step1-recall-normalize-v9-blind"
DEFAULT_MODEL = os.environ.get("REVIEW_MODEL", "gpt-6-luna-2026-09-22")
DEFAULT_ROUTE = os.environ.get("REVIEW_ROUTE", "litellm")

CLAUSE_LABEL = {
    "appulse": "Moon or planet measured against a Normal Star",
    "in_sign": "Planet in a named sign",
    "last_app": "Last appearance in a sign",
    "eclipse": "Eclipse of a stated type",
    "entry": "Planet enters a sign",
    "planet_appulse": "Moon measured against a planet",
    "station_appulse": "Station measured against a Normal Star",
    "planet_planet_appulse": "Planet measured against a planet",
    "station": "Planet stations in a sign",
    "first_app": "First appearance in a sign",
    "season": "Solstice or equinox on a day",
    "sirius": "Sirius date on a day",
    "acronychal": "Acronychal rising on a day",
}

FACT_LABEL = {
    "appulse": "Normal-Star relation",
    "planet_appulse": "Moon-planet relation",
    "station_appulse": "Station-star relation",
    "planet_planet_appulse": "Planet-planet relation",
    "in_sign": "Planet in sign",
    "last_app": "Last appearance sign",
    "entry": "Sign entry",
    "eclipse": "Typed eclipse",
    "season": "Solstice or equinox",
    "sirius": "Sirius scheme date",
    "acronychal": "Acronychal rising",
    "first_app": "First appearance sign",
    "station": "Station in sign",
}

FACT_ORDER = [
    "appulse", "planet_appulse", "station_appulse",
    "planet_planet_appulse", "in_sign", "last_app", "entry",
    "eclipse", "season", "sirius", "acronychal", "first_app",
    "station",
]

KINDS = set(FACT_ORDER)
MEASURED_KINDS = {
    "appulse", "planet_appulse", "station_appulse",
    "planet_planet_appulse",
}
BODY_NAMES = {
    "", "moon", "mercury", "venus", "mars", "jupiter", "saturn",
    "sun", "sirius", "lunar", "solar",
}
DIRECTIONS = {None, "above", "below", "in front of", "behind"}
STATION_PHASES = {None, "east", "west"}
UNITS = {None, "cubit", "cubits", "finger", "fingers", "cubit+finger"}

SYSTEM_PROMPT = r"""
You are a semantic auditor for English translations of Babylonian Astronomical
Diaries. A deterministic parser has already extracted ONE structured
astronomical statement. Review whether that structured statement accurately
represents what the supplied source text actually says.

You have authority to return exactly one of three actions:

KEEP   - the deterministic structured statement is semantically adequate.
MODIFY - the source contains the astronomical claim, but one or more semantic
         fields should be corrected. Return the complete corrected semantic
         fields.
DROP   - the parser match itself is not a testable astronomical claim or the
         parser has assigned a claim that the matched source does not assert.

SOURCE-GROUNDING RULES -- IMPORTANT:
1. context_before + parser_matched_source + context_after are contiguous pieces
   of the SAME source translation.
2. parser_matched_source is GUARANTEED by the calling code to be an EXACT
   literal substring of the source translation. Treat that provenance as a
   hard fact, not something to verify. It is primary evidence. Do NOT require
   the same words to occur again in context_before or context_after, and never
   claim that parser_matched_source is absent from the source.
3. If parser_matched_source directly states the parsed astronomical relation,
   do not DROP it merely because surrounding context does not repeat the planet,
   star, sign, number, or relation.
4. DROP only when POSITIVE textual evidence shows that the matched source is
   not actually the structured astronomical assertion (for example the regex
   crossed incompatible syntax, attached the wrong grammatical subject/target,
   or matched non-astronomical/editorial language). Mere absence of repetition
   in context_before/context_after is NEVER a valid DROP reason.
5. For MODIFY or DROP, evidence_quote must be copied literally from the
   supplied source, from the SENTENCE that contains parser_matched_source, and
   it must contain the corrected value. Calling code rejects a quote taken from
   another sentence. For KEEP it may be empty.

ATOMIC NORMALIZATION -- IMPORTANT:
6. One source clause may intentionally be decomposed by the deterministic parser
   into MULTIPLE atomic Statement objects. Review ONLY the one atomic Statement
   you were given. A statement is NOT incomplete or wrong merely because the
   same parser_matched_source also asserts another planet or another conjunct.
7. In particular, parser_rule="coordinated_sign" intentionally turns a clause
   such as "Saturn and Mars were in Leo" into TWO valid atomic statements:
   "Saturn was in Leo" and "Mars was in Leo". parser_rule=
   "triple_coordinated_sign" does the same for three planets. KEEP each atomic
   statement if its own planet and sign are asserted by the coordinated source.
   Do NOT DROP it because another coordinated planet is absent from that one
   structured object.
8. For kind="sirius", planet="sirius" names the actual source subject. Cached
   legacy planet="sun" values are migrated deterministically, not reinterpreted
   as a solar observation. The source phenomenon is Sirius first appearance,
   last appearance, or acronychal rising, represented by event_subtype.
   If parser_matched_source literally says a recognized Sirius appearance/rising,
   that is itself a valid astronomical claim; surrounding context need only supply
   inherited month/day information.

ANSWER-BLINDNESS:
9. Judge ONLY from the supplied source text and parser representation. You are
   deliberately not given a BCE year, candidate years, astronomical
   calculations, or published answer. Never infer or propose a chronological
   year.
9a. Year numbers and ruler names in the translation are withheld and appear as
   {year} and {ruler}. "record" is an opaque token. None of them matter here.

RESTORATION, OMISSION, AND UNCERTAINTY:
10. restored and omitted are INDEPENDENT concepts.
11. restored=true means that a relevant word, measurement, subject, target,
   sign, or governing temporal context depends on editorial restoration or
   bracketed reconstruction.
12. omitted (the scribe's mark that the event was not observed, such as "I did
   not watch") is set by deterministic code with the parser's own rule. Return
   the deterministic value unchanged; code ignores any change to it.
13. year_offset is set by deterministic code from year headings. Return the
   deterministic value unchanged; code ignores any change to it.
14. A real phenomenon may be reported even when it was not watched. "I did not
   watch" is never a reason to DROP.

TEMPORAL CONTEXT:
15. month/day/year_offset may have been inherited by the deterministic parser
    from a governing heading earlier in the tablet, possibly outside the local
    excerpt shown here. Therefore ABSENCE of a month/day heading in this excerpt
    is NOT evidence that an existing month/day/year_offset is wrong.
16. Modify month/day/year_offset only when the supplied text positively supports
    a different value, shows that the parser attached the wrong nearby heading,
    or otherwise directly contradicts the parser's context assignment.
17. The same conservative rule applies to an existing restored=true caused by
    governing context outside the excerpt: do not clear it merely because the
    local matched phrase itself is unbracketed.

MEASURED RELATIONS -- INTERNAL NORMALIZATION:
18. For measured relations the deterministic Statement.amount is an INTERNAL
    normalized value in CUBITS, even when the source unit is fingers. Example:
    source "10 fingers" is represented internally as 10/24 = 0.4166667 cubit.
    This is NOT an error.
19. The payload separately supplies source_measurement with literal source-unit
    components: measurement_cubits and measurement_fingers. Review and, if
    necessary, correct THESE source-unit components. Do not try to make the
    internal normalized amount numerically equal to the literal finger count.
20. 24 fingers = 1 cubit. The downstream code will deterministically convert
    the final source-unit components to normalized cubits and degrees.
21. For a source "4 cubits", final measurement_cubits=4 and
    measurement_fingers=null. For "10 fingers", measurement_cubits=null and
    measurement_fingers=10. For "1 cubit 4 fingers", use 1 and 4 respectively.
22. For measured relations preserve/correct subject body, target star/planet,
    direction, and source measurement only according to the literal source.

GENERAL EDITING POLICY:
23. Do not change a field merely to be stylistically different. MODIFY only when
    the supplied source supports a concrete semantic correction.
23a. Every proposed MODIFY will be independently checked field by field. Do not
    bundle an unsupported cleanup with a supported correction. Do NOT clear
    month/day/restored merely because those values are not visible in this
    excerpt.
24. If the parser has recognized a real claim but the supplied text is
    insufficient to prove a correction, choose KEEP rather than inventing one.
25. Keep explanations concise and specific.

Field meanings:
- kind: one of appulse, planet_appulse, station_appulse,
  planet_planet_appulse, in_sign, last_app, entry, eclipse, season, sirius,
  acronychal, first_app, station.
- planet: lower-case subject body. For typed eclipses it is "lunar" or "solar";
  for season/Sirius it is "sun".
- target_planet: lower-case target for measured planet-to-planet relations.
- star: Normal-Star label if applicable.
- sign: canonical zodiac sign if applicable, else empty string.
- month/day/year_offset: governing Babylonian calendar context when preserved.
- measurement_cubits / measurement_fingers: LITERAL source-unit components,
  not the parser's internally normalized amount. Both null for non-measured
  statement kinds.
- direction: above, below, in front of, or behind for measured relations.
- station_phase: east or west when explicitly supplied.
- omitted: set by code from the scribe's marks; copy the deterministic value.
- restored: relevant evidence/context depends on editorial restoration.
- event_subtype: for season use solstice/equinox; for sirius use
  rising/setting/acronychal; otherwise null.
""".strip()


REVIEW_FORMAT = {
    "type": "json_schema",
    "name": "statement_semantic_review",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["KEEP", "MODIFY", "DROP"]},
            "explanation": {"type": "string"},
            "evidence_quote": {"type": "string"},
            "final": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": sorted(KINDS)},
                    "planet": {"type": "string"},
                    "target_planet": {"type": ["string", "null"]},
                    "star": {"type": ["string", "null"]},
                    "sign": {"type": "string"},
                    "month": {"type": ["string", "null"]},
                    "day": {"type": ["integer", "null"]},
                    "year_offset": {"type": "integer"},
                    "measurement_cubits": {"type": ["number", "null"]},
                    "measurement_fingers": {"type": ["number", "null"]},
                    "direction": {"type": ["string", "null"]},
                    "station_phase": {"type": ["string", "null"]},
                    "omitted": {"type": "boolean"},
                    "restored": {"type": "boolean"},
                    "event_subtype": {
                        "type": ["string", "null"],
                        "enum": [None, "solstice", "equinox", "rising", "setting", "acronychal"],
                    },
                },
                "required": [
                    "kind", "planet", "target_planet", "star", "sign", "month",
                    "day", "year_offset", "measurement_cubits",
                    "measurement_fingers", "direction", "station_phase",
                    "omitted", "restored", "event_subtype",
                ],
            },
        },
        "required": ["action", "explanation", "evidence_quote", "final"],
    },
}


# ---------------------------------------------------------------------------
# Raw-text recall stage: discover claims S.extract() missed completely
# ---------------------------------------------------------------------------

RECALL_DISCOVERY_PROMPT = r"""
You are performing a HIGH-RECALL audit of an English translation of a
Babylonian Astronomical Diary.

A deterministic parser has already produced the atomic statements listed in
existing_structured_statements. Your task is to read the RAW SOURCE directly
and find TESTABLE ASTRONOMICAL claims whose evidence STARTS inside core_text and
which are NOT already represented by those existing statements.

This is a discovery task, not a repository-classification task. For each missed
claim, return a short natural-language claim plus an exact evidence quote. Do
not assign an internal kind such as appulse/in_sign here; a separate
normalization pass will do that.

IMPORTANT:
1. Find genuinely missing ASTRONOMICAL claims, not stylistic restatements of an
   existing atomic statement.
2. EXCLUDE ordinary weather, river levels, commodity prices, animal events,
   births/deaths, political events, and other non-astronomical diary content.
   Weather may be mentioned only as part of the evidence for an astronomical
   observation's status (for example, "I did not watch because of clouds").
3. One coordinated source clause may contain multiple atomic claims. Prefer to
   return the atomic conjuncts separately. If you do return a coordinated claim,
   preserve all subjects explicitly; the normalization stage is allowed to
   expand it into multiple repository Statements.
4. Resolve a local pronoun such as "it" only when its antecedent is unambiguous
   in the supplied source context.
5. Editorial brackets, broken words, line breaks, and restoration markup may
   interrupt a word or clause. Do not ignore an otherwise clear astronomical
   claim merely because it is restored or damaged.
6. "I did not watch", "I did not see", or equivalent does NOT erase the
   astronomical claim. It is observational status that will be represented
   later.
7. Include elliptical but grammatically clear astronomical assertions, e.g.
   "Jupiter in Gemini" in a context where this means Jupiter was in Gemini.
8. Include measured relations and preserve coordinated subjects, e.g.
   "Mars and Mercury were 4 cubits in front of α Leonis" contains TWO atomic
   astronomical claims.
9. Include a real astronomical claim even if the current repository schema may
   not support it. The later normalization pass must label such a claim
   unsupported_but_astronomical rather than forcing it into a wrong kind.
10. Do NOT infer a BCE year and do not use any astronomical calculation,
    published date, candidate ranking, or answer. Year numbers and ruler names
    are withheld and appear as {year} and {ruler}; "record" is an opaque token.
11. evidence_quote MUST be copied VERBATIM from the supplied source fields. It
    should be long enough to identify the proposition and its grammatical
    subject/target, normally the whole local clause.
12. Return only claims whose evidence_quote begins in core_text. context_before
    and context_after are supplied only to resolve local grammar/context.
13. Before returning, scan core_text from beginning to end a second time and
    explicitly check for EACH of these high-value constructions: first appearance,
    last appearance, stationary/station, planet in a zodiac sign, entry/reaching a
    sign, measured cubit/finger relations, Moon-planet relations, planet-planet
    relations, eclipse statements, solstice/equinox, Sirius events, and acronychal
    risings. Do not stop after finding several claims; the goal is exhaustive recall
    within the supplied core window.
14. If there are no missed testable astronomical claims, return an empty claims
    array.
""".strip()

RECALL_DISCOVERY_FORMAT = {
    "type": "json_schema",
    "name": "raw_tablet_recall_discovery",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["claim", "evidence_quote", "explanation"],
                },
            },
            "overall_explanation": {"type": "string"},
        },
        "required": ["claims", "overall_explanation"],
    },
}


RECALL_QUOTE_REPAIR_PROMPT = r"""
You are repairing SOURCE GROUNDING ONLY for a raw-text recall audit.

The candidate claims were already proposed. Do NOT reinterpret, expand, or
paraphrase them. For each candidate listed in bad_claims, either:

- USE_QUOTE: copy one exact literal substring from context_before/core_text/
  context_after that directly supports that candidate claim; or
- DROP: if no exact source substring supports that candidate claim.

Hard requirements:
1. evidence_quote for USE_QUOTE must be copied VERBATIM from the supplied source.
2. Preserve brackets, spaces, line breaks, punctuation, Greek letters, and
   damaged/restored text exactly as they occur.
3. Do not repair a quote by paraphrasing the source.
4. Do not introduce any new candidate claims.
5. If the previous evidence quote was a paraphrase of a real source statement,
   copy the actual source wording instead.
6. If no literal source span supports the claim, choose DROP rather than guess.
""".strip()


RECALL_QUOTE_REPAIR_FORMAT = {
    "type": "json_schema",
    "name": "raw_tablet_recall_quote_repair",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "repairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim_index": {"type": "integer"},
                        "action": {"type": "string", "enum": ["USE_QUOTE", "DROP"]},
                        "evidence_quote": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["claim_index", "action", "evidence_quote", "explanation"],
                },
            }
        },
        "required": ["repairs"],
    },
}


RECALL_NORMALIZE_PROMPT = r"""
You are the SECOND stage of a raw-text recall audit for Babylonian Astronomical
Diaries. Another reviewer discovered ONE candidate astronomical claim directly
from source text. Decide whether that candidate should become one OR MORE
repository Statements.

Return ADD when the source really asserts one or more testable astronomical
claims that can be represented by supported repository kinds. A coordinated
candidate MUST be atomized: for example, "Venus and Mercury were in Leo" yields
TWO finals, one for Venus and one for Mercury; "Mars and Mercury were 4 cubits
in front of α Leonis" likewise yields TWO measured finals.

Return SKIP with skip_reason="unsupported_but_astronomical" when the candidate
is genuinely astronomical but the current repository ontology cannot represent
it safely. Return another SKIP reason when it is non-testable, too ambiguous,
not actually asserted, or otherwise unsuitable. NEVER force an unsupported
claim into a wrong supported kind.

The exact evidence_quote is primary source evidence. Every non-null semantic
field in every ADD must be grounded in that quote. Do not import a sign, star,
planet, target, measurement, or relation from nearby text unless that same
field is literally supported by the evidence_quote. The calling code performs
an independent mechanical grounding check after your response. It also
deterministically canonicalizes measured-relation kinds from the literal body
topology (Moon-to-planet, planet-to-planet, or body-to-Normal-Star), so do not
force a relation into the wrong kind merely to satisfy the schema.

The source may contain editorial breaks inside names: e.g. "Ge ] mini" still
supports Gemini, "α [ Leonis ... ]" still supports α Leonis, and
"Mer [ cury" still supports Mercury. This does NOT license importing a field
that is absent from the quote.

SUPPORTED KINDS AND FIELD RULES:
- in_sign: a planet is in a named zodiac sign. Requires planet+sign. No star,
  target_planet, measurement, or direction.
- first_app / last_app: first or last appearance of a planet in a zodiac sign.
  Requires planet+sign. Phrasing such as "first appearance in the end of
  Sagittarius" still means first_app in Sagittarius. If the source gives only
  an asterism such as the Chariot and no zodiac sign, do NOT invent a sign;
  return unsupported_but_astronomical unless another supported kind fits.
- entry: a planet reaches/enters a named zodiac sign. Requires planet+sign.
- station: a planet becomes stationary in a named zodiac sign. Requires
  planet+sign. station_phase may be east/west only when that phase itself is
  part of the source claim; otherwise null.
- appulse: INTERNAL category for a measured Moon/planet relation to a catalogued
  Normal Star. Requires planet+star+literal source measurement+one of the
  repository directions: above, below, in front of, behind. It does NOT mean
  generic closeness and MUST NOT be used for "Jupiter in Gemini".
- planet_appulse: measured Moon-to-planet relation. Requires moon as subject,
  target_planet, literal measurement, and repository direction.
- planet_planet_appulse: measured visible-planet-to-visible-planet relation.
  Requires subject planet, target_planet, literal measurement, and direction.
- station_appulse: measured relation from a stationary outer planet to a
  catalogued Normal Star.
- eclipse: typed lunar/solar eclipse. planet must be "lunar" or "solar".
- season: solstice/equinox. planet is internally "sun" and event_subtype must be
  solstice/equinox.
- sirius: Sirius first/last appearance or acronychal scheme date. planet is
  "sirius"; star MUST be null. event_subtype is rising for Sirius first
  appearance, setting for Sirius last appearance, acronychal for acronychal
  rising.
- acronychal: acronychal rising of Mars/Jupiter/Saturn.

ATOMICITY AND SUBJECT BINDING:
- finals is an ARRAY. A simple candidate normally yields one final.
- A coordinated clause yields one final per atomic supported proposition.
- Do not merge two subjects into one final.
- Do not create extra atoms that the evidence quote does not assert.
- EVERY final must include atomic_evidence_quote: the SMALLEST contiguous
  substring of evidence_quote that directly supports that one atomic claim. It
  must be copied verbatim from evidence_quote. Do not use a broad span merely
  because the needed words occur somewhere inside it.
- EVERY final must include subject_binding, one of:
    DIRECT: the grammatical subject is explicitly named in atomic_evidence_quote.
    COORDINATED: the subject is explicitly named as one conjunct in the same
      coordinated clause as the predicate/target/sign.
    ANAPHORIC: the atomic clause uses a local pronoun (normally "it") whose
      immediately preceding antecedent is unambiguous. In this case provide
      antecedent_quote, copied verbatim from the supplied context/evidence, and
      keep atomic_evidence_quote focused on the anaphoric clause itself.
- For DIRECT and COORDINATED, antecedent_quote must be null.
- For ANAPHORIC, antecedent_quote must name the subject and must occur BEFORE
  atomic_evidence_quote in close local context. Never use a later sentence as
  an antecedent.
- Grammatical source roles and repository storage roles are not always the same.
  In a star-first relation such as "δ Scorpii ... it stood 1 2/3 cubits in
  front of Mercury", the grammatical subject is the star, but the repository
  canonically stores the visible planet in the planet field and the Normal Star
  in the star field. Preserve a local ANAPHORIC binding to the star antecedent;
  deterministic code will canonicalize the stored roles and invert the relation
  direction when necessary (star in front of Mercury -> Mercury behind star).
- A planet name appearing only in a FOLLOWING independent sentence does NOT
  support assigning that planet to an earlier damaged clause. For example:
    "[...] first appearance in the end of Pisces. Night of the 29th,
     Mercury's first appearance ... in Aries."
  does NOT support Mercury as the subject of the Pisces clause. Return SKIP for
  that unsupported proposition unless an earlier/local antecedent genuinely
  establishes the subject.

MEASUREMENTS:
Record literal source units, not normalized internal values. Thus "10 fingers"
means measurement_fingers=10 and measurement_cubits=null; "1 cubit 4 fingers"
means 1 and 4. 24 fingers = 1 cubit; deterministic code performs conversion.
Do not copy a different measurement from elsewhere in a long quote.

NULLS:
For fields that do not apply, return JSON null (or the empty string for sign as
required by the schema). Never return the literal strings "null", "None",
"unknown", or "N/A" as semantic values.

RESTORATION / OMISSION:
- restored=true only when a relevant part of this claim depends on editorial
  restoration/bracketed reconstruction.
- omitted (the scribe's mark that the event was not observed) is assigned by
  calling code with the parser's own rule. Return false; code ignores it.

Year numbers and ruler names are withheld and appear as {year} and {ruler};
"record" is an opaque token.

Do not guess month/day/year_offset. The calling code assigns temporal context
from source offsets using the same deterministic context machinery as the
original parser.
""".strip()

RECALL_FINAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": sorted(KINDS)},
        "planet": {"type": "string"},
        "target_planet": {"type": ["string", "null"]},
        "star": {"type": ["string", "null"]},
        "sign": {"type": "string"},
        "measurement_cubits": {"type": ["number", "null"]},
        "measurement_fingers": {"type": ["number", "null"]},
        "direction": {"type": ["string", "null"]},
        "station_phase": {"type": ["string", "null"]},
        "omitted": {"type": "boolean"},
        "restored": {"type": "boolean"},
        "event_subtype": {
            "type": ["string", "null"],
            "enum": [None, "solstice", "equinox", "rising", "setting", "acronychal"],
        },
        "atomic_evidence_quote": {"type": "string"},
        "subject_binding": {
            "type": "string",
            "enum": ["DIRECT", "COORDINATED", "ANAPHORIC"],
        },
        "antecedent_quote": {"type": ["string", "null"]},
    },
    "required": [
        "kind", "planet", "target_planet", "star", "sign",
        "measurement_cubits", "measurement_fingers", "direction",
        "station_phase", "omitted", "restored", "event_subtype",
        "atomic_evidence_quote", "subject_binding", "antecedent_quote",
    ],
}


RECALL_NORMALIZE_FORMAT = {
    "type": "json_schema",
    "name": "raw_recall_candidate_normalization",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["ADD", "SKIP"]},
            "skip_reason": {
                "type": ["string", "null"],
                "enum": [
                    None,
                    "not_testable",
                    "too_ambiguous",
                    "unsupported_repository_kind",
                    "unsupported_but_astronomical",
                    "not_asserted_by_source",
                    "other",
                ],
            },
            "explanation": {"type": "string"},
            "finals": {
                "type": "array",
                "items": RECALL_FINAL_SCHEMA,
                "maxItems": 8,
            },
        },
        "required": ["action", "skip_reason", "explanation", "finals"],
    },
}

FALSE_NEGATIVE_CASES: list[dict[str, Any]] = [
    {
        "tablet": "X102811",
        "label": "first appearance with 'in the end of' phrasing",
        "expected": [{"kind": "first_app", "planet": "mars", "sign": "Sagittarius"}],
    },
    {
        "tablet": "X104181",
        "label": "last appearance with 'in the end of' phrasing",
        "expected": [{"kind": "last_app", "planet": "saturn", "sign": "Pisces"}],
    },
    {
        "tablet": "X104630",
        "label": "coordinated planets with editorial markup",
        "expected": [
            {"kind": "in_sign", "planet": "venus", "sign": "Leo"},
            {"kind": "in_sign", "planet": "mercury", "sign": "Leo"},
        ],
    },
    {
        "tablet": "X102940",
        "label": "anaphoric 'it' in station statement",
        "expected": [{"kind": "station", "planet": "saturn", "sign": "Gemini"}],
    },
    {
        "tablet": "X103222",
        "label": "sign split by editorial brackets/spacing",
        "expected": [{"kind": "in_sign", "planet": "saturn", "sign": "Gemini"}],
    },
    {
        "tablet": "X300993",
        "label": "copula elided: 'Jupiter in Gemini'",
        "expected": [{"kind": "in_sign", "planet": "jupiter", "sign": "Gemini"}],
    },
    {
        "tablet": "X105670",
        "label": "coordinated measured subjects",
        "expected": [
            {
                "kind": "appulse", "planet": "mars", "star": "α Leonis",
                "amount": 4.0, "direction": "in front of",
            },
            {
                "kind": "appulse", "planet": "mercury", "star": "α Leonis",
                "amount": 4.0, "direction": "in front of",
            },
        ],
    },
]


RECALL_SEMANTIC_REGRESSION_CASES: list[dict[str, Any]] = [
    {
        "tablet": "X300993",
        "label": "recalled explicit unwatched status is preserved",
        "expected": [
            {
                "kind": "last_app", "planet": "saturn",
                "sign": "Capricorn", "omitted": True,
            }
        ],
    },
    {
        "tablet": "X104181",
        "label": "Moon-to-planet measured relation uses planet_appulse",
        "expected": [
            {
                "kind": "planet_appulse", "planet": "moon",
                "target_planet": "mars", "amount": 3.0 / 24.0,
                "direction": "in front of",
            }
        ],
    },
    {
        "tablet": "X102940",
        "label": "star-subject appulse is canonically inverted to planet-relative-to-star",
        "expected": [
            {
                "kind": "appulse", "planet": "mercury",
                "star": "δ Scorpii", "amount": 1.0 + 2.0 / 3.0,
                "direction": "behind",
            }
        ],
    },
]


DROP_VERIFY_PROMPT = r"""
You are the SECOND-PASS verifier for a proposed DROP of a deterministic
astronomical statement extracted from an English translation of a Babylonian
Astronomical Diary.

The first-pass reviewer proposed DROP. Your task is narrower: determine whether
there is POSITIVE textual evidence that the extracted atomic astronomical claim
should really be removed.

HARD FACTS:
1. parser_matched_source is GUARANTEED by the calling code to be an exact
   literal substring of the source translation. Never question or re-verify
   that provenance.
2. context_before + parser_matched_source + context_after are contiguous source
   text. parser_matched_source is itself primary evidence.
3. The statement may be an intentional ATOMIC decomposition of a larger clause.
   A coordinated clause such as "Saturn and Mars were in Leo" legitimately
   yields both "Saturn in Leo" and "Mars in Leo" as separate statements.
4. Month/day/year_offset and restored status may be inherited from governing
   context outside the local excerpt. Their absence from the excerpt is not a
   reason to DROP.
5. Bracketed/restored text is not the same as an omitted observation.
6. "I did not watch" or equivalent does not erase the astronomical report; it
   normally means the statement should be kept with omitted=true, not dropped.
7. Do not use published dates, candidate years, or astronomical calculations.

CONFIRM_DROP only if the supplied source positively demonstrates one of these
problems:
- negated_or_nonassertive: the matched words occur, but the source explicitly
  negates or does not actually assert the astronomical proposition.
- wrong_subject_or_target: syntax positively shows that the parser attached the
  wrong grammatical subject, star, planet, or sign.
- crossed_clause_boundary: the regex combined words belonging to incompatible
  clauses into a claim the source does not make.
- editorial_or_meta_text: the match comes from editorial/meta-discussion rather
  than a Diary astronomical assertion.
- non_astronomical_usage: the matched wording has a non-astronomical meaning in
  context.
- duplicated_or_artifactual_match: the extracted object is demonstrably an
  artifact/duplicate rather than a distinct asserted atomic claim.
- incompatible_grammar: the literal words are present but their grammar
  positively contradicts the parser's structured relation.
- other_positive_contradiction: another concrete textual fact positively shows
  the parsed assertion is wrong.

Return KEEP if the matched source directly asserts the atomic claim and there is
no such positive contradiction. In particular, NEVER confirm DROP merely because
the same wording is not repeated in context_before/context_after.

For CONFIRM_DROP, evidence_quote must be an exact literal substring copied from
the SENTENCE that contains parser_matched_source, and it must demonstrate the
positive contradiction. Name the contradicted field in contradicted_field and
copy into source_value the exact words of evidence_quote that state what the
text says instead: the planet, star, sign, direction, measurement or
phenomenon it actually names, or, for contradicted_field="assertion", the
negating words (such as "not"). Calling code rejects the DROP if the quote
comes from another sentence, if source_value is not in the quote, or if
source_value agrees with the parser's value. A sentence elsewhere in the
excerpt about another body or another day never contradicts this clause. If
no such quote exists, return KEEP with contradicted_field="none".

Year numbers and ruler names are withheld and appear as {year} and {ruler};
"record" is an opaque token.
""".strip()


DROP_VERIFY_FORMAT = {
    "type": "json_schema",
    "name": "statement_drop_verification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": ["KEEP", "CONFIRM_DROP"]},
            "reason": {
                "type": "string",
                "enum": [
                    "no_positive_reason",
                    "negated_or_nonassertive",
                    "wrong_subject_or_target",
                    "crossed_clause_boundary",
                    "editorial_or_meta_text",
                    "non_astronomical_usage",
                    "duplicated_or_artifactual_match",
                    "incompatible_grammar",
                    "other_positive_contradiction",
                ],
            },
            "contradicted_field": {
                "type": "string",
                "enum": [
                    "none", "planet", "target_planet", "star", "sign", "kind",
                    "measurement", "direction", "assertion",
                ],
            },
            "source_value": {"type": "string"},
            "explanation": {"type": "string"},
            "evidence_quote": {"type": "string"},
        },
        "required": [
            "verdict", "reason", "contradicted_field", "source_value",
            "explanation", "evidence_quote",
        ],
    },
}


MODIFY_VERIFY_PROMPT = r"""
You are the SECOND-PASS verifier for a proposed MODIFY of one deterministic
astronomical statement extracted from an English translation of a Babylonian
Astronomical Diary.

The first-pass reviewer proposed changing one or more semantic fields. Review
EACH proposed field change independently. Accept only changes supported by
positive source evidence; reject unsupported cleanup or inference.

HARD FACTS:
1. parser_matched_source is guaranteed to be exact literal source text.
2. context_before + parser_matched_source + context_after are contiguous source
   text. The matched source is itself primary evidence.
3. Existing month/day/year_offset/restored values may come from governing
   context outside the local excerpt. Absence of that context here is NOT
   evidence that those values are wrong.
4. In particular, do not accept month/day changing from a known value to null,
   or restored changing true->false, merely because the local phrase lacks the
   heading/brackets that originally established those values.
5. restored and omitted are independent. Brackets/restoration never imply
   omitted=true by themselves.
6. The unobserved mark (omitted) is set by code with the parser's rule and is
   not among the changes you review.
7. For measured relations, measurement_cubits and measurement_fingers are
   literal source-unit components; the internal normalized cubit amount is not
   being reviewed here.
8. Do not use published dates, candidate years, or astronomical calculations.

For every proposed changed field, return one decision:
- ACCEPT: positive source evidence supports the proposed new value.
- REJECT: the proposed change is unsupported, based only on absence from the
  local excerpt, or otherwise not justified.

Use reason="explicit_source_support" when an exact source phrase directly
supports the new value, and "positive_contradiction_of_original" when source
text positively shows the original value is wrong. Use the rejection reasons
when the change lacks positive support or improperly tries to undo inherited
context.

For every ACCEPT decision, evidence_quote MUST be an exact literal substring of
the SENTENCE that contains parser_matched_source, and it must contain the new
value itself (the planet, star, sign, number, direction, month or day). Calling
code rejects an ACCEPT whose quote comes from another sentence or lacks the new
value. For REJECT it may be empty. omitted and year_offset are set by code and
are never reviewed here.

Year numbers and ruler names are withheld and appear as {year} and {ruler};
"record" is an opaque token.
""".strip()


MODIFY_VERIFY_FORMAT = {
    "type": "json_schema",
    "name": "statement_modify_verification",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overall_explanation": {"type": "string"},
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": [
                                "kind", "planet", "target_planet", "star", "sign",
                                "month", "day", "year_offset", "measurement_cubits",
                                "measurement_fingers", "direction", "station_phase",
                                "omitted", "restored", "event_subtype",
                            ],
                        },
                        "decision": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
                        "reason": {
                            "type": "string",
                            "enum": [
                                "explicit_source_support",
                                "positive_contradiction_of_original",
                                "mandatory_unwatched",
                                "no_positive_support",
                                "inherited_context_protected",
                                "restoration_not_omission",
                                "measurement_normalization_protected",
                                "other",
                            ],
                        },
                        "explanation": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                    },
                    "required": [
                        "field", "decision", "reason", "explanation", "evidence_quote"
                    ],
                },
            },
        },
        "required": ["overall_explanation", "decisions"],
    },
}


# ---------------------------------------------------------------------------
# Model client: an OpenAI-compatible chat-completions endpoint
# ---------------------------------------------------------------------------

ROUTES = {
    # route: (key name, base URL)
    "litellm": ("LITELLM_API_KEY",
                os.environ.get("LITELLM_BASE_URL",
                               "https://litellm.cloud.osu.edu").rstrip("/") + "/v1"),
    "gemini": ("GOOGLE_API_KEY",
               "https://generativelanguage.googleapis.com/v1beta/openai/"),
}
# Reasoning settings tried in order; the first the endpoint accepts is kept.
REASONING_EFFORTS = ("minimal", "low", None)


def load_api_key(name: str) -> str:
    """Read a key from the environment, a .env at the repository root, or ~/.env.

    The value is never printed or written anywhere.
    """
    if os.environ.get(name):
        return os.environ[name]
    for env in (HERE / ".env", Path.home() / ".env"):
        if not env.exists():
            continue
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise RuntimeError(
        f"{name} not found. Set it in the environment, or put {name}=... in a "
        ".env at the repository root or in your home directory.")


class ModelRequestRejected(RuntimeError):
    """The endpoint refused the content of one request (HTTP 400).

    A refusal is local to one clause or one search window. Callers record it
    and keep the parser's clause, rather than stopping a corpus-scale run.
    """


class AdaptiveLimiter:
    """Bound the requests in flight, growing slowly and halving on HTTP 429.

    The limit starts at ``start`` and rises by a quarter after every two
    rounds of successful calls, up to ``ceiling``. A 429 halves it (at most
    once per cool-down) and pauses all new requests for the cool-down, so a
    rate limit does not turn into a storm of retries.
    """

    def __init__(self, start: int, ceiling: int):
        self.limit = self.start = max(1, int(start))
        self.ceiling = max(self.limit, int(ceiling))
        self.in_flight = 0
        self.peak = self.limit
        self.rate_limited = 0
        self.history: list[tuple[float, str, int]] = []
        self._ok_streak = 0
        self._cooldown_until = 0.0
        self._last_cut = -1e9
        self._t0 = time.monotonic()
        self._cond = threading.Condition()

    def acquire(self) -> None:
        with self._cond:
            while True:
                wait = self._cooldown_until - time.monotonic()
                if wait <= 0 and self.in_flight < self.limit:
                    break
                self._cond.wait(timeout=wait if wait > 0 else 1.0)
            self.in_flight += 1

    def release(self, *, ok: bool, throttled: bool = False,
                retry_after: float | None = None) -> None:
        with self._cond:
            self.in_flight -= 1
            now = time.monotonic()
            if throttled:
                self.rate_limited += 1
                self._ok_streak = 0
                pause = min(60.0, max(2.0, retry_after or 5.0))
                self._cooldown_until = max(self._cooldown_until, now + pause)
                if now - self._last_cut > pause:
                    self._last_cut = now
                    self.limit = max(1, self.limit // 2)
                    self.history.append((round(now - self._t0, 1), "429_halve", self.limit))
            elif ok:
                self._ok_streak += 1
                if self._ok_streak >= 2 * self.limit and self.limit < self.ceiling:
                    self.limit = min(self.ceiling, self.limit + max(1, self.limit // 4))
                    self.peak = max(self.peak, self.limit)
                    self._ok_streak = 0
                    self.history.append((round(now - self._t0, 1), "grow", self.limit))
            self._cond.notify_all()


class RunStats:
    """Thread-safe totals of every model call made in this process."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        self.failures: dict[str, int] = {}
        self.started = time.monotonic()

    def add_call(self, meta: dict[str, Any]) -> None:
        with self._lock:
            self.calls.append(meta)

    def add_failure(self, kind: str) -> None:
        with self._lock:
            self.failures[kind] = self.failures.get(kind, 0) + 1

    def summary(self, limiter: AdaptiveLimiter | None = None) -> dict[str, Any]:
        with self._lock:
            calls = list(self.calls)
            failures = dict(self.failures)
        def total(key):
            return int(sum(c.get(key) or 0 for c in calls))
        costs = [c.get("cost_usd") for c in calls]
        by_stage: dict[str, dict[str, Any]] = {}
        for c in calls:
            row = by_stage.setdefault(c["stage"], {"calls": 0, "prompt_tokens": 0,
                                                   "completion_tokens": 0, "cost_usd": 0.0})
            row["calls"] += 1
            row["prompt_tokens"] += c.get("prompt_tokens") or 0
            row["completion_tokens"] += c.get("completion_tokens") or 0
            row["cost_usd"] += c.get("cost_usd") or 0.0
        latencies = sorted(c["latency_s"] for c in calls if c.get("latency_s") is not None)
        out = {
            "calls": len(calls),
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "reasoning_tokens": total("reasoning_tokens"),
            "cached_tokens": total("cached_tokens"),
            "cost_usd_reported": round(sum(c for c in costs if c is not None), 6),
            "calls_without_reported_cost": sum(c is None for c in costs),
            "throttled_retries": total("throttled_retries"),
            "transport_retries": total("transport_retries"),
            "truncated_calls": sum(c.get("finish_reason") == "length" for c in calls),
            "latency_s_median": latencies[len(latencies) // 2] if latencies else None,
            "latency_s_p90": latencies[int(len(latencies) * 0.9)] if latencies else None,
            "models_reported": sorted({str(c.get("model_reported")) for c in calls}),
            "response_formats": sorted({str(c.get("response_format")) for c in calls}),
            "reasoning_efforts": sorted({str(c.get("reasoning_effort")) for c in calls}),
            "by_stage": by_stage,
            "failures": failures,
            "wall_time_s": round(time.monotonic() - self.started, 1),
        }
        if limiter is not None:
            out["concurrency"] = {
                "start": limiter.start,
                "final_limit": limiter.limit, "peak_limit": limiter.peak,
                "ceiling": limiter.ceiling, "http_429": limiter.rate_limited,
                "history": limiter.history[-40:],
            }
        return out


STATS = RunStats()
_LEDGER = threading.local()


@contextmanager
def call_ledger():
    """Collect the metadata of every call made by this thread inside the block."""
    previous = getattr(_LEDGER, "calls", None)
    _LEDGER.calls = []
    try:
        yield _LEDGER.calls
    finally:
        _LEDGER.calls = previous


def _ledger_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [c.get("cost_usd") for c in calls]
    return {
        "n_calls": len(calls),
        "prompt_tokens": int(sum(c.get("prompt_tokens") or 0 for c in calls)),
        "completion_tokens": int(sum(c.get("completion_tokens") or 0 for c in calls)),
        "cost_usd": (round(sum(c for c in costs if c is not None), 8)
                     if any(c is not None for c in costs) else None),
        "models_reported": sorted({str(c.get("model_reported")) for c in calls}),
    }


def _schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Check a decoded answer against the subset of JSON Schema used here.

    Strict structured output makes this redundant; it matters when the endpoint
    only offers JSON mode and the schema is enforced in code instead.
    """
    errors: list[str] = []
    types = schema.get("type")
    types = [types] if isinstance(types, str) else list(types or [])
    kinds = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
    }
    if types and not any(kinds[t](value) for t in types):
        return [f"{path}: expected {types}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: missing")
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{k}: not allowed" for k in value if k not in props)
        for name, sub in props.items():
            if name in value:
                errors.extend(_schema_errors(value[name], sub, f"{path}.{name}"))
    if isinstance(value, list) and "items" in schema:
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        for i, item in enumerate(value):
            errors.extend(_schema_errors(item, schema["items"], f"{path}[{i}]"))
    return errors


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


class ModelClient:
    """One endpoint and its settings, shared by every worker thread."""

    def __init__(self, route: str = DEFAULT_ROUTE, limiter: AdaptiveLimiter | None = None):
        from openai import OpenAI
        if route not in ROUTES:
            raise ValueError(f"unknown route {route!r}; choose one of {sorted(ROUTES)}")
        key_name, base_url = ROUTES[route]
        self.route = route
        self.base_url = base_url
        self.client = OpenAI(api_key=load_api_key(key_name), base_url=base_url,
                             max_retries=0, timeout=300)
        self.limiter = limiter or AdaptiveLimiter(4, 8)
        self._lock = threading.Lock()
        self.json_schema_ok = True
        self.effort_index = 0
        self.token_param = "max_completion_tokens"

    def _settings(self) -> tuple[bool, str | None, str]:
        with self._lock:
            return self.json_schema_ok, REASONING_EFFORTS[self.effort_index], self.token_param

    def _downgrade(self, message: str, used: tuple[bool, str | None, str]) -> bool:
        """Adapt to a 400 about a parameter. Returns True if a retry makes sense."""
        text = message.lower()
        json_schema, effort, token_param = used
        with self._lock:
            if "reasoning" in text and effort is not None:
                if REASONING_EFFORTS[self.effort_index] == effort:
                    self.effort_index += 1
                return True
            if json_schema and ("response_format" in text or "json_schema" in text
                                or "schema" in text):
                self.json_schema_ok = False
                return True
            if token_param == "max_completion_tokens" and "max_completion_tokens" in text:
                self.token_param = "max_tokens"
                return True
        return False

    def complete(self, *, model: str, instructions: str, user: str,
                 fmt: dict[str, Any], max_tokens: int, stage: str
                 ) -> tuple[str | None, str | None]:
        """One call. Returns (content, finish_reason) and records its metadata."""
        import openai
        throttled = transport = 0
        while True:
            used = self._settings()
            json_schema, effort, token_param = used
            system = instructions
            if json_schema:
                response_format = {"type": "json_schema", "json_schema": {
                    "name": fmt["name"], "strict": True, "schema": fmt["schema"]}}
            else:
                response_format = {"type": "json_object"}
                system = (instructions + "\n\nReturn only one JSON object that "
                          "satisfies this JSON Schema:\n"
                          + json.dumps(fmt["schema"], ensure_ascii=False))
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "response_format": response_format,
                token_param: max_tokens,
            }
            if effort is not None:
                kwargs["reasoning_effort"] = effort
            self.limiter.acquire()
            started = time.monotonic()
            ok = was_throttled = False
            retry_after = None
            try:
                raw = self.client.chat.completions.with_raw_response.create(**kwargs)
                ok = True
            except openai.RateLimitError as exc:
                was_throttled = True
                throttled += 1
                header = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    retry_after = float(header.get("retry-after"))
                except (TypeError, ValueError):
                    retry_after = None
                if throttled > 12:
                    raise
                error = exc
            except openai.BadRequestError as exc:
                error = exc
            except (openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as exc:
                transport += 1
                if transport > 6:
                    raise
                error = exc
            finally:
                self.limiter.release(ok=ok, throttled=was_throttled, retry_after=retry_after)
            if not ok:
                if isinstance(error, openai.BadRequestError):
                    if self._downgrade(str(error), used):
                        continue
                    STATS.add_failure(f"{stage}:request_rejected")
                    raise ModelRequestRejected(str(error)[:500]) from error
                delay = (retry_after or min(60.0, 2.0 ** min(throttled + transport, 5)))
                time.sleep(delay * (0.5 + random.random()))
                continue
            completion = raw.parse()
            choice = completion.choices[0] if completion.choices else None
            usage = completion.usage
            details = getattr(usage, "completion_tokens_details", None)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cost = raw.headers.get("x-litellm-response-cost")
            try:
                cost = float(cost) if cost not in (None, "", "None") else None
            except ValueError:
                cost = None
            meta = {
                "stage": stage,
                "model_requested": model,
                "model_reported": getattr(completion, "model", None),
                "route": self.route,
                "response_format": response_format["type"],
                "reasoning_effort": effort,
                "max_tokens": max_tokens,
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "cached_tokens": getattr(prompt_details, "cached_tokens", None),
                "cost_usd": cost,
                "latency_s": round(time.monotonic() - started, 3),
                "throttled_retries": throttled,
                "transport_retries": transport,
            }
            STATS.add_call(meta)
            ledger = getattr(_LEDGER, "calls", None)
            if ledger is not None:
                ledger.append(meta)
            content = getattr(getattr(choice, "message", None), "content", None)
            return content, meta["finish_reason"]


def get_client(route: str = DEFAULT_ROUTE, limiter: AdaptiveLimiter | None = None):
    return ModelClient(route, limiter)


def _structured_request(
    client: ModelClient,
    model: str,
    *,
    instructions: str,
    payload: dict[str, Any],
    fmt: dict[str, Any],
    budgets: tuple[int, ...],
    stage: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ask for one schema-conforming JSON object, retrying a truncated answer.

    Returns (answer, None), or (None, error) when every output budget gave a
    truncated, empty, malformed or non-conforming answer. Raises
    ModelRequestRejected when the endpoint refuses the request itself.
    """
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error = None
    for budget in budgets:
        content, finish = client.complete(model=model, instructions=instructions,
                                          user=user, fmt=fmt, max_tokens=budget,
                                          stage=stage)
        if finish == "length":
            last_error = f"incomplete_model_output at budget {budget}"
            continue
        if not content:
            last_error = f"empty_model_output (finish_reason={finish!r}) at budget {budget}"
            continue
        try:
            answer = json.loads(_strip_json_fence(content))
        except json.JSONDecodeError as exc:
            last_error = (f"malformed_model_json: {exc.msg} at line {exc.lineno} "
                          f"column {exc.colno}; max_tokens={budget}")
            continue
        problems = _schema_errors(answer, fmt["schema"])
        if problems:
            last_error = "schema_violation: " + "; ".join(problems[:5])
            continue
        return answer, None
    STATS.add_failure(f"{stage}:output_failure")
    return None, last_error


# ---------------------------------------------------------------------------
# Answer blinding: an opaque record token and masked chronology
# ---------------------------------------------------------------------------

# Year formulas, after tools/eras.py and scripts/source/build_adart_iv_corpus.py:
# "Year 38", "year 1 ] 37" (a number broken by restoration), "which is year
# 234", "37th year". Only the number is withheld; the word "Year" stays, since
# a year heading inside a tablet is relative context the parser also reads.
_YEAR_NUMBER = re.compile(
    r"\byear(?:\s*[\[\]])*\s*(\d{1,3}(?:(?:\s*[\[\]])+\s*\d{1,3})?)\b"
    r"|\b(\d{1,3}(?:st|nd|rd|th))\s+year\b"
    r"|\bS\.?\s?E\.?\s*(\d{1,3})\b",
    re.IGNORECASE)
# Rulers and royal women named in ADART I-III, and the ADART IV list
# (Seleucus, Antiochus, Demetrius, Artaxerxes, Arsaces, Alexander, Darius,
# Xerxes). The edition's spellings are kept.
RULER_NAMES = (
    "Šamaš-šuma-ukin", "Kandalanu", "Nabopolassar", "Nebuchadnezzar",
    "Amēl-Marduk", "Neriglissar", "Nabonidus", "Cyrus", "Cambyses", "Bardiya",
    "Darius", "Xerxes", "Artaxerxes", "Umakuš", "Artašata", "Arses",
    "Alexander", "Philip", "Antigonus", "Antigonos", "Seleucus", "Antiochus",
    "Antiochos", "Demetrius", "Arsaces", "Mithradates", "Mitradata", "Mitradatā",
    "Miṭratu", "Mitratu", "Phraates", "Orodes", "Artabanus", "Artabana",
    "Hyspaosines", "Aspasinē", "Aspasine", "Gotarzes", "Gutarza", "Tigranes",
    "Tigranu", "Timarchus", "Laodice", "Laodike", "Stratonice", "Stratonike",
)
_EDIT_GAP = r"(?:\s*[\[\]]\s*)?"


def _ruler_pattern() -> re.Pattern:
    """Ruler names, also when a restoration bracket splits or ends them.

    "Se [ leucus" is matched whole. A name the break cuts off, such as
    "Seleu [ ...", is matched by a prefix of at least five letters that is
    followed by a bracket ("Tigr [" may be the river Tigris).
    """
    whole = [_EDIT_GAP.join(re.escape(ch) for ch in name)
             for name in sorted(RULER_NAMES, key=len, reverse=True)]
    prefixes = sorted({name[:n] for name in RULER_NAMES if len(name) >= 6
                       for n in range(5, len(name))}, key=len, reverse=True)
    cut = [re.escape(p) + r"(?=\s*[\[\]])" for p in prefixes]
    return re.compile(r"(?<![^\W\d_])(?:" + "|".join(whole + cut) + r")(?![^\W\d_])")


_RULER = _ruler_pattern()


class BlindText:
    """A translation with chronology withheld, and maps between the offsets.

    ``masked`` replaces each year number with "{year}" and each ruler name
    with "{ruler}". A bracket inside a withheld span is kept after the token,
    so restoration marks survive. Text outside the withheld spans is
    unchanged, character for character. A passage the model cites in the
    masked text maps back to the original offsets; a passage that ends inside
    a token covers the whole withheld span.
    """

    def __init__(self, original: str, masked: str,
                 spans: tuple[tuple[int, int, int, int], ...]):
        self.original = original
        self.masked = masked
        # (original start, original end, masked start, masked end) per token
        self.spans = spans

    def to_original_start(self, pos: int) -> int:
        shift = 0
        for o0, o1, m0, m1 in self.spans:
            if pos < m0:
                break
            if pos < m1:
                return o0
            shift = o1 - m1
        return pos + shift

    def to_original_end(self, pos: int) -> int:
        shift = 0
        for o0, o1, m0, m1 in self.spans:
            if pos <= m0:
                break
            if pos <= m1:
                return o1
            shift = o1 - m1
        return pos + shift

    def to_masked_start(self, pos: int) -> int:
        shift = 0
        for o0, o1, m0, m1 in self.spans:
            if pos < o0:
                break
            if pos < o1:
                return m0
            shift = m1 - o1
        return pos + shift

    def to_masked_end(self, pos: int) -> int:
        shift = 0
        for o0, o1, m0, m1 in self.spans:
            if pos <= o0:
                break
            if pos <= o1:
                return m1
            shift = m1 - o1
        return pos + shift

    def masked_slice(self, start: int, end: int) -> str:
        """The masked text of an original span."""
        return self.masked[self.to_masked_start(start):self.to_masked_end(end)]


def blind(text: str) -> BlindText:
    text = text or ""
    found: list[tuple[int, int, str]] = []
    for m in _YEAR_NUMBER.finditer(text):
        group = next(g for g in (1, 2, 3) if m.group(g) is not None)
        found.append((m.start(group), m.end(group), "{year}"))
    for m in _RULER.finditer(text):
        found.append((m.start(), m.end(), "{ruler}"))
    found.sort()
    pieces: list[str] = []
    spans: list[tuple[int, int, int, int]] = []
    cursor = 0
    length = 0
    for start, end, token in found:
        if start < cursor:
            continue
        pieces.append(text[cursor:start])
        length += start - cursor
        brackets = "".join(ch for ch in text[start:end] if ch in "[]")
        replacement = token + (" " + " ".join(brackets) if brackets else "")
        spans.append((start, end, length, length + len(replacement)))
        pieces.append(replacement)
        length += len(replacement)
        cursor = end
    pieces.append(text[cursor:])
    return BlindText(text, "".join(pieces), tuple(spans))


def blind_string(text: str | None) -> str:
    return blind(text or "").masked


def record_token(tablet: str) -> str:
    """An opaque per-record label; it does not encode the ADsD identifier.

    The salt keeps the old name step1, because the token is part of what the
    model was shown, and so of every committed response.
    """
    return "R" + hashlib.sha256(f"step1-blind-v1:{tablet}".encode()).hexdigest()[:8]


_SOURCE_NUMBER = r"((?:\d+\s+)?(?:\d+(?:/|⁄)\d+)|\d+)"
_SOURCE_CUBIT = re.compile(_SOURCE_NUMBER + r"\s*cubits?\b", re.IGNORECASE)
_SOURCE_FINGER = re.compile(_SOURCE_NUMBER + r"\s*fingers?\b", re.IGNORECASE)


def _source_number(value: str) -> float | None:
    """Parse a literal Babylonian distance component using the repo parser."""
    return S.parse_amount(value.replace("⁄", "/"))


def source_measurement_components(source: str) -> dict[str, float | None]:
    """Recover literal cubit/finger components from the exact matched source.

    Statement.amount is normalized to total cubits by tools.statements.  This
    helper exposes the source-unit quantities separately so the LLM never has
    to guess whether 0.4166667 means ten fingers or a parser error.
    """
    clean = re.sub(r"[\[\]]", " ", source or "")
    clean = " ".join(clean.split())
    cubit_match = _SOURCE_CUBIT.search(clean)
    finger_match = _SOURCE_FINGER.search(clean)
    return {
        "measurement_cubits": (
            _source_number(cubit_match.group(1)) if cubit_match else None
        ),
        "measurement_fingers": (
            _source_number(finger_match.group(1)) if finger_match else None
        ),
    }


def jsonable_statement(st: S.Statement) -> dict[str, Any]:
    d = asdict(st)
    # amount is explicitly named as normalized cubits because the original
    # Statement.unit retains a source/provenance label and can otherwise mislead
    # a language model (e.g. 10 fingers -> amount 10/24 cubit).
    return {
        "kind": d["kind"],
        "planet": d["planet"],
        "target_planet": d["target_planet"],
        "star": d["star"],
        "sign": d["sign"],
        "month": d["month"],
        "day": d["day"],
        "year_offset": d["year_offset"],
        "normalized_amount_cubits": d["amount"],
        "source_unit_label": d["unit"],
        "direction": d["direction"],
        "station_phase": d["station_phase"],
        "omitted": bool(d["omitted"]),
        "restored": bool(d["restored"]),
        "parser_rule": d["parser_rule"],
        "offset_degrees": d["offset"],
        "latitudinal": bool(d["latitudinal"]),
        "normalized_text": d["text"],
    }


def infer_event_subtype(st: S.Statement) -> str | None:
    rule = st.parser_rule.removesuffix("_bracket")
    if st.kind == "season" and rule.startswith("season_"):
        return rule.removeprefix("season_")
    if st.kind == "sirius" and rule.startswith("sirius_"):
        return rule.removeprefix("sirius_")
    return None


def parser_rule_note(st: S.Statement) -> str:
    """Explain intentional parser normalization that an LLM must not undo."""
    rule = st.parser_rule.removesuffix("_bracket")
    if rule == "coordinated_sign":
        return (
            "This rule intentionally decomposes a two-planet coordinated sign "
            "clause into one atomic in_sign Statement per planet. The source may "
            "therefore name TWO planets while this Statement correctly contains "
            "only one of them. Do not DROP for that reason."
        )
    if rule == "triple_coordinated_sign":
        return (
            "This rule intentionally decomposes a three-planet coordinated sign "
            "clause into one atomic in_sign Statement per planet. The source may "
            "therefore name THREE planets while this Statement correctly contains "
            "only one of them. Do not DROP for that reason."
        )
    if rule in {"monthly_path", "monthly_path_after"}:
        return (
            "This rule intentionally decomposes a monthly path such as 'Mars was "
            "in Aries, at the end of the month, in Taurus' into one atomic in_sign "
            "Statement for each sign. The source therefore names TWO signs while "
            "this Statement correctly contains only one of them. Do not DROP for "
            "that reason."
        )
    if rule.startswith("sirius_"):
        return (
            "This rule directly recognizes a Sirius first appearance, last "
            "appearance, or acronychal rising. kind='sirius' with planet='sirius' "
            "identifies the observed subject; its seasonal evidence family is "
            "assigned separately by the scoring layer. The literal "
            "Sirius phrase in parser_matched_source is sufficient evidence for the "
            "event; month/day may be inherited from context."
        )
    return "No special parser normalization note for this rule."


def review_payload(source: BlindText, token: str, index: int,
                   st: S.Statement) -> dict[str, Any]:
    """The model-facing review request for one parser clause.

    Every text field comes from the masked translation, and ``record`` is an
    opaque token, so neither the ADsD identifier nor a year formula or ruler
    name reaches the model.
    """
    pad = 500
    masked = source.masked
    m_start = source.to_masked_start(st.source_start)
    m_end = source.to_masked_end(st.source_end)
    start = max(0, m_start - pad)
    end = min(len(masked), m_end + pad)
    matched = masked[m_start:m_end]
    original = jsonable_statement(st)
    original["normalized_text"] = blind_string(original["normalized_text"])
    original["event_subtype"] = infer_event_subtype(st)
    return {
        "record": token,
        "statement_index": index,
        "context_before": masked[start:m_start],
        "parser_matched_source": matched,
        "context_after": masked[m_end:end],
        "source_measurement": source_measurement_components(matched),
        "deterministic_statement": original,
        "parser_rule_note": parser_rule_note(st),
        "important_context_note": (
            "parser_matched_source is itself exact source text. Existing month/day/"
            "year_offset/restored values may derive from governing context outside "
            "this local excerpt; absence of such a heading here is not evidence "
            "that those values are wrong."
        ),
    }


def cache_key(tablet: str, text: str, index: int, st: S.Statement, model: str) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "tablet": tablet,
        "tablet_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "statement_index": index,
        "statement": jsonable_statement(st),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A run stopped mid-write leaves one partial last line. Its work is
            # simply redone; any other damage is reported.
            print(f"WARNING: skipping unreadable cache line {number} in {path}")
            continue
        out[row["cache_key"]] = row
    return out


def normalize_body(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


@lru_cache(maxsize=1)
def scoring_fingerprint() -> str:
    """Invalidate numerical caches when any candidate-verdict dependency changes."""
    paths = ["tools/statements.py", "tools/sky.py", "tools/cal.py", "tools/eclipses.py",
             "scripts/agent/model_review.py",
             "data/physical/ephemeris.parquet", "data/physical/calendar.parquet",
             "data/physical/eclipses.parquet", "data/physical/normal_stars.parquet"]
    hashes = {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def statement_fingerprint(st: S.Statement) -> str:
    """Bind a numerical cache row to all semantic and joint-source fields."""
    raw = json.dumps(asdict(st), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def derive_parser_rule(original: S.Statement, final: dict[str, Any]) -> str:
    """Keep provenance when possible; use a safe canonical rule after a kind change."""
    kind = final["kind"]
    subtype = final.get("event_subtype")

    if kind == "season":
        if subtype not in {"solstice", "equinox"}:
            raise ValueError("season correction requires event_subtype=solstice/equinox")
        return f"season_{subtype}"
    if kind == "sirius":
        if subtype not in {"rising", "setting", "acronychal"}:
            raise ValueError("sirius correction requires event_subtype=rising/setting/acronychal")
        return f"sirius_{subtype}"
    if kind == original.kind:
        return original.parser_rule

    canonical = {
        "eclipse": "typed_eclipse",
        "appulse": "measured_normal_star",
        "planet_appulse": "measured_moon_planet",
        "station_appulse": "measured_station_normal_star",
        "planet_planet_appulse": "measured_planet_planet",
        "in_sign": "in_sign",
        "station": "station",
        "entry": "entry",
        "last_app": "last_app",
        "first_app": "first_app",
        "acronychal": "acronychal_rising",
    }
    if kind not in canonical:
        raise ValueError(f"no canonical parser rule for corrected kind {kind!r}")
    return canonical[kind]


def derive_measurement(
    final: dict[str, Any],
    original: S.Statement,
) -> tuple[float | None, str | None, float | None, bool]:
    """Normalize LLM-reviewed literal source units to Statement internals.

    The LLM edits literal cubit/finger components.  Only deterministic code
    converts them to Statement.amount (total cubits) and offset degrees.
    """
    if final["kind"] not in MEASURED_KINDS:
        return None, None, None, False

    cubits = final.get("measurement_cubits")
    fingers = final.get("measurement_fingers")
    direction = final.get("direction")
    if cubits is None and fingers is None:
        raise ValueError("measured relation requires cubit and/or finger source amount")
    if direction not in DIRECTIONS or direction is None:
        raise ValueError(f"measured relation has unsupported direction {direction!r}")

    cubit_value = 0.0 if cubits is None else float(cubits)
    finger_value = 0.0 if fingers is None else float(fingers)
    if cubit_value < 0 or finger_value < 0:
        raise ValueError("measurement components cannot be negative")
    total_cubits = cubit_value + finger_value / 24.0
    if total_cubits <= 0:
        raise ValueError("measured relation requires a positive distance")

    if cubits is not None and fingers is not None:
        unit = "cubit+finger"
    elif cubits is not None:
        unit = "cubit" if cubit_value == 1 else "cubits"
    else:
        unit = "finger" if finger_value == 1 else "fingers"

    degrees = total_cubits * 2.0
    latitudinal = direction in {"above", "below"}
    if latitudinal:
        offset = 0.0
    elif direction == "in front of":
        offset = -degrees
    elif direction == "behind":
        offset = degrees
    else:
        raise ValueError(f"unsupported longitudinal direction {direction!r}")

    return total_cubits, unit, offset, latitudinal


SIGN_KINDS = {"in_sign", "entry", "last_app", "first_app", "station"}


def validate_final(final: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one clause's fields before it may be used.

    Stars must be in the Normal-Star catalogue and signs must be zodiac signs,
    both in the repository's canonical spelling, as in the search path. A sign
    on a kind that takes none is cleared, so later code never meets one.
    """
    final = dict(final)
    final["planet"] = normalize_body(final.get("planet")) or ""
    final["target_planet"] = normalize_body(final.get("target_planet"))
    if final["kind"] not in KINDS:
        raise ValueError(f"unknown kind {final['kind']!r}")
    if final["planet"] not in BODY_NAMES:
        raise ValueError(f"unknown planet/body {final['planet']!r}")
    if final["target_planet"] not in BODY_NAMES - {""} and final["target_planet"] is not None:
        raise ValueError(f"unknown target planet {final['target_planet']!r}")
    if final.get("direction") not in DIRECTIONS:
        raise ValueError(f"unknown direction {final.get('direction')!r}")
    if final.get("station_phase") not in STATION_PHASES:
        raise ValueError(f"unknown station phase {final.get('station_phase')!r}")
    if final.get("day") is not None and not (1 <= int(final["day"]) <= 30):
        raise ValueError(f"invalid Babylonian day {final['day']!r}")
    final["star"] = _canonical_star_label(final.get("star"))
    kind = final["kind"]
    final["sign"] = _canonical_sign_name(final.get("sign")) if kind in SIGN_KINDS else ""

    if kind in MEASURED_KINDS:
        if (final.get("measurement_cubits") is None
                and final.get("measurement_fingers") is None):
            raise ValueError(f"{kind} requires a literal source measurement")
        if final.get("direction") is None:
            raise ValueError(f"{kind} requires a direction")
    else:
        # Non-measured claims should not carry an invented source measurement.
        final["measurement_cubits"] = None
        final["measurement_fingers"] = None
        final["direction"] = None

    if kind in SIGN_KINDS and not final.get("sign"):
        raise ValueError(f"{kind} requires a zodiac sign")
    if kind in {"appulse", "station_appulse"} and not final.get("star"):
        raise ValueError(f"{kind} requires a Normal Star")
    if kind in {"planet_appulse", "planet_planet_appulse"} and not final.get("target_planet"):
        raise ValueError(f"{kind} requires target_planet")
    if kind == "eclipse" and final["planet"] not in {"lunar", "solar"}:
        raise ValueError("eclipse requires planet='lunar' or 'solar'")
    if kind == "season":
        final["planet"] = "sun"
    if kind == "sirius":
        final["planet"] = "sirius"
    if kind == "planet_appulse":
        final["planet"] = "moon"
    return final


def parser_unwatched(text: str, kind: str, original: S.Statement) -> bool:
    """The unobserved mark, by the parser's rule only.

    A clause that keeps its kind keeps the parser's own mark. A clause whose
    kind the review changed gets the mark the parser's rule gives that kind at
    the same passage. The model's value is never used.
    """
    if kind == original.kind:
        return bool(original.omitted)
    return S.unwatched(text, kind, original.source_start, original.source_end)


def apply_review(original: S.Statement, review: dict[str, Any],
                 text: str) -> S.Statement | None:
    action = review["action"]
    if action == "DROP":
        return None
    if action == "KEEP":
        return original
    if action != "MODIFY":
        raise ValueError(f"invalid action {action!r}")

    final = validate_final(review["final"])
    amount, unit, offset, latitudinal = derive_measurement(final, original)
    parser_rule = derive_parser_rule(original, final)

    # Keep source offsets and exact parser-matched text as provenance. The
    # semantic fields may change, except the unobserved mark and the relative
    # year, which only the parser's rules set.
    return replace(
        original,
        kind=final["kind"],
        planet=final["planet"],
        target_planet=final["target_planet"],
        star=final["star"],
        sign=final["sign"],
        month=final["month"],
        day=final["day"],
        year_offset=original.year_offset,
        amount=amount,
        unit=unit,
        direction=final["direction"],
        station_phase=final["station_phase"],
        omitted=parser_unwatched(text, final["kind"], original),
        restored=bool(final["restored"]),
        offset=offset,
        latitudinal=latitudinal,
        parser_rule=parser_rule,
    )


def _normalize_grounding_text(value: str) -> str:
    """Normalize harmless formatting differences for quote grounding.

    This does NOT remove words or punctuation generally.  It only normalizes
    Unicode compatibility forms, common quote/dash variants, and runs of
    whitespace so that line wrapping or typographic punctuation does not cause
    a false validation failure.
    """
    value = unicodedata.normalize("NFKC", value or "")
    table = str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2010": "-", "\u2011": "-", "\u2012": "-",
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u00a0": " ",
    })
    value = value.translate(table)
    return " ".join(value.split())


def _normalized_grounding_with_map(value: str) -> tuple[str, list[int]]:
    """Normalize source text while retaining an index back to original chars.

    This mirrors _normalize_grounding_text closely enough for quote alignment:
    Unicode compatibility forms, common quote/dash variants, NBSP, and collapsed
    whitespace.  It is used only to recover the exact original substring after a
    harmless formatting mismatch; it is not a fuzzy semantic matcher.
    """
    table = str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2010": "-", "\u2011": "-", "\u2012": "-",
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u00a0": " ",
    })

    chars: list[str] = []
    origin: list[int] = []
    pending_space_index: int | None = None

    for original_index, original_char in enumerate(value or ""):
        expanded = unicodedata.normalize("NFKC", original_char).translate(table)
        for ch in expanded:
            if ch.isspace():
                if chars and pending_space_index is None:
                    pending_space_index = original_index
                continue
            if pending_space_index is not None:
                chars.append(" ")
                origin.append(pending_space_index)
                pending_space_index = None
            chars.append(ch)
            origin.append(original_index)

    return "".join(chars), origin


def _safe_grounding_quote_variants(quote: str) -> list[str]:
    """Return only harmless formatting variants of a model-provided quote.

    JSON strings occasionally contain literal ``\\n``/``\\t`` characters rather
    than real line breaks/tabs.  Treating those escape spellings as whitespace is
    safe because it changes formatting only; it never inserts/removes semantic
    words or punctuation.
    """
    raw = str(quote or "")
    variants = [raw]
    if any(token in raw for token in (r"\r\n", r"\n", r"\r", r"\t")):
        deescaped = (
            raw.replace(r"\r\n", "\n")
               .replace(r"\n", "\n")
               .replace(r"\r", "\n")
               .replace(r"\t", "\t")
        )
        if deescaped not in variants:
            variants.append(deescaped)
    return variants


def _quote_spans(quote: str, source: str) -> list[tuple[int, int]]:
    """Every (start, end) at which a quote occurs in source, exact spans first.

    Only harmless formatting differences are allowed: Unicode/whitespace
    normalization and literal JSON whitespace escapes (``\\n``/``\\t``).
    """
    if not quote or not str(quote).strip():
        return []
    spans: list[tuple[int, int]] = []
    for variant in _safe_grounding_quote_variants(str(quote)):
        pos = source.find(variant)
        while pos >= 0:
            spans.append((pos, pos + len(variant)))
            pos = source.find(variant, pos + 1)
        if spans:
            return spans
    snorm, origin = _normalized_grounding_with_map(source)
    for variant in _safe_grounding_quote_variants(str(quote)):
        qnorm = _normalize_grounding_text(variant)
        if not qnorm or not origin:
            continue
        npos = snorm.find(qnorm)
        while npos >= 0:
            nend = npos + len(qnorm) - 1
            if nend < len(origin):
                spans.append((origin[npos], origin[nend] + 1))
            npos = snorm.find(qnorm, npos + 1)
        if spans:
            return spans
    return []


def _ground_quote_to_exact_source(
    quote: str, source: str, *, before: int | None = None,
) -> tuple[str, int, int] | None:
    """Resolve a grounded quote to the exact source substring and offsets.

    The first occurrence is returned, unless ``before`` is given: then the
    last occurrence that ends at or before that offset is returned, so an
    antecedent binds to the mention nearest the clause that refers back to
    it. If none ends before it, the first occurrence is returned and the
    caller's ordering check rejects it.
    """
    spans = _quote_spans(quote, source)
    if not spans:
        return None
    start, end = spans[0]
    if before is not None:
        earlier = [span for span in spans if span[1] <= before]
        if earlier:
            start, end = max(earlier, key=lambda span: span[1])
    return source[start:end], start, end


# A sentence ends at a semicolon or at a full stop that is not part of a run
# of dots (a lacuna). Line breaks wrap the edition's lines, not sentences.
_SENTENCE_END = re.compile(r"(?<!\.)\.(?!\.)|;")


def _sentence_span(text: str, start: int, end: int, limit: int = 240) -> tuple[int, int]:
    """Offsets of the sentence containing text[start:end], at most ``limit``
    characters beyond the clause on either side."""
    left = max(0, start - limit)
    for match in _SENTENCE_END.finditer(text, left, start):
        left = match.end()
    right = min(len(text), end + limit)
    match = _SENTENCE_END.search(text, end, right)
    if match:
        right = match.start()
    return left, right


def _clause_window(payload: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    """The supplied source excerpt and the clause's own sentence within it."""
    before = payload["context_before"]
    matched = payload["parser_matched_source"]
    window = before + matched + payload["context_after"]
    return window, _sentence_span(window, len(before), len(before) + len(matched))


def _quote_in_clause_sentence(payload: dict[str, Any], quote: str) -> bool:
    """Whether some occurrence of the quote overlaps the clause's sentence."""
    window, (left, right) = _clause_window(payload)
    return any(start < right and left < end for start, end in _quote_spans(quote, window))


def _repair_recall_quotes(
    client,
    model: str,
    payload: dict[str, Any],
    claims: list[dict[str, Any]],
    bad_indexes: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Dedicated exact-quote repair for ungrounded discovery candidates.

    Returns (kept_claims, rejected_claim_records). A stubborn bad quote rejects
    only that candidate; it never aborts the rest of the source chunk.
    """
    source = payload["context_before"] + payload["core_text"] + payload["context_after"]
    repair_payload = {
        "record": payload.get("record"),
        "context_before": payload["context_before"],
        "core_text": payload["core_text"],
        "context_after": payload["context_after"],
        "bad_claims": [
            {
                "claim_index": i,
                "claim": claims[i].get("claim", ""),
                "previous_evidence_quote": claims[i].get("evidence_quote", ""),
                "previous_explanation": claims[i].get("explanation", ""),
            }
            for i in bad_indexes
        ],
    }

    try:
        result, error = _structured_request(
            client, model, instructions=RECALL_QUOTE_REPAIR_PROMPT,
            payload=repair_payload, fmt=RECALL_QUOTE_REPAIR_FORMAT,
            budgets=(5000,), stage="recall_quote_repair")
    except ModelRequestRejected as exc:
        result, error = None, f"quote_repair_request_rejected: {exc}"
    if result is None:
        # Do not destroy the entire chunk if the repair call itself fails.
        rejected = [
            {
                "claim_index": i,
                "claim": claims[i].get("claim", ""),
                "evidence_quote": claims[i].get("evidence_quote", ""),
                "reason": f"quote_repair_failed: {error}",
            }
            for i in bad_indexes
        ]
        return [c for i, c in enumerate(claims) if i not in set(bad_indexes)], rejected

    repairs = {int(r["claim_index"]): r for r in result.get("repairs", [])}
    bad_set = set(bad_indexes)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for i, claim in enumerate(claims):
        if i not in bad_set:
            kept.append(claim)
            continue

        repair = repairs.get(i)
        if not repair or repair.get("action") != "USE_QUOTE":
            rejected.append({
                "claim_index": i,
                "claim": claim.get("claim", ""),
                "evidence_quote": claim.get("evidence_quote", ""),
                "reason": "quote_repair_dropped",
                "repair_explanation": "" if not repair else repair.get("explanation", ""),
            })
            continue

        grounded = _ground_quote_to_exact_source(repair.get("evidence_quote", ""), source)
        if grounded is None:
            rejected.append({
                "claim_index": i,
                "claim": claim.get("claim", ""),
                "evidence_quote": repair.get("evidence_quote", ""),
                "reason": "quote_repair_still_ungrounded",
                "repair_explanation": repair.get("explanation", ""),
            })
            continue

        exact_quote, _, _ = grounded
        fixed = dict(claim)
        fixed["evidence_quote"] = exact_quote
        fixed["quote_repair_explanation"] = repair.get("explanation", "")
        kept.append(fixed)

    return kept, rejected


def _one_review_request(
    client,
    model: str,
    payload: dict[str, Any],
    *,
    instructions: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """One structured review. A failed or refused request keeps the clause."""
    try:
        result, error = _structured_request(
            client, model, instructions=instructions, payload=payload,
            fmt=REVIEW_FORMAT, budgets=(4000, 8000), stage="review")
    except ModelRequestRejected as exc:
        result, error = None, f"request_rejected: {exc}"
    if result is not None:
        return result
    # Safety-first fallback for a post-parser review: if the model cannot return
    # valid structured output, preserve the deterministic statement unchanged.
    return {
        "action": "KEEP",
        "explanation": (
            "Conservative KEEP because the model failed to return complete "
            f"valid structured JSON after retries ({error})."
        ),
        "evidence_quote": "",
        "final": _original_review_final(payload),
        "_model_output_failure": error,
    }


_REVIEW_FIELDS = (
    "kind", "planet", "target_planet", "star", "sign", "month", "day",
    "year_offset", "measurement_cubits", "measurement_fingers", "direction",
    "station_phase", "omitted", "restored", "event_subtype",
)


def _original_review_final(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert the deterministic payload into the same field space as review.final."""
    st = payload["deterministic_statement"]
    measurement = payload.get("source_measurement") or {}
    return {
        "kind": st["kind"],
        "planet": st["planet"],
        "target_planet": st.get("target_planet"),
        "star": st.get("star"),
        "sign": st.get("sign", ""),
        "month": st.get("month"),
        "day": st.get("day"),
        "year_offset": int(st.get("year_offset", 0)),
        "measurement_cubits": measurement.get("measurement_cubits"),
        "measurement_fingers": measurement.get("measurement_fingers"),
        "direction": st.get("direction"),
        "station_phase": st.get("station_phase"),
        "omitted": bool(st.get("omitted", False)),
        "restored": bool(st.get("restored", False)),
        "event_subtype": st.get("event_subtype"),
    }


def _proposed_changes(payload: dict[str, Any], review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    original = _original_review_final(payload)
    final = review["final"]
    return {
        field: {"original": original[field], "proposed": final[field]}
        for field in _REVIEW_FIELDS
        if original[field] != final[field]
    }


# Fields only deterministic code sets. A model proposal to change one is
# recorded and ignored: the unobserved mark follows the parser's rule
# (statements.unwatched), and a relative year cannot be shown by a quoted value.
CODE_FIELDS = ("omitted", "year_offset")


def _hold_code_fields(payload: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Reset omitted/year_offset to the parser's values, logging any proposal."""
    final = review.get("final")
    if not isinstance(final, dict):
        return review
    original = _original_review_final(payload)
    held = {
        name: {"proposed": final.get(name), "kept": original[name]}
        for name in CODE_FIELDS if final.get(name) != original[name]
    }
    if not held:
        return review
    fixed = dict(review)
    fixed["final"] = dict(final, **{name: original[name] for name in CODE_FIELDS})
    fixed["_code_field_overrides"] = held
    return fixed


def _canonical_proposal(final: dict[str, Any]) -> dict[str, Any]:
    """Spell a proposed star or sign canonically before comparing it.

    "alpha Leonis" and "α Leonis" are one value, and a change between them is
    no change. A value that cannot be canonicalized stays as given, and the
    later validation rejects it.
    """
    out = dict(final)
    try:
        out["star"] = _canonical_star_label(out.get("star"))
    except ValueError:
        pass
    if out.get("kind") not in SIGN_KINDS:
        # validate_final clears it anyway; a sign on such a kind is no change.
        out["sign"] = ""
    else:
        try:
            out["sign"] = _canonical_sign_name(out.get("sign"))
        except ValueError:
            pass
    return out


def _field_value_errors(field: str, value: Any, final: dict[str, Any], quote: str) -> list[str]:
    """Reasons the quote does not contain this field's new value.

    Semantic fields are checked with the same grounding rules as the search
    (``_recall_field_grounding_errors``), one field at a time. Month and day
    must occur as a heading in the quote, and restored=true needs a bracket.
    """
    if value is None:
        return []
    if field == "month":
        months = {S.month_from_match(m) for m in S.month_matches(quote)}
        return [] if S.canonical_month(str(value)) in months else [
            f"month {value!r} is not a month heading in the quote"]
    if field == "day":
        days = {int(m.group(1)) for m in S.day_matches(quote, 0, len(quote))}
        return [] if int(value) in days else [f"day {value!r} is not a day heading in the quote"]
    if field == "restored":
        return [] if (value is False or "[" in quote or "]" in quote) else [
            "restored=true needs a bracket in the quote"]
    if field in CODE_FIELDS:
        return [f"{field} is set by code"]
    probe: dict[str, Any] = {
        "kind": None, "planet": "", "target_planet": None, "star": None,
        "sign": "", "measurement_cubits": None, "measurement_fingers": None,
        "direction": None, "station_phase": None, "event_subtype": None,
    }
    if field in {"kind", "event_subtype", "planet"}:
        probe["kind"] = final.get("kind")
        probe["event_subtype"] = final.get("event_subtype")
    probe[field] = value
    return _recall_field_grounding_errors(probe, {"evidence_quote": quote})


def _one_modify_verification_request(
    client,
    model: str,
    payload: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    changes = _proposed_changes(payload, review)
    verify_payload = {
        "record": payload.get("record"),
        "statement_index": payload.get("statement_index"),
        "context_before": payload["context_before"],
        "parser_matched_source": payload["parser_matched_source"],
        "context_after": payload["context_after"],
        "parser_rule_note": payload.get("parser_rule_note", ""),
        "original_fields": _original_review_final(payload),
        "proposed_final": review["final"],
        "proposed_changes": changes,
        "first_pass_modify": {
            "explanation": review.get("explanation", ""),
            "evidence_quote": review.get("evidence_quote", ""),
        },
    }
    try:
        result, error = _structured_request(
            client, model, instructions=MODIFY_VERIFY_PROMPT, payload=verify_payload,
            fmt=MODIFY_VERIFY_FORMAT, budgets=(3000, 6000), stage="modify_verify")
    except ModelRequestRejected as exc:
        result, error = None, f"request_rejected: {exc}"
    if result is not None:
        return result
    # Empty decisions make verify_modify reject every proposed semantic change.
    return {
        "overall_explanation": (
            "Verifier returned malformed/incomplete structured output; all "
            f"proposed changes are conservatively rejected ({error})."
        ),
        "decisions": [],
        "_model_output_failure": error,
    }


def verify_modify(
    client, model: str, payload: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    """Accept or reject each proposed field change on its own.

    A change survives only if the verifier accepts it AND code finds its quote
    in the clause's own sentence AND the quote contains the new value.
    """
    if review.get("action") != "MODIFY":
        return review

    original = _original_review_final(payload)
    changes = _proposed_changes(payload, review)

    if not changes:
        fixed = dict(review)
        fixed["action"] = "KEEP"
        fixed["final"] = original
        fixed["explanation"] = (
            "MODIFY proposed no semantic field changes; converted to KEEP. "
            + str(review.get("explanation", ""))
        )
        fixed["_modify_verification"] = {
            "overall_explanation": "No semantic differences to verify.",
            "decisions": [],
        }
        return fixed

    verification = _one_modify_verification_request(client, model, payload, review)
    by_field = {}
    for decision in verification.get("decisions", []):
        name = decision.get("field")
        if name in changes and name not in by_field:
            by_field[name] = decision

    accepted = dict(original)
    audited_decisions = []

    for name, change in changes.items():
        decision = dict(by_field.get(name) or {
            "field": name,
            "decision": "REJECT",
            "reason": "no_positive_support",
            "explanation": "Verifier did not return a decision for this proposed field change.",
            "evidence_quote": "",
        })

        # Conservative deterministic protections for inherited context.
        if name in {"month", "day"} and change["original"] is not None and change["proposed"] is None:
            decision.update({
                "decision": "REJECT",
                "reason": "inherited_context_protected",
                "explanation": (
                    "Existing month/day may be inherited from governing context outside "
                    "the excerpt; absence locally cannot justify clearing it to null."
                ),
                "evidence_quote": "",
            })
        elif name == "restored" and change["original"] is True and change["proposed"] is False:
            decision.update({
                "decision": "REJECT",
                "reason": "inherited_context_protected",
                "explanation": (
                    "restored=true may derive from relevant bracketed/governing context "
                    "outside the local match; absence of local brackets cannot prove false."
                ),
                "evidence_quote": "",
            })

        if decision.get("decision") == "ACCEPT":
            quote = decision.get("evidence_quote", "")
            problems: list[str] = []
            if not _quote_spans(quote, _clause_window(payload)[0]):
                problems.append("its evidence_quote is not in the supplied source")
            elif not _quote_in_clause_sentence(payload, quote):
                problems.append("its evidence_quote is not in the clause's own sentence")
            else:
                problems.extend(_field_value_errors(name, change["proposed"],
                                                    review["final"], quote))
            if problems:
                decision.update({
                    "decision": "REJECT",
                    "reason": "no_positive_support",
                    "explanation": "ACCEPT was rejected by code because "
                                   + "; ".join(problems) + ".",
                    "model_explanation": decision.get("explanation", ""),
                    "rejected_quote": quote,
                    "evidence_quote": "",
                })
            else:
                accepted[name] = change["proposed"]

        audited_decisions.append(decision)

    fixed = dict(review)
    fixed["_modify_verification"] = {
        "overall_explanation": verification.get("overall_explanation", ""),
        "decisions": audited_decisions,
    }
    if verification.get("_model_output_failure"):
        fixed["_modify_verification"]["_model_output_failure"] = verification["_model_output_failure"]

    if all(accepted[name] == original[name] for name in _REVIEW_FIELDS):
        fixed["action"] = "KEEP"
        fixed["final"] = original
        fixed["explanation"] = (
            "Second-pass MODIFY verification rejected all proposed field changes. "
            + str(verification.get("overall_explanation", ""))
            + " | First-pass explanation: "
            + str(review.get("explanation", ""))
        )
        fixed["evidence_quote"] = ""
        return fixed

    # Validate the mixed accepted/reverted final object before allowing it into cache.
    try:
        validated = validate_final(dict(accepted))
    except Exception as exc:
        fixed["action"] = "KEEP"
        fixed["final"] = original
        fixed["explanation"] = (
            "Second-pass MODIFY verification produced a partial combination that "
            f"failed semantic validation ({exc}); reverted to KEEP."
        )
        fixed["evidence_quote"] = ""
        fixed["_modify_verification"]["validation_error"] = str(exc)
        return fixed

    accepted_fields = [
        name for name in _REVIEW_FIELDS if validated[name] != original[name]
    ]
    if not accepted_fields:
        fixed["action"] = "KEEP"
        fixed["final"] = original
        fixed["explanation"] = ("The accepted changes vanish once canonicalized; "
                                "kept as parsed. " + str(review.get("explanation", "")))
        fixed["evidence_quote"] = ""
        return fixed
    fixed["action"] = "MODIFY"
    fixed["final"] = validated
    fixed["explanation"] = (
        "Second-pass MODIFY verification accepted only these field changes: "
        + ", ".join(accepted_fields)
        + ". "
        + str(verification.get("overall_explanation", ""))
        + " | First-pass explanation: "
        + str(review.get("explanation", ""))
    )
    grounded_quotes = [
        d.get("evidence_quote", "")
        for d in audited_decisions
        if d.get("decision") == "ACCEPT" and d.get("evidence_quote")
    ]
    fixed["evidence_quote"] = grounded_quotes[0] if grounded_quotes else review.get("evidence_quote", "")
    return fixed


def _one_drop_verification_request(
    client,
    model: str,
    payload: dict[str, Any],
    initial_review: dict[str, Any],
    *,
    repair_note: str | None = None,
) -> dict[str, Any]:
    """Second-pass verification for a proposed DROP."""
    verify_payload = {
        "record": payload.get("record"),
        "statement_index": payload.get("statement_index"),
        "context_before": payload["context_before"],
        "parser_matched_source": payload["parser_matched_source"],
        "context_after": payload["context_after"],
        "deterministic_statement": payload["deterministic_statement"],
        "source_measurement": payload.get("source_measurement"),
        "parser_rule_note": payload.get("parser_rule_note", ""),
        "first_pass_drop": {
            "explanation": initial_review.get("explanation", ""),
            "evidence_quote": initial_review.get("evidence_quote", ""),
        },
    }
    if repair_note:
        verify_payload["repair_note"] = repair_note
    try:
        result, error = _structured_request(
            client, model, instructions=DROP_VERIFY_PROMPT, payload=verify_payload,
            fmt=DROP_VERIFY_FORMAT, budgets=(2500, 5000), stage="drop_verify")
    except ModelRequestRejected as exc:
        result, error = None, f"request_rejected: {exc}"
    if result is not None:
        return result
    # A verifier failure can never be positive evidence for DROP.
    return {
        "verdict": "KEEP",
        "reason": "no_positive_reason",
        "contradicted_field": "none",
        "source_value": "",
        "explanation": (
            "DROP verifier returned malformed/incomplete structured output; "
            f"DROP is conservatively rejected ({error})."
        ),
        "evidence_quote": "",
        "_model_output_failure": error,
    }


_NEGATION = re.compile(r"\b(?:not|no|nor|never|without)\b", re.IGNORECASE)


def drop_check_errors(payload: dict[str, Any], verification: dict[str, Any]) -> list[str]:
    """Reasons code rejects a confirmed DROP; empty if the DROP may stand.

    The quote must lie in the source, overlap the clause's own sentence and
    contain ``source_value``: the value the text gives for the contradicted
    field instead of the parser's, or for "assertion" the words that negate
    the claim. The value must differ from the parser's value.
    """
    quote = str(verification.get("evidence_quote") or "")
    field = str(verification.get("contradicted_field") or "none")
    value = str(verification.get("source_value") or "").strip()
    window, _sentence = _clause_window(payload)
    if not _quote_spans(quote, window):
        return ["evidence_quote is not in the supplied source"]
    errors: list[str] = []
    if not _quote_in_clause_sentence(payload, quote):
        errors.append("evidence_quote is not in the clause's own sentence")
    if field == "none" or not value:
        return errors + ["no contradicted field and source value were named"]
    if not _normalized_contains(quote, value) and not _literal_field_present(quote, value):
        errors.append(f"source_value {value!r} is not in evidence_quote")
    original = payload["deterministic_statement"]
    measurement = payload.get("source_measurement") or {}
    # Editorial breaks inside a name ("Mer [ cury's ]") do not change it.
    joined = re.sub(r"[’'`]s?$", "", re.sub(r"\s*[\[\]]\s*", "", value)).strip()
    spaced = " ".join(re.sub(r"[\[\]]", " ", value).split())
    if field in {"planet", "target_planet"}:
        body = normalize_body(joined)
        if body not in BODY_NAMES - {""}:
            errors.append(f"source_value {value!r} is not a body name")
        elif body == normalize_body(original.get(field)):
            errors.append(f"source_value {value!r} agrees with the parser's {field}")
    elif field == "star":
        star = _catalogued_star_or_none(spaced)
        if star is None:
            errors.append(f"source_value {value!r} is not a catalogued Normal Star")
        elif star == original.get("star"):
            errors.append("source_value agrees with the parser's star")
    elif field == "sign":
        try:
            sign = _canonical_sign_name(joined)
        except ValueError:
            sign = ""
        if not sign:
            errors.append(f"source_value {value!r} is not a zodiac sign")
        elif sign == original.get("sign"):
            errors.append("source_value agrees with the parser's sign")
        elif not _sign_field_present(quote, sign):
            errors.append(f"sign {sign!r} is not in evidence_quote")
    elif field == "direction":
        if value.casefold() not in {d for d in DIRECTIONS if d}:
            errors.append(f"source_value {value!r} is not a direction")
        elif value.casefold() == str(original.get("direction") or "").casefold():
            errors.append("source_value agrees with the parser's direction")
    elif field == "measurement":
        cubits, fingers = _all_source_measurements(value)
        if not cubits and not fingers:
            errors.append(f"source_value {value!r} is not a measurement")
        elif (cubits[:1] or [None])[0] == measurement.get("measurement_cubits") and \
                (fingers[:1] or [None])[0] == measurement.get("measurement_fingers"):
            errors.append("source_value agrees with the parser's measurement")
    elif field == "assertion":
        if not _NEGATION.search(value):
            errors.append("an assertion is contradicted only by negating words")
    elif field == "kind":
        if _source_compact(value) and _source_compact(value) in _source_compact(
                payload["parser_matched_source"]):
            errors.append("source_value is part of the parser's own matched words")
    return errors


def verify_drop(
    client, model: str, payload: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    """Require a positive, local contradiction before an LLM DROP may survive."""
    verification = _one_drop_verification_request(client, model, payload, review)

    if (verification.get("verdict") == "CONFIRM_DROP"
            and verification.get("reason") != "no_positive_reason"
            and not _quote_spans(verification.get("evidence_quote", ""),
                                 _clause_window(payload)[0])):
        verification = _one_drop_verification_request(
            client,
            model,
            payload,
            review,
            repair_note=(
                "Your previous CONFIRM_DROP evidence_quote could not be found "
                "literally in the supplied source. If there is genuine positive "
                "evidence for DROP, copy an exact substring of the clause's own "
                "sentence. Otherwise return KEEP with reason=no_positive_reason."
            ),
        )

    if (
        verification.get("verdict") != "CONFIRM_DROP"
        or verification.get("reason") == "no_positive_reason"
    ):
        fixed = dict(review)
        fixed["action"] = "KEEP"
        fixed["explanation"] = (
            "Second-pass DROP verification rejected the proposed DROP. "
            f"Verifier reason={verification.get('reason')!r}: "
            + str(verification.get("explanation", ""))
            + " | First-pass explanation: "
            + str(review.get("explanation", ""))
        )
        fixed["evidence_quote"] = ""
        fixed["_drop_verification"] = verification
        return fixed

    problems = drop_check_errors(payload, verification)
    if problems:
        fixed = dict(review)
        fixed["action"] = "KEEP"
        fixed["explanation"] = (
            "Code rejected the confirmed DROP: " + "; ".join(problems) + ". "
            f"Verifier reason={verification.get('reason')!r}: "
            + str(verification.get("explanation", ""))
            + " | First-pass explanation: "
            + str(review.get("explanation", ""))
        )
        fixed["evidence_quote"] = ""
        fixed["_drop_verification"] = dict(verification, code_rejection=problems)
        return fixed

    confirmed = dict(review)
    confirmed["action"] = "DROP"
    confirmed["explanation"] = (
        "Second-pass DROP verification CONFIRMED the proposed DROP. "
        f"Reason={verification.get('reason')}; {verification.get('contradicted_field')} "
        f"is {verification.get('source_value')!r} in the text: "
        + str(verification.get("explanation", ""))
        + " | First-pass explanation: "
        + str(review.get("explanation", ""))
    )
    confirmed["evidence_quote"] = verification.get("evidence_quote", "")
    confirmed["_drop_verification"] = verification
    return confirmed


def _normalized_contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return (
        _normalize_grounding_text(needle).lower()
        in _normalize_grounding_text(haystack).lower()
    )


def _recognized_sirius_literal(matched: str, subtype: str | None) -> bool:
    """Return True only for the repository's formal Sirius scheme-date phrases.

    IMPORTANT: a generic clause such as ``Sirius set`` is *not* equivalent to
    ``Sirius' last appearance``.  The downstream ``sirius`` checker evaluates
    a civil-calendar scheme date via ``S.sirius_scheme_date``; coercing an
    ordinary stellar setting during an eclipse into that family would change
    the scientific meaning of the source.
    """
    value = _normalize_grounding_text(matched).lower()
    if "sirius" not in value:
        return False
    required = {
        "rising": "first appearance",
        "setting": "last appearance",
        "acronychal": "acronychal rising",
    }.get(subtype)
    return bool(required and required in value)


def _generic_sirius_setting_literal(value: str) -> bool:
    """Detect a literal ordinary Sirius setting, distinct from last appearance.

    This deliberately matches the standalone verbs set/sets/setting and does
    not match words such as ``sunset``.  Formal ``last appearance`` clauses are
    excluded because those *are* supported by the repository's Sirius scheme
    family.
    """
    text = _normalize_grounding_text(value).casefold()
    if not re.search(r"\bsirius\b", text):
        return False
    if re.search(r"\blast\s+appearance\b", text):
        return False
    return bool(re.search(r"\bsirius\b(?:(?![.;]).){0,80}?\b(?:set|sets|setting)\b", text))




def _deterministic_anaphoric_station_add(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Recover a narrow, mechanically provable anaphoric station-in-sign clause.

    Example source:
        "when Saturn became stationary in the west , it became sta [ tionary ] in Gemini"

    The local anaphoric clause supplies the station predicate and zodiac sign.
    Exactly one close preceding visible-planet antecedent supplies only the
    planet identity. In particular, wording such as "in the west" in the
    antecedent is not copied into the generic measured-relation ``direction``
    field.
    """
    evidence = str(payload.get("evidence_quote") or "")
    if not evidence:
        return None

    for match in re.finditer(r"\bit\b", evidence, flags=re.IGNORECASE):
        atomic_start = match.start()

        # Keep the anaphoric atom local while preserving exact source text.
        # A semicolon is a safe strong boundary; ordinary periods are not used
        # because editorial lacunae such as "[ ... ]" contain periods.
        semicolon = evidence.find(";", atomic_start)
        atomic_end = len(evidence) if semicolon < 0 else semicolon
        atomic = evidence[atomic_start:atomic_end].strip()
        if not atomic:
            continue

        if "station" not in _source_compact(atomic):
            continue

        signs = [
            sign for sign in sky.SIGNS
            if _sign_field_present(atomic, sign)
        ]
        if len(signs) != 1:
            continue

        # Only a close preceding antecedent is eligible. Trim to the last strong
        # source boundary so an unrelated earlier planet cannot bind the pronoun.
        window_start = max(0, atomic_start - 240)
        preceding = evidence[window_start:atomic_start]
        boundary = max(
            preceding.rfind(";"),
            preceding.rfind("."),
            preceding.rfind("\n"),
        )
        antecedent = preceding[boundary + 1:].strip(" \t\r\n,")
        planets = _literal_planet_mentions(antecedent)
        if len(planets) != 1:
            continue
        if "station" not in _source_compact(antecedent):
            continue

        return {
            "action": "ADD",
            "skip_reason": None,
            "explanation": (
                "Deterministic anaphoric-station recovery: the local 'it' clause "
                "literally supplies a station predicate and zodiac sign, while "
                f"the close preceding antecedent names exactly one planet "
                f"({planets[0]}). Only the planet identity is inherited."
            ),
            "finals": [{
                "kind": "station",
                "planet": planets[0],
                "target_planet": None,
                "star": None,
                "sign": signs[0],
                "measurement_cubits": None,
                "measurement_fingers": None,
                "direction": None,
                "station_phase": None,
                "omitted": False,
                "restored": ("[" in atomic or "]" in atomic),
                "event_subtype": None,
                "atomic_evidence_quote": atomic,
                "subject_binding": "ANAPHORIC",
                "antecedent_quote": antecedent,
            }],
            "_deterministic_anaphoric_station_guard": True,
        }

    return None

def _deterministic_generic_sirius_skip(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Classify ordinary Sirius setting as real astronomy outside current ontology.

    The special repository ``sirius`` kind is reserved for Sirius first/last
    appearance and acronychal scheme dates.  An ordinary setting used as an
    eclipse-phase time marker is therefore preserved in the audit as
    ``unsupported_but_astronomical`` rather than being mis-scored as a scheme
    date.
    """
    quote = str(payload.get("evidence_quote") or "")
    if not _generic_sirius_setting_literal(quote):
        return None
    return {
        "action": "SKIP",
        "skip_reason": "unsupported_but_astronomical",
        "explanation": (
            "The source literally reports an ordinary Sirius setting (for example, "
            "'Sirius set') rather than Sirius' formal last appearance. The current "
            "repository kind='sirius' is a scheme-date family checked with "
            "sirius_scheme_date(), so mapping this eclipse-relative stellar setting "
            "to event_subtype='setting' would change the source semantics. Preserve "
            "it as genuine but currently unsupported astronomical evidence."
        ),
        "finals": [],
        "_deterministic_generic_sirius_guard": True,
    }


def guard_known_normalizations(
    payload: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    """Prevent two known category errors while retaining general LLM authority.

    These guards only protect parser transformations whose correctness is
    mechanically provable from the exact parser-matched source.
    """
    if review.get("action") != "DROP":
        return review

    st = payload["deterministic_statement"]
    rule = str(st.get("parser_rule") or "").removesuffix("_bracket")
    matched = payload.get("parser_matched_source", "")

    # Coordinated sign clauses intentionally yield one atomic Statement per
    # planet. If this atom's planet and sign are both literally in the matched
    # clause, dropping it merely because another conjunct is not in the same
    # Statement object is necessarily incorrect.
    if rule in {"coordinated_sign", "triple_coordinated_sign",
                "monthly_path", "monthly_path_after"}:
        planet = str(st.get("planet") or "")
        sign = str(st.get("sign") or "")
        if _normalized_contains(matched, planet) and _sign_field_present(matched, sign):
            fixed = dict(review)
            fixed["action"] = "KEEP"
            fixed["explanation"] = (
                "Deterministic guard overrode DROP: this parser rule intentionally "
                "decomposes a coordinated sign clause or a monthly path into atomic "
                "planet-in-sign predicates, and the matched source literally "
                "contains this atom's planet and sign. "
                "Original LLM explanation: " + str(review.get("explanation", ""))
            )
            fixed["evidence_quote"] = matched
            return fixed

    # The Sirius regex itself directly matches the named event phrase. If that
    # literal phrase agrees with the subtype, absence of repetition in nearby
    # context cannot invalidate the event.
    if rule.startswith("sirius_"):
        subtype = st.get("event_subtype")
        if _recognized_sirius_literal(matched, subtype):
            fixed = dict(review)
            fixed["action"] = "KEEP"
            fixed["explanation"] = (
                "Deterministic guard overrode DROP: parser_matched_source literally "
                "contains the recognized Sirius event phrase; surrounding context is "
                "only needed for inherited temporal context. Original LLM explanation: "
                + str(review.get("explanation", ""))
            )
            fixed["evidence_quote"] = matched
            return fixed

    return review



def _first_pass(client, model: str, payload: dict[str, Any], *,
                instructions: str = SYSTEM_PROMPT) -> dict[str, Any]:
    review = _one_review_request(client, model, payload, instructions=instructions)
    review["final"] = _canonical_proposal(review.get("final") or _original_review_final(payload))
    return _hold_code_fields(payload, review)


def call_review(client, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Review one parser statement with independent DROP/MODIFY safeguards."""
    context = _clause_window(payload)[0]

    review = _first_pass(client, model, payload)
    initial_action = review.get("action")
    overrides = review.get("_code_field_overrides")

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        result["_initial_action"] = initial_action
        if overrides and "_code_field_overrides" not in result:
            result["_code_field_overrides"] = overrides
        return result

    # Known mechanically provable normalizations can reject a bad DROP without
    # spending a second API call. All other DROP proposals receive generic
    # positive-evidence verification.
    review = guard_known_normalizations(payload, review)
    if review["action"] == "DROP":
        review = verify_drop(client, model, payload, review)
    if review["action"] in {"KEEP", "DROP"}:
        return finish(review)

    # MODIFY first needs a grounded first-pass quote. If not grounded, give the
    # reviewer one repair chance before field-by-field verification.
    if not _quote_spans(review.get("evidence_quote", ""), context):
        repair_payload = {
            **payload,
            "previous_review": {k: v for k, v in review.items() if not k.startswith("_")},
            "repair_instruction": (
                "Your previous MODIFY decision did not provide an evidence_quote "
                "that could be located in the supplied source context. Review the "
                "decision again. If MODIFY is still warranted, evidence_quote MUST "
                "be copied verbatim from the sentence that contains "
                "parser_matched_source, and contain the corrected value. Do not use "
                "ellipses, paraphrase, normalized spelling, or added words. If the "
                "source does not contain a literal substring supporting the "
                "correction, return KEEP instead. Do not clear month/day/restored "
                "merely because their governing context is not visible locally."
            ),
        }
        repair_instructions = (
            SYSTEM_PROMPT
            + "\n\nIMPORTANT REPAIR PASS:\n"
            + repair_payload["repair_instruction"]
        )
        review = _first_pass(client, model, repair_payload, instructions=repair_instructions)
        overrides = overrides or review.get("_code_field_overrides")
        review = guard_known_normalizations(payload, review)
        if review["action"] == "DROP":
            review = verify_drop(client, model, payload, review)
        if review["action"] in {"KEEP", "DROP"}:
            return finish(review)
        if not _quote_spans(review.get("evidence_quote", ""), context):
            # A persistently ungrounded MODIFY is a candidate-local model failure,
            # not evidence that the deterministic parser statement is wrong.
            bad_quote = review.get("evidence_quote", "")
            return finish({
                "action": "KEEP",
                "explanation": (
                    "Conservative KEEP because the model proposed MODIFY but its "
                    "evidence_quote remained ungrounded after the dedicated repair "
                    "pass. The unsupported semantic correction was discarded. "
                    f"Rejected quote={bad_quote!r}. "
                    f"Repair-pass explanation: {review.get('explanation', '')}"
                ),
                "evidence_quote": "",
                "final": _original_review_final(payload),
                "_modify_grounding_guard": {"applied": True, "rejected_quote": bad_quote},
            })

    return finish(verify_modify(client, model, payload, review))


def build_review_record(
    client,
    model: str,
    tablet: str,
    source: BlindText,
    index: int,
    st: S.Statement,
) -> dict[str, Any]:
    payload = review_payload(source, record_token(tablet), index, st)
    with call_ledger() as calls:
        review = call_review(client, model, payload)
    final_st = apply_review(st, review, source.original)  # validates before it can enter cache
    return {
        "cache_key": cache_key(tablet, source.original, index, st, model),
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "tablet": tablet,
        "record_token": payload["record"],
        "statement_index": index,
        "initial_action": review.get("_initial_action", review["action"]),
        "action": review["action"],
        "explanation": review["explanation"],
        "evidence_quote": review["evidence_quote"],
        "drop_verification": review.get("_drop_verification"),
        "modify_verification": review.get("_modify_verification"),
        "code_field_overrides": review.get("_code_field_overrides"),
        "model_output_failure": review.get("_model_output_failure"),
        "parser_matched_source": payload["parser_matched_source"],
        "source_measurement": payload["source_measurement"],
        "parser_rule_note": payload["parser_rule_note"],
        "original_statement": jsonable_statement(st),
        "review_final_fields": review["final"],
        "final_statement": None if final_st is None else jsonable_statement(final_st),
        "calls": calls,
        "call_totals": _ledger_totals(calls),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def record_to_statement(original: S.Statement, record: dict[str, Any],
                        text: str) -> S.Statement | None:
    return apply_review(original, {"action": record["action"],
                                   "final": record["review_final_fields"]}, text)


# ---------------------------------------------------------------------------
# Targeted review filters
# ---------------------------------------------------------------------------


def _statement_local_context(text: str, st: S.Statement, radius: int = 600) -> str:
    """Return the exact parser span plus nearby source context for filtering."""
    start = max(0, int(getattr(st, "source_start", -1)))
    end = int(getattr(st, "source_end", -1))
    if start < 0 or end < start:
        return getattr(st, "text", "") or ""
    return text[max(0, start - radius): min(len(text), end + radius)]


def filter_extracted(
    extracted: dict[str, tuple[str, list[S.Statement]]],
    *,
    tablets: list[str] | None = None,
    contains: list[str] | None = None,
    kinds: list[str] | None = None,
    parser_rules: list[str] | None = None,
) -> dict[str, tuple[str, list[S.Statement]]]:
    """Filter statements for an answer-blind targeted audit.

    Filters are ANDed across categories and ORed within a repeated category.
    For example::

        --kind season --contains "I did not watch"

    keeps only season statements whose exact matched span or +/-600 characters
    of local source context contains that phrase (case-insensitive).
    """
    tablet_set = {v.strip() for v in (tablets or []) if v.strip()}
    kind_set = {v.strip() for v in (kinds or []) if v.strip()}
    rule_set = {v.strip() for v in (parser_rules or []) if v.strip()}
    needles = [v.casefold() for v in (contains or []) if v.strip()]

    out: dict[str, tuple[str, list[S.Statement]]] = {}
    for tablet, (text, sts) in extracted.items():
        if tablet_set and tablet not in tablet_set:
            continue
        selected: list[S.Statement] = []
        for st in sts:
            if kind_set and st.kind not in kind_set:
                continue
            if rule_set and st.parser_rule not in rule_set:
                continue
            if needles:
                haystack = _statement_local_context(text, st).casefold()
                if not any(needle in haystack for needle in needles):
                    continue
            selected.append(st)
        if selected:
            out[tablet] = (text, selected)
    return out


# ---------------------------------------------------------------------------
# Review all parser-extracted statements answer-blind
# ---------------------------------------------------------------------------


# The paper reports ADART I-III; ADART V-VI records carry no reference year.
SCOPES = {
    "adart-i-iii": ("adsd/adart1", "adsd/adart2", "adsd/adart3"),
    "all": None,
}


def load_tablets(scope: str = "adart-i-iii") -> pd.DataFrame:
    """Record ids and translations only; published years are not loaded here."""
    tablets = pd.read_parquet(HERE / "data" / "corpus" / "tablets.parquet",
                              columns=["tablet", "project", "text"])
    tablets = tablets[tablets.text.notna()]
    projects = SCOPES[scope]
    if projects is not None:
        tablets = tablets[tablets.project.isin(projects)]
    return tablets[["tablet", "text"]].copy()


def extract_all(tablets: pd.DataFrame) -> dict[str, tuple[str, list[S.Statement]]]:
    out = {}
    for row in tablets.itertuples(index=False):
        text = row.text if isinstance(row.text, str) else ""
        out[row.tablet] = (text, S.extract(text))
    return out


# One client per process, created on the first cache miss. The rebuild scripts
# replace get_client with a function that raises, so they never pay for a call.
_CLIENT: dict[str, Any] = {}


def shared_client(route: str, workers: int, max_workers: int) -> ModelClient:
    if "client" not in _CLIENT:
        _CLIENT["client"] = get_client(route, AdaptiveLimiter(workers, max_workers))
    return _CLIENT["client"]


def _run_parallel(function, items, max_workers: int):
    """Map in parallel; the shared limiter bounds the requests in flight."""
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        yield from pool.map(function, items)


def review_all(
    extracted: dict[str, tuple[str, list[S.Statement]]],
    model: str,
    cache_path: Path,
    workers: int,
    force: bool,
    pilot: int | None,
    *,
    route: str = DEFAULT_ROUTE,
    max_workers: int | None = None,
) -> tuple[dict[str, list[S.Statement]], list[dict[str, Any]], dict[str, list[S.Statement]]]:
    """Review every parser clause, from cache where possible.

    Returns the reviewed clauses per tablet, the review records used, and the
    parser clauses the review dropped, which the search treats as existing.
    """
    cache = {} if force else load_cache(cache_path)
    max_workers = max_workers or workers
    items = []
    records_by_key: dict[str, dict[str, Any]] = dict(cache)

    for tablet, (text, sts) in extracted.items():
        for index, st in enumerate(sts):
            key = cache_key(tablet, text, index, st, model)
            if key not in records_by_key:
                items.append((tablet, text, index, st))

    total = sum(len(sts) for _, sts in extracted.values())
    print(f"{total:,} deterministic statements; {len(items):,} require model review; "
          f"{total - len(items):,} reusable from cache")

    if pilot is not None:
        items = items[:pilot]
        print(f"pilot mode: reviewing {len(items):,} uncached statements and then exiting")

    if items:
        client = shared_client(route, workers, max_workers)
        sources = {tablet: blind(text) for tablet, (text, _) in extracted.items()}

        def one(item):
            tablet, _text, index, st = item
            return build_review_record(client, model, tablet, sources[tablet], index, st)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for number, record in enumerate(_run_parallel(one, items, max_workers), start=1):
            records_by_key[record["cache_key"]] = record
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if number % 25 == 0 or number == len(items):
                print(f"  reviewed {number:,}/{len(items):,} new statements")

    if pilot is not None:
        return {}, list(records_by_key.values()), {}

    reviewed: dict[str, list[S.Statement]] = {}
    dropped: dict[str, list[S.Statement]] = {}
    used_records = []
    for tablet, (text, sts) in extracted.items():
        final_sts = []
        for index, st in enumerate(sts):
            key = cache_key(tablet, text, index, st, model)
            if key not in records_by_key:
                raise RuntimeError(f"missing cached LLM review for {tablet} statement {index}")
            record = records_by_key[key]
            final = record_to_statement(st, record, text)
            used_records.append(record)
            if final is not None:
                final_sts.append(final)
            else:
                dropped.setdefault(tablet, []).append(st)
        reviewed[tablet] = S.bind_joint_positions(text, final_sts)
    return reviewed, used_records, dropped


# ---------------------------------------------------------------------------
# Raw-text recall: discovery -> normalization -> deterministic Statement
# ---------------------------------------------------------------------------

_GREEK_WORDS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
}

_NULL_STRINGS = {
    "", "null", "none", "n/a", "na", "unknown", "not applicable",
    "not_applicable", "nil",
}


def _noneish(value: Any) -> Any:
    """Convert model placeholder strings to real nulls before validation."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().casefold() in _NULL_STRINGS:
        return None
    return value


def _source_compact(value: Any) -> str:
    """Compact source text for literal-field support across editorial breaks.

    Removes punctuation/brackets/whitespace but not letters/digits, converts
    spelled Greek-letter names to symbols, and folds common theta/rho variants.
    Thus `Ge ] mini` supports Gemini and `α [ Leonis ... ]` supports α Leonis.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("ϑ", "θ").replace("ϱ", "ρ")
    for word, symbol in _GREEK_WORDS.items():
        text = re.sub(rf"\b{word}\b", symbol, text, flags=re.IGNORECASE)
    return "".join(ch for ch in text if ch.isalnum() or ch in set(_GREEK_WORDS.values()))


def _literal_field_present(quote: str, value: Any) -> bool:
    value = _noneish(value)
    if value is None:
        return True
    needle = _source_compact(value)
    return bool(needle) and needle in _source_compact(quote)


def _sign_field_present(quote: str, sign: Any) -> bool:
    """Ground canonical zodiac labels using only controlled spelling aliases.

    This deliberately allows orthographic variants such as Scorpius/Scorpio and
    Capricorn/Capricornus, but does NOT map asterisms (e.g. Chariot, Twins) onto
    zodiac signs.
    """
    sign = _noneish(sign)
    if sign is None or str(sign).strip() == "":
        return True
    raw = str(sign).strip()
    folded = raw.casefold()
    aliases = [raw]
    if folded in {"scorpio", "scorpius"}:
        aliases.extend(["Scorpio", "Scorpius"])
    if folded in {"capricorn", "capricornus"}:
        aliases.extend(["Capricorn", "Capricornus"])
    q = _source_compact(quote)
    return any(_source_compact(alias) in q for alias in aliases if alias)


def _all_source_measurements(source: str) -> tuple[list[float], list[float]]:
    clean = re.sub(r"[\[\]]", " ", source or "")
    clean = " ".join(clean.split())
    cubits = [_source_number(m.group(1)) for m in _SOURCE_CUBIT.finditer(clean)]
    fingers = [_source_number(m.group(1)) for m in _SOURCE_FINGER.finditer(clean)]
    return cubits, fingers


def _number_literal_present(values: list[float], wanted: Any) -> bool:
    if wanted is None:
        return True
    try:
        target = float(wanted)
    except Exception:
        return False
    return any(math.isclose(float(v), target, rel_tol=0.0, abs_tol=1e-8) for v in values)


# A measurement belongs to a relation only when it stands directly before the
# direction word, as in the parser's own grammar: "2 1/2 cubits behind α
# Virginis", "1 cubit 8 fingers above Mercury". Spaces, editorial brackets
# and an uncertainty mark may intervene.
_MEASURE_GAP = r"[\s\[\]?]*"
_MEASURE_NUMBER = r"(?:\d+\s+)?\d+/\d+|\d+"
_BOUND_MEASURE = re.compile(
    rf"(?P<c>{_MEASURE_NUMBER}){_MEASURE_GAP}cubits?"
    rf"(?:{_MEASURE_GAP}(?P<cf>{_MEASURE_NUMBER}){_MEASURE_GAP}fingers?)?"
    rf"{_MEASURE_GAP}(?P<cd>above|below|in\s+front\s+of|behind)\b"
    rf"|(?P<f>{_MEASURE_NUMBER}){_MEASURE_GAP}fingers?"
    rf"{_MEASURE_GAP}(?P<fd>above|below|in\s+front\s+of|behind)\b",
    re.IGNORECASE)


def _bound_measurements(quote: str) -> list[tuple[float | None, float | None, str]]:
    """(cubits, fingers, direction) of each measurement attached to a relation.

    A number written after "+", "x", "nn" or "?" (as in "x +1 cubits" or
    "1 ? 1/2 cubits") is not a definite distance, and a finger count that
    completes a cubit count ("2 ? [ cubits ] 6 fingers") is not a distance on
    its own. Neither is returned.
    """
    found = []
    for m in _BOUND_MEASURE.finditer(quote or ""):
        group = "c" if m.group("c") is not None else "f"
        prefix = quote[:m.start(group)].rstrip(" \t\r\n[]")
        if prefix.endswith(("+", "?", "x", "nn", "/")) or prefix[-1:].isdigit():
            continue
        if group == "c":
            cubits = _source_number(m.group("c"))
            fingers = _source_number(m.group("cf")) if m.group("cf") else None
            direction = m.group("cd")
        else:
            if re.search(r"cubits?$", prefix.rstrip(" \t\r\n[]?"), re.IGNORECASE):
                continue
            cubits, fingers, direction = None, _source_number(m.group("f")), m.group("fd")
        found.append((cubits, fingers, " ".join(direction.lower().split())))
    return found


def _measurement_binding_errors(final: dict[str, Any], quote: str) -> list[str]:
    cubits = final.get("measurement_cubits")
    fingers = final.get("measurement_fingers")
    direction = final.get("direction")
    if final.get("kind") not in MEASURED_KINDS or (cubits is None and fingers is None):
        return []
    allowed = {direction, _INVERT_DIRECTION.get(direction)}

    def same(a, b):
        return (a is None and b is None) or (
            a is not None and b is not None and math.isclose(float(a), float(b), abs_tol=1e-8))

    for c, f, d in _bound_measurements(quote):
        if same(c, cubits) and same(f, fingers) and d in allowed:
            return []
    return [
        f"measurement cubits={cubits!r} fingers={fingers!r} {direction!r} is not the "
        "distance written directly before the relation in evidence_quote"
    ]


def _recall_field_grounding_errors(final: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """Mechanically require every substantive ADD field to be source-grounded."""
    quote = str(payload.get("evidence_quote") or "")
    errors: list[str] = []
    kind = final.get("kind")
    planet = final.get("planet")
    target = final.get("target_planet")
    star = final.get("star")
    sign = final.get("sign")
    subtype = final.get("event_subtype")

    # Internal bookkeeping exceptions are intentionally narrow.
    if kind not in {"season", "sirius"} and planet not in {"lunar", "solar"}:
        if planet and not _literal_field_present(quote, planet):
            errors.append(f"planet {planet!r} is not supported by evidence_quote")
    elif kind == "eclipse":
        q = _source_compact(quote)
        if planet == "lunar" and not any(x in q for x in ("moon", "lunar", "eclipse", "eclipsed")):
            errors.append("lunar eclipse subject is not supported by evidence_quote")
        if planet == "solar" and not any(x in q for x in ("sun", "solar", "eclipse", "eclipsed")):
            errors.append("solar eclipse subject is not supported by evidence_quote")

    if target and not _literal_field_present(quote, target):
        errors.append(f"target_planet {target!r} is not supported by evidence_quote")
    if star and not _literal_field_present(quote, star):
        errors.append(f"star {star!r} is not supported by evidence_quote")
    if sign and not _sign_field_present(quote, sign):
        errors.append(f"sign {sign!r} is not supported by evidence_quote")

    cubits, fingers = _all_source_measurements(quote)
    if final.get("measurement_cubits") is not None and not _number_literal_present(cubits, final.get("measurement_cubits")):
        errors.append(
            f"measurement_cubits={final.get('measurement_cubits')!r} is not literally supported by evidence_quote"
        )
    if final.get("measurement_fingers") is not None and not _number_literal_present(fingers, final.get("measurement_fingers")):
        errors.append(
            f"measurement_fingers={final.get('measurement_fingers')!r} is not literally supported by evidence_quote"
        )

    direction = final.get("direction")
    if direction and not _literal_field_present(quote, direction):
        errors.append(f"direction {direction!r} is not supported by evidence_quote")
    errors.extend(_measurement_binding_errors(final, quote))

    phase = final.get("station_phase")
    if phase and not _literal_field_present(quote, phase):
        errors.append(f"station_phase {phase!r} is not supported by evidence_quote")

    qcompact = _source_compact(quote)
    if kind == "first_app" and "firstappearance" not in qcompact:
        errors.append("first_app is not supported by a literal first-appearance phrase")
    if kind == "last_app" and "lastappearance" not in qcompact:
        errors.append("last_app is not supported by a literal last-appearance phrase")
    if kind == "station" and "station" not in qcompact:
        errors.append("station is not supported by a literal station phrase")
    if kind == "entry" and not any(token in qcompact for token in ("enter", "entered", "reached", "reach")):
        errors.append("entry is not supported by a literal enter/reach phrase")
    if kind == "season" and subtype and subtype not in qcompact:
        errors.append(f"season subtype {subtype!r} is not supported by evidence_quote")
    if kind == "sirius":
        if "sirius" not in qcompact:
            errors.append("sirius kind requires literal Sirius in evidence_quote")
        if subtype == "rising" and not any(x in qcompact for x in ("firstappearance", "rising")):
            errors.append("Sirius rising/first appearance subtype is not supported")
        if subtype == "setting" and "lastappearance" not in qcompact:
            errors.append(
                "Sirius scheme setting requires a literal last-appearance phrase; "
                "generic 'Sirius set/setting' is a distinct unsupported event"
            )
        if subtype == "acronychal" and "acronychal" not in qcompact:
            errors.append("Sirius acronychal subtype is not supported")
    if kind == "acronychal" and "acronychal" not in qcompact:
        errors.append("acronychal kind is not supported by evidence_quote")

    return errors




def _atomic_clause_segments(text: str) -> list[str]:
    """Split an atomic-evidence span at strong independent-clause boundaries.

    Newlines alone are not boundaries because ADART translations frequently wrap
    a single clause across lines. Semicolons are boundaries. A period is treated
    as a boundary when the following text visibly starts a new dated/independent
    diary clause. This is deliberately conservative: the goal is to prevent a
    subject in a later sentence from licensing an earlier damaged proposition.
    """
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(
        r"\s*;\s*|\.\s+(?=(?:Night|The|Month|Around|Year|At|On|When|From|Mercury|Venus|Mars|Jupiter|Saturn|Sirius|the\s+moon|the\s+sun|\[|\())",
        text,
        flags=re.IGNORECASE,
    )
    return [p.strip() for p in parts if p and p.strip()]


def _semantic_atom_without_binding(atom: dict[str, Any]) -> dict[str, Any]:
    out = dict(atom)
    for key in ("atomic_evidence_quote", "subject_binding", "antecedent_quote"):
        out.pop(key, None)
    return out


_INVERT_DIRECTION = {
    "above": "below",
    "below": "above",
    "in front of": "behind",
    "behind": "in front of",
}


def _catalogued_star_or_none(value: Any) -> str | None:
    value = _noneish(value)
    if value is None:
        return None
    try:
        return _canonical_star_label(str(value))
    except Exception:
        return None


def _source_direction_occurrences(text: str) -> list[tuple[int, str]]:
    compact = _source_compact(text)
    found: list[tuple[int, str]] = []
    for direction in ("above", "below", "in front of", "behind"):
        needle = _source_compact(direction)
        start = 0
        while needle:
            pos = compact.find(needle, start)
            if pos < 0:
                break
            found.append((pos, direction))
            start = pos + max(1, len(needle))
    return sorted(found)


def _relation_source_role(
    atom: dict[str, Any],
    atomic_quote: str,
    antecedent_quote: str | None,
    binding: str,
) -> tuple[str | None, str | None]:
    """Infer grammatical subject role and literal source direction.

    Returns (role, direction) where role is ``planet``, ``star``, or
    ``target_planet``.  This uses literal source ordering only; it does not make
    astronomical inferences.
    """
    planet = _noneish(atom.get("planet"))
    star = _noneish(atom.get("star"))
    target = _noneish(atom.get("target_planet"))
    directions = _source_direction_occurrences(atomic_quote)
    source_direction = directions[0][1] if len(directions) == 1 else None

    if binding == "ANAPHORIC" and antecedent_quote:
        ant_roles: list[str] = []
        if planet and _literal_field_present(antecedent_quote, planet):
            ant_roles.append("planet")
        if star and _literal_field_present(antecedent_quote, star):
            ant_roles.append("star")
        if target and _literal_field_present(antecedent_quote, target):
            ant_roles.append("target_planet")
        if len(ant_roles) == 1:
            return ant_roles[0], source_direction
        return None, source_direction

    compact = _source_compact(atomic_quote)
    role_positions: list[tuple[int, str]] = []
    for role, value in (("planet", planet), ("star", star), ("target_planet", target)):
        if value:
            needle = _source_compact(value)
            p = compact.find(needle)
            if p >= 0:
                role_positions.append((p, role))
    if source_direction is None:
        return None, None
    dpos = next((p for p, d in directions if d == source_direction), None)
    if dpos is None:
        return None, source_direction
    before = [(p, role) for p, role in role_positions if p < dpos]
    after = [(p, role) for p, role in role_positions if p > dpos]
    if before and after:
        # The closest semantic entity before the relation phrase is the safest
        # grammatical-subject cue for the simple ADART relation clauses handled
        # here ("Mars was ... below α Leonis" / "α Leonis was ... below Mars").
        return max(before)[1], source_direction
    return None, source_direction


def _canonicalize_star_planet_source_roles(
    atom: dict[str, Any],
    atomic_quote: str,
    antecedent_quote: str | None,
    binding: str,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Canonicalize star↔planet storage roles after resolving source grammar.

    The repository always stores a Normal-Star relation as ``planet=<body>`` and
    ``star=<Normal Star>`` even when the translation makes the STAR the
    grammatical subject.  When that happens, relation direction is inverted so
    it remains expressed as canonical planet relative to star.
    """
    out = dict(atom)
    visible = {"mercury", "venus", "mars", "jupiter", "saturn", "moon"}

    # Salvage a model output that put a Normal Star into the generic planet slot
    # and the visible body into target_planet.
    raw_planet = _noneish(out.get("planet"))
    raw_target = _noneish(out.get("target_planet"))
    star_from_planet = _catalogued_star_or_none(raw_planet)
    target_body = str(raw_target or "").strip().casefold()
    if star_from_planet and target_body in visible:
        out["kind"] = "appulse"
        out["planet"] = target_body
        out["star"] = star_from_planet
        out["target_planet"] = None
        out["sign"] = ""

    # Also salvage the inverse malformed layout: visible body in planet and a
    # catalogued star accidentally placed in target_planet.
    if _noneish(out.get("star")) is None:
        target_star = _catalogued_star_or_none(out.get("target_planet"))
        body = str(_noneish(out.get("planet")) or "").strip().casefold()
        if target_star and body in visible:
            out["kind"] = "appulse"
            out["star"] = target_star
            out["target_planet"] = None
            out["sign"] = ""

    role, source_direction = _relation_source_role(
        out, atomic_quote, antecedent_quote, binding
    )
    if out.get("kind") in {"appulse", "station_appulse"} and out.get("star"):
        if source_direction:
            if role == "star":
                out["direction"] = _INVERT_DIRECTION[source_direction]
            elif role == "planet":
                out["direction"] = source_direction
        out["target_planet"] = None
        out["sign"] = ""
    return out, role, source_direction


def _semantic_for_source_grounding(
    semantic: dict[str, Any],
    *,
    source_role: str | None,
    source_direction: str | None,
) -> dict[str, Any]:
    """Convert canonical storage fields back to literal source orientation only
    for the purpose of grounding checks.
    """
    local = dict(semantic)
    if (
        local.get("kind") in {"appulse", "station_appulse"}
        and source_role == "star"
        and source_direction
    ):
        local["direction"] = source_direction
    return local


def _atomic_binding_validate_and_canonicalize(
    atom: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Require one atom to be supported by one coherent local source span.

    DIRECT/COORDINATED atoms must jointly ground all semantic fields inside one
    clause segment. ANAPHORIC atoms may inherit exactly one semantic role
    (planet, Normal Star, or target planet) from a close *preceding* antecedent.
    Source grammatical roles are resolved before canonical repository storage;
    therefore star-subject clauses remain valid and have their direction
    inverted when stored as planet→star relations.
    """
    out = dict(atom)
    errors: list[str] = []
    evidence = str(payload.get("evidence_quote") or "")
    atomic_quote = str(out.get("atomic_evidence_quote") or "").strip()
    binding = str(out.get("subject_binding") or "").strip().upper()
    antecedent = _noneish(out.get("antecedent_quote"))

    if binding not in {"DIRECT", "COORDINATED", "ANAPHORIC"}:
        errors.append(f"invalid subject_binding {binding!r}")
        return out, errors
    if not atomic_quote:
        errors.append("atomic_evidence_quote is empty")
        return out, errors

    grounded_atomic = _ground_quote_to_exact_source(atomic_quote, evidence)
    if grounded_atomic is None:
        errors.append("atomic_evidence_quote is not an exact grounded substring of evidence_quote")
        return out, errors
    exact_atomic, atomic_local_start, _atomic_local_end = grounded_atomic
    out["atomic_evidence_quote"] = exact_atomic
    clauses = _atomic_clause_segments(exact_atomic) or [exact_atomic]

    exact_ant: str | None = None
    inherited_role: str | None = None
    if binding in {"DIRECT", "COORDINATED"}:
        if antecedent is not None:
            errors.append(f"{binding} requires antecedent_quote=null")
    else:
        if antecedent is None or not str(antecedent).strip():
            errors.append("ANAPHORIC requires antecedent_quote")
            return out, errors
        full_context = (
            str(payload.get("context_before") or "")
            + evidence
            + str(payload.get("context_after") or "")
        )
        evidence_global_start = len(str(payload.get("context_before") or ""))
        atomic_global_start = evidence_global_start + atomic_local_start
        # Bind to the mention nearest before the clause: the same antecedent
        # wording can occur more than once in the context (X500075).
        grounded_ant = _ground_quote_to_exact_source(
            str(antecedent), full_context, before=atomic_global_start)
        if grounded_ant is None:
            errors.append("antecedent_quote is not grounded in supplied local context")
            return out, errors
        exact_ant, _ant_start, ant_end = grounded_ant
        out["antecedent_quote"] = exact_ant
        if ant_end > atomic_global_start:
            errors.append("ANAPHORIC antecedent must occur before atomic_evidence_quote")
        elif atomic_global_start - ant_end > 240:
            errors.append("ANAPHORIC antecedent is too distant (>240 chars) from atomic clause")

        atomic_compact = _normalize_grounding_text(exact_atomic).casefold()
        if not re.search(r"\b(?:it|this|that|same)\b", atomic_compact):
            errors.append("ANAPHORIC atomic clause lacks an explicit local anaphor such as 'it'")

    # Resolve source grammatical roles, then canonicalize star↔planet storage.
    out, source_role, source_direction = _canonicalize_star_planet_source_roles(
        out, exact_atomic, exact_ant, binding
    )
    semantic = _semantic_atom_without_binding(out)
    source_semantic = _semantic_for_source_grounding(
        semantic, source_role=source_role, source_direction=source_direction
    )

    if binding in {"DIRECT", "COORDINATED"}:
        clause_errors = [
            _recall_field_grounding_errors(source_semantic, {"evidence_quote": clause})
            for clause in clauses
        ]
        if not any(not e for e in clause_errors):
            shortest = min(clause_errors, key=len) if clause_errors else ["no atomic clause"]
            errors.append(
                "semantic fields are not jointly grounded in one atomic clause: "
                + "; ".join(shortest)
            )
        if binding == "COORDINATED":
            compact = _normalize_grounding_text(exact_atomic).casefold()
            if " and " not in f" {compact} " and "< and >" not in exact_atomic.casefold():
                errors.append("COORDINATED binding lacks an explicit coordination marker")

    else:
        assert exact_ant is not None
        role_values = {
            "planet": semantic.get("planet"),
            "star": semantic.get("star"),
            "target_planet": semantic.get("target_planet"),
        }
        ant_roles = [
            role for role, value in role_values.items()
            if value and _literal_field_present(exact_ant, value)
        ]
        if len(ant_roles) != 1:
            errors.append(
                "ANAPHORIC antecedent must bind exactly one semantic role "
                f"(found {ant_roles or 'none'})"
            )
        else:
            inherited_role = ant_roles[0]

        # Everything except the inherited grammatical role must be explicitly
        # supported by the anaphoric clause itself. This allows, for example,
        # star antecedent + "it stood ... in front of Mercury" while still
        # forbidding a later Mercury sentence from supplying an earlier subject.
        local_semantic = dict(source_semantic)
        if inherited_role:
            local_semantic[inherited_role] = None if inherited_role != "planet" else ""
        clause_errors = [
            _recall_field_grounding_errors(local_semantic, {"evidence_quote": clause})
            for clause in clauses
        ]
        if not any(not e for e in clause_errors):
            shortest = min(clause_errors, key=len) if clause_errors else ["no atomic clause"]
            errors.append(
                "non-inherited semantic fields are not jointly grounded in the anaphoric clause: "
                + "; ".join(shortest)
            )

    return out, errors


def _canonical_sign_name(value: str | None) -> str:
    value = _noneish(value)
    raw = (value or "").strip()
    if not raw:
        return ""
    raw = S.canonical_sign(raw)
    for sign in sky.SIGNS:
        if sign.casefold() == raw.casefold():
            return sign
    raise ValueError(f"unknown zodiac sign {value!r}")


def _canonical_star_label(value: str | None) -> str | None:
    value = _noneish(value)
    if value is None:
        return None
    raw = " ".join(str(value).strip().split())
    if not raw:
        return None
    parts = raw.split(" ", 1)
    if parts[0].casefold() in _GREEK_WORDS and len(parts) == 2:
        raw = _GREEK_WORDS[parts[0].casefold()] + " " + parts[1]
    raw = raw.replace("ϱ ", "ρ ")
    # sky.star() is the repository's own catalogue gate.
    if sky.star(raw) is None:
        raise ValueError(f"Normal Star {raw!r} is not catalogued")
    return raw


def _strict_recall_final(final: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize and validate a raw-recall normalization result.

    This is intentionally stricter than validate_final(): impossible hybrids
    such as kind='appulse' + sign='Gemini' are rejected rather than silently
    allowed into deterministic astronomy.
    """
    out = dict(final)
    # Mechanical null canonicalization prevents strings such as "null" from
    # becoming invented stars/signs/targets in the scientific pipeline.
    for key in ("target_planet", "star", "direction", "station_phase", "event_subtype"):
        out[key] = _noneish(out.get(key))
    out["sign"] = _canonical_sign_name(_noneish(out.get("sign")))
    out["star"] = _canonical_star_label(out.get("star"))
    out.update({"month": None, "day": None, "year_offset": 0})
    out = validate_final(out)

    kind = out["kind"]
    planet = out["planet"]
    target = out.get("target_planet")
    star = out.get("star")
    sign = out.get("sign", "")
    subtype = out.get("event_subtype")
    visible_planets = {"mercury", "venus", "mars", "jupiter", "saturn"}

    if kind in {"in_sign", "first_app", "last_app", "entry", "station"}:
        if planet not in visible_planets:
            raise ValueError(f"{kind} requires a visible planet subject")
        if not sign:
            raise ValueError(f"{kind} requires a zodiac sign")
        if star is not None or target is not None:
            raise ValueError(f"{kind} cannot carry star/target_planet")
        if out.get("measurement_cubits") is not None or out.get("measurement_fingers") is not None:
            raise ValueError(f"{kind} cannot carry a measurement")
        if kind != "station" and out.get("station_phase") is not None:
            raise ValueError(f"{kind} cannot carry station_phase")
        if subtype is not None:
            raise ValueError(f"{kind} cannot carry event_subtype")

    elif kind == "appulse":
        if planet not in visible_planets | {"moon"}:
            raise ValueError("appulse requires Moon or a visible planet")
        if star is None or target is not None or sign:
            raise ValueError("appulse requires star and forbids target_planet/sign")
        if subtype is not None or out.get("station_phase") is not None:
            raise ValueError("appulse cannot carry event_subtype/station_phase")

    elif kind == "station_appulse":
        if planet not in {"mars", "jupiter", "saturn"}:
            raise ValueError("station_appulse requires an outer-planet subject")
        if star is None or target is not None or sign:
            raise ValueError("station_appulse requires star and forbids target_planet/sign")
        if subtype is not None:
            raise ValueError("station_appulse cannot carry event_subtype")

    elif kind == "planet_appulse":
        if planet != "moon" or target not in visible_planets:
            raise ValueError("planet_appulse requires Moon -> visible planet")
        if star is not None or sign:
            raise ValueError("planet_appulse forbids star/sign")
        if subtype is not None or out.get("station_phase") is not None:
            raise ValueError("planet_appulse cannot carry event_subtype/station_phase")

    elif kind == "planet_planet_appulse":
        if planet not in visible_planets or target not in visible_planets or planet == target:
            raise ValueError("planet_planet_appulse requires two distinct visible planets")
        if star is not None or sign:
            raise ValueError("planet_planet_appulse forbids star/sign")
        if subtype is not None or out.get("station_phase") is not None:
            raise ValueError("planet_planet_appulse cannot carry event_subtype/station_phase")

    elif kind == "eclipse":
        if planet not in {"lunar", "solar"}:
            raise ValueError("eclipse requires lunar/solar")
        if star is not None or target is not None or sign or subtype is not None:
            raise ValueError("eclipse carries no star/target/sign/event_subtype")
        if out.get("station_phase") is not None:
            raise ValueError("eclipse cannot carry station_phase")

    elif kind == "season":
        if subtype not in {"solstice", "equinox"}:
            raise ValueError("season requires solstice/equinox subtype")
        if star is not None or target is not None or sign or out.get("station_phase") is not None:
            raise ValueError("season carries no star/target/sign/station_phase")

    elif kind == "sirius":
        if subtype not in {"rising", "setting", "acronychal"}:
            raise ValueError("sirius requires rising/setting/acronychal subtype")
        if star is not None or target is not None or sign or out.get("station_phase") is not None:
            raise ValueError("sirius carries no star/target/sign/station_phase")

    elif kind == "acronychal":
        if planet not in {"mars", "jupiter", "saturn"}:
            raise ValueError("acronychal requires Mars/Jupiter/Saturn")
        if star is not None or target is not None or sign or subtype is not None:
            raise ValueError("acronychal carries no star/target/sign/event_subtype")
        if out.get("station_phase") is not None:
            raise ValueError("acronychal cannot carry station_phase")

    return out


def _recall_parser_rule(final: dict[str, Any]) -> str:
    kind = final["kind"]
    compound = (
        final.get("measurement_cubits") is not None
        and final.get("measurement_fingers") is not None
    )
    if kind == "appulse":
        return "measured_normal_star_compound" if compound else "measured_normal_star"
    if kind == "planet_appulse":
        return "measured_moon_planet_compound" if compound else "measured_moon_planet"
    if kind == "planet_planet_appulse":
        return "measured_planet_planet_compound" if compound else "measured_planet_planet"
    if kind == "station_appulse":
        return "measured_station_normal_star_compound" if compound else "measured_station_normal_star"
    if kind == "eclipse":
        return "typed_eclipse"
    if kind == "season":
        return f"season_{final['event_subtype']}"
    if kind == "sirius":
        return f"sirius_{final['event_subtype']}"
    if kind == "acronychal":
        return "acronychal_rising"
    return kind


def _recall_context(text: str, pos: int, kind: str) -> tuple[str | None, int | None, int, bool]:
    if kind in MEASURED_KINDS:
        return S.appulse_context(text, pos)
    if kind in {"eclipse", "season", "sirius", "acronychal"}:
        return S.eclipse_context(text, pos)
    month, year_offset, restored = S.positional_context(text, pos)
    return month, None, year_offset, restored


def _make_recalled_statement(
    text: str,
    source_start: int,
    source_end: int,
    normalized: dict[str, Any],
) -> S.Statement:
    """A Statement for one accepted search atom, in original-text offsets.

    Month, day and year come from the parser's context machinery at the
    passage, and the unobserved mark from the parser's rule. The model's own
    omitted value is never used.
    """
    final = _strict_recall_final(normalized)
    month, day, year_offset, context_restored = _recall_context(
        text, source_start, final["kind"]
    )
    amount, unit, offset, latitudinal = derive_measurement(final, None)  # type: ignore[arg-type]
    return S.Statement(
        kind=final["kind"],
        planet=final["planet"],
        sign=final["sign"],
        text=re.sub(r"\s+", " ", text[source_start:source_end]).strip(),
        month=month,
        day=day,
        year_offset=year_offset,
        star=final.get("star"),
        target_planet=final.get("target_planet"),
        offset=offset,
        latitudinal=latitudinal,
        omitted=S.unwatched(text, final["kind"], source_start, source_end),
        restored=bool(final.get("restored")) or bool(context_restored),
        source_start=source_start,
        source_end=source_end,
        parser_rule=_recall_parser_rule(final),
        amount=amount,
        unit=unit,
        direction=final.get("direction"),
        station_phase=final.get("station_phase"),
    )


def _compact_statement(st: S.Statement, source: BlindText) -> dict[str, Any]:
    quote = ""
    if 0 <= st.source_start <= st.source_end <= len(source.original):
        quote = source.masked_slice(st.source_start, st.source_end)
    return {
        "kind": st.kind,
        "planet": st.planet,
        "target_planet": st.target_planet,
        "star": st.star,
        "sign": st.sign,
        "amount_cubits_internal": st.amount,
        "direction": st.direction,
        "station_phase": st.station_phase,
        "omitted": bool(st.omitted),
        "restored": bool(st.restored),
        "source_quote": quote,
    }


def _recall_chunks(
    text: str, core_chars: int, context_chars: int, overlap_chars: int = 350
):
    """Generate overlapping ownership windows for exhaustive raw-text recall.

    Large single-tablet prompts encouraged the model to stop after finding many
    claims and overlook later ones. Smaller overlapping cores reduce that failure
    mode while context on both sides still supports local anaphora/restoration.
    """
    if core_chars < 1000:
        raise ValueError("--recall-core-chars must be at least 1000")
    if overlap_chars < 0:
        raise ValueError("--recall-overlap-chars cannot be negative")
    if overlap_chars >= core_chars:
        raise ValueError("--recall-overlap-chars must be smaller than --recall-core-chars")
    step = core_chars - overlap_chars
    starts = list(range(0, len(text), step)) or [0]
    for core_start in starts:
        core_end = min(len(text), core_start + core_chars)
        yield {
            "core_start": core_start,
            "core_end": core_end,
            "context_start": max(0, core_start - context_chars),
            "context_end": min(len(text), core_end + context_chars),
        }
        if core_end >= len(text):
            break


_RECALL_TRIGGER_PATTERNS = [
    re.compile(r"\bfirst.{0,24}?appearance\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\blast.{0,24}?appearance\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bacronychal.{0,24}?rising\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:became\s+)?stationary\b|\bsta\s*\[\s*tionary\s*\]", re.IGNORECASE),
    re.compile(r"\beclipse(?:d)?\b", re.IGNORECASE),
    re.compile(r"\b(?:solstice|equinox|Sirius)\b", re.IGNORECASE),
]


def _trigger_covered_by_existing(position: int, statements: list[S.Statement]) -> bool:
    for st in statements:
        s = int(getattr(st, "source_start", -1))
        e = int(getattr(st, "source_end", -1))
        if s >= 0 and e >= s and s - 40 <= position <= e + 40:
            return True
    return False


def _recall_trigger_chunks(
    text: str, statements: list[S.Statement], context_chars: int
):
    """Supplement broad recall with small windows around uncovered key phrases."""
    seen_positions: set[int] = set()
    trigger_core = 1400
    for pattern in _RECALL_TRIGGER_PATTERNS:
        for match in pattern.finditer(text):
            pos = match.start()
            if pos in seen_positions or _trigger_covered_by_existing(pos, statements):
                continue
            seen_positions.add(pos)
            core_start = max(0, pos - 350)
            core_end = min(len(text), core_start + trigger_core)
            # Keep the trigger comfortably inside the core near end-of-tablet.
            if core_end - core_start < trigger_core and core_end == len(text):
                core_start = max(0, core_end - trigger_core)
            yield {
                "core_start": core_start,
                "core_end": core_end,
                "context_start": max(0, core_start - context_chars),
                "context_end": min(len(text), core_end + context_chars),
                "trigger_centered": True,
            }


def _planned_recall_chunks(
    text: str,
    statements: list[S.Statement],
    core_chars: int,
    context_chars: int,
    overlap_chars: int,
):
    """Return unique broad + trigger-centered windows for one tablet."""
    planned: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for chunk in _recall_chunks(text, core_chars, context_chars, overlap_chars):
        key = (chunk["core_start"], chunk["core_end"])
        if key not in seen:
            seen.add(key)
            planned.append(chunk)
    for chunk in _recall_trigger_chunks(text, statements, context_chars):
        key = (chunk["core_start"], chunk["core_end"])
        if key not in seen:
            seen.add(key)
            planned.append(chunk)
    planned.sort(key=lambda x: (x["core_start"], x["core_end"]))
    return planned


def _recall_discovery_key(
    tablet: str,
    text: str,
    chunk: dict[str, int],
    existing: list[dict[str, Any]],
    model: str,
) -> str:
    payload = {
        "stage": "discovery",
        "prompt_version": RECALL_DISCOVERY_PROMPT_VERSION,
        "model": model,
        "tablet": tablet,
        "tablet_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "core_start": chunk["core_start"],
        "core_end": chunk["core_end"],
        "existing": existing,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _recall_normalize_key(
    tablet: str,
    text: str,
    claim: dict[str, Any],
    source_start: int,
    model: str,
) -> str:
    payload = {
        "stage": "normalize",
        "prompt_version": RECALL_NORMALIZE_PROMPT_VERSION,
        "model": model,
        "tablet": tablet,
        "tablet_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_start": source_start,
        "claim": claim,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _one_recall_discovery_request(client, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result, error = _structured_request(
            client, model, instructions=RECALL_DISCOVERY_PROMPT, payload=payload,
            fmt=RECALL_DISCOVERY_FORMAT, budgets=(6000, 12000), stage="recall_discovery")
    except ModelRequestRejected as exc:
        result, error = None, f"discovery_request_rejected: {exc}"
    if result is None:
        # A failed discovery window must not abort all tablets. It is recorded
        # as an empty, audited window, and its failure counted in the summary.
        return {
            "claims": [],
            "overall_explanation": (
                "Discovery produced no usable structured output after retries. "
                "No malformed model text was accepted into recall."
            ),
            "grounding_rejections": [{
                "claim_index": None,
                "claim": "",
                "evidence_quote": "",
                "reason": error or "discovery_output_failure",
                "repair_explanation": "",
            }],
            "_model_output_failure": error,
        }

    source = payload["context_before"] + payload["core_text"] + payload["context_after"]
    claims = list(result.get("claims", []))

    # Canonicalize harmless Unicode/whitespace differences back to the exact
    # source substring; send the rest to one quote-repair call.
    bad_indexes: list[int] = []
    canonical: list[dict[str, Any]] = []
    for i, claim in enumerate(claims):
        grounded = _ground_quote_to_exact_source(claim.get("evidence_quote", ""), source)
        if grounded is None:
            bad_indexes.append(i)
            canonical.append(claim)
            continue
        exact_quote, _, _ = grounded
        fixed = dict(claim)
        fixed["evidence_quote"] = exact_quote
        canonical.append(fixed)

    rejected: list[dict[str, Any]] = []
    if bad_indexes:
        canonical, rejected = _repair_recall_quotes(
            client, model, payload, canonical, bad_indexes
        )

    result["claims"] = canonical
    result["grounding_rejections"] = rejected
    if rejected:
        result["overall_explanation"] = (
            str(result.get("overall_explanation", ""))
            + f" Grounding validation rejected {len(rejected)} candidate(s) "
              "that could not be tied to an exact source substring."
        ).strip()
    return result


def _literal_planet_mentions(text: str) -> list[str]:
    """Return repository-visible planet names explicitly present in source text."""
    found = []
    for planet in ("mercury", "venus", "mars", "jupiter", "saturn"):
        if re.search(rf"\b{planet}\b", text or "", flags=re.IGNORECASE):
            found.append(planet)
    return found


def _canonicalize_measured_relation_kind(
    final: dict[str, Any],
    quote: str,
) -> tuple[dict[str, Any], list[str]]:
    """Choose measured-relation kind mechanically from subject/target topology.

    The model may identify the right bodies but choose appulse vs
    planet_appulse vs planet_planet_appulse incorrectly. Once the literal bodies
    are known, that distinction is deterministic and should not be delegated to
    the model.
    """
    out = dict(final)
    notes: list[str] = []
    visible = {"mercury", "venus", "mars", "jupiter", "saturn"}
    kind = str(out.get("kind") or "")
    planet = str(_noneish(out.get("planet")) or "").strip().casefold()
    target_raw = _noneish(out.get("target_planet"))
    target = str(target_raw or "").strip().casefold()
    star = _noneish(out.get("star"))
    has_measurement = (
        out.get("measurement_cubits") is not None
        or out.get("measurement_fingers") is not None
    )

    if not (kind in MEASURED_KINDS or (has_measurement and (star or target))):
        return out, notes

    qcompact = _source_compact(quote)
    if star:
        desired = (
            "station_appulse"
            if planet in {"mars", "jupiter", "saturn"} and "station" in qcompact
            else "appulse"
        )
        if kind != desired:
            notes.append(f"canonicalized measured relation kind {kind!r} -> {desired!r} from star target topology")
            out["kind"] = desired
        out["target_planet"] = None
        out["sign"] = ""
        if desired != "station_appulse":
            out["station_phase"] = None
        return out, notes

    if target:
        if planet == "moon" and target in visible:
            desired = "planet_appulse"
        elif planet in visible and target in visible and planet != target:
            desired = "planet_planet_appulse"
        else:
            return out, notes
        if kind != desired:
            notes.append(f"canonicalized measured relation kind {kind!r} -> {desired!r} from subject/target body topology")
            out["kind"] = desired
        out["star"] = None
        out["sign"] = ""
        out["station_phase"] = None
    return out, notes


def _structural_salvage_recall_final(
    final: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Apply only mechanically provable fixes to an invalid recalled atom."""
    out = dict(final)
    notes: list[str] = []
    quote = str(out.get("atomic_evidence_quote") or payload.get("evidence_quote") or "")

    # Canonicalize placeholder strings before any structural reasoning.
    for key in ("target_planet", "star", "direction", "station_phase", "event_subtype"):
        before = out.get(key)
        after = _noneish(before)
        if before != after:
            out[key] = after
            notes.append(f"canonicalized placeholder {key}={before!r} to null")
    if _noneish(out.get("sign")) is None and out.get("sign") not in (None, ""):
        notes.append(f"canonicalized placeholder sign={out.get('sign')!r} to empty")
        out["sign"] = ""

    out, topology_notes = _canonicalize_measured_relation_kind(out, quote)
    notes.extend(topology_notes)

    kind = str(out.get("kind") or "")
    planet = str(_noneish(out.get("planet")) or "").strip().casefold()
    target = _noneish(out.get("target_planet"))
    mentions = _literal_planet_mentions(quote)

    if kind == "planet_appulse" and not target:
        if planet == "moon" and len(mentions) == 1:
            out["target_planet"] = mentions[0]
            notes.append(
                "filled missing target_planet from the unique visible planet "
                f"literally named in evidence_quote: {mentions[0]}"
            )
        elif planet != "moon" and "moon" in _source_compact(quote) and len(mentions) == 1:
            out["planet"] = "moon"
            out["target_planet"] = mentions[0]
            notes.append(
                "canonicalized planet_appulse to Moon -> visible planet using "
                f"literal evidence_quote mentions: moon -> {mentions[0]}"
            )

    elif kind == "planet_planet_appulse" and not target:
        if planet in mentions:
            others = [p for p in mentions if p != planet]
            if len(others) == 1:
                out["target_planet"] = others[0]
                notes.append(
                    "filled missing target_planet from the unique other visible "
                    f"planet literally named in evidence_quote: {others[0]}"
                )

    elif kind == "sirius":
        # Sirius is the public source subject; evidence-family bookkeeping
        # is handled separately in tools.statements.evidence_families().
        if out.get("star") is not None:
            out["star"] = None
            notes.append("cleared star for special sirius repository kind")
        out["planet"] = "sirius"
        q = _source_compact(quote)
        if _noneish(out.get("event_subtype")) is None:
            if "acronychal" in q:
                out["event_subtype"] = "acronychal"
                notes.append("derived Sirius subtype=acronychal from literal phrase")
            elif "firstappearance" in q or "rising" in q:
                out["event_subtype"] = "rising"
                notes.append("derived Sirius subtype=rising from literal phrase")
            elif "lastappearance" in q:
                out["event_subtype"] = "setting"
                notes.append("derived Sirius subtype=setting from formal last-appearance phrase")

    return out, notes


def _normalization_failure_skip(
    last_result: dict[str, Any] | None,
    last_error: str | None,
    *,
    atom_rejections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert one persistently invalid recall candidate into an audited SKIP."""
    return {
        "action": "SKIP",
        "skip_reason": "other",
        "explanation": (
            "Candidate rejected after normalization and one validation-repair "
            "attempt because it still could not satisfy repository Statement "
            f"semantics/source grounding. Validation error: {last_error or 'unknown validation error'}"
        ),
        "finals": [],
        "_validation_error": last_error,
        "_invalid_finals": (
            last_result.get("finals") if isinstance(last_result, dict) else None
        ),
        "_atom_rejections": atom_rejections or [],
    }


def _validate_recall_normalization_result(
    result: dict[str, Any],
    payload: dict[str, Any],
    *,
    allow_partial: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Validate/ground every normalized atom; optionally retain valid subset."""
    if result.get("action") == "SKIP":
        result = dict(result)
        result["finals"] = []
        return result, []
    if result.get("action") != "ADD":
        raise ValueError(f"unknown recall normalization action {result.get('action')!r}")

    finals = result.get("finals") or []
    if not finals:
        raise ValueError("ADD requires at least one atomic final")

    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    salvage_all: list[str] = []
    seen = set()

    for index, atom in enumerate(finals):
        atom_error = None
        salvage_notes: list[str] = []

        # First enforce atomic source binding. This is intentionally before any
        # structural salvage so a planet from a neighboring/later clause cannot
        # be used to repair an earlier damaged proposition.
        candidate, binding_errors = _atomic_binding_validate_and_canonicalize(
            dict(atom), payload
        )
        if binding_errors:
            atom_error = "; ".join(binding_errors)

        if atom_error is None:
            candidate, topology_notes = _canonicalize_measured_relation_kind(
                candidate, str(candidate.get("atomic_evidence_quote") or "")
            )
            salvage_notes.extend(topology_notes)
            try:
                candidate = _strict_recall_final(candidate)
            except Exception as exc:
                atom_error = str(exc)
                candidate, salvage_notes = _structural_salvage_recall_final(candidate, payload)
                # Salvage is allowed to correct topology/nulls, but it must still
                # satisfy the same atomic binding gate afterward.
                candidate, rebinding_errors = _atomic_binding_validate_and_canonicalize(
                    candidate, payload
                )
                if rebinding_errors:
                    atom_error = "; ".join(rebinding_errors)
                else:
                    try:
                        candidate = _strict_recall_final(candidate)
                        atom_error = None
                    except Exception as exc2:
                        atom_error = str(exc2)

        if atom_error is not None:
            rejected.append({
                "atom_index": index,
                "error": atom_error,
                "proposed_final": atom,
                "structural_salvage": salvage_notes,
            })
            continue

        for key in ("month", "day", "year_offset"):
            candidate.pop(key, None)
        # In-response deduplication is semantic; two differently worded evidence
        # spans for the same atom should not create duplicate finals.
        core = json.dumps(
            _semantic_atom_without_binding(candidate),
            sort_keys=True, ensure_ascii=False
        )
        if core in seen:
            continue
        seen.add(core)
        validated.append(candidate)
        salvage_all.extend(salvage_notes)

    if rejected and not allow_partial:
        details = " | ".join(
            f"atom {r['atom_index']}: {r['error']}" for r in rejected
        )
        raise ValueError(details)

    if not validated:
        details = " | ".join(
            f"atom {r['atom_index']}: {r['error']}" for r in rejected
        ) or "no valid atomic finals"
        raise ValueError(details)

    fixed = dict(result)
    fixed["finals"] = validated
    fixed["skip_reason"] = None
    if salvage_all:
        fixed["_structural_salvage"] = salvage_all
    if rejected:
        fixed["_atom_rejections"] = rejected
        fixed["explanation"] = (
            str(fixed.get("explanation", ""))
            + f" {len(rejected)} proposed atom(s) were mechanically rejected "
              "for invalid or ungrounded fields; only the validated atoms were retained."
        ).strip()
    return fixed, [r["error"] for r in rejected]


# Candidate-local fallbacks are appended here as JSON lines. main() points it
# at the run's output directory; nothing is ever written into the source tree.
FALLBACK_LOG: dict[str, Path | None] = {"path": None}
_FALLBACK_LOCK = threading.Lock()


def log_fallback(event: str, outcome: str, *, tablet: str | None = None,
                 claim: str | None = None, error: Any = None) -> None:
    STATS.add_failure(f"fallback:{event}:{outcome}")
    path = FALLBACK_LOG["path"]
    if path is None:
        return
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event, "outcome": outcome, "tablet": tablet, "claim": claim,
        "error": None if error is None else str(error)[:1000],
    }
    with _FALLBACK_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_once(client, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one claim: a request, then at most one repair or regeneration.

    Malformed or truncated output is retried with a larger budget and then
    regenerated once. An invalid ADD gets one validation-repair request. If the
    candidate still fails, an audited SKIP is returned. ModelRequestRejected
    propagates to the caller.
    """
    last_error = None
    last_result: dict[str, Any] | None = None
    last_rejections: list[dict[str, Any]] = []
    request_payload = dict(payload)
    instructions = RECALL_NORMALIZE_PROMPT
    output_failure = False

    for attempt in range(2):
        result, error = _structured_request(
            client, model, instructions=instructions, payload=request_payload,
            fmt=RECALL_NORMALIZE_FORMAT, budgets=(4500, 9000),
            stage="recall_normalize" if attempt == 0 else "recall_normalize_repair")
        if result is None:
            last_error, output_failure = error, True
        else:
            output_failure = False
            last_result = result
            # A model-declared unsupported astronomical claim is a valid,
            # auditable outcome and is never forced into a repository kind.
            if result.get("action") == "SKIP":
                result["finals"] = []
                return result
            try:
                # First pass: any bad atom triggers one focused repair so a
                # coordinated candidate gets a chance to recover all atoms.
                fixed, _ = _validate_recall_normalization_result(
                    result, payload, allow_partial=(attempt == 1))
                return fixed
            except (ValueError, TypeError, KeyError) as exc:
                last_error = str(exc)
                last_rejections = result.get("_atom_rejections", [])

        if attempt == 0:
            request_payload = dict(payload)
            request_payload["validation_error"] = last_error
            if output_failure:
                instructions = (
                    RECALL_NORMALIZE_PROMPT
                    + "\n\nFORMAT REGENERATION: The previous response was truncated, "
                      "incomplete, or invalid JSON. Re-read the ORIGINAL source payload "
                      "and regenerate the answer from scratch. Return one complete "
                      "response matching the required JSON schema."
                )
            else:
                request_payload["previous_invalid_response"] = last_result
                instructions = (
                    RECALL_NORMALIZE_PROMPT
                    + "\n\nVALIDATION REPAIR: Your previous ADD contained one or more "
                      "invalid or SOURCE-UNGROUNDED atomic finals. Correct each atom "
                      "independently. Coordinated clauses must still yield all supported "
                      "atomic conjuncts. For EVERY atom provide the smallest exact "
                      "atomic_evidence_quote plus a correct DIRECT/COORDINATED/ANAPHORIC "
                      "subject_binding. Resolve the GRAMMATICAL source subject first; a Normal Star "
                      "may be the grammatical antecedent even though the repository stores "
                      "the visible planet in its planet field. Remove any field not jointly "
                      "supported inside that atomic clause (except the one locally verified "
                      "ANAPHORIC role). If a genuine astronomical proposition cannot be "
                      "represented by the supported kinds, return SKIP with "
                      "skip_reason='unsupported_but_astronomical'. Never invent a sign, "
                      "star, planet, target, measurement, or direction merely to satisfy "
                      "the schema. Validation error: "
                      f"{last_error}"
                )

    failure = _normalization_failure_skip(last_result, last_error, atom_rejections=last_rejections)
    if output_failure:
        failure["_model_output_failure"] = last_error
        failure["explanation"] = (
            "Candidate rejected because the model failed to return complete valid "
            "structured JSON after retries/regeneration. No malformed output was "
            "accepted into the scientific pipeline. " + failure["explanation"]
        )
    return failure


def _trim_context(payload: dict[str, Any], keep: int) -> dict[str, Any]:
    out = dict(payload)
    out["context_before"] = payload.get("context_before", "")[-keep:] if keep else ""
    out["context_after"] = payload.get("context_after", "")[:keep] if keep else ""
    return out


def _one_recall_normalize_request(
    client,
    model: str,
    payload: dict[str, Any],
    *,
    tablet: str | None = None,
) -> dict[str, Any]:
    """Normalize one recalled claim; no single candidate can stop the run.

    A mechanically provable anaphoric station is recovered without a call. A
    refused request (HTTP 400, for example a content filter) is retried with
    160 characters of context, then with the quote alone, and is otherwise
    recorded as an audited SKIP. Each fallback is logged.
    """
    station = _deterministic_anaphoric_station_add(payload)
    if station is not None:
        # Deterministic recovery must pass the same binding and validation
        # gates as a model ADD; if it does not, the model is asked instead.
        try:
            return _validate_recall_normalization_result(station, payload, allow_partial=False)[0]
        except ValueError as exc:
            log_fallback("deterministic_station_invalid", "model_normalization",
                         tablet=tablet, claim=payload.get("discovered_atomic_claim"), error=exc)

    sirius = _deterministic_generic_sirius_skip(payload)
    if sirius is not None:
        return sirius

    for label, request in (("full_context", payload),
                           ("reduced_context", _trim_context(payload, 160)),
                           ("evidence_only", _trim_context(payload, 0))):
        try:
            result = _normalize_once(client, model, request)
        except ModelRequestRejected as exc:
            log_fallback(f"request_rejected_{label}", "retry", tablet=tablet,
                         claim=payload.get("discovered_atomic_claim"), error=exc)
            continue
        if label != "full_context":
            log_fallback(f"request_rejected_{label}", "success", tablet=tablet,
                         claim=payload.get("discovered_atomic_claim"))
            result["_context_fallback"] = label
        return result
    skipped = _normalization_failure_skip(
        None, "request refused with full, reduced and evidence-only context",
        atom_rejections=[])
    skipped["_request_rejected_skip"] = True
    log_fallback("request_rejected_all", "audited_skip", tablet=tablet,
                 claim=payload.get("discovered_atomic_claim"))
    return skipped


def _semantic_core(st: S.Statement) -> tuple[Any, ...]:
    subtype = infer_event_subtype(st)
    return (
        st.kind,
        st.planet,
        st.target_planet,
        st.star,
        st.sign,
        None if st.amount is None else round(float(st.amount), 8),
        st.direction,
        st.station_phase,
        subtype,
    )


def _source_spans_overlap(a: S.Statement, b: S.Statement) -> bool:
    if min(a.source_start, b.source_start) < 0:
        return False
    return a.source_start < b.source_end and b.source_start < a.source_end


def _is_recall_duplicate(candidate: S.Statement, existing: list[S.Statement]) -> bool:
    core = _semantic_core(candidate)
    for st in existing:
        if _semantic_core(st) != core:
            continue
        # Same atomic semantics in the same/overlapping local source is a duplicate.
        if _source_spans_overlap(candidate, st):
            return True
        if (
            candidate.source_start >= 0 and st.source_start >= 0
            and abs(candidate.source_start - st.source_start) <= 80
        ):
            return True
    return False


def _continues_existing_eclipse(text: str, candidate: S.Statement,
                                existing: list[S.Statement]) -> bool:
    """Whether an eclipse addition only continues an eclipse report already held.

    "lunar eclipse ; when it began ... ( the moon ) became eclipsed in Libra"
    reports one event. A later mention of the same eclipse type counts as the
    same event when it follows within 400 characters with no day or month
    heading in between.
    """
    if candidate.kind != "eclipse":
        return False
    for st in existing:
        if st.kind != "eclipse" or st.planet != candidate.planet:
            continue
        gap = text[st.source_end:candidate.source_start]
        if 0 <= candidate.source_start - st.source_start <= 400 and not any(
                S.day_matches(gap, 0, len(gap))) and not any(S.month_matches(gap)):
            return True
    return False


def _existing_for_chunk(
    source: BlindText,
    statements: list[S.Statement],
    chunk: dict[str, int],
) -> list[dict[str, Any]]:
    lo, hi = chunk["context_start"], chunk["context_end"]
    out = []
    for st in statements:
        if st.source_end < lo or st.source_start > hi:
            continue
        out.append(_compact_statement(st, source))
    return out


def _locate_in_core(masked: str, quote: str, core_start: int,
                    core_end: int) -> tuple[int, int] | None:
    """The span of a discovery quote that starts inside the window's core."""
    for start, end in _quote_spans(quote, masked):
        if core_start <= start < core_end:
            return start, end
    return None


def run_raw_recall(
    tablets: pd.DataFrame,
    base_statements: dict[str, list[S.Statement]],
    *,
    model: str,
    cache_path: Path,
    workers: int,
    force: bool,
    core_chars: int,
    context_chars: int,
    overlap_chars: int = 350,
    tablet_ids: set[str] | None = None,
    dropped_statements: dict[str, list[S.Statement]] | None = None,
    route: str = DEFAULT_ROUTE,
    max_workers: int | None = None,
) -> tuple[dict[str, list[S.Statement]], list[dict[str, Any]]]:
    """Search the text for clauses missing from the reviewed parser output.

    Windows are planned in original offsets and sent masked. A parser clause
    that the review dropped counts as existing: it is listed to the model with
    the others, and a search proposal that restates it is recorded as a
    conflict with the review rather than added back. Every accepted passage
    is mapped back to original offsets before the Statement is built.
    """
    text_by_tablet = tablets.set_index("tablet").text.to_dict()
    selected_ids = set(text_by_tablet) if tablet_ids is None else set(tablet_ids)
    dropped_statements = dropped_statements or {}
    max_workers = max_workers or workers
    cache = {} if force else load_cache(cache_path)
    records_by_key = dict(cache)
    sources: dict[str, BlindText] = {}

    chunks_to_run = []
    chunk_meta = []
    for tablet in sorted(selected_ids):
        text = text_by_tablet.get(tablet)
        if not isinstance(text, str):
            continue
        source = sources[tablet] = blind(text)
        known = base_statements.get(tablet, []) + dropped_statements.get(tablet, [])
        for chunk in _planned_recall_chunks(
            text, known, core_chars, context_chars, overlap_chars
        ):
            existing = _existing_for_chunk(source, known, chunk)
            key = _recall_discovery_key(tablet, text, chunk, existing, model)
            meta = (tablet, chunk, existing, key)
            chunk_meta.append(meta)
            if key not in records_by_key:
                chunks_to_run.append(meta)

    print(
        f"raw recall: {len(chunk_meta):,} source chunks; "
        f"{len(chunks_to_run):,} require discovery calls; "
        f"{len(chunk_meta) - len(chunks_to_run):,} reusable from recall cache"
    )

    def masked_core(tablet: str, chunk: dict[str, int]) -> tuple[int, int, int, int]:
        source = sources[tablet]
        return (source.to_masked_start(chunk["context_start"]),
                source.to_masked_start(chunk["core_start"]),
                source.to_masked_start(chunk["core_end"]),
                source.to_masked_end(chunk["context_end"]))

    if chunks_to_run:
        client = shared_client(route, workers, max_workers)

        def discover(meta):
            tablet, chunk, existing, key = meta
            masked = sources[tablet].masked
            c0, k0, k1, c1 = masked_core(tablet, chunk)
            payload = {
                "record": record_token(tablet),
                "context_before": masked[c0:k0],
                "core_text": masked[k0:k1],
                "context_after": masked[k1:c1],
                "existing_structured_statements": existing,
            }
            with call_ledger() as calls:
                result = _one_recall_discovery_request(client, model, payload)
            return {
                "cache_key": key,
                "record_type": "discovery",
                "prompt_version": RECALL_DISCOVERY_PROMPT_VERSION,
                "model": model,
                "tablet": tablet,
                "record_token": payload["record"],
                "core_start": chunk["core_start"],
                "core_end": chunk["core_end"],
                "result": result,
                "calls": calls,
                "call_totals": _ledger_totals(calls),
            }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for number, record in enumerate(_run_parallel(discover, chunks_to_run, max_workers), start=1):
            records_by_key[record["cache_key"]] = record
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if number % 20 == 0 or number == len(chunks_to_run):
                print(f"  recall discovery {number:,}/{len(chunks_to_run):,} chunks")

    # Gather grounded candidate claims whose evidence starts in each chunk's
    # core. Offsets here are in the masked text the model saw.
    candidates = []
    raw_seen = set()
    for tablet, chunk, existing, key in chunk_meta:
        record = records_by_key[key]
        _c0, k0, k1, _c1 = masked_core(tablet, chunk)
        for claim in record["result"].get("claims", []):
            quote = claim.get("evidence_quote", "")
            span = _locate_in_core(sources[tablet].masked, quote, k0, k1)
            if span is None:
                # It belongs to overlap/context and will be owned by another core.
                continue
            raw_key = (tablet, span[0], quote, claim.get("claim", "").casefold())
            if raw_key in raw_seen:
                continue
            raw_seen.add(raw_key)
            candidates.append((tablet, span[0], span[1], claim))

    print(f"raw recall: {len(candidates):,} candidate missed atomic claims discovered")

    pad = 650

    def normalize_payload(tablet: str, start: int, end: int, claim: dict[str, Any]) -> dict[str, Any]:
        masked = sources[tablet].masked
        return {
            "record": record_token(tablet),
            "discovered_atomic_claim": claim.get("claim", ""),
            "evidence_quote": claim.get("evidence_quote", ""),
            "context_before": masked[max(0, start - pad):start],
            "context_after": masked[end:min(len(masked), end + pad)],
            "source_measurement": source_measurement_components(claim.get("evidence_quote", "")),
        }

    normalize_to_run = []
    normalize_meta = []
    for tablet, start, end, claim in candidates:
        key = _recall_normalize_key(tablet, text_by_tablet[tablet], claim, start, model)
        meta = (tablet, start, end, claim, key)
        normalize_meta.append(meta)
        if key not in records_by_key:
            normalize_to_run.append(meta)

    if normalize_to_run:
        client = shared_client(route, workers, max_workers)

        def normalize(meta):
            tablet, start, end, claim, key = meta
            payload = normalize_payload(tablet, start, end, claim)
            with call_ledger() as calls:
                result = _one_recall_normalize_request(client, model, payload, tablet=tablet)
            return {
                "cache_key": key,
                "record_type": "normalization",
                "prompt_version": RECALL_NORMALIZE_PROMPT_VERSION,
                "model": model,
                "tablet": tablet,
                "record_token": payload["record"],
                "masked_source_start": start,
                "masked_source_end": end,
                "source_start": sources[tablet].to_original_start(start),
                "source_end": sources[tablet].to_original_end(end),
                "claim": claim,
                "result": result,
                "calls": calls,
                "call_totals": _ledger_totals(calls),
            }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for number, record in enumerate(_run_parallel(normalize, normalize_to_run, max_workers), start=1):
            records_by_key[record["cache_key"]] = record
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if number % 20 == 0 or number == len(normalize_to_run):
                print(f"  recall normalization {number:,}/{len(normalize_to_run):,} candidates")

    additions: dict[str, list[S.Statement]] = {tablet: [] for tablet in selected_ids}
    audit_rows: list[dict[str, Any]] = []

    # Preserve quote-grounding failures in the audit instead of aborting the run.
    for tablet, chunk, existing, key in chunk_meta:
        record = records_by_key[key]
        for rejected in record.get("result", {}).get("grounding_rejections", []):
            audit_rows.append({
                "tablet": tablet,
                "source_start": None,
                "source_end": None,
                "discovered_claim": rejected.get("claim", ""),
                "evidence_quote": rejected.get("evidence_quote", ""),
                "discovery_explanation": rejected.get("repair_explanation", ""),
                "normalization_action": None,
                "normalization_skip_reason": rejected.get("reason", "ungrounded_evidence"),
                "normalization_explanation": "",
                "normalization_validation_error": None,
                "normalization_structural_salvage": "[]",
                "status": "rejected_ungrounded_evidence",
                "final_statement_json": "",
            })

    for tablet, start, end, claim, key in normalize_meta:
        result = records_by_key[key]["result"]
        source = sources[tablet]
        text = source.original
        context = normalize_payload(tablet, start, end, claim)
        base_row = {
            "tablet": tablet,
            "source_start": source.to_original_start(start),
            "source_end": source.to_original_end(end),
            "discovered_claim": claim.get("claim", ""),
            "evidence_quote": claim.get("evidence_quote", ""),
            "discovery_explanation": claim.get("explanation", ""),
            "normalization_action": result.get("action"),
            "normalization_skip_reason": result.get("skip_reason"),
            "normalization_explanation": result.get("explanation", ""),
            "normalization_validation_error": result.get("_validation_error"),
            "normalization_structural_salvage": json.dumps(result.get("_structural_salvage", []), ensure_ascii=False),
            "normalization_atom_rejections": json.dumps(result.get("_atom_rejections", []), ensure_ascii=False),
            "atom_index": None,
            "atomic_evidence_quote": "",
            "subject_binding": "",
            "antecedent_quote": "",
            "atomic_source_start": None,
            "atomic_source_end": None,
            "model_omitted": None,
            "parser_unwatched": None,
            "status": None,
            "final_statement_json": "",
        }
        if result.get("action") != "ADD":
            row = dict(base_row)
            if result.get("skip_reason") == "unsupported_but_astronomical":
                row["status"] = "unsupported_but_astronomical"
            else:
                row["status"] = "skipped_by_normalizer"
            audit_rows.append(row)
            continue

        finals = result.get("finals") or []
        if not finals:
            row = dict(base_row)
            row["status"] = "skipped_by_normalizer"
            row["normalization_validation_error"] = (
                row.get("normalization_validation_error") or "ADD returned no validated atomic finals"
            )
            audit_rows.append(row)
            continue

        for atom_index, final in enumerate(finals):
            row = dict(base_row)
            row["atom_index"] = atom_index
            row["atomic_evidence_quote"] = str(final.get("atomic_evidence_quote") or "")
            row["subject_binding"] = str(final.get("subject_binding") or "")
            row["antecedent_quote"] = str(final.get("antecedent_quote") or "")
            row["model_omitted"] = final.get("omitted")
            try:
                # Re-run every gate at the final acceptance boundary, atom by
                # atom: cached output cannot bypass them, and one invalid atom
                # does not discard the valid atoms of the same candidate.
                checked, binding_errors = _atomic_binding_validate_and_canonicalize(
                    dict(final), context)
                if binding_errors:
                    raise ValueError("; ".join(binding_errors))
                atomic_ground = _ground_quote_to_exact_source(
                    str(checked.get("atomic_evidence_quote") or ""),
                    claim.get("evidence_quote", ""),
                )
                if atomic_ground is None:
                    raise ValueError("atomic_evidence_quote lost exact grounding before final acceptance")
                exact_atomic, atomic_local_start, atomic_local_end = atomic_ground
                atomic_start = source.to_original_start(start + atomic_local_start)
                atomic_end = source.to_original_end(start + atomic_local_end)
                row["atomic_source_start"] = atomic_start
                row["atomic_source_end"] = atomic_end
                row["atomic_evidence_quote"] = text[atomic_start:atomic_end]

                checked, _topology_notes = _canonicalize_measured_relation_kind(
                    checked, exact_atomic
                )
                if checked.get("kind") == "sirius":
                    subtype = checked.get("event_subtype")
                    if not _recognized_sirius_literal(exact_atomic, subtype):
                        raise ValueError(
                            "repository sirius kind requires a formal Sirius first/last "
                            "appearance or acronychal-rising phrase; generic Sirius "
                            "rise/set timing is unsupported_but_astronomical"
                        )
                checked = _strict_recall_final(checked)
                st = _make_recalled_statement(text, atomic_start, atomic_end, checked)
            except (ValueError, TypeError, KeyError) as exc:
                row["status"] = "rejected_at_final_grounding_gate"
                row["normalization_validation_error"] = str(exc)
                row["final_statement_json"] = json.dumps(final, ensure_ascii=False, sort_keys=True)
                audit_rows.append(row)
                continue

            row["parser_unwatched"] = bool(st.omitted)
            row["final_statement_json"] = json.dumps(jsonable_statement(st), ensure_ascii=False, sort_keys=True)
            all_existing = base_statements.get(tablet, []) + additions.setdefault(tablet, [])
            if (_is_recall_duplicate(st, all_existing)
                    or _continues_existing_eclipse(text, st, all_existing)):
                row["status"] = "duplicate"
            elif _is_recall_duplicate(st, dropped_statements.get(tablet, [])):
                # The review removed this clause with a checked, local
                # contradiction. The search may not quietly restore it.
                row["status"] = "conflicts_with_review_drop"
                log_fallback("search_readds_dropped_clause", "not_added", tablet=tablet,
                             claim=claim.get("claim", ""))
            else:
                additions[tablet].append(st)
                row["status"] = "added"
            audit_rows.append(row)

    statuses = pd.Series([r["status"] for r in audit_rows], dtype="object").value_counts().to_dict()
    added_count = sum(len(v) for v in additions.values())
    print(
        f"raw recall final: {added_count:,} added statements; "
        + "; ".join(f"{n:,} {status}" for status, n in sorted(statuses.items()))
    )
    return additions, audit_rows


def merge_recall_additions(
    base: dict[str, list[S.Statement]],
    additions: dict[str, list[S.Statement]],
    source_texts: dict[str, str] | None = None,
) -> dict[str, list[S.Statement]]:
    out = {tablet: list(sts) for tablet, sts in base.items()}
    for tablet, sts in additions.items():
        out.setdefault(tablet, []).extend(sts)
        out[tablet].sort(key=lambda st: (st.source_start, st.source_end, st.kind, st.planet))
    if source_texts is not None:
        out = {tablet: S.bind_joint_positions(source_texts[tablet], sts)
               for tablet, sts in out.items()}
    return out


def write_recall_audit(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"LLM raw-recall audit: {len(rows):,} candidates -> {path}")


def _norm_compare(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _statement_matches_gold(st: S.Statement, expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key == "amount":
            if st.amount is None or not math.isclose(float(st.amount), float(value), abs_tol=1e-8):
                return False
        elif key == "star":
            if _norm_compare(st.star).replace("alpha ", "α ") != _norm_compare(value).replace("alpha ", "α "):
                return False
        else:
            actual = getattr(st, key)
            if _norm_compare(actual) != _norm_compare(value):
                return False
    return True


def _score_gold(statements: list[S.Statement], expected: list[dict[str, Any]]) -> tuple[int, list[bool]]:
    used = set()
    hits = []
    for gold in expected:
        found = False
        for i, st in enumerate(statements):
            if i in used:
                continue
            if _statement_matches_gold(st, gold):
                used.add(i)
                found = True
                break
        hits.append(found)
    return sum(hits), hits


def run_false_negative_benchmark(
    extracted_all: dict[str, tuple[str, list[S.Statement]]],
    combined: dict[str, list[S.Statement]],
    recall_rows: list[dict[str, Any]] | None = None,
) -> bool:
    total_gold = 0
    parser_hits = 0
    final_hits = 0
    full_cases = 0
    print("\nKnown false-negative end-to-end benchmark")
    print("=" * 72)
    for case in FALSE_NEGATIVE_CASES:
        tablet = case["tablet"]
        expected = case["expected"]
        parser_sts = extracted_all.get(tablet, ("", []))[1]
        final_sts = combined.get(tablet, [])
        p_hits, _ = _score_gold(parser_sts, expected)
        f_hits, vector = _score_gold(final_sts, expected)
        total_gold += len(expected)
        parser_hits += p_hits
        final_hits += f_hits
        full_cases += int(f_hits == len(expected))
        status = "PASS" if f_hits == len(expected) else "FAIL"
        print(f"{tablet} — {case['label']}")
        print(f"  deterministic parser: {p_hits}/{len(expected)}")
        print(f"  final main pipeline:   {f_hits}/{len(expected)}  [{status}]")
        if f_hits != len(expected):
            print(f"  gold hit vector: {vector}")
    print("=" * 72)
    print(f"Deterministic baseline: {parser_hits}/{total_gold}")
    print(f"Final main-pipeline recall: {final_hits}/{total_gold}")
    print(f"Cases fully recovered: {full_cases}/{len(FALSE_NEGATIVE_CASES)}")

    regression_ok = True
    print("\nRecall semantic regression checks")
    print("-" * 72)
    for case in RECALL_SEMANTIC_REGRESSION_CASES:
        tablet = case["tablet"]
        expected = case["expected"]
        final_sts = combined.get(tablet, [])
        hits, vector = _score_gold(final_sts, expected)
        status = "PASS" if hits == len(expected) else "FAIL"
        regression_ok = regression_ok and (hits == len(expected))
        print(f"{tablet} — {case['label']}")
        print(f"  final main pipeline:   {hits}/{len(expected)}  [{status}]")
        if hits != len(expected):
            print(f"  regression hit vector: {vector}")
    print("-" * 72)

    sirius_generic_ok = True
    if recall_rows is not None:
        # Only candidates that actually reached normalization test the ontology
        # boundary. Discovery-stage grounding rejections may contain a broad
        # quote with the words "Sirius set", but normalization_action=None means
        # they never entered the normalization/Statement path.
        sirius_set_rows = [
            r for r in recall_rows
            if r.get("tablet") == "X300993"
            and r.get("normalization_action") is not None
            and _generic_sirius_setting_literal(str(r.get("evidence_quote") or ""))
        ]
        sirius_generic_ok = bool(sirius_set_rows) and all(
            r.get("status") == "unsupported_but_astronomical"
            for r in sirius_set_rows
        )
        print("\nSirius generic-setting ontology guard")
        print("-" * 72)
        status = "PASS" if sirius_generic_ok else "FAIL"
        print(
            "generic 'Sirius set' remains distinct from scheme-date Sirius: "
            f"{len(sirius_set_rows)} normalized candidate(s)  [{status}]"
        )
        print("-" * 72)

    atomic_binding_ok = True
    if recall_rows is not None:
        added_rows = [r for r in recall_rows if r.get("status") == "added"]
        bad_rows: list[str] = []
        for row in added_rows:
            binding = str(row.get("subject_binding") or "")
            atomic_quote = str(row.get("atomic_evidence_quote") or "")
            antecedent_quote = str(row.get("antecedent_quote") or "")
            try:
                final_json = json.loads(str(row.get("final_statement_json") or "{}"))
            except Exception:
                final_json = {}
            planet = str(final_json.get("planet") or "").casefold()
            kind = str(final_json.get("kind") or "")
            star = final_json.get("star")
            target_planet = final_json.get("target_planet")

            if not atomic_quote or binding not in {"DIRECT", "COORDINATED", "ANAPHORIC"}:
                bad_rows.append(f"{row.get('tablet')} atom {row.get('atom_index')}: missing/invalid binding metadata")
                continue
            if kind not in {"season", "sirius", "eclipse"} and planet in {"mercury", "venus", "mars", "jupiter", "saturn", "moon"}:
                if binding in {"DIRECT", "COORDINATED"} and not _literal_field_present(atomic_quote, planet):
                    bad_rows.append(f"{row.get('tablet')} atom {row.get('atom_index')}: canonical planet/body absent from atomic evidence")
                if binding == "ANAPHORIC":
                    role_values = {
                        "planet": final_json.get("planet"),
                        "star": star,
                        "target_planet": target_planet,
                    }
                    ant_roles = [
                        role for role, value in role_values.items()
                        if value and _literal_field_present(antecedent_quote, value)
                    ]
                    if len(ant_roles) != 1:
                        bad_rows.append(
                            f"{row.get('tablet')} atom {row.get('atom_index')}: "
                            f"anaphoric antecedent binds {ant_roles or 'no'} semantic role"
                        )
                    if not re.search(r"\b(?:it|this|that|same)\b", _normalize_grounding_text(atomic_quote).casefold()):
                        bad_rows.append(f"{row.get('tablet')} atom {row.get('atom_index')}: anaphoric atomic evidence lacks local pronoun")

        atomic_binding_ok = not bad_rows
        print("\nAtomic evidence / subject-binding acceptance gate")
        print("-" * 72)
        status = "PASS" if atomic_binding_ok else "FAIL"
        print(f"validated added recall atoms: {len(added_rows) - len(bad_rows)}/{len(added_rows)}  [{status}]")
        for detail in bad_rows[:10]:
            print(f"  {detail}")
        if len(bad_rows) > 10:
            print(f"  ... plus {len(bad_rows) - 10} more binding failures")
        print("-" * 72)

    return (
        final_hits == total_gold
        and regression_ok
        and sirius_generic_ok
        and atomic_binding_ok
    )


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Table 5
# ---------------------------------------------------------------------------


def build_table5(
    tablets_text_only: pd.DataFrame,
    reviewed: dict[str, list[S.Statement]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # IMPORTANT: published years are loaded here, only after LLM interpretation
    # has already been frozen in `reviewed`.
    references = pd.read_parquet(
        HERE / "data" / "corpus" / "tablets.parquet",
        columns=["tablet", "bce_first", "bce_second"],
    ).set_index("tablet")
    text_by_tablet = tablets_text_only.set_index("tablet").text.to_dict()

    fact_rows = []
    for tablet, sts in reviewed.items():
        text = text_by_tablet.get(tablet)
        if not isinstance(text, str):
            continue
        months = S.months_in(text)
        year = references.loc[tablet, "bce_first"] if tablet in references.index else None
        for index, st in enumerate(sts):
            if pd.isna(year):
                verdict = "not_evaluated"
                detail = "tablet has no independent reference year"
            else:
                checked = S.check(st, yr.bce_to_astro(int(year)), months)
                verdict = checked.result
                detail = checked.detail
            fact_rows.append({
                "tablet": tablet,
                "statement_index": index,
                "kind": st.kind,
                "planet": st.planet,
                "parser_rule": st.parser_rule,
                "restored": bool(st.restored),
                "omitted": bool(st.omitted),
                "semantic_tier": "decisive" if S.is_decisive(st) else "supporting",
                "forward_verdict": verdict,
                "forward_detail": detail,
            })

    facts = pd.DataFrame(fact_rows)
    rows = []
    for kind in FACT_ORDER:
        group = facts[facts.kind.eq(kind)]
        if group.empty:
            continue
        evaluated = group[group.forward_verdict.ne("not_evaluated")]
        testable = evaluated[evaluated.forward_verdict.isin(["agree", "contradict"])]
        agree = int(testable.forward_verdict.eq("agree").sum())
        contradict = int(testable.forward_verdict.eq("contradict").sum())
        untestable = int(evaluated.forward_verdict.eq("untestable").sum())
        rows.append({
            "Observation type": FACT_LABEL[kind],
            "Clauses": len(group),
            "Testable": len(testable),
            "Agree": agree,
            "Contradict": contradict,
            "Untestable": untestable,
            "Agreement %": (100.0 * agree / len(testable)) if len(testable) else None,
            "kind": kind,
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["Agreement %"] = table["Agreement %"].round(1)
    print(f"Table 5: {len(facts):,} reviewed clauses across {facts.tablet.nunique():,} tablets")
    return table, facts


# ---------------------------------------------------------------------------
# Audit output
# ---------------------------------------------------------------------------


def write_change_audit(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for record in records:
        initial_action = record.get("initial_action", record["action"])
        overrides = record.get("code_field_overrides") or {}
        if record["action"] == "KEEP" and initial_action == "KEEP" and not overrides:
            continue
        verification = record.get("drop_verification") or {}
        modify_verification = record.get("modify_verification") or {}
        rows.append({
            "tablet": record["tablet"],
            "statement_index": record["statement_index"],
            "initial_action": initial_action,
            "action": record["action"],
            "drop_verification_verdict": verification.get("verdict"),
            "drop_verification_reason": verification.get("reason"),
            "drop_contradicted_field": verification.get("contradicted_field"),
            "drop_source_value": verification.get("source_value"),
            "drop_code_rejection": "; ".join(verification.get("code_rejection") or []),
            "drop_verification_explanation": verification.get("explanation"),
            "modify_verification_json": json.dumps(
                modify_verification, ensure_ascii=False, sort_keys=True
            ) if modify_verification else "",
            "code_field_overrides_json": json.dumps(
                overrides, ensure_ascii=False, sort_keys=True) if overrides else "",
            "model_output_failure": record.get("model_output_failure"),
            "model": record["model"],
            "explanation": record["explanation"],
            "evidence_quote": record["evidence_quote"],
            "parser_matched_source": record.get("parser_matched_source", ""),
            "original_statement_json": json.dumps(
                record["original_statement"], ensure_ascii=False, sort_keys=True),
            "final_statement_json": json.dumps(
                record["final_statement"], ensure_ascii=False, sort_keys=True),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"LLM intervention audit: {len(rows):,} rows -> {path}")


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-")


def default_out_dir(model: str) -> Path:
    return HERE / "data" / "clauses" / f"model_review_{model_slug(model)}"


def _record_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calls, tokens and cost stored in cache records, new and reused."""
    calls = [c for r in records for c in (r.get("calls") or [])]
    return _ledger_totals(calls)


def write_run_summary(path: Path, *, args, tablets_run: list[str] | None,
                      review_records: list[dict[str, Any]],
                      recall_rows: list[dict[str, Any]],
                      recall_records: list[dict[str, Any]]) -> dict[str, Any]:
    client = _CLIENT.get("client")
    actions = pd.Series([r["action"] for r in review_records], dtype="object")
    initial = pd.Series([r.get("initial_action", r["action"]) for r in review_records], dtype="object")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "route": args.route,
        "scope": args.scope,
        "endpoint": getattr(client, "base_url", None),
        "prompt_versions": {"review": PROMPT_VERSION,
                            "recall_discovery": RECALL_DISCOVERY_PROMPT_VERSION,
                            "recall_normalize": RECALL_NORMALIZE_PROMPT_VERSION},
        "tablets": tablets_run,
        "review": {
            "clauses": len(review_records),
            "final_actions": actions.value_counts().to_dict(),
            "first_pass_actions": initial.value_counts().to_dict(),
            "code_field_overrides": sum(bool(r.get("code_field_overrides")) for r in review_records),
            "model_output_failures": sum(bool(r.get("model_output_failure")) for r in review_records),
        },
        "recall": pd.Series([r["status"] for r in recall_rows], dtype="object")
                    .value_counts().to_dict(),
        "this_process_calls": STATS.summary(getattr(client, "limiter", None)),
        "all_cached_record_calls": _record_totals(review_records + recall_records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    print(f"run summary -> {path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--route", choices=sorted(ROUTES), default=DEFAULT_ROUTE,
                        help="litellm: the course proxy; gemini: the Gemini API fallback")
    parser.add_argument("--workers", type=int, default=8,
                        help="requests in flight at the start; grows while no 429 arrives")
    parser.add_argument("--max-workers", type=int, default=32,
                        help="ceiling on requests in flight")
    parser.add_argument("--scope", choices=sorted(SCOPES), default="adart-i-iii",
                        help="records to review; the paper reports ADART I-III")
    parser.add_argument("--out-dir", type=Path,
                        help="output directory; default data/clauses/model_review_<model>")
    parser.add_argument("--force", action="store_true",
                        help="ignore prior review cache and call the model again")
    parser.add_argument("--dry-run", action="store_true",
                        help="extract/count statements but make no API calls")
    parser.add_argument("--pilot", type=int,
                        help="review this many uncached statements, cache them, then exit")
    parser.add_argument("--tablet", action="append", default=[],
                        help="review and search only this record; repeatable")
    parser.add_argument("--contains", action="append", default=[],
                        help="review only clauses whose +/-600-char context contains this text; repeatable")
    parser.add_argument("--kind", action="append", default=[],
                        help="review only one statement kind (e.g. season, appulse); repeatable")
    parser.add_argument("--parser-rule", action="append", default=[],
                        help="review only one parser_rule (e.g. coordinated_sign); repeatable")
    parser.add_argument("--false-negative-test", action="store_true",
                        help="run the search only on the seven known false-negative tablets and score the hidden 9-atom benchmark")
    parser.add_argument("--no-recall", action="store_true",
                        help="ablation: skip the search for missed clauses")
    parser.add_argument("--force-recall", action="store_true",
                        help="ignore prior search cache and call the model again")
    parser.add_argument("--recall-core-chars", type=int, default=2400,
                        help="source characters in each search core")
    parser.add_argument("--recall-context-chars", type=int, default=700,
                        help="source context supplied on each side of a search core")
    parser.add_argument("--recall-overlap-chars", type=int, default=350,
                        help="overlap between neighboring search cores")
    args = parser.parse_args()

    out = (args.out_dir or default_out_dir(args.model)).resolve()
    paths = {
        "cache": out / "review_cache.jsonl",
        "recall_cache": out / "recall_cache.jsonl",
        "recall_out": out / "search_proposals.csv",
        "changes_out": out / "review_changes.csv",
        "summary": out / "run_summary.json",
    }
    FALLBACK_LOG["path"] = out / "fallbacks.jsonl"
    concurrency = {"route": args.route, "workers": args.workers,
                   "max_workers": max(args.workers, args.max_workers)}
    print(f"model {args.model} via {args.route}; outputs in {out}")

    # Phase A: answer-blind source material only.
    tablets = load_tablets("all" if args.false_negative_test else args.scope)
    extracted_all = extract_all(tablets)
    total_all = sum(len(sts) for _, sts in extracted_all.values())
    print(f"deterministic parser extracted {total_all:,} statements from {len(tablets):,} tablets")

    def recall(tablet_frame, base, dropped, ids=None):
        return run_raw_recall(
            tablets=tablet_frame, base_statements=base, model=args.model,
            cache_path=paths["recall_cache"], force=args.force_recall,
            core_chars=args.recall_core_chars, context_chars=args.recall_context_chars,
            overlap_chars=args.recall_overlap_chars, tablet_ids=ids,
            dropped_statements=dropped, **concurrency)

    # End-to-end search benchmark on the seven known false-negative tablets.
    # The hidden gold expectations are used only after model output.
    if args.false_negative_test:
        wanted = {case["tablet"] for case in FALSE_NEGATIVE_CASES}
        base = {tablet: list(sts) for tablet, (_, sts) in extracted_all.items()
                if tablet in wanted}
        if args.dry_run:
            print("false-negative benchmark dry-run: no API calls")
            run_false_negative_benchmark(extracted_all, base)
            return 0
        additions, recall_rows = recall(tablets[tablets.tablet.isin(wanted)].copy(),
                                        base, {}, wanted)
        combined = merge_recall_additions(
            base, additions, dict(zip(tablets.tablet, tablets.text)))
        write_recall_audit(recall_rows, paths["recall_out"])
        passed = run_false_negative_benchmark(extracted_all, combined, recall_rows)
        return 0 if passed else 2

    statement_filters = bool(args.contains or args.kind or args.parser_rule)
    targeted_mode = bool(args.tablet or statement_filters)
    extracted = (
        filter_extracted(
            extracted_all,
            tablets=args.tablet,
            contains=args.contains,
            kinds=args.kind,
            parser_rules=args.parser_rule,
        )
        if targeted_mode else extracted_all
    )
    if args.tablet and not statement_filters:
        # Records without parser clauses are still searched.
        for tablet in args.tablet:
            if tablet not in extracted and tablet in extracted_all:
                extracted[tablet] = extracted_all[tablet]
    total = sum(len(sts) for _, sts in extracted.values())
    if targeted_mode:
        print(f"targeted selection: {total:,} statements from {len(extracted):,} tablets")

    if args.dry_run:
        cache = load_cache(paths["cache"])
        needed = sum(cache_key(tablet, text, index, st, args.model) not in cache
                     for tablet, (text, sts) in extracted.items()
                     for index, st in enumerate(sts))
        print(f"model: {args.model}")
        print(f"cached parser-statement reviews reusable: {total - needed:,}")
        print(f"parser-statement reviews required: {needed:,}")
        if not statement_filters and not args.no_recall:
            ids = set(extracted)
            chunk_count = sum(
                len(_planned_recall_chunks(row.text, extracted_all.get(row.tablet, ("", []))[1],
                                           args.recall_core_chars, args.recall_context_chars,
                                           args.recall_overlap_chars))
                for row in tablets.itertuples(index=False) if row.tablet in ids)
            print(f"search discovery windows planned: {chunk_count:,}")
            print("normalization calls are data-dependent: one per discovered candidate")
        return 0

    reviewed, records, dropped = review_all(
        extracted=extracted,
        model=args.model,
        cache_path=paths["cache"],
        force=args.force,
        pilot=args.pilot,
        **concurrency,
    )
    if args.pilot is not None:
        print(f"pilot reviews cached in {paths['cache']}; no tables were generated")
        write_run_summary(paths["summary"], args=args, tablets_run=None,
                          review_records=records, recall_rows=[], recall_records=[])
        return 0
    write_change_audit(records, paths["changes_out"])

    recall_rows: list[dict[str, Any]] = []
    if args.no_recall or statement_filters:
        combined = reviewed
        print("search for missed clauses not run")
    else:
        ids = set(extracted) if targeted_mode else None
        frame = tablets[tablets.tablet.isin(ids)].copy() if ids else tablets
        additions, recall_rows = recall(frame, reviewed, dropped, ids)
        combined = merge_recall_additions(
            reviewed, additions, dict(zip(tablets.tablet, tablets.text)))
        added = sum(len(v) for v in additions.values())
        before = sum(len(v) for v in reviewed.values())
        print(f"combined statement set: {before:,} reviewed parser statements + "
              f"{added:,} additions = {before + added:,} statements")
        write_recall_audit(recall_rows, paths["recall_out"])

    recall_records = [r for r in load_cache(paths["recall_cache"]).values()
                      if r.get("model") == args.model
                      and (not targeted_mode or r.get("tablet") in set(extracted))]
    if targeted_mode:
        write_run_summary(paths["summary"], args=args, tablets_run=sorted(extracted),
                          review_records=records, recall_rows=recall_rows,
                          recall_records=recall_records)
        print("targeted run: audits written; no tables were generated")
        return 0

    write_run_summary(paths["summary"], args=args, tablets_run=None,
                      review_records=records, recall_rows=recall_rows,
                      recall_records=recall_records)
    print("\nWrote:")
    for path in paths.values():
        print(f"  {path}")
    print("Next, score the final clauses with scripts/agent/rescore_model_review.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
