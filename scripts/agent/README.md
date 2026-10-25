# agent

These scripts are bounded language-model interfaces. A model here sees translated text with the published year withheld, and may not supply a number or change a verdict. The package holds the clause review that the paper reports, and the MCP server that gives any agent the read-only tools.

## Clause review

| Script | What it does |
|---|---|
| `model_review.py` | The paid review. A language model checks each parser clause against the text and keeps, modifies or drops it. It is called through the proxy of the Ohio State University, or through the Gemini API with `--route gemini`. A second call must confirm each change. Code accepts the change only if the cited passage lies in the clause's own sentence and contains the new value. A drop must name the value that the text gives instead. The model then searches the text for clauses the parser missed. The model sees an opaque record token in place of the ADsD id. In its translation, `{year}` and `{ruler}` replace year numbers and ruler names. Only the parser's rule, `tools.statements.unwatched`, marks a clause unobserved. Every call's reported model, tokens and cost are cached. Each run writes to its own directory, `data/clauses/model_review_<model>/`. It holds the caches, `review_changes.csv` and `search_proposals.csv` (the records behind SI Table S1), `fallbacks.jsonl` and `run_summary.json`. `--scope` defaults to ADART I to III |
| `rescore_model_review.py` | Rebuild the final clauses of one run from its caches alone, and score every record that has one. Writes `reviewed_clause_bits.parquet` and `reviewed_clause_verdicts.parquet` into the run directory. `scripts/paper/build_paper_counts.py` filters these to ADART I to III |
| `compare_models.py` | Test the pilot of three models on ten records, in `data/clauses/model_review_pilot/`, at the published years, and write `comparison.json` there. It makes no model call |

The rescore replaces the model client with one that raises, so a cache miss stops it before any money is spent. The published year is loaded only after the reviewed clauses are final, and only for the verdicts at the published year.

The cache is keyed on the model name, the prompt version, each clause as the parser extracts it and the tablet text. The prompt versions, such as `step1-review-v7-blind`, keep the old name of the review, step1. A new name there would make every committed response a cache miss. A parser change that alters a clause turns its review into a cache miss, and the rescore stops until the paid review is topped up. The saved scores carry a fingerprint of `tools/statements.py`, `sky.py`, `cal.py` and `eclipses.py`, `model_review.py`, and the four physical tables. Editing any of those files, even a comment, invalidates the saved scores until the rescore is rerun.

The committed run is `data/clauses/model_review_gpt-6-luna-2026-09-22/`, made on 24 October 2026. Its 15,016 calls cost US$3.92.

## MCP server

| Script | What it does |
|---|---|
| `mcp_server.py` | The paper's tools for any agent, through MCP over stdio. The server reuses `blind_case_tools.py` and the decision of `inference/date_all.py`. It is read-only and opens only the tables in `data/physical`, so it knows no published date. Its seven tools read the clauses of a passage, give a clause's verdict in one year or in all 701, and score a passage. They also convert Babylonian and Julian dates, give the sign of a body and list the eclipses near a date. Run `python scripts/agent/mcp_server.py` |
| `blind_case_tools.py` | The read-only tools of a date-blind workspace, which `mcp_server.py` imports |

## Client

| Script | What it does |
|---|---|
| `llm.py` | Small client for the language-model proxy. It reads `LITELLM_API_KEY` from the environment, a `.env` at the repository root, or `~/.env`, and returns token and cost metadata with each response. Nothing is cached. `source/classify_evidence.py --llm` also uses it |

`__init__.py` only marks the package.
