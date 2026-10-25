# tests

These tests run offline, and none needs the network or a model. Each is a plain `unittest` module that runs from the root.

```bash
python tests/test_joint_positions.py
python tests/test_model_review.py
python tests/test_result_figures.py
python tests/test_paper_numbers.py
python tests/test_mcp_server.py
python tests/test_rules.py
```

| File | What it covers |
|---|---|
| `test_joint_positions.py` | 17 tests of clauses that place several planets in one sign at once. Each planet becomes its own clause, linked to the others only within one source statement. Restored joints and inner planets stay supporting. A missing month leaves the clause untestable. The planets must share a sign at the same time, and touching a boundary does not count. Sirius is labelled as Sirius without changing the physics. A sign crossing inside one day is found |
| `test_mcp_server.py` | 10 tests of the read-only MCP server in `scripts/agent/mcp_server.py`. The server lists seven read-only tools, and a call through the protocol returns what the function returns. The parser reads a short synthetic passage into clauses with their fields and rules. A lunar eclipse clause agrees in its year, and its verdicts over the 701 years add up. The score is the decision of `date_all.py`. A Babylonian date converts to a Julian date and back. Bad input is refused. No output has a field named like an answer, such as `bce_first`, `published`, `truth` or `reference`. The tools read only `data/physical`, and the source names no answer column |
| `test_model_review.py` | 15 tests of what code decides in the model review, whatever the model says. Year numbers, ruler names and the record id never reach the model, and cited passages map back to the original offsets. Only the parser's rule marks a clause unobserved, and the model cannot change the mark. A drop or a modification must cite the clause's own sentence and contain the value it claims. A search distance must stand directly before its relation, a later mention of an eclipse already held is the same event, and an added clause whose passage holds a bracket or a question mark counts as restored. An antecedent binds to its nearest earlier mention, and stars and signs are canonical. JSON-mode answers are checked against the schema, and the request limit halves on HTTP 429 |
| `test_paper_numbers.py` | 16 tests that recompute every number the paper reports from `data/` and require the matching phrase in `paper/PNAS_Paper.tex`. A rerun that moves a result fails, and so does an edit to the paper that no longer matches the data. Tables S3 and S4 are compared cell by cell with `data/paper_counts/`. The module skips when the paper is absent, as in the public repository |
| `test_rules.py` | 8 tests of edge cases of the rules. The Sirius scheme gives all three phases in every candidate year, and the last appearance counts from the solstice of the year before. A shift of the time scale reaches the cached solstices and Sirius dates. The month window starts on the civil day before the month, so it holds the opening night, and it ends at the next month. A position without a month stays decisive and is tested over its year. The ADART IV filter cuts year numbers and rulers anywhere in a line, and keeps the rest of the line exactly |
| `test_result_figures.py` | 7 tests of the paper's figures. The curve of Fig. 4a reaches the full number of candidate years. The sensitivity figure labels exactly the configurations the audit runs. The ADART IV year cells sit at the right centres in a stable order. Labels stay inside the figure and apart from each other, and the labels of Fig. 1 stay inside their panels |

The tests use short synthetic texts and patch most data reads, though some verdicts still load the committed calendar and ephemeris. `test_paper_numbers.py` checks the reported numbers, and the others check behaviour. The modules are run as scripts, so there is no `__init__.py`.
