# Tests

`make check` runs each `test_*.py` file, then `verify.py`. None of them needs the network or a paid service, and together they take less than a minute.

| File | What it checks |
|---|---|
| `verify.py` | The whole repository. Each SHA-256 that the results record matches its file. Each summary follows from its rows. The benchmark, the ranking, the sensitivity audit and the held-out test come from one run. A record whose year is known gets that year. The tree holds only the files that the READMEs and `data/columns.csv` describe, and each README names each file in its folder. |
| `test_rules.py` | The dates of the Sirius scheme, the Julian interval of a Babylonian month, the effect of a time offset on saved dates, and the removal of year numbers and rulers from the ADART IV text. |
| `test_joint_positions.py` | Joint clauses such as "Jupiter and Mars were in Leo". The parser splits them into linked clauses, and the tools test them as one claim, with shared bits and one tier. |
| `test_model_review.py` | The rules that the code applies in the language-model review, such as what the model sees and when a cited passage may change or drop a clause. It calls no model. |
| `test_mcp_server.py` | Each tool of the MCP server in `scripts/agent/mcp_server.py`, on a short text. It checks the outputs, the refusal of bad input, and that no output gives a published date. |
| `test_result_figures.py` | The figures, drawn from small examples. It checks the curve of Fig. 4a, the rows of Fig. S1, the year cells of Fig. S2, and that the labels of Figs. 1 to 3 do not overlap. |

You can also run `python3 tests/verify.py` alone. It prints one line per check and ends with the number of checks that failed. Exit code 0 means that each check passed.
