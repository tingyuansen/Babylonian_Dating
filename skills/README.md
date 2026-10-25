# skills

| File | What it is |
|---|---|
| `dating_protocol.md` | The written protocol of steps 1 to 3, supplied unchanged to each agent and human auditor. It holds rules and no dates or results. |

`scripts/inference/date_all.py` records the protocol's SHA-256 in every result, so an edit to it invalidates the committed results until the pipeline is rerun. `verify.py` checks that the copy handed to a blind agent workspace is byte-identical to this file.
