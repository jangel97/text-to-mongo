# eval/ — 4-Layer Evaluation Harness

Validates generated MongoDB queries through four independent layers. Each layer checks a different aspect of correctness. Layers run sequentially — if syntax fails, later layers are skipped.

## Entrypoint

The harness is called from the training pipeline (`training/baseline.py` and `training/compare.py`), not directly. The main function is `harness.evaluate()`.

## Files

| File | What it does |
|---|---|
| `syntax.py` | **Layer 1** — Validates structural correctness: valid JSON, top-level dict, has `type` field (`"find"` or `"aggregate"`), well-formed body (`filter` dict for find, `pipeline` list for aggregate where each stage has exactly one `$`-prefixed key). |
| `operators.py` | **Layer 2** — Recursively extracts every `$`-prefixed key from the query. Checks that all operators are in the allowed list and that no unsafe operators (`$where`, `$function`, `$merge`, `$out`) appear. Skips Extended JSON wrappers (`$date`, `$oid`, etc.) since those are value literals, not query operators. |
| `fields.py` | **Layer 3** — Extracts every field reference from the query (dict keys in filters + `$`-prefixed strings in expressions). Handles the subtlety that `$group`/`$bucket` output keys are aliases, not field references. Flags any field that doesn't exist in the schema as hallucinated. |
| `generalization.py` | **Layer 4** — Compares pass rates between training-schema examples and held-out-schema examples. Flags potential overfitting if the gap exceeds 5% on any metric. |
| `harness.py` | Orchestrator. For each (example, prediction) pair, runs layers 1→2→3. Aggregates results into an `EvalReport` with per-layer pass rates. Optionally runs layer 4 across the full result set. |

## Evaluation Flow

```
Raw model output
    → Layer 1: Syntax (valid JSON + correct structure?)
        → Layer 2: Operators (only allowed ops? no unsafe ops?)
            → Layer 3: Fields (all fields exist in schema?)
                → Layer 4: Generalization (train vs held-out gap < 5%?)
```
