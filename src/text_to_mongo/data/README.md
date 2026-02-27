# data/ — Synthetic Dataset Generation

Generates the full training dataset from scratch — no human labeling, no LLM in the loop. Deterministic (seed 42), runs in under a second.

## Entrypoint

```bash
python -m text_to_mongo.data
```

This runs `generator.py`, which orchestrates the full pipeline and writes `data/train.jsonl`, `data/eval.jsonl`, and `data/held_out.jsonl`.

## Files

| File | What it does |
|---|---|
| `schemas.py` | 19 hand-crafted MongoDB collection schemas across 8 domains. Each schema defines fields with typed semantic roles (`identifier`, `measure`, `timestamp`, `category`, `enum`, `boolean`, `text`). 16 are used for training, 3 are held out (`museum_exhibits`, `weather_stations`, `fleet_vehicles`). |
| `intents.py` | 10 generator functions that produce (natural language intent, MongoDB query) pairs. Each reads a schema's fields, picks fields matching required roles, and builds both a randomized English question and the corresponding ground-truth query. Covers: filters, aggregations, time ranges, top-N, counts, exists checks, enum filters, date bucketing. |
| `augment.py` | Four augmentation strategies that multiply base examples: field name shuffling (6 passes), negative examples with hallucinated fields (15%), date variation (4 passes), and operator subsetting (4 passes). |
| `generator.py` | Orchestrator. Matches all schemas to all applicable intent generators, applies augmentation, then calls `export.py` to write the splits. |
| `export.py` | Splits examples into JSONL files. Held-out collection examples go to `held_out.jsonl`. The rest are shuffled and split 90/10 into `train.jsonl` / `eval.jsonl`. |

## Pipeline

```
schemas.py (19 schemas)
    → intents.py (10 patterns × applicable schemas → ~235 base examples)
        → augment.py (shuffle + negatives + dates + ops → ~1,800 examples)
            → export.py (split → train/eval/held_out JSONL)
```
