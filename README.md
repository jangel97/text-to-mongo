# Text-to-MongoDB LoRA

Fine-tuned language model that translates natural language questions into MongoDB queries.

Given a collection schema, allowed operators, and a plain-English intent, the model produces a valid `find` or `aggregate` query as JSON — running locally on a single GPU with ~1 second inference latency.

## Results

**Base model**: Qwen2.5-Coder-7B-Instruct (4-bit QLoRA)

| Model | Split | Syntax | Operators | Fields | Overall |
|---|---|---|---|---|---|
| Qwen 7B **baseline** (zero-shot) | eval (145) | 51.0% | 51.0% | 40.0% | 40.0% |
| Qwen 7B **baseline** (zero-shot) | held-out (368) | 53.8% | 53.5% | 39.9% | 39.9% |
| Qwen 7B **LoRA r=8** | eval (145) | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| Qwen 7B **LoRA r=8** | held-out (368) | **98.9%** | **98.9%** | **98.9%** | **98.9%** |

The held-out set uses 3 collection schemas the model never saw during training, testing generalization to unseen domains.

### 4-Layer Evaluation

Each generated query is validated through four independent layers. Layers are applied sequentially — if syntax fails, later layers are skipped since there's nothing to analyze.

**Layer 1 — Syntax**: Parses the raw model output and validates structural correctness: valid JSON, top-level dict, has a `type` field (`"find"` or `"aggregate"`), and the query body is well-formed (`find` needs a `filter` dict, `aggregate` needs a `pipeline` list where each stage has exactly one `$`-prefixed key).

**Layer 2 — Operators**: Recursively extracts every `$`-prefixed key from the query and checks two things: (1) every operator used was in the `allowed_ops` list given to the model, and (2) no unsafe operators (`$where`, `$merge`, `$out`, etc.) appear. Extended JSON type wrappers like `$date` and `$oid` are excluded — they're value literals, not query operators.

**Layer 3 — Fields**: Extracts every field reference from the query — both dict keys in filters (e.g. `"status"` in `{"status": "active"}`) and `$`-prefixed strings in expressions (e.g. `"$salary"`). The key subtlety: inside `$group`/`$bucket` stages, output keys like `"total"` in `{"total": {"$sum": "$salary"}}` are aliases, not field references, so they're skipped. Any field that doesn't exist in the schema is flagged as hallucinated.

**Layer 4 — Generalization**: Splits results by whether the example's collection schema was in the training set or the held-out set (3 collections the model never saw: `museum_exhibits`, `weather_stations`, `fleet_vehicles`). Compares pass rates across splits — if the gap exceeds 5% on any metric, it flags potential overfitting. This is what proves the model learned to read schemas rather than memorize query patterns.

## How It Works

```
User: "Find orders over $100 shipped last month"
         │
         ▼
┌─────────────────────────────────┐
│  Schema: orders collection      │
│  Fields: order_id, total,       │
│          status, created_at     │
│  Allowed ops: $match, $gte...   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Qwen 7B + LoRA adapter        │
│  (4-bit quantized, ~5GB VRAM)  │
└─────────────────────────────────┘
         │
         ▼
{
  "type": "find",
  "filter": {
    "total": {"$gt": 100},
    "status": "shipped",
    "created_at": {"$gte": {"$date": "2025-01-01T00:00:00Z"}}
  }
}
```

The model learns to read the schema (field names, types, roles, enum values) and the allowed operators, then composes a query that respects both. It generalizes to schemas it has never seen.

## Project Structure

```
text-to-mongo-lora/
├── src/text_to_mongo/
│   ├── schema.py              # Pydantic models (SchemaDef, FieldDef, TrainingExample, etc.)
│   ├── prompt.py              # ChatML prompt template builder
│   ├── data/                  # Synthetic dataset generation
│   │   ├── schemas.py         #   16 train + 3 held-out collection schemas
│   │   ├── intents.py         #   10 intent patterns per schema
│   │   ├── augment.py         #   Field shuffling, date variation, operator subsets, negatives
│   │   ├── generator.py       #   Orchestrates schema → intent → query generation
│   │   └── export.py          #   Exports to train/eval/held_out JSONL splits
│   ├── eval/                  # 4-layer evaluation harness
│   │   ├── syntax.py          #   JSON structure validation
│   │   ├── operators.py       #   Operator allowlist checking
│   │   ├── fields.py          #   Schema field reference validation
│   │   ├── generalization.py  #   Train vs held-out gap detection
│   │   └── harness.py         #   Runs all layers, produces EvalReport
│   ├── training/              # QLoRA fine-tuning pipeline
│   │   ├── config.py          #   Model, LoRA, and training configs
│   │   ├── dataset.py         #   JSONL → HuggingFace Dataset (prompt-completion format)
│   │   ├── trainer.py         #   SFTTrainer orchestration with QLoRA
│   │   ├── inference.py       #   Model loading (PeftModel) and batched generation
│   │   ├── baseline.py        #   Zero-shot evaluation (no adapter)
│   │   ├── compare.py         #   Post-training eval + comparison tables
│   │   └── cli.py             #   Unified CLI: baseline, train, eval, compare
│   └── serve/                 # FastAPI inference microservice
│       ├── app.py             #   POST /predict endpoint
│       └── models.py          #   Request/response models
├── tests/                     # 89 pytest tests (no GPU required)
├── pyproject.toml             # Package config with [train] and [serve] optional deps
└── data/                      # Generated JSONL (not committed, reproducible)
    ├── train.jsonl            #   ~1,312 examples
    ├── eval.jsonl             #   ~145 examples
    └── held_out.jsonl         #   ~368 examples (unseen schemas)
```

## Setup

```bash
# Clone and install base package
git clone <repo-url>
cd text-to-mongo-lora
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (no GPU needed)
pytest
```

### Generate Training Data

```bash
python -m text_to_mongo.data
```

Produces `data/train.jsonl`, `data/eval.jsonl`, and `data/held_out.jsonl` from 16 synthetic collection schemas with augmentation (field shuffling, date variations, operator subsets, 15% negative examples).

### Train (requires GPU)

```bash
pip install -e ".[train]"

# Run zero-shot baseline
python -m text_to_mongo.training baseline --model qwen2.5-coder-7b

# Fine-tune with LoRA
python -m text_to_mongo.training train --model qwen2.5-coder-7b --lora-r 8

# Evaluate the adapter
python -m text_to_mongo.training eval --model qwen2.5-coder-7b --lora-r 8

# Compare all runs
python -m text_to_mongo.training compare
```

Training takes ~20 minutes on an RTX 5090 (3 epochs, batch size 4, gradient accumulation 4).

### Serve (requires GPU)

```bash
pip install -e ".[serve]"

# Start inference service
ADAPTER_PATH=runs/qwen2.5-coder-7b_r8/adapter \
uvicorn text_to_mongo.serve.app:app --host 0.0.0.0 --port 8080
```

#### API

**POST /predict**

```json
{
  "schema": {
    "collection": "orders",
    "domain": "ecommerce",
    "fields": [
      {"name": "total", "type": "double", "role": "measure", "description": "Order total"},
      {"name": "status", "type": "string", "role": "enum", "description": "Order status",
       "enum_values": ["pending", "shipped", "delivered"]}
    ]
  },
  "allowed_ops": {
    "stage_operators": ["$match", "$group"],
    "expression_operators": ["$sum", "$avg", "$gt"]
  },
  "intent": "Find all pending orders"
}
```

**Response**

```json
{
  "query": {"type": "find", "filter": {"status": "pending"}},
  "raw_output": "{\"type\": \"find\", \"filter\": {\"status\": \"pending\"}}",
  "syntax_valid": true,
  "errors": [],
  "latency_ms": 782
}
```

**GET /health**

```json
{"status": "ok", "model": "qwen2.5-coder-7b", "adapter": "runs/qwen2.5-coder-7b_r8/adapter", "device": "cuda:0"}
```

## Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Quantization | 4-bit NF4 + double quantization |
| LoRA rank | 8 (alpha=16) |
| LoRA targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj |
| Epochs | 3 |
| Effective batch size | 16 (batch=4, grad_accum=4) |
| Learning rate | 2e-4 (cosine schedule, 5% warmup) |
| Max sequence length | 512 |
| Optimizer | paged_adamw_8bit |
| Training framework | trl SFTTrainer (prompt-completion format) |
| Loss | Completion-only (prompt tokens masked) |
| VRAM usage | ~5 GB (4-bit model + LoRA) |

## Key Design Decisions

**PeftModel inference (no merge)**: LoRA adapters cannot be merged into 4-bit quantized weights — `merge_and_unload()` silently produces garbage. The adapter is kept as a PeftModel wrapper at runtime.

**Prompt-completion format**: Uses trl's prompt-completion dataset format (separate `prompt` and `completion` columns) which automatically masks prompt tokens from the loss. No custom data collator needed.

**Field semantic roles**: Each field has a role (`identifier`, `measure`, `timestamp`, `category`, `text`, `enum`, `boolean`) that helps the model understand how to use it in queries.

**Short schema descriptions**: The model is sensitive to prompt length. Field descriptions should be 2-5 words — longer descriptions cause degraded output quality.

## License

MIT
