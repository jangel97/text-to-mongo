# Architecture

## System Overview

```mermaid
graph LR
    subgraph "Data Generation"
        S[19 Schemas] --> IG[Intent Generators]
        IG --> BE[~235 Base Examples]
        BE --> AUG[Augmentation]
        AUG --> DS[~1,800 Examples]
        DS --> SP{Split}
        SP --> TR[train.jsonl]
        SP --> EV[eval.jsonl]
        SP --> HO[held_out.jsonl]
    end

    subgraph "Training"
        TR --> FMT[Prompt Builder]
        FMT --> SFT[SFTTrainer + QLoRA]
        SFT --> AD[LoRA Adapter]
    end

    subgraph "Evaluation"
        EV --> INF[Batch Inference]
        HO --> INF
        AD --> INF
        INF --> HAR[4-Layer Harness]
        HAR --> REP[EvalReport]
    end

    subgraph "Serving"
        AD --> SRV[FastAPI Service]
        SRV --> API[POST /predict]
    end
```

## Data Generation Pipeline

```mermaid
graph TD
    SCH["19 Collection Schemas
    (8 domains)"] --> GEN

    subgraph GEN["Intent Generation (10 patterns)"]
        G1[Filter Only]
        G2[Aggregate Single]
        G3[Aggregate Filtered]
        G4[Time Range]
        G5[Top N]
        G6[Multi-Group]
        G7[Count]
        G8[Exists Check]
        G9[Enum Filter]
        G10[Date Bucket]
    end

    GEN -->|"~235 examples"| BASE[Base Examples]

    subgraph AUG["Augmentation"]
        A1["Field Shuffling (6 passes)
        salary → compensation"]
        A2["Negatives (15%)
        hallucinated fields → error"]
        A3["Date Variation (4 passes)
        random concrete dates"]
        A4["Operator Subset (4 passes)
        remove unused ops"]
    end

    BASE --> AUG
    AUG -->|"~1,800 total"| SPLIT

    subgraph SPLIT["Export Splits"]
        TRAIN["train.jsonl (~1,312)
        16 training schemas, 90%"]
        EVAL["eval.jsonl (~145)
        16 training schemas, 10%"]
        HELD["held_out.jsonl (~368)
        3 unseen schemas, 100%"]
    end
```

Each generator reads the schema's field roles to decide applicability. A schema needs a `measure` and a `category` field for aggregation patterns, a `timestamp` for time range queries, an `enum` for `$in` filters, etc. The intent is generated from randomized natural language templates, and the ground-truth MongoDB query is built programmatically — no LLM in the loop.

## Training Example Format

Each example is a JSON object with four components:

```json
{
  "schema": {
    "collection": "orders",
    "domain": "ecommerce",
    "fields": [
      {"name": "total_amount", "type": "double", "role": "measure", "description": "Order total in USD"},
      {"name": "status", "type": "string", "role": "enum", "description": "Order status",
       "enum_values": ["pending", "shipped", "delivered", "cancelled"]}
    ]
  },
  "allowed_ops": {
    "stage_operators": ["$match", "$group", "$sort"],
    "expression_operators": ["$sum", "$avg", "$eq", "$gt"]
  },
  "intent": "Find all pending orders",
  "output": {"type": "find", "filter": {"status": "pending"}}
}
```

The prompt builder renders this into ChatML format:

```
<|im_start|>system
You are a MongoDB query generator. Given a collection schema...
<|im_end|>
<|im_start|>user
Collection: orders
Fields:
  - total_amount (double, measure): Order total in USD
  - status (string, enum): Order status [values: pending, shipped, delivered, cancelled]

Allowed stage operators: $match, $group, $sort
Allowed expression operators: $sum, $avg, $eq, $gt

Intent: Find all pending orders
<|im_end|>
<|im_start|>assistant
{"type": "find", "filter": {"status": "pending"}}<|im_end|>
```

During training, prompt tokens are masked from the loss — the model only learns to generate the query JSON.

## Training Pipeline

```mermaid
graph LR
    subgraph "QLoRA Setup"
        BASE["Qwen2.5-Coder-7B
        (4-bit NF4)"] --> LORA["LoRA Adapter
        r=8, alpha=16
        q/k/v/o/gate/up"]
    end

    subgraph "Training Loop"
        DATA[train.jsonl] --> PC["Prompt-Completion
        Split"]
        PC --> SFT["SFTTrainer
        3 epochs, lr=2e-4
        cosine schedule"]
        LORA --> SFT
        SFT --> SAVE["Adapter Weights
        (~20MB)"]
    end
```

The base model weights stay frozen in 4-bit precision. Only the LoRA adapter parameters (~0.1% of total) are trained. The adapter is saved separately and loaded at inference time via `PeftModel` — it cannot be merged into quantized weights.

## Evaluation Harness

```mermaid
graph TD
    PRED[Model Prediction] --> L1

    subgraph "Layer 1: Syntax"
        L1{Valid JSON?} -->|No| FAIL1[FAIL]
        L1 -->|Yes| L1B{Has type field?}
        L1B -->|No| FAIL1
        L1B -->|Yes| L1C{"find or aggregate?"}
        L1C -->|No| FAIL1
        L1C -->|Yes| L1D{Body well-formed?}
        L1D -->|No| FAIL1
        L1D -->|Yes| PASS1[PASS]
    end

    PASS1 --> L2

    subgraph "Layer 2: Operators"
        L2{"Extract all $-keys
        (skip $date, $oid)"} --> L2B{All in allowed list?}
        L2B -->|No| FAIL2[FAIL]
        L2B -->|Yes| L2C{"No unsafe ops?
        ($where, $merge, $out)"}
        L2C -->|No| FAIL2
        L2C -->|Yes| PASS2[PASS]
    end

    PASS2 --> L3

    subgraph "Layer 3: Fields"
        L3{"Extract field refs
        (skip $group aliases)"} --> L3B{All fields in schema?}
        L3B -->|No| FAIL3["FAIL (hallucinated)"]
        L3B -->|Yes| PASS3[PASS]
    end

    PASS3 --> OVERALL[Overall: PASS]
```

Layers are sequential — if syntax fails, operators and fields are skipped. The fourth layer (generalization) operates across the full result set, comparing pass rates between training and held-out schemas.

## Inference Service

```mermaid
sequenceDiagram
    participant C as Client
    participant S as FastAPI Service
    participant M as Qwen 7B + LoRA

    C->>S: POST /predict {schema, allowed_ops, intent}
    S->>S: Build ChatML prompt
    S->>M: Tokenize + generate (greedy, max 256 tokens)
    M-->>S: Raw token output
    S->>S: Extract JSON (brace-depth matching)
    S->>S: Validate syntax (Layer 1)
    S-->>C: {query, raw_output, syntax_valid, errors, latency_ms}
```

The service loads the base model in 4-bit precision and applies the LoRA adapter on startup. Generation uses greedy decoding (no sampling) for deterministic output. The `extract_json` function handles noisy model output by finding the first balanced JSON object using brace-depth tracking.

## Key Constraints

**Short schema descriptions**: The model was trained on field descriptions of 2-5 words. Longer descriptions cause it to hallucinate operator lists instead of generating queries. This is a hard constraint — inference prompts must match training prompt length.

**No merge for 4-bit models**: `merge_and_unload()` on a 4-bit quantized model silently produces garbage weights. The adapter must stay as a `PeftModel` wrapper at runtime.

**Field semantic roles**: The role annotation (`identifier`, `measure`, `timestamp`, `category`, `enum`, `boolean`, `text`) is what teaches the model which fields to group by, which to sum, which to filter on. Without roles, the model has to guess from field names alone.
