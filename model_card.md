---
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
library_name: peft
pipeline_tag: text-generation
license: mit
language:
- en
tags:
- base_model:adapter:Qwen/Qwen2.5-Coder-7B-Instruct
- lora
- qlora
- sft
- transformers
- trl
- mongodb
- text-to-query
- code-generation
datasets:
- jmorenas/text-to-mongo-dataset-qlora
---

# Text-to-MongoDB QLoRA

LoRA adapter that translates natural language questions into MongoDB `find` and `aggregate` queries.

Given a collection schema, allowed operators, and a plain-English intent, the model produces a valid MongoDB query as JSON — running locally on a single GPU with ~1 second inference latency.

**[Try it on Google Colab](https://colab.research.google.com/drive/1jb22dXx0k-5aiZq0O9gGS7emnvRdlhHb)** | **[Dataset](https://huggingface.co/datasets/jmorenas/text-to-mongo-dataset-qlora)** | **[GitHub](https://github.com/jmorenas/text-to-mongo-lora)**

## Results

| Model | Split | Syntax | Operators | Fields | Overall |
|---|---|---|---|---|---|
| Qwen 7B **baseline** (zero-shot) | eval (171) | 51.0% | 51.0% | 40.0% | 40.0% |
| Qwen 7B **baseline** (zero-shot) | held-out (423) | 53.8% | 53.5% | 39.9% | 39.9% |
| Qwen 7B **+ LoRA r=8** | eval (171) | **98.8%** | **98.8%** | **98.8%** | **98.8%** |
| Qwen 7B **+ LoRA r=8** | held-out (423) | **98.6%** | **98.6%** | **98.6%** | **98.6%** |

The held-out set uses 3 collection schemas the model never saw during training, testing generalization to unseen domains.

### 4-Layer Evaluation

Each generated query is validated through four layers applied sequentially:

1. **Syntax** — Valid JSON, correct structure (`find` needs `filter`, `aggregate` needs `pipeline`)
2. **Operators** — Every `$`-operator used is in the allowed list, no unsafe operators (`$where`, `$merge`, `$out`)
3. **Fields** — Every field reference exists in the schema (catches hallucinated field names)
4. **Generalization** — Compares train vs held-out pass rates to detect overfitting

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter = "jmorenas/text-to-mongo-qlora"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)

schema = """Collection: orders (ecommerce)
Fields:
- order_id: string [identifier] — Order ID
- total: double [measure] — Order total
- status: string [enum] — Order status (pending, shipped, delivered)
- created_at: date [timestamp] — Creation date"""

allowed_ops = """Stage: $match, $group, $sort, $limit, $project
Expression: $sum, $avg, $gt, $gte, $lt, $lte, $in, $eq"""

intent = "Find all pending orders over $100"

messages = [
    {"role": "system", "content": f"You are a MongoDB query generator.\n\nSchema:\n{schema}\n\nAllowed operators:\n{allowed_ops}"},
    {"role": "user", "content": intent},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

**Output:**
```json
{"type": "find", "filter": {"status": "pending", "total": {"$gt": 100}}}
```

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

The model reads the schema (field names, types, roles, enum values) and the allowed operators, then composes a query that respects both. It generalizes to schemas it has never seen.

## Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Quantization | 4-bit NF4 + double quantization |
| LoRA rank | 8 (alpha=16) |
| LoRA targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj |
| LoRA dropout | 0.05 |
| Epochs | 3 |
| Effective batch size | 16 (batch=4, grad_accum=4) |
| Learning rate | 2e-4 (cosine schedule, 5% warmup) |
| Max sequence length | 512 |
| Optimizer | paged_adamw_8bit |
| Framework | trl SFTTrainer (prompt-completion format) |
| Loss | Completion-only (prompt tokens masked) |
| Training time | ~11 minutes on RTX 5090 |
| VRAM usage | ~5 GB |
| Train loss | 0.013 |

### Training Data

[jmorenas/text-to-mongo-dataset-qlora](https://huggingface.co/datasets/jmorenas/text-to-mongo-dataset-qlora) — 1,544 train / 171 eval / 423 held-out examples across 19 collection schemas (16 train + 3 held-out).

The dataset is fully synthetic — generated deterministically from hand-crafted schemas and intent patterns. 12 generator patterns produce query types including filters, aggregations, projections, time ranges, top-N, counts, exists checks, enum filters, and date bucketing. Augmentation strategies (field name shuffling, date variation, operator subsetting, negatives) multiply the base examples ~7x.

## Limitations

- Designed for `find` and `aggregate` queries only — does not generate `update`, `delete`, or `insertOne`
- Field descriptions in the schema should be 2-5 words; longer descriptions degrade output quality
- The adapter cannot be merged into 4-bit weights (`merge_and_unload()` produces garbage) — must be used as a PeftModel wrapper

## License

MIT
