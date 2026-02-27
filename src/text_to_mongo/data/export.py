"""JSONL export with train/eval/held_out splits."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from text_to_mongo.schema import TrainingExample


def example_to_record(example: TrainingExample) -> dict[str, Any]:
    """Convert a TrainingExample to a flat dict for JSONL export."""
    schema_dict = {
        "collection": example.schema_def.collection,
        "domain": example.schema_def.domain,
        "fields": [
            {
                "name": f.name,
                "type": f.type,
                "role": f.role.value,
                "description": f.description,
                **({"enum_values": f.enum_values} if f.enum_values else {}),
            }
            for f in example.schema_def.fields
        ],
    }

    allowed_ops_dict = {
        "stage_operators": example.allowed_ops.stage_operators,
        "expression_operators": example.allowed_ops.expression_operators,
    }

    return {
        "schema": schema_dict,
        "allowed_ops": allowed_ops_dict,
        "intent": example.intent,
        "output": example.output,
        "is_negative": example.is_negative,
    }


def export_splits(
    examples: list[TrainingExample],
    output_dir: Path,
    held_out_collections: set[str],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Split examples into train/eval/held_out and write JSONL files.

    Returns dict with count per split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    held_out: list[TrainingExample] = []
    rest: list[TrainingExample] = []

    for ex in examples:
        if ex.schema_def.collection in held_out_collections:
            held_out.append(ex)
        else:
            rest.append(ex)

    # Shuffle then split rest into train/eval
    rng.shuffle(rest)
    eval_count = max(1, int(len(rest) * eval_ratio))
    eval_set = rest[:eval_count]
    train_set = rest[eval_count:]

    counts = {}
    for split_name, split_data in [
        ("train", train_set),
        ("eval", eval_set),
        ("held_out", held_out),
    ]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(example_to_record(ex), ensure_ascii=False) + "\n")
        counts[split_name] = len(split_data)

    return counts
