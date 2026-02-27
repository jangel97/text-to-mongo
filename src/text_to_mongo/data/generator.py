"""Main dataset generation orchestrator.

Produces TrainingExample instances by matching schemas to intent templates,
then applying augmentation.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from text_to_mongo.schema import AllowedOps, TrainingExample
from text_to_mongo.data.augment import run_all_augmentations
from text_to_mongo.data.export import export_splits
from text_to_mongo.data.intents import ALL_GENERATORS
from text_to_mongo.data.schemas import (
    HELD_OUT_COLLECTIONS,
    HELD_OUT_SCHEMAS,
    TRAIN_SCHEMAS,
)

# Default allowed operators for generated examples
DEFAULT_ALLOWED_OPS = AllowedOps(
    stage_operators=[
        "$match", "$group", "$sort", "$limit", "$skip",
        "$project", "$unwind", "$lookup", "$addFields", "$count",
        "$bucket", "$bucketAuto", "$facet", "$replaceRoot",
        "$sortByCount", "$sample", "$set", "$unset",
    ],
    expression_operators=[
        "$sum", "$avg", "$min", "$max", "$first", "$last",
        "$push", "$addToSet",
        "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
        "$in", "$nin", "$and", "$or", "$not", "$nor",
        "$exists", "$type", "$regex",
        "$year", "$month", "$dayOfMonth", "$dayOfWeek",
        "$add", "$subtract", "$multiply", "$divide",
        "$concat", "$substr", "$toLower", "$toUpper",
        "$cond", "$ifNull", "$switch",
        "$size", "$slice", "$filter", "$map",
        "$dateToString", "$toDate",
    ],
)


def generate_base_examples(seed: int = 42) -> list[TrainingExample]:
    """Generate base examples by matching every schema to every intent pattern."""
    rng = random.Random(seed)
    all_schemas = TRAIN_SCHEMAS + HELD_OUT_SCHEMAS
    examples: list[TrainingExample] = []

    for schema in all_schemas:
        for gen_fn in ALL_GENERATORS:
            pairs = gen_fn(schema, rng)
            for intent, query in pairs:
                examples.append(TrainingExample(
                    schema=schema,
                    allowed_ops=DEFAULT_ALLOWED_OPS,
                    intent=intent,
                    output=query,
                    is_negative=False,
                ))

    return examples


def generate_dataset(
    seed: int = 42,
    output_dir: str | Path = "data",
) -> dict[str, int]:
    """Full pipeline: generate base examples, augment, and export splits.

    Returns dict with counts per split.
    """
    base = generate_base_examples(seed)
    augmented = run_all_augmentations(base, seed=seed)
    all_examples = base + augmented

    counts = export_splits(
        all_examples,
        output_dir=Path(output_dir),
        held_out_collections=HELD_OUT_COLLECTIONS,
        eval_ratio=0.1,
        seed=seed,
    )

    return counts


def main() -> None:
    output_dir = Path("data")
    print(f"Generating dataset to {output_dir}/...")
    counts = generate_dataset(output_dir=output_dir)
    print(f"Done! Generated:")
    for split, count in counts.items():
        print(f"  {split}: {count} examples")


if __name__ == "__main__":
    main()
