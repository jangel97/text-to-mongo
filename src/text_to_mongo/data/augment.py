"""Data augmentation strategies for training examples."""
from __future__ import annotations

import copy
import json
import random
import re
from datetime import datetime, timedelta
from typing import Any

from text_to_mongo.schema import (
    AllowedOps,
    FieldDef,
    SchemaDef,
    TrainingExample,
)

# ---------------------------------------------------------------------------
# Field name synonyms for shuffling
# ---------------------------------------------------------------------------
FIELD_SYNONYMS: dict[str, list[str]] = {
    "price": ["unit_price", "price_per_item", "cost", "unit_cost"],
    "amount": ["value", "total", "sum_amount", "quantity_value"],
    "total_amount": ["order_total", "grand_total", "total_value", "total_cost"],
    "name": ["full_name", "display_name", "label", "title"],
    "status": ["state", "current_status", "condition"],
    "category": ["group", "classification", "segment", "type_class"],
    "region": ["area", "zone", "territory", "locale"],
    "department": ["division", "unit", "section", "team"],
    "salary": ["compensation", "pay", "wage", "annual_pay"],
    "score": ["rating", "grade_score", "evaluation", "mark"],
    "balance": ["current_balance", "account_balance", "available_funds"],
    "weight_kg": ["mass_kg", "gross_weight", "shipping_weight"],
    "likes": ["upvotes", "reactions", "like_count", "thumbs_up"],
    "reading": ["measurement", "sensor_value", "data_point"],
    "charge": ["bill_amount", "visit_cost", "fee_amount"],
    "duration_sec": ["elapsed_seconds", "time_spent", "session_length"],
    "grade": ["final_score", "course_grade", "academic_score"],
    "mileage": ["odometer", "total_miles", "distance_traveled"],
    "capacity": ["max_capacity", "storage_limit", "total_slots"],
    "visitor_count": ["attendance", "total_visitors", "footfall"],
    "temperature_c": ["temp_celsius", "air_temp", "reading_celsius"],
    "humidity_pct": ["relative_humidity", "moisture_pct", "rh_percent"],
    "fuel_cost": ["gas_expense", "fuel_expense", "fuel_spend"],
    "response_time_ms": ["latency_ms", "reply_time_ms", "processing_time"],
    "lifetime_value": ["ltv", "total_spend", "customer_value"],
    "rating": ["avg_rating", "review_score", "star_rating"],
}

# Fake field names for negative examples (fields that don't exist)
HALLUCINATED_FIELDS = [
    "profit_margin", "tax_rate", "discount_pct", "refund_amount",
    "ip_address", "user_agent", "session_token", "api_key",
    "latitude", "longitude", "altitude", "elevation",
    "cpu_usage", "memory_mb", "disk_io", "network_bps",
    "blood_pressure", "heart_rate", "bmi", "cholesterol",
]


def augment_field_names(
    examples: list[TrainingExample],
    rng: random.Random,
    ratio: float = 0.5,
) -> list[TrainingExample]:
    """Create new examples by renaming fields to synonyms.

    For each eligible example (with fields that have synonyms), produce a variant
    with renamed fields in both the schema and the query output.
    """
    augmented: list[TrainingExample] = []
    for ex in examples:
        if rng.random() > ratio:
            continue

        # Find fields with available synonyms
        renameable = [
            f for f in ex.schema_def.fields if f.name in FIELD_SYNONYMS
        ]
        if not renameable:
            continue

        # Pick 1-2 fields to rename
        to_rename = rng.sample(renameable, min(rng.randint(1, 2), len(renameable)))
        rename_map: dict[str, str] = {}
        for f in to_rename:
            rename_map[f.name] = rng.choice(FIELD_SYNONYMS[f.name])

        # Build new schema with renamed fields
        new_fields = []
        for f in ex.schema_def.fields:
            if f.name in rename_map:
                new_f = f.model_copy(update={"name": rename_map[f.name]})
                new_fields.append(new_f)
            else:
                new_fields.append(f)
        new_schema = ex.schema_def.model_copy(update={"fields": new_fields})

        # Rename fields in the output query
        new_output = _rename_in_obj(ex.output, rename_map)

        # Rename fields in the intent string
        new_intent = ex.intent
        for old_name, new_name in rename_map.items():
            new_intent = new_intent.replace(old_name, new_name)

        augmented.append(TrainingExample(
            schema=new_schema,
            allowed_ops=ex.allowed_ops,
            intent=new_intent,
            output=new_output,
            is_negative=ex.is_negative,
        ))

    return augmented


def _rename_in_obj(obj: Any, rename_map: dict[str, str]) -> Any:
    """Recursively rename field references in a query object."""
    if isinstance(obj, str):
        # Handle "$field" references
        if obj.startswith("$") and not obj.startswith("$$"):
            field_path = obj[1:]
            parts = field_path.split(".")
            if parts[0] in rename_map:
                parts[0] = rename_map[parts[0]]
            return "$" + ".".join(parts)
        return obj
    elif isinstance(obj, dict):
        new_dict: dict[str, Any] = {}
        for key, value in obj.items():
            new_key = key
            if key.startswith("$"):
                # Operator key — keep as is but recurse value
                new_dict[key] = _rename_in_obj(value, rename_map)
            else:
                # Could be a field name as key
                root = key.split(".")[0]
                if root in rename_map:
                    parts = key.split(".")
                    parts[0] = rename_map[parts[0]]
                    new_key = ".".join(parts)
                new_dict[new_key] = _rename_in_obj(value, rename_map)
        return new_dict
    elif isinstance(obj, list):
        return [_rename_in_obj(item, rename_map) for item in obj]
    else:
        return obj


def generate_negatives(
    examples: list[TrainingExample],
    rng: random.Random,
    ratio: float = 0.1,
) -> list[TrainingExample]:
    """Generate negative examples by referencing hallucinated fields."""
    negatives: list[TrainingExample] = []
    for ex in examples:
        if rng.random() > ratio:
            continue

        # Pick a hallucinated field
        bad_field = rng.choice(HALLUCINATED_FIELDS)
        intent = f"Show all {ex.schema_def.collection} where {bad_field} is greater than 100"
        output = {"error": f"Field '{bad_field}' does not exist in {ex.schema_def.collection}"}

        negatives.append(TrainingExample(
            schema=ex.schema_def,
            allowed_ops=ex.allowed_ops,
            intent=intent,
            output=output,
            is_negative=True,
        ))

    return negatives


def augment_date_placeholders(
    examples: list[TrainingExample],
    rng: random.Random,
) -> list[TrainingExample]:
    """Replace fixed date values with random concrete dates."""
    augmented: list[TrainingExample] = []
    for ex in examples:
        output_str = json.dumps(ex.output)
        if '"$date"' not in output_str:
            continue

        # Generate random date range
        base = datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 730))
        start = base.strftime("%Y-%m-%dT00:00:00Z")
        end = (base + timedelta(days=rng.randint(30, 365))).strftime("%Y-%m-%dT23:59:59Z")

        # Replace date values in output
        new_output = copy.deepcopy(ex.output)
        new_output = _replace_dates(new_output, [start, end], rng)

        # Update intent with new dates
        new_intent = ex.intent
        new_intent = re.sub(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            lambda m, dates=iter([start, end]): next(dates, m.group()),
            new_intent,
        )

        augmented.append(TrainingExample(
            schema=ex.schema_def,
            allowed_ops=ex.allowed_ops,
            intent=new_intent,
            output=new_output,
            is_negative=ex.is_negative,
        ))

    return augmented


def _replace_dates(obj: Any, dates: list[str], rng: random.Random) -> Any:
    if isinstance(obj, dict):
        if "$date" in obj:
            if dates:
                return {"$date": dates.pop(0)}
            # Generate a random date if we run out
            d = datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 730))
            return {"$date": d.strftime("%Y-%m-%dT%H:%M:%SZ")}
        return {k: _replace_dates(v, dates, rng) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_dates(item, dates, rng) for item in obj]
    return obj


def augment_operator_subset(
    examples: list[TrainingExample],
    rng: random.Random,
    ratio: float = 0.15,
) -> list[TrainingExample]:
    """Randomly restrict allowed operators and verify query is still valid."""
    augmented: list[TrainingExample] = []
    for ex in examples:
        if rng.random() > ratio or ex.is_negative:
            continue

        # Find operators used in the query
        from text_to_mongo.eval.operators import extract_operators
        used_ops = extract_operators(ex.output)

        all_ops = set(ex.allowed_ops.all_operators)
        removable = all_ops - used_ops  # ops we can safely remove

        if len(removable) < 2:
            continue

        # Remove 1-2 unused operators
        to_remove = set(rng.sample(list(removable), min(2, len(removable))))
        new_stage = [op for op in ex.allowed_ops.stage_operators if op not in to_remove]
        new_expr = [op for op in ex.allowed_ops.expression_operators if op not in to_remove]

        augmented.append(TrainingExample(
            schema=ex.schema_def,
            allowed_ops=AllowedOps(stage_operators=new_stage, expression_operators=new_expr),
            intent=ex.intent,
            output=ex.output,
            is_negative=ex.is_negative,
        ))

    return augmented


def run_all_augmentations(
    examples: list[TrainingExample],
    seed: int = 42,
) -> list[TrainingExample]:
    """Run all augmentation strategies and return combined results.

    Runs multiple passes of field-name shuffling and operator subsetting
    with different RNG states to reach ~1,500-2,000 total examples from
    ~235 base examples.
    """
    augmented: list[TrainingExample] = []

    # Multiple passes of field-name shuffling (biggest multiplier)
    for i in range(6):
        rng = random.Random(seed + i)
        augmented.extend(augment_field_names(examples, rng, ratio=0.8))

    # Negative examples
    rng = random.Random(seed + 100)
    augmented.extend(generate_negatives(examples, rng, ratio=0.15))

    # Date variation — multiple passes
    for i in range(4):
        rng = random.Random(seed + 200 + i)
        augmented.extend(augment_date_placeholders(examples, rng))

    # Operator subset — multiple passes
    for i in range(4):
        rng = random.Random(seed + 300 + i)
        augmented.extend(augment_operator_subset(examples, rng, ratio=0.4))

    return augmented
