"""Intent templates keyed by query pattern.

Each template defines:
- `pattern`: unique identifier
- `intent_templates`: list of natural language templates (with {placeholders})
- `required_roles`: field roles the schema must have for this pattern to apply
- `build_query`: function that produces the expected MongoDB query dict
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from text_to_mongo.schema import FieldDef, FieldRole, SchemaDef

# Aggregation operators to use in generated queries
AGG_OPS = {"average": "$avg", "total": "$sum", "maximum": "$max", "minimum": "$min"}
AGG_OP_NAMES = list(AGG_OPS.keys())


@dataclass
class IntentTemplate:
    pattern: str
    intent_templates: list[str]
    required_roles: list[FieldRole]
    build_query: Callable[..., dict[str, Any]]
    # Optional: additional roles that must exist for certain slots
    optional_roles: list[FieldRole] = field(default_factory=list)


def _pick_field(schema: SchemaDef, role: FieldRole, rng: random.Random) -> FieldDef | None:
    candidates = schema.fields_by_role(role)
    return rng.choice(candidates) if candidates else None


def _pick_fields(schema: SchemaDef, role: FieldRole, n: int, rng: random.Random) -> list[FieldDef]:
    candidates = schema.fields_by_role(role)
    return rng.sample(candidates, min(n, len(candidates)))


def _sample_enum_values(field_def: FieldDef, rng: random.Random) -> list[str]:
    if field_def.enum_values:
        k = rng.randint(1, min(3, len(field_def.enum_values)))
        return rng.sample(field_def.enum_values, k)
    return ["value_a", "value_b"]


def _sample_enum_value(field_def: FieldDef, rng: random.Random) -> str:
    if field_def.enum_values:
        return rng.choice(field_def.enum_values)
    return "some_value"


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _build_filter_only(schema: SchemaDef, cat: FieldDef, value: str, **_: Any) -> dict:
    return {
        "type": "find",
        "filter": {cat.name: value},
    }


def _build_aggregate_single(
    schema: SchemaDef, measure: FieldDef, cat: FieldDef, agg_op: str, **_: Any
) -> dict:
    mongo_op = AGG_OPS[agg_op]
    return {
        "type": "aggregate",
        "pipeline": [
            {"$group": {"_id": f"${cat.name}", agg_op: {mongo_op: f"${measure.name}"}}},
        ],
    }


def _build_aggregate_filtered(
    schema: SchemaDef, measure: FieldDef, cat: FieldDef, agg_op: str,
    filter_field: FieldDef, filter_value: str, **_: Any,
) -> dict:
    mongo_op = AGG_OPS[agg_op]
    return {
        "type": "aggregate",
        "pipeline": [
            {"$match": {filter_field.name: filter_value}},
            {"$group": {"_id": f"${cat.name}", agg_op: {mongo_op: f"${measure.name}"}}},
        ],
    }


def _build_time_range(
    schema: SchemaDef, measure: FieldDef, ts: FieldDef,
    start: str, end: str, **_: Any,
) -> dict:
    return {
        "type": "find",
        "filter": {
            ts.name: {
                "$gte": {"$date": start},
                "$lte": {"$date": end},
            },
        },
        "projection": {measure.name: 1, ts.name: 1},
    }


def _build_top_n(
    schema: SchemaDef, measure: FieldDef, n: int, **_: Any,
) -> dict:
    return {
        "type": "aggregate",
        "pipeline": [
            {"$sort": {measure.name: -1}},
            {"$limit": n},
        ],
    }


def _build_multi_group(
    schema: SchemaDef, measure: FieldDef, cat1: FieldDef, cat2: FieldDef,
    agg_op: str, **_: Any,
) -> dict:
    mongo_op = AGG_OPS[agg_op]
    return {
        "type": "aggregate",
        "pipeline": [
            {"$group": {
                "_id": {cat1.name: f"${cat1.name}", cat2.name: f"${cat2.name}"},
                agg_op: {mongo_op: f"${measure.name}"},
            }},
        ],
    }


def _build_count(
    schema: SchemaDef, filter_field: FieldDef, filter_value: str, **_: Any,
) -> dict:
    return {
        "type": "aggregate",
        "pipeline": [
            {"$match": {filter_field.name: filter_value}},
            {"$count": "total"},
        ],
    }


def _build_exists_check(schema: SchemaDef, target: FieldDef, **_: Any) -> dict:
    return {
        "type": "find",
        "filter": {target.name: {"$exists": True, "$ne": None}},
    }


def _build_enum_filter(
    schema: SchemaDef, enum_field: FieldDef, values: list[str], **_: Any,
) -> dict:
    return {
        "type": "find",
        "filter": {enum_field.name: {"$in": values}},
    }


def _build_date_bucket(
    schema: SchemaDef, measure: FieldDef, ts: FieldDef,
    agg_op: str, time_unit: str, **_: Any,
) -> dict:
    mongo_op = AGG_OPS[agg_op]
    date_trunc = {
        "year": {"$year": f"${ts.name}"},
        "month": {"$month": f"${ts.name}"},
        "day": {"$dayOfMonth": f"${ts.name}"},
    }
    return {
        "type": "aggregate",
        "pipeline": [
            {"$group": {
                "_id": date_trunc.get(time_unit, {"$month": f"${ts.name}"}),
                agg_op: {mongo_op: f"${measure.name}"},
            }},
            {"$sort": {"_id": 1}},
        ],
    }


# ---------------------------------------------------------------------------
# Template generation helpers — each returns (intent_str, query_dict) pairs
# ---------------------------------------------------------------------------

def generate_filter_only(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    # Use category fields and enum fields
    for role in (FieldRole.category, FieldRole.enum):
        for f in schema.fields_by_role(role):
            value = _sample_enum_value(f, rng) if f.enum_values else f"sample_{f.name}"
            templates = [
                f"Show all {schema.collection} where {f.name} is {value}",
                f"Find {schema.collection} with {f.name} equal to {value}",
                f"List {schema.collection} that have {f.name} set to {value}",
            ]
            intent = rng.choice(templates)
            query = _build_filter_only(schema, f, value)
            results.append((intent, query))
    return results


def generate_aggregate_single(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    measures = schema.fields_by_role(FieldRole.measure)
    cats = schema.fields_by_role(FieldRole.category) + schema.fields_by_role(FieldRole.enum)
    for m in measures:
        for c in cats:
            agg_op = rng.choice(AGG_OP_NAMES)
            templates = [
                f"What is the {agg_op} {m.name} per {c.name}?",
                f"Calculate the {agg_op} of {m.name} grouped by {c.name}",
                f"Show {agg_op} {m.name} for each {c.name}",
            ]
            intent = rng.choice(templates)
            query = _build_aggregate_single(schema, m, c, agg_op)
            results.append((intent, query))
    return results


def generate_aggregate_filtered(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    measures = schema.fields_by_role(FieldRole.measure)
    cats = schema.fields_by_role(FieldRole.category) + schema.fields_by_role(FieldRole.enum)
    filter_candidates = schema.fields_by_role(FieldRole.enum) + schema.fields_by_role(FieldRole.category)
    if not measures or len(cats) < 1 or not filter_candidates:
        return results
    for _ in range(min(3, len(measures) * len(cats))):
        m = rng.choice(measures)
        c = rng.choice(cats)
        ff = rng.choice(filter_candidates)
        if ff.name == c.name and len(cats) > 1:
            c = rng.choice([x for x in cats if x.name != ff.name])
        fv = _sample_enum_value(ff, rng)
        agg_op = rng.choice(AGG_OP_NAMES)
        templates = [
            f"What is the {agg_op} {m.name} for {ff.name} = {fv}, grouped by {c.name}?",
            f"Show {agg_op} {m.name} by {c.name} where {ff.name} is {fv}",
        ]
        intent = rng.choice(templates)
        query = _build_aggregate_filtered(schema, m, c, agg_op, ff, fv)
        results.append((intent, query))
    return results


def generate_time_range(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    ts_fields = schema.fields_by_role(FieldRole.timestamp)
    measures = schema.fields_by_role(FieldRole.measure)
    if not ts_fields or not measures:
        return results
    ts = rng.choice(ts_fields)
    m = rng.choice(measures)
    start = "2024-01-01T00:00:00Z"
    end = "2024-06-30T23:59:59Z"
    templates = [
        f"Show {m.name} between {start} and {end}",
        f"Get {m.name} from {ts.name} ranging {start} to {end}",
    ]
    intent = rng.choice(templates)
    query = _build_time_range(schema, m, ts, start, end)
    results.append((intent, query))
    return results


def generate_top_n(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    measures = schema.fields_by_role(FieldRole.measure)
    for m in measures:
        n = rng.choice([3, 5, 10])
        templates = [
            f"Top {n} {schema.collection} by {m.name}",
            f"Show the {n} highest {m.name} in {schema.collection}",
        ]
        intent = rng.choice(templates)
        query = _build_top_n(schema, m, n)
        results.append((intent, query))
    return results


def generate_multi_group(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    measures = schema.fields_by_role(FieldRole.measure)
    cats = schema.fields_by_role(FieldRole.category) + schema.fields_by_role(FieldRole.enum)
    if len(cats) < 2 or not measures:
        return results
    m = rng.choice(measures)
    pair = rng.sample(cats, 2)
    agg_op = rng.choice(AGG_OP_NAMES)
    templates = [
        f"{agg_op} of {m.name} by {pair[0].name} and {pair[1].name}",
        f"Group {schema.collection} by {pair[0].name} and {pair[1].name}, show {agg_op} {m.name}",
    ]
    intent = rng.choice(templates)
    query = _build_multi_group(schema, m, pair[0], pair[1], agg_op)
    results.append((intent, query))
    return results


def generate_count(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    filter_candidates = schema.fields_by_role(FieldRole.enum) + schema.fields_by_role(FieldRole.boolean)
    for f in filter_candidates:
        if f.role == FieldRole.boolean:
            fv: Any = True
            templates = [
                f"How many {schema.collection} have {f.name} set to true?",
                f"Count {schema.collection} where {f.name} is true",
            ]
        else:
            fv = _sample_enum_value(f, rng)
            templates = [
                f"How many {schema.collection} have {f.name} equal to {fv}?",
                f"Count {schema.collection} where {f.name} is {fv}",
            ]
        intent = rng.choice(templates)
        query = _build_count(schema, f, fv)
        results.append((intent, query))
    return results


def generate_exists_check(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    # Pick a non-identifier field
    candidates = [f for f in schema.fields if f.role != FieldRole.identifier]
    if not candidates:
        return results
    target = rng.choice(candidates)
    templates = [
        f"Which {schema.collection} have a {target.name}?",
        f"Find {schema.collection} where {target.name} exists",
    ]
    intent = rng.choice(templates)
    query = _build_exists_check(schema, target)
    results.append((intent, query))
    return results


def generate_enum_filter(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    enum_fields = schema.fields_by_role(FieldRole.enum)
    for f in enum_fields:
        values = _sample_enum_values(f, rng)
        if len(values) < 2 and f.enum_values and len(f.enum_values) >= 2:
            values = rng.sample(f.enum_values, 2)
        templates = [
            f"Show {schema.collection} where {f.name} is one of {', '.join(values)}",
            f"Find {schema.collection} with {f.name} in [{', '.join(values)}]",
        ]
        intent = rng.choice(templates)
        query = _build_enum_filter(schema, f, values)
        results.append((intent, query))
    return results


def generate_date_bucket(schema: SchemaDef, rng: random.Random) -> list[tuple[str, dict]]:
    results = []
    ts_fields = schema.fields_by_role(FieldRole.timestamp)
    measures = schema.fields_by_role(FieldRole.measure)
    if not ts_fields or not measures:
        return results
    ts = rng.choice(ts_fields)
    m = rng.choice(measures)
    time_unit = rng.choice(["year", "month", "day"])
    agg_op = rng.choice(AGG_OP_NAMES)
    templates = [
        f"{agg_op} of {m.name} by {time_unit} from {ts.name}",
        f"Show {agg_op} {m.name} bucketed by {time_unit} using {ts.name}",
    ]
    intent = rng.choice(templates)
    query = _build_date_bucket(schema, m, ts, agg_op, time_unit)
    results.append((intent, query))
    return results


# ---------------------------------------------------------------------------
# All generators in order
# ---------------------------------------------------------------------------
ALL_GENERATORS = [
    generate_filter_only,
    generate_aggregate_single,
    generate_aggregate_filtered,
    generate_time_range,
    generate_top_n,
    generate_multi_group,
    generate_count,
    generate_exists_check,
    generate_enum_filter,
    generate_date_bucket,
]
