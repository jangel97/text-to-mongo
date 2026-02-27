from __future__ import annotations

from typing import Any

from text_to_mongo.schema import FieldResult, SchemaDef

# Fields that are always valid (implicit in every collection)
IMPLICIT_FIELDS: frozenset[str] = frozenset({"_id"})

# Operators whose direct child keys are output aliases, not field references.
# For these, we recurse into values but don't treat keys as field names.
_ALIAS_OPERATORS: frozenset[str] = frozenset({"$group", "$bucket", "$bucketAuto"})


def extract_field_refs(obj: Any, *, _inside_alias_op: bool = False) -> set[str]:
    """Recursively extract field references from a MongoDB query.

    Handles:
    - String values starting with "$" (field references like "$price", "$addr.city")
    - Dict keys that are plain field names (non-operator) in $match, $project, etc.
    - Skips output alias keys in $group/$bucket stages
    """
    refs: set[str] = set()
    if isinstance(obj, str):
        if obj.startswith("$") and not obj.startswith("$$"):
            field_path = obj[1:]
            root_field = field_path.split(".")[0]
            refs.add(root_field)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key.startswith("$"):
                # Operator — recurse into its value
                is_alias = key in _ALIAS_OPERATORS
                refs.update(extract_field_refs(value, _inside_alias_op=is_alias))
            elif _inside_alias_op:
                # Inside $group etc.: keys are output aliases, not field refs.
                # Still recurse into values to find $-prefixed field references.
                refs.update(extract_field_refs(value))
            else:
                # Plain field name used as key (e.g., in $match: {"status": "active"})
                root_field = key.split(".")[0]
                refs.add(root_field)
                refs.update(extract_field_refs(value))
    elif isinstance(obj, list):
        for item in obj:
            refs.update(extract_field_refs(item, _inside_alias_op=_inside_alias_op))
    return refs


def eval_fields(query: dict[str, Any], schema: SchemaDef) -> FieldResult:
    # Extract from the body of the query (pipeline or filter), not the top-level wrapper
    if "pipeline" in query:
        body = query["pipeline"]
    elif "filter" in query:
        body = query["filter"]
    else:
        body = query
    refs = extract_field_refs(body)

    # Remove implicit fields and the "type" key (not a real field ref)
    refs -= IMPLICIT_FIELDS
    refs.discard("type")

    schema_fields = schema.field_names
    hallucinated = refs - schema_fields - IMPLICIT_FIELDS

    coverage = len(refs & schema_fields) / len(schema_fields) if schema_fields else 0.0

    return FieldResult(
        referenced_fields=refs,
        hallucinated_fields=hallucinated,
        coverage=coverage,
        passed=len(hallucinated) == 0,
    )
