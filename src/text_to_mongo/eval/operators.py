from __future__ import annotations

from typing import Any

from text_to_mongo.schema import OperatorResult

UNSAFE_OPERATORS: frozenset[str] = frozenset({
    "$where",
    "$function",
    "$accumulator",
    "$merge",
    "$out",
    "$currentOp",
    "$collStats",
    "$indexStats",
    "$planCacheStats",
})

# Extended JSON type wrappers — these are value literals, not query operators.
_EJSON_KEYS: frozenset[str] = frozenset({
    "$date", "$oid", "$numberLong", "$numberInt", "$numberDouble",
    "$numberDecimal", "$binary", "$timestamp", "$regex", "$undefined",
    "$minKey", "$maxKey", "$dbPointer", "$symbol", "$code",
})


def extract_operators(obj: Any) -> set[str]:
    """Recursively extract all $-prefixed keys from a nested structure.

    Skips Extended JSON type wrapper keys (e.g. ``$date``, ``$oid``).
    """
    ops: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.startswith("$") and key not in _EJSON_KEYS:
                ops.add(key)
            ops.update(extract_operators(value))
    elif isinstance(obj, list):
        for item in obj:
            ops.update(extract_operators(item))
    return ops


def eval_operators(query: dict[str, Any], allowed: list[str]) -> OperatorResult:
    used = extract_operators(query)
    allowed_set = set(allowed)
    violations = used - allowed_set
    unsafe = used & UNSAFE_OPERATORS

    passed = len(violations) == 0 and len(unsafe) == 0
    return OperatorResult(
        used_operators=used,
        violations=violations,
        unsafe_operators=unsafe,
        passed=passed,
    )
