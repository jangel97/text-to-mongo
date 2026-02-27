from __future__ import annotations

import json
from typing import Any

from text_to_mongo.schema import SyntaxResult


def eval_syntax(raw_output: str) -> SyntaxResult:
    result = SyntaxResult()

    # 1. Valid JSON
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        result.errors.append("Invalid JSON")
        return result
    result.valid_json = True

    if not isinstance(parsed, dict):
        result.errors.append("Top-level value must be an object")
        return result

    # 2. Has `type` field
    if "type" not in parsed:
        result.errors.append("Missing 'type' field")
        return result
    result.has_type = True
    result.type_value = parsed["type"]

    # 3. type is aggregate or find
    if parsed["type"] not in ("aggregate", "find"):
        result.errors.append(f"Invalid type '{parsed['type']}'; expected 'aggregate' or 'find'")
        return result

    # 4. Body present
    if parsed["type"] == "aggregate":
        if "pipeline" not in parsed:
            result.errors.append("Aggregate query missing 'pipeline'")
            return result
        result.has_body = True
        pipeline = parsed["pipeline"]

        # 5. Pipeline well-formedness
        if not isinstance(pipeline, list):
            result.errors.append("'pipeline' must be a list")
            return result

        for i, stage in enumerate(pipeline):
            if not isinstance(stage, dict):
                result.errors.append(f"Pipeline stage {i} is not an object")
                return result
            dollar_keys = [k for k in stage if k.startswith("$")]
            if len(dollar_keys) != 1:
                result.errors.append(
                    f"Pipeline stage {i} must have exactly one $-prefixed key, got {len(dollar_keys)}"
                )
                return result
        result.pipeline_well_formed = True

    elif parsed["type"] == "find":
        if "filter" not in parsed:
            result.errors.append("Find query missing 'filter'")
            return result
        result.has_body = True
        if not isinstance(parsed["filter"], dict):
            result.errors.append("'filter' must be an object")
            return result
        # find queries don't have pipeline — mark as well-formed
        result.pipeline_well_formed = True

    result.passed = True
    return result
