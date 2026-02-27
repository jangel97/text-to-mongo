"""Prompt template builder for ChatML (Qwen2.5-Coder) format."""
from __future__ import annotations

import json

from text_to_mongo.schema import TrainingExample


SYSTEM_PROMPT = (
    "You are a MongoDB query generator. Given a collection schema, a list of "
    "allowed operators, and a natural language intent, produce a valid MongoDB "
    "query as JSON. The output must be a JSON object with a 'type' field "
    "('aggregate' or 'find') and the corresponding query body. "
    "If the intent references fields not in the schema, respond with "
    '{"error": "<reason>"}. '
    "Use only the allowed operators."
)


def _render_schema(example: TrainingExample) -> str:
    schema = example.schema_def
    lines = [f"Collection: {schema.collection}"]
    lines.append("Fields:")
    for f in schema.fields:
        parts = [f"  - {f.name} ({f.type}, {f.role.value})"]
        if f.description:
            parts.append(f": {f.description}")
        if f.enum_values:
            parts.append(f" [values: {', '.join(f.enum_values)}]")
        lines.append("".join(parts))
    return "\n".join(lines)


def _render_allowed_ops(example: TrainingExample) -> str:
    ops = example.allowed_ops
    lines = ["Allowed stage operators: " + ", ".join(ops.stage_operators)]
    lines.append("Allowed expression operators: " + ", ".join(ops.expression_operators))
    return "\n".join(lines)


def _render_user_message(example: TrainingExample) -> str:
    parts = [
        _render_schema(example),
        "",
        _render_allowed_ops(example),
        "",
        f"Intent: {example.intent}",
    ]
    return "\n".join(parts)


def build_prompt(
    example: TrainingExample,
    include_output: bool = False,
) -> str:
    """Build a ChatML-formatted prompt string.

    Args:
        example: The training example to render.
        include_output: If True, append the expected output (for training data).

    Returns:
        Formatted prompt string.
    """
    user_msg = _render_user_message(example)
    output_str = json.dumps(example.output, ensure_ascii=False)

    parts = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>",
        f"<|im_start|>user\n{user_msg}<|im_end|>",
    ]
    if include_output:
        parts.append(f"<|im_start|>assistant\n{output_str}<|im_end|>")
    else:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
