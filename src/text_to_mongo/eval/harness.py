from __future__ import annotations

import json
from typing import Any

from text_to_mongo.schema import (
    EvalReport,
    EvalResult,
    FieldResult,
    OperatorResult,
    SyntaxResult,
    TrainingExample,
)
from text_to_mongo.eval.fields import eval_fields
from text_to_mongo.eval.generalization import eval_generalization
from text_to_mongo.eval.operators import eval_operators
from text_to_mongo.eval.syntax import eval_syntax


def _eval_one(example: TrainingExample, prediction: str) -> EvalResult:
    # Layer 1: Syntax
    syntax = eval_syntax(prediction)

    # Default results for layers that depend on valid parse
    operators = OperatorResult()
    fields = FieldResult()

    if syntax.passed:
        parsed = json.loads(prediction)

        # Layer 2: Operators
        operators = eval_operators(parsed, example.allowed_ops.all_operators)

        # Layer 3: Fields
        fields = eval_fields(parsed, example.schema_def)

    passed_all = syntax.passed and operators.passed and fields.passed

    return EvalResult(
        example=example,
        prediction=prediction,
        syntax=syntax,
        operators=operators,
        fields=fields,
        passed_all=passed_all,
    )


def run_eval(
    examples: list[TrainingExample],
    predictions: list[str],
    held_out_schemas: set[str] | None = None,
) -> EvalReport:
    if len(examples) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(examples)} examples vs {len(predictions)} predictions"
        )

    results = [_eval_one(ex, pred) for ex, pred in zip(examples, predictions)]
    total = len(results)

    if total == 0:
        return EvalReport(results=results, total=0)

    syntax_pass = sum(1 for r in results if r.syntax.passed) / total
    ops_pass = sum(1 for r in results if r.operators.passed) / total
    field_pass = sum(1 for r in results if r.fields.passed) / total
    overall_pass = sum(1 for r in results if r.passed_all) / total

    # Layer 4: Generalization (if held-out schemas specified)
    generalization = None
    if held_out_schemas:
        train_results = [
            r for r in results if r.example.schema_def.collection not in held_out_schemas
        ]
        held_results = [
            r for r in results if r.example.schema_def.collection in held_out_schemas
        ]
        if held_results:
            generalization = eval_generalization(train_results, held_results)

    return EvalReport(
        results=results,
        total=total,
        syntax_pass_rate=syntax_pass,
        operator_pass_rate=ops_pass,
        field_pass_rate=field_pass,
        overall_pass_rate=overall_pass,
        generalization=generalization,
    )
