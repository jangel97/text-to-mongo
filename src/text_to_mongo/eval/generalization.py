from __future__ import annotations

from text_to_mongo.schema import EvalResult, GeneralizationResult


def _pass_rate(results: list[EvalResult], attr: str) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if getattr(getattr(r, attr), "passed", False)) / len(results)


def eval_generalization(
    train_results: list[EvalResult],
    held_out_results: list[EvalResult],
) -> GeneralizationResult:
    train_syntax = _pass_rate(train_results, "syntax")
    held_syntax = _pass_rate(held_out_results, "syntax")
    train_ops = _pass_rate(train_results, "operators")
    held_ops = _pass_rate(held_out_results, "operators")
    train_fields = _pass_rate(train_results, "fields")
    held_fields = _pass_rate(held_out_results, "fields")

    gaps = {
        "syntax": train_syntax - held_syntax,
        "operators": train_ops - held_ops,
        "fields": train_fields - held_fields,
    }

    flagged = any(abs(g) > 0.05 for g in gaps.values())

    return GeneralizationResult(
        train_syntax_pass_rate=train_syntax,
        held_out_syntax_pass_rate=held_syntax,
        train_operator_pass_rate=train_ops,
        held_out_operator_pass_rate=held_ops,
        train_field_pass_rate=train_fields,
        held_out_field_pass_rate=held_fields,
        gaps=gaps,
        flagged=flagged,
    )
