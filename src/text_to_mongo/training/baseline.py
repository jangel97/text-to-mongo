"""Zero-shot baseline evaluation (no adapter)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from text_to_mongo.schema import TrainingExample, EvalReport
from text_to_mongo.eval.harness import run_eval
from text_to_mongo.data.schemas import HELD_OUT_COLLECTIONS
from text_to_mongo.training.config import ModelConfig, MODELS
from text_to_mongo.training.dataset import load_examples
from text_to_mongo.training.inference import (
    load_model_for_inference,
    generate_predictions,
    extract_json,
)

logger = logging.getLogger(__name__)


def _save_predictions(
    examples: list[TrainingExample],
    predictions: list[str],
    path: Path,
) -> None:
    """Save predictions to a JSONL file for inspection."""
    with open(path, "w") as f:
        for ex, pred in zip(examples, predictions):
            record = {
                "collection": ex.schema_def.collection,
                "intent": ex.intent,
                "expected": ex.output,
                "prediction": pred,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_report(report: EvalReport, path: Path) -> None:
    """Save an EvalReport as JSON."""
    with open(path, "w") as f:
        f.write(report.model_dump_json(indent=2))


def _run_split(
    model,
    tokenizer,
    model_config: ModelConfig,
    examples: list[TrainingExample],
    split_name: str,
    run_dir: Path,
    batch_size: int,
    held_out_schemas: set[str] | None = None,
) -> EvalReport:
    """Generate predictions and evaluate for a single split."""
    logger.info("Evaluating %s split (%d examples)", split_name, len(examples))

    raw_predictions = generate_predictions(
        model, tokenizer, examples, model_config, batch_size=batch_size,
    )

    # Extract JSON from raw output
    predictions = [extract_json(p) for p in raw_predictions]

    # Run eval harness
    report = run_eval(examples, predictions, held_out_schemas=held_out_schemas)

    # Save artifacts
    _save_predictions(examples, predictions, run_dir / f"{split_name}_predictions.jsonl")
    _save_report(report, run_dir / f"{split_name}_report.json")

    logger.info(
        "%s results — syntax: %.1f%%, operators: %.1f%%, fields: %.1f%%, overall: %.1f%%",
        split_name,
        report.syntax_pass_rate * 100,
        report.operator_pass_rate * 100,
        report.field_pass_rate * 100,
        report.overall_pass_rate * 100,
    )

    return report


def run_baseline(
    model_name: str,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("runs"),
    batch_size: int = 8,
) -> dict[str, EvalReport]:
    """Run zero-shot baseline evaluation for a model.

    Args:
        model_name: Key from MODELS dict (e.g. "qwen2.5-coder-7b").
        data_dir: Directory containing eval.jsonl and held_out.jsonl.
        output_dir: Parent directory for run output.
        batch_size: Generation batch size.

    Returns:
        Dict mapping split name to EvalReport.
    """
    model_config = MODELS[model_name]
    run_dir = output_dir / f"{model_name}_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running baseline for %s", model_name)
    logger.info("Output: %s", run_dir)

    # Load model (no adapter)
    model, tokenizer = load_model_for_inference(model_config, adapter_path=None)

    # Load eval splits
    eval_examples = load_examples(data_dir / "eval.jsonl")
    held_out_examples = load_examples(data_dir / "held_out.jsonl")

    reports: dict[str, EvalReport] = {}

    reports["eval"] = _run_split(
        model, tokenizer, model_config, eval_examples,
        "eval", run_dir, batch_size,
    )

    reports["held_out"] = _run_split(
        model, tokenizer, model_config, held_out_examples,
        "held_out", run_dir, batch_size,
        held_out_schemas=HELD_OUT_COLLECTIONS,
    )

    logger.info("Baseline complete for %s", model_name)
    return reports
