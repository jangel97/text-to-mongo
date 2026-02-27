"""Post-training evaluation and comparison table generation."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from text_to_mongo.schema import EvalReport
from text_to_mongo.eval.harness import run_eval
from text_to_mongo.data.schemas import HELD_OUT_COLLECTIONS
from text_to_mongo.training.config import ModelConfig, MODELS
from text_to_mongo.training.dataset import load_examples
from text_to_mongo.training.inference import (
    load_model_for_inference,
    generate_predictions,
    extract_json,
)
from text_to_mongo.training.baseline import _save_predictions, _save_report, _run_split

logger = logging.getLogger(__name__)


def run_post_training_eval(
    model_name: str,
    adapter_path: str,
    run_name: str,
    data_dir: Path = Path("data"),
    output_dir: Path = Path("runs"),
    batch_size: int = 8,
) -> dict[str, EvalReport]:
    """Evaluate a fine-tuned adapter on eval and held-out splits.

    Args:
        model_name: Key from MODELS dict.
        adapter_path: Path to the saved adapter directory.
        run_name: Name for this run (used as output subdirectory).
        data_dir: Directory containing eval.jsonl and held_out.jsonl.
        output_dir: Parent directory for run output.
        batch_size: Generation batch size.

    Returns:
        Dict mapping split name to EvalReport.
    """
    model_config = MODELS[model_name]
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running post-training eval: %s (adapter: %s)", run_name, adapter_path)

    # Load model with adapter
    model, tokenizer = load_model_for_inference(model_config, adapter_path=adapter_path)

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

    return reports


def build_comparison_table(runs_dir: Path = Path("runs")) -> str:
    """Scan run directories and build a markdown comparison table.

    Looks for *_report.json files in each subdirectory of runs_dir.

    Args:
        runs_dir: Directory containing run subdirectories.

    Returns:
        Markdown-formatted comparison table.
    """
    rows: list[dict[str, str | float]] = []

    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue

        for report_file in sorted(run_path.glob("*_report.json")):
            split = report_file.stem.replace("_report", "")
            try:
                report = EvalReport.model_validate_json(report_file.read_text())
            except Exception as e:
                logger.warning("Failed to load %s: %s", report_file, e)
                continue

            rows.append({
                "run": run_path.name,
                "split": split,
                "total": report.total,
                "syntax": report.syntax_pass_rate,
                "operators": report.operator_pass_rate,
                "fields": report.field_pass_rate,
                "overall": report.overall_pass_rate,
            })

    if not rows:
        return "No results found."

    # Build markdown table
    lines = [
        "# Training Comparison Results",
        "",
        "| Run | Split | N | Syntax | Operators | Fields | Overall |",
        "|-----|-------|---|--------|-----------|--------|---------|",
    ]

    for r in rows:
        lines.append(
            f"| {r['run']} | {r['split']} | {r['total']} "
            f"| {r['syntax']:.1%} | {r['operators']:.1%} "
            f"| {r['fields']:.1%} | {r['overall']:.1%} |"
        )

    return "\n".join(lines)
