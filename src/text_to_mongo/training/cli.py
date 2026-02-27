"""Unified CLI for the training pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from text_to_mongo.training.config import MODELS, LoraConfig, TrainingConfig


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), required=True,
        help="Model to use",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory containing JSONL data files (default: data/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs"),
        help="Parent directory for run outputs (default: runs/)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Generation batch size (default: 8)",
    )


def cmd_baseline(args: argparse.Namespace) -> None:
    """Run zero-shot baseline evaluation."""
    from text_to_mongo.training.baseline import run_baseline

    reports = run_baseline(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    for split, report in reports.items():
        print(f"\n{split}: syntax={report.syntax_pass_rate:.1%}, "
              f"operators={report.operator_pass_rate:.1%}, "
              f"fields={report.field_pass_rate:.1%}, "
              f"overall={report.overall_pass_rate:.1%}")


def cmd_train(args: argparse.Namespace) -> None:
    """Run LoRA fine-tuning."""
    from text_to_mongo.training.trainer import run_training

    model_config = MODELS[args.model]
    lora_config = LoraConfig(r=args.lora_r, alpha=args.lora_alpha)
    training_config = TrainingConfig(
        model=model_config,
        lora=lora_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
    )

    adapter_dir = run_training(training_config)
    print(f"\nAdapter saved to: {adapter_dir}")
    print(f"Run eval with: python -m text_to_mongo.training eval "
          f"--model {args.model} --adapter {adapter_dir} "
          f"--run-name {training_config.run_name}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run post-training evaluation."""
    from text_to_mongo.training.compare import run_post_training_eval

    reports = run_post_training_eval(
        model_name=args.model,
        adapter_path=args.adapter,
        run_name=args.run_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    for split, report in reports.items():
        print(f"\n{split}: syntax={report.syntax_pass_rate:.1%}, "
              f"operators={report.operator_pass_rate:.1%}, "
              f"fields={report.field_pass_rate:.1%}, "
              f"overall={report.overall_pass_rate:.1%}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Build comparison table from all runs."""
    from text_to_mongo.training.compare import build_comparison_table

    table = build_comparison_table(args.output_dir)
    print(table)

    # Also save to file
    out_path = args.output_dir / "comparison.md"
    out_path.write_text(table)
    print(f"\nSaved to {out_path}")


def main(argv: list[str] | None = None) -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        prog="text-to-mongo-train",
        description="Text-to-MongoDB LoRA training pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # baseline
    p_baseline = subparsers.add_parser("baseline", help="Run zero-shot baseline evaluation")
    _add_common_args(p_baseline)
    p_baseline.set_defaults(func=cmd_baseline)

    # train
    p_train = subparsers.add_parser("train", help="Run LoRA fine-tuning")
    _add_common_args(p_train)
    p_train.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
    p_train.add_argument("--lora-alpha", type=int, default=0, help="LoRA alpha (default: 2*r)")
    p_train.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    p_train.add_argument("--train-batch-size", type=int, default=4, help="Training batch size (default: 4)")
    p_train.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    p_train.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run post-training evaluation")
    _add_common_args(p_eval)
    p_eval.add_argument("--adapter", type=str, required=True, help="Path to saved adapter directory")
    p_eval.add_argument("--run-name", type=str, required=True, help="Name for this eval run")
    p_eval.set_defaults(func=cmd_eval)

    # compare
    p_compare = subparsers.add_parser("compare", help="Build comparison table from all runs")
    p_compare.add_argument(
        "--output-dir", type=Path, default=Path("runs"),
        help="Directory containing run subdirectories (default: runs/)",
    )
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)

    # Handle lora-alpha default: 0 means auto (2*r)
    if hasattr(args, "lora_alpha") and args.lora_alpha == 0:
        args.lora_alpha = 2 * args.lora_r

    args.func(args)
