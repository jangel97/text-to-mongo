"""JSONL data loading and HuggingFace Dataset construction."""
from __future__ import annotations

import json
from pathlib import Path

from text_to_mongo.schema import TrainingExample
from text_to_mongo.prompt import build_prompt
from text_to_mongo.training.config import ModelConfig


def load_examples(path: Path) -> list[TrainingExample]:
    """Read a JSONL file and return TrainingExample objects."""
    examples: list[TrainingExample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(TrainingExample.model_validate(json.loads(line)))
    return examples


def format_example(example: TrainingExample, model_config: ModelConfig, include_output: bool = True) -> str:
    """Format a single example using the prompt builder."""
    return build_prompt(example, include_output=include_output)


def format_examples(
    examples: list[TrainingExample],
    model_config: ModelConfig,
    include_output: bool = True,
) -> list[str]:
    """Format a list of examples into prompt strings."""
    return [format_example(ex, model_config, include_output) for ex in examples]


def format_prompt_completion(
    example: TrainingExample,
    model_config: ModelConfig,
) -> tuple[str, str]:
    """Split an example into prompt and completion strings.

    Returns:
        (prompt, completion) tuple for prompt-completion dataset format.
    """
    import json as _json

    prompt = build_prompt(example, include_output=False)
    completion = _json.dumps(example.output, ensure_ascii=False) + "<|im_end|>"

    return prompt, completion


def build_hf_dataset(
    data_dir: Path,
    model_config: ModelConfig,
    split: str = "train",
):
    """Build a HuggingFace Dataset with 'prompt' and 'completion' columns for SFTTrainer.

    trl >= 0.29 uses prompt-completion format for completion-only loss masking.

    Args:
        data_dir: Directory containing {split}.jsonl files.
        model_config: Model config for prompt formatting.
        split: One of 'train', 'eval', or 'held_out'.

    Returns:
        datasets.Dataset with 'prompt' and 'completion' columns.
    """
    from datasets import Dataset

    path = data_dir / f"{split}.jsonl"
    examples = load_examples(path)

    prompts = []
    completions = []
    for ex in examples:
        p, c = format_prompt_completion(ex, model_config)
        prompts.append(p)
        completions.append(c)

    return Dataset.from_dict({"prompt": prompts, "completion": completions})
