"""Model loading for inference and batched generation."""
from __future__ import annotations

import logging
import re

from text_to_mongo.schema import TrainingExample
from text_to_mongo.prompt import build_prompt
from text_to_mongo.training.config import ModelConfig

logger = logging.getLogger(__name__)


def load_model_for_inference(model_config: ModelConfig, adapter_path: str | None = None):
    """Load a 4-bit base model with optional LoRA adapter merged.

    Args:
        model_config: Model configuration.
        adapter_path: Path to a saved LoRA adapter directory. If None, loads base model only.

    Returns:
        (model, tokenizer) tuple ready for generation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if adapter_path:
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        # Do NOT merge_and_unload — merging into 4-bit quantized weights
        # silently fails. Keep PeftModel for inference instead.
        logger.info("Loaded adapter from %s (PeftModel, not merged)", adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Loaded base model (no adapter): %s", model_config.hf_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.hf_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def generate_predictions(
    model,
    tokenizer,
    examples: list[TrainingExample],
    model_config: ModelConfig,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> list[str]:
    """Generate predictions for a list of examples using batched greedy decoding.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        examples: List of TrainingExample to generate for.
        model_config: Model config (for prompt format).
        max_new_tokens: Maximum tokens to generate per example.
        batch_size: Number of examples per batch.

    Returns:
        List of generated strings (one per example).
    """
    import torch

    predictions: list[str] = []

    # Format prompts (without output — inference mode)
    prompts = [build_prompt(ex, include_output=False) for ex in examples]

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        logger.info("Generating batch %d/%d (%d examples)", i // batch_size + 1, (len(prompts) + batch_size - 1) // batch_size, len(batch_prompts))

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated portion (exclude prompt tokens)
        for j, output_ids in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated_ids = output_ids[prompt_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            predictions.append(text)

    return predictions


def extract_json(text: str) -> str:
    """Extract the first JSON object from potentially noisy model output.

    Uses brace-depth matching to find the first complete JSON object.

    Args:
        text: Raw model output string.

    Returns:
        The extracted JSON string, or the original text if no JSON found.
    """
    # Try to find the start of a JSON object
    start = text.find("{")
    if start == -1:
        return text.strip()

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]

        if escape:
            escape = False
            continue

        if c == "\\":
            if in_string:
                escape = True
            continue

        if c == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # If we never balanced, return from start to end
    return text[start:]
