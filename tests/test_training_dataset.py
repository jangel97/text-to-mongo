"""Tests for training data loading and formatting (no GPU required)."""
import json
from pathlib import Path

import pytest

from text_to_mongo.schema import TrainingExample
from text_to_mongo.training.config import MODELS
from text_to_mongo.training.dataset import load_examples, format_example, format_examples, format_prompt_completion
from text_to_mongo.training.inference import extract_json


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def sample_record() -> dict:
    """A minimal training example as a raw dict."""
    return {
        "schema": {
            "collection": "orders",
            "domain": "ecommerce",
            "fields": [
                {"name": "order_id", "type": "string", "role": "identifier", "description": "Order ID"},
                {"name": "total", "type": "double", "role": "measure", "description": "Order total"},
                {"name": "status", "type": "string", "role": "enum", "description": "Order status",
                 "enum_values": ["pending", "shipped", "delivered"]},
            ],
        },
        "allowed_ops": {
            "stage_operators": ["$match", "$group"],
            "expression_operators": ["$sum", "$avg", "$eq"],
        },
        "intent": "Find all pending orders",
        "output": {"type": "find", "filter": {"status": "pending"}},
        "is_negative": False,
    }


@pytest.fixture
def sample_example(sample_record: dict) -> TrainingExample:
    return TrainingExample.model_validate(sample_record)


class TestLoadExamples:
    @pytest.mark.skipif(not DATA_DIR.exists(), reason="data/ not generated")
    def test_load_train_examples(self):
        examples = load_examples(DATA_DIR / "train.jsonl")
        assert len(examples) > 100
        assert all(isinstance(ex, TrainingExample) for ex in examples)

    @pytest.mark.skipif(not DATA_DIR.exists(), reason="data/ not generated")
    def test_load_eval_examples(self):
        examples = load_examples(DATA_DIR / "eval.jsonl")
        assert len(examples) > 10

    @pytest.mark.skipif(not DATA_DIR.exists(), reason="data/ not generated")
    def test_load_held_out_examples(self):
        examples = load_examples(DATA_DIR / "held_out.jsonl")
        assert len(examples) > 10

    def test_load_from_tmp(self, tmp_path: Path, sample_record: dict):
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(sample_record) + "\n")
        examples = load_examples(path)
        assert len(examples) == 1
        assert examples[0].intent == "Find all pending orders"


class TestFormatExamples:
    def test_chatml_format_has_markers(self, sample_example: TrainingExample):
        text = format_example(sample_example, MODELS["qwen2.5-coder-7b"], include_output=True)
        assert "<|im_start|>system" in text
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text
        assert "<|im_end|>" in text
        assert "pending" in text  # from the output

    def test_chatml_no_output(self, sample_example: TrainingExample):
        text = format_example(sample_example, MODELS["qwen2.5-coder-7b"], include_output=False)
        assert text.endswith("<|im_start|>assistant\n")
        # Output should not be in the text
        assert '"filter"' not in text

    def test_response_template_findable(self, sample_example: TrainingExample):
        """Critical: DataCollatorForCompletionOnlyLM must find the response boundary."""
        model_config = MODELS["qwen2.5-coder-7b"]
        text = format_example(sample_example, model_config, include_output=True)
        assert model_config.response_template in text

    def test_format_examples_batch(self, sample_example: TrainingExample):
        texts = format_examples([sample_example] * 3, MODELS["qwen2.5-coder-7b"])
        assert len(texts) == 3
        assert all(isinstance(t, str) for t in texts)

    def test_schema_fields_in_prompt(self, sample_example: TrainingExample):
        text = format_example(sample_example, MODELS["qwen2.5-coder-7b"])
        assert "order_id" in text
        assert "total" in text
        assert "status" in text
        assert "orders" in text  # collection name

    def test_operators_in_prompt(self, sample_example: TrainingExample):
        text = format_example(sample_example, MODELS["qwen2.5-coder-7b"])
        assert "$match" in text
        assert "$group" in text
        assert "$sum" in text

    def test_intent_in_prompt(self, sample_example: TrainingExample):
        text = format_example(sample_example, MODELS["qwen2.5-coder-7b"])
        assert "Find all pending orders" in text


class TestPromptCompletion:
    """Tests for the prompt-completion split used by trl >= 0.29."""

    def test_prompt_ends_with_assistant_tag(self, sample_example: TrainingExample):
        prompt, completion = format_prompt_completion(sample_example, MODELS["qwen2.5-coder-7b"])
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_completion_has_end_token(self, sample_example: TrainingExample):
        prompt, completion = format_prompt_completion(sample_example, MODELS["qwen2.5-coder-7b"])
        assert completion.endswith("<|im_end|>")

    def test_completion_is_json(self, sample_example: TrainingExample):
        prompt, completion = format_prompt_completion(sample_example, MODELS["qwen2.5-coder-7b"])
        # Strip end token and parse
        json_str = completion.replace("<|im_end|>", "")
        parsed = json.loads(json_str)
        assert parsed["type"] == "find"

    def test_prompt_has_no_output(self, sample_example: TrainingExample):
        prompt, _ = format_prompt_completion(sample_example, MODELS["qwen2.5-coder-7b"])
        assert '"filter"' not in prompt

    def test_completion_has_output(self, sample_example: TrainingExample):
        _, completion = format_prompt_completion(sample_example, MODELS["qwen2.5-coder-7b"])
        assert '"filter"' in completion
        assert "pending" in completion


class TestExtractJson:
    def test_clean_json(self):
        text = '{"type": "find", "filter": {"status": "pending"}}'
        assert extract_json(text) == text

    def test_json_with_prefix(self):
        text = 'Here is the query: {"type": "find", "filter": {}}'
        assert extract_json(text) == '{"type": "find", "filter": {}}'

    def test_json_with_suffix(self):
        text = '{"type": "find", "filter": {}} some extra text'
        assert extract_json(text) == '{"type": "find", "filter": {}}'

    def test_nested_json(self):
        text = '{"type": "aggregate", "pipeline": [{"$match": {"x": 1}}]}'
        result = extract_json(text)
        parsed = json.loads(result)
        assert parsed["type"] == "aggregate"
        assert len(parsed["pipeline"]) == 1

    def test_no_json(self):
        text = "I don't know how to do that"
        assert extract_json(text) == text.strip()

    def test_json_with_strings_containing_braces(self):
        text = '{"error": "Field {foo} not found"}'
        result = extract_json(text)
        assert json.loads(result)["error"] == "Field {foo} not found"
