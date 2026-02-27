"""Tests for training configuration models."""
from pathlib import Path

from text_to_mongo.training.config import (
    MODELS,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)


class TestModelConfig:
    def test_models_dict_has_qwen(self):
        assert "qwen2.5-coder-7b" in MODELS

    def test_qwen_config(self):
        m = MODELS["qwen2.5-coder-7b"]
        assert "<|im_start|>assistant" in m.response_template
        assert m.hf_id == "Qwen/Qwen2.5-Coder-7B-Instruct"

    def test_response_template_not_empty(self):
        for name, m in MODELS.items():
            assert len(m.response_template) > 0, f"{name} has empty response_template"


class TestLoraConfig:
    def test_defaults(self):
        lora = LoraConfig()
        assert lora.r == 8
        assert lora.alpha == 16
        assert lora.dropout == 0.05
        assert "q_proj" in lora.target_modules
        assert "v_proj" in lora.target_modules
        assert len(lora.target_modules) == 6

    def test_alpha_auto_scales_with_r(self):
        lora = LoraConfig(r=16)
        assert lora.alpha == 32  # 2 * 16

    def test_alpha_explicit_override(self):
        lora = LoraConfig(r=16, alpha=64)
        assert lora.alpha == 64


class TestTrainingConfig:
    def test_run_name(self):
        config = TrainingConfig(model=MODELS["qwen2.5-coder-7b"])
        assert config.run_name == "qwen2.5-coder-7b_r8"

    def test_run_name_r16(self):
        config = TrainingConfig(
            model=MODELS["qwen2.5-coder-7b"],
            lora=LoraConfig(r=16),
        )
        assert config.run_name == "qwen2.5-coder-7b_r16"

    def test_run_dir(self):
        config = TrainingConfig(
            model=MODELS["qwen2.5-coder-7b"],
            output_dir=Path("runs"),
        )
        assert config.run_dir == Path("runs/qwen2.5-coder-7b_r8")

    def test_hyperparameter_defaults(self):
        config = TrainingConfig(model=MODELS["qwen2.5-coder-7b"])
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.grad_accum_steps == 4
        assert config.learning_rate == 2e-4
        assert config.max_seq_len == 512
        assert config.bf16 is True
        assert config.use_4bit is True
