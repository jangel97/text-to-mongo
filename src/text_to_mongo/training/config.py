"""Configuration dataclasses for the training pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Base model identity and prompt format."""

    name: str
    hf_id: str
    response_template: str  # boundary marker for DataCollatorForCompletionOnlyLM


MODELS: dict[str, ModelConfig] = {
    "qwen2.5-coder-7b": ModelConfig(
        name="qwen2.5-coder-7b",
        hf_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        response_template="\n<|im_start|>assistant\n",
    ),
}


@dataclass
class LoraConfig:
    """LoRA adapter hyperparameters."""

    r: int = 8
    alpha: int = 16  # 2x r by default
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        ]
    )

    def __post_init__(self) -> None:
        if self.alpha == 16 and self.r != 8:
            # Auto-set alpha = 2 * r unless explicitly overridden
            self.alpha = 2 * self.r


@dataclass
class TrainingConfig:
    """Full training run configuration."""

    model: ModelConfig
    lora: LoraConfig = field(default_factory=LoraConfig)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("runs"))

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_len: int = 512
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Quantization
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    @property
    def run_name(self) -> str:
        return f"{self.model.name}_r{self.lora.r}"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name
