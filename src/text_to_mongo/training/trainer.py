"""Quantized model loading, LoRA setup, and SFTTrainer orchestration."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from text_to_mongo.training.config import TrainingConfig
from text_to_mongo.training.dataset import build_hf_dataset

logger = logging.getLogger(__name__)


def load_quantized_model(config: TrainingConfig):
    """Load a 4-bit quantized model and tokenizer.

    Returns:
        (model, tokenizer) tuple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.hf_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def _build_peft_config(config: TrainingConfig):
    """Build a PEFT LoRA config to pass directly to SFTTrainer."""
    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

def run_training(config: TrainingConfig) -> Path:
    """Execute the full training pipeline.

    Uses trl >= 0.29 API: peft_config passed to SFTTrainer, prompt-completion
    dataset format for completion-only loss.

    Args:
        config: Complete training configuration.

    Returns:
        Path to the saved adapter directory.
    """
    from trl import SFTConfig, SFTTrainer

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = run_dir / "adapter"

    logger.info("Starting training run: %s", config.run_name)
    logger.info("Output directory: %s", run_dir)

    # 1. Load quantized model + tokenizer
    logger.info("Loading quantized model: %s", config.model.hf_id)
    model, tokenizer = load_quantized_model(config)

    # 2. Build LoRA config (SFTTrainer applies it internally)
    peft_config = _build_peft_config(config)
    logger.info("LoRA config: r=%d, alpha=%d, targets=%s", config.lora.r, config.lora.alpha, config.lora.target_modules)

    # 3. Load datasets (prompt-completion format)
    logger.info("Loading datasets from %s", config.data_dir)
    train_dataset = build_hf_dataset(config.data_dir, config.model, split="train")
    eval_dataset = build_hf_dataset(config.data_dir, config.model, split="eval")
    logger.info("Train: %d examples, Eval: %d examples", len(train_dataset), len(eval_dataset))

    # 4. SFTConfig
    sft_config = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        optim="paged_adamw_8bit",
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        max_grad_norm=1.0,
        max_length=config.max_seq_len,
        packing=False,
    )

    # 5. SFTTrainer — peft_config passed directly, prompt-completion format
    #    auto-enables completion-only loss
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # 6. Save adapter + tokenizer + config
    logger.info("Saving adapter to %s", adapter_dir)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save training config as JSON for reproducibility
    config_record = {
        "model": config.model.name,
        "hf_id": config.model.hf_id,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "lora_target_modules": config.lora.target_modules,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "effective_batch_size": config.batch_size * config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler": config.lr_scheduler,
        "warmup_ratio": config.warmup_ratio,
        "max_seq_len": config.max_seq_len,
        "bf16": config.bf16,
        "train_metrics": train_result.metrics,
    }
    with open(run_dir / "training_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    logger.info("Run complete: %s", config.run_name)
    return adapter_dir
