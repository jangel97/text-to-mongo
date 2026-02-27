# training/ — QLoRA Fine-Tuning Pipeline

Handles everything from loading data to training the LoRA adapter, evaluating results, and comparing runs.

## Entrypoint

```bash
python -m text_to_mongo.training {baseline,train,eval,compare}
```

The CLI dispatches to the appropriate module. All commands require a GPU.

## Commands

| Command | What it does |
|---|---|
| `baseline --model qwen2.5-coder-7b` | Loads the base model without any adapter and evaluates it zero-shot on eval and held-out splits. Establishes the "before" accuracy. |
| `train --model qwen2.5-coder-7b --lora-r 8` | Fine-tunes the model using QLoRA. Saves the LoRA adapter to `runs/<model>_r<rank>/adapter/`. |
| `eval --model qwen2.5-coder-7b --adapter runs/.../adapter` | Loads the base model + adapter and evaluates on eval and held-out splits. Saves predictions and eval reports. |
| `compare` | Reads all eval reports from `runs/` and prints a markdown comparison table. |

## Files

| File | What it does |
|---|---|
| `cli.py` | Argument parser and command dispatch. Supports `--model`, `--lora-r`, `--lora-alpha`, `--epochs`, `--batch-size`, `--lr`, `--adapter`, `--run-name`. |
| `__main__.py` | Entry point for `python -m text_to_mongo.training`. |
| `config.py` | Dataclasses for model, LoRA, and training configuration. Defines the `MODELS` dict with supported base models (currently Qwen2.5-Coder-7B). |
| `dataset.py` | Loads JSONL examples, formats them into ChatML prompts, splits into prompt/completion pairs, and builds HuggingFace `Dataset` objects for SFTTrainer. |
| `trainer.py` | Core training logic. Loads the base model in 4-bit (NF4 + double quant), attaches LoRA adapters to attention and MLP layers, runs SFTTrainer with completion-only loss masking, saves the adapter. |
| `inference.py` | Model loading and generation. Loads PeftModel (adapter cannot be merged into 4-bit weights), runs batched greedy decoding, extracts JSON from raw output via brace-depth matching. |
| `baseline.py` | Zero-shot evaluation. Loads base model without adapter, generates predictions, runs the eval harness, saves results. |
| `compare.py` | Post-training evaluation and comparison tables. Reads eval reports from all runs and formats a markdown table. |

## Training Flow

```
JSONL data
    → dataset.py (prompt/completion split)
        → trainer.py (QLoRA: 4-bit base + LoRA adapters)
            → runs/<model>_r<rank>/adapter/ (saved adapter, ~20MB)
                → inference.py (load PeftModel + generate)
                    → eval harness (4-layer validation)
                        → compare.py (markdown table)
```
