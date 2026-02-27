## Evaluating and Training Small Language Models for Structured MongoDB Query Generation

Large language models can generate MongoDB queries from natural language, but they often hallucinate fields, misuse operators, and depend on costly external APIs. In this talk, we present an end-to-end pipeline for training and evaluating small, self-hosted language models to reliably generate structured MongoDB find and aggregate queries.

We fine-tune Qwen2.5-Coder-7B using QLoRA on ~1,300 synthetic examples, with no human-labeled data, and improve accuracy from 40% to 98.9% on held-out collection schemas never seen during training. Correctness is enforced through a four-layer evaluation harness that validates JSON syntax, operator safety, field usage, and generalization.

Training completes in under 20 minutes on a single GPU, and inference runs locally at ~1s latency using 5GB of VRAM. The session covers synthetic data generation, fine-tuning, evaluation methodology, and deployment as a local inference service, showing how small language models can be made reliable for structured query generation.