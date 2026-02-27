"""FastAPI inference service for the LoRA fine-tuned text-to-MongoDB model."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from text_to_mongo.eval.syntax import eval_syntax
from text_to_mongo.prompt import build_prompt
from text_to_mongo.schema import TrainingExample
from text_to_mongo.serve.models import InferenceRequest, InferenceResponse
from text_to_mongo.training.config import MODELS
from text_to_mongo.training.inference import extract_json, load_model_for_inference

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder-7b")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "runs/qwen2.5-coder-7b_r8/adapter")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

_model = None
_tokenizer = None
_model_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _model_config

    _model_config = MODELS[MODEL_NAME]
    logger.info("Loading model %s with adapter %s ...", MODEL_NAME, ADAPTER_PATH)

    start = time.time()
    _model, _tokenizer = load_model_for_inference(_model_config, adapter_path=ADAPTER_PATH)
    elapsed = time.time() - start
    logger.info("Model loaded in %.1fs", elapsed)

    yield

    logger.info("Shutting down inference service.")


app = FastAPI(
    title="Text-to-Mongo LoRA Inference",
    description="Serves LoRA fine-tuned model for text-to-MongoDB query generation",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "adapter": ADAPTER_PATH,
        "device": str(next(_model.parameters()).device),
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    import torch

    if _model is None or _tokenizer is None or _model_config is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    start = time.time()

    # Build a TrainingExample to reuse prompt builder
    example = TrainingExample(
        schema=request.schema_def,
        allowed_ops=request.allowed_ops,
        intent=request.intent,
        output={},  # dummy — not used when include_output=False
        is_negative=False,
    )

    prompt = build_prompt(example, include_output=False)

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    raw_output = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Extract JSON and validate syntax
    json_str = extract_json(raw_output)
    syntax = eval_syntax(json_str)

    query = None
    errors = list(syntax.errors)
    if syntax.passed:
        try:
            query = json.loads(json_str)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")

    elapsed_ms = int((time.time() - start) * 1000)

    return InferenceResponse(
        query=query,
        raw_output=raw_output,
        syntax_valid=syntax.passed,
        errors=errors,
        latency_ms=elapsed_ms,
    )
