# serve/ — FastAPI Inference Microservice

Serves the fine-tuned LoRA model over HTTP. Loads the base model in 4-bit precision, applies the LoRA adapter on startup, and exposes a prediction endpoint.

## Entrypoint

```bash
ADAPTER_PATH=runs/qwen2.5-coder-7b_r8/adapter \
uvicorn text_to_mongo.serve.app:app --host 0.0.0.0 --port 8080
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `qwen2.5-coder-7b` | Key from the `MODELS` dict in `training/config.py` |
| `ADAPTER_PATH` | `runs/qwen2.5-coder-7b_r8/adapter` | Path to the saved LoRA adapter directory |
| `MAX_NEW_TOKENS` | `256` | Maximum tokens to generate per request |

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Takes a schema, allowed ops, and intent. Returns the generated query, raw output, syntax validity, errors, and latency. |
| `/health` | GET | Returns model name, adapter path, device, and readiness status. |

## Files

| File | What it does |
|---|---|
| `app.py` | FastAPI application. Loads model on startup via lifespan handler. Builds a ChatML prompt from the request, runs greedy decoding, extracts JSON, validates syntax, and returns the response. |
| `models.py` | Pydantic request/response models. `InferenceRequest` accepts schema (with alias `"schema"`), allowed ops, and intent. `InferenceResponse` returns the parsed query, raw text, validation status, errors, and latency in milliseconds. |