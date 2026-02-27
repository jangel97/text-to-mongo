"""Request/response models for the LoRA inference service."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from text_to_mongo.schema import AllowedOps, SchemaDef


class InferenceRequest(BaseModel):
    schema_def: SchemaDef = Field(alias="schema")
    allowed_ops: AllowedOps
    intent: str

    model_config = {"populate_by_name": True}


class InferenceResponse(BaseModel):
    query: dict[str, Any] | None = None
    raw_output: str = ""
    syntax_valid: bool = False
    errors: list[str] = Field(default_factory=list)
    latency_ms: int = 0
