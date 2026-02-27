"""Generic utilities for text-to-MongoDB query tools.

This module contains all reusable functions for interacting with the LoRA
inference service and MongoDB. It has no hardcoded schemas or collection
data — those are loaded from a config JSON file at runtime.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import requests
from bson import ObjectId

# ---------------------------------------------------------------------------
# Configuration (env vars)
# ---------------------------------------------------------------------------

LORA_URL = os.getenv("LORA_URL", "http://192.168.1.138:8080")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongoadmin:mongopassword@localhost:27017")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load a tool config JSON file.

    Expected keys: schemas, allowed_ops, collection_keywords.
    Optional keys: mongo_db, suggestions.
    """
    with open(path) as f:
        return json.load(f)


def _field(name: str, type: str, role: str, description: str = "", **kwargs) -> dict:
    """Build a schema field dict."""
    d: dict[str, Any] = {"name": name, "type": type, "role": role, "description": description}
    if "enum_values" in kwargs:
        d["enum_values"] = kwargs["enum_values"]
    return d


# ---------------------------------------------------------------------------
# Collection resolver (keyword-based, parameterized)
# ---------------------------------------------------------------------------


def resolve_collection(question: str, keywords_map: dict[str, list[str] | set[str]]) -> str | None:
    """Score each collection by keyword hits and return the best match."""
    q = question.lower()
    scores: dict[str, int] = {}
    for collection, keywords in keywords_map.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[collection] = score
    if not scores:
        return None
    return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LoRA inference client
# ---------------------------------------------------------------------------


def call_inference(schema: dict, allowed_ops: dict, intent: str,
                   lora_url: str | None = None) -> dict:
    """Call the LoRA inference service and return the response."""
    url = lora_url or LORA_URL
    payload = {
        "schema": schema,
        "allowed_ops": allowed_ops,
        "intent": intent,
    }
    resp = requests.post(f"{url}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Query executor
# ---------------------------------------------------------------------------


def _convert_extended_json(obj: Any) -> Any:
    """Convert Extended JSON v2 dates to Python datetime objects."""
    if isinstance(obj, dict):
        if "$date" in obj and len(obj) == 1:
            try:
                date_str = obj["$date"]
                if date_str.endswith("Z"):
                    date_str = date_str[:-1] + "+00:00"
                return datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                return obj
        return {k: _convert_extended_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_extended_json(item) for item in obj]
    return obj


def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def execute_query(db, collection_name: str, query: dict) -> list[dict]:
    """Execute a LoRA-generated query against MongoDB."""
    collection = db[collection_name]
    query = _convert_extended_json(query)
    query_type = query.get("type")

    if query_type == "find":
        filter_doc = query.get("filter", {})
        sort = query.get("sort")
        limit = min(query.get("limit", MAX_RESULTS), MAX_RESULTS)
        projection = query.get("projection")

        cursor = collection.find(filter_doc, projection)
        if sort:
            if isinstance(sort, dict):
                cursor = cursor.sort(list(sort.items()))
            elif isinstance(sort, list):
                cursor = cursor.sort(sort)
        cursor = cursor.limit(limit)
        return list(cursor)

    elif query_type == "aggregate":
        pipeline = query.get("pipeline", [])
        has_limit = any("$limit" in stage for stage in pipeline if isinstance(stage, dict))
        if not has_limit:
            pipeline.append({"$limit": MAX_RESULTS})
        return list(collection.aggregate(pipeline))

    else:
        return [{"error": f"Unknown query type: {query_type}"}]


# ---------------------------------------------------------------------------
# Schema display
# ---------------------------------------------------------------------------


def print_schema(name: str, schemas: dict) -> None:
    """Print a schema's fields to stdout."""
    schema = schemas.get(name)
    if not schema:
        print(f"  Unknown collection: {name}")
        print(f"  Available: {', '.join(schemas.keys())}")
        return
    print(f"  Collection: {schema['collection']}")
    print(f"  Fields:")
    for f in schema["fields"]:
        parts = [f"    - {f['name']} ({f['type']}, {f['role']})"]
        if f.get("description"):
            parts.append(f": {f['description']}")
        if f.get("enum_values"):
            parts.append(f" [{', '.join(f['enum_values'])}]")
        print("".join(parts))
