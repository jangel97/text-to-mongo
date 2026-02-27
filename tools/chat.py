#!/usr/bin/env python3
"""Interactive CLI chat that connects to the LoRA inference service and MongoDB.

Usage:
    python tools/chat.py

Environment variables:
    LORA_URL   - LoRA inference service URL (default: http://192.168.1.139:8080)
    MONGO_URI  - MongoDB connection string (default: mongodb://mongoadmin:mongopassword@localhost:27017)
    MONGO_DB   - Database name (default: cicd_dashboard)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any

import requests
from bson import ObjectId
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LORA_URL = os.getenv("LORA_URL", "http://192.168.1.139:8080")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongoadmin:mongopassword@localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "dashboard")
MAX_RESULTS = 50

# ---------------------------------------------------------------------------
# Schema definitions (same as dashboard lora_agent)
# ---------------------------------------------------------------------------


def _field(name: str, type: str, role: str, description: str = "", **kwargs) -> dict:
    d: dict[str, Any] = {"name": name, "type": type, "role": role, "description": description}
    if "enum_values" in kwargs:
        d["enum_values"] = kwargs["enum_values"]
    return d


SCHEMAS: dict[str, dict] = {
    "products": {
        "collection": "products",
        "domain": "cicd_dashboard",
        "fields": [
            _field("key", "string", "enum", "Product key",
                   enum_values=["rhel-ai", "rhaiis", "base-images", "builder-images"]),
            _field("product_name", "string", "text", "Full product name"),
            _field("short_name", "string", "text", "Short product name"),
            _field("supported_versions", "array", "category", "Supported versions"),
            _field("konflux_namespace", "string", "category", "Build namespace"),
            _field("drop_strategy", "string", "enum", "Drop collection strategy",
                   enum_values=["gitlab-tags", "artifact-commits"]),
            _field("last_updated", "date", "timestamp", "Last sync time"),
        ],
    },
    "drops": {
        "collection": "drops",
        "domain": "cicd_dashboard",
        "fields": [
            _field("key", "string", "identifier", "Drop key"),
            _field("name", "string", "text", "Drop version name"),
            _field("product_key", "string", "enum", "Product key",
                   enum_values=["rhel-ai", "rhaiis", "base-images", "builder-images"]),
            _field("product_version", "string", "text", "Semantic version"),
            _field("created_at", "date", "timestamp", "Creation time"),
            _field("announced_at", "date", "timestamp", "Announcement time"),
            _field("published_at", "date", "timestamp", "Publication time"),
        ],
    },
    "artifacts": {
        "collection": "artifacts",
        "domain": "cicd_dashboard",
        "fields": [
            _field("key", "string", "identifier", "Artifact ID"),
            _field("type", "string", "enum", "Artifact type",
                   enum_values=["containers", "disk-images", "cloud-disk-images",
                                "disk-image-containers", "cloud-containers",
                                "base-images", "wheels-collections", "instructlab",
                                "models", "model-cars"]),
            _field("product_key", "string", "enum", "Product key",
                   enum_values=["rhel-ai", "rhaiis", "base-images", "builder-images"]),
            _field("variant", "string", "category", "Accelerator+OS combo"),
            _field("archs", "array", "category", "CPU architectures",
                   enum_values=["x86_64", "aarch64", "s390x", "ppc64le"]),
            _field("commit", "string", "identifier", "Git commit SHA"),
            _field("created_at", "date", "timestamp", "Build timestamp"),
            _field("environments", "array", "enum", "Deployment environments",
                   enum_values=["stage", "production"]),
            _field("drop_keys", "array", "identifier", "Drop keys"),
            _field("sha_digest", "string", "identifier", "Image digest"),
            _field("series", "string", "category", "Version series"),
            _field("alternative_names", "array", "text", "Alt registry names"),
        ],
    },
    "git_repositories": {
        "collection": "git_repositories",
        "domain": "cicd_dashboard",
        "fields": [
            _field("key", "string", "identifier", "Repository key"),
            _field("url", "string", "text", "Git URL"),
            _field("type", "string", "enum", "Repository type",
                   enum_values=["containers", "disk-images", "wheels-collections",
                                "cloud-disk-images", "base-images", "instructlab",
                                "models", "model-cars"]),
            _field("product_keys", "array", "enum", "Product keys",
                   enum_values=["rhel-ai", "rhaiis", "base-images", "builder-images"]),
            _field("gitlab_project_id", "int", "identifier", "GitLab project ID"),
        ],
    },
}

ALLOWED_OPS: dict = {
    "stage_operators": [
        "$match", "$group", "$sort", "$limit", "$project",
        "$unwind", "$count", "$addFields", "$bucket",
    ],
    "expression_operators": [
        "$sum", "$avg", "$min", "$max", "$first", "$last",
        "$push", "$addToSet", "$size",
        "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
        "$in", "$nin", "$exists", "$regex",
        "$and", "$or", "$not",
        "$dateToString", "$year", "$month", "$dayOfMonth",
        "$substr", "$toLower", "$toUpper",
        "$cond", "$ifNull",
    ],
}

# ---------------------------------------------------------------------------
# Collection resolver (keyword-based)
# ---------------------------------------------------------------------------

_COLLECTION_KEYWORDS: dict[str, set[str]] = {
    "artifacts": {
        "artifact", "container", "image", "wheel", "build", "sbom",
        "disk", "iso", "qcow", "ami", "vhd", "base-image", "base image",
        "accelerator", "cuda", "rocm", "cpu", "variant", "digest",
        "instructlab", "model-car",
    },
    "drops": {
        "drop", "release", "version", "announced", "published",
        "series", "nightly", "rc", "freeze",
    },
    "products": {
        "product", "rhel", "rhaiis", "supported", "namespace",
        "konflux", "rhelai", "rhel ai", "inference server",
    },
    "git_repositories": {
        "repo", "repository", "git", "branch", "tag", "gitlab",
        "commit",
    },
}


def resolve_collection(question: str) -> str | None:
    q = question.lower()
    scores: dict[str, int] = {}
    for collection, keywords in _COLLECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[collection] = score
    if not scores:
        return None
    return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LoRA inference client
# ---------------------------------------------------------------------------


def call_inference(schema: dict, intent: str) -> dict:
    """Call the LoRA inference service and return the response."""
    payload = {
        "schema": schema,
        "allowed_ops": ALLOWED_OPS,
        "intent": intent,
    }
    resp = requests.post(f"{LORA_URL}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Query executor (pymongo, sync)
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
# REPL
# ---------------------------------------------------------------------------


def print_schema(name: str) -> None:
    schema = SCHEMAS.get(name)
    if not schema:
        print(f"  Unknown collection: {name}")
        print(f"  Available: {', '.join(SCHEMAS.keys())}")
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


def main() -> None:
    # Connect to MongoDB
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[MONGO_DB]
    except Exception as e:
        print(f"Failed to connect to MongoDB at {MONGO_URI}: {e}")
        sys.exit(1)

    # Check LoRA service health
    try:
        health = requests.get(f"{LORA_URL}/health", timeout=5).json()
        if health.get("status") != "ok":
            print(f"LoRA service not ready: {health}")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to connect to LoRA service at {LORA_URL}: {e}")
        sys.exit(1)

    print(f"Connected to MongoDB ({MONGO_DB}) and LoRA service ({LORA_URL})")
    print(f"Collections: {', '.join(SCHEMAS.keys())}")
    print(f"Commands: collections, schema <name>, exit")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if user_input.lower() == "collections":
            print(f"  Available: {', '.join(SCHEMAS.keys())}")
            print()
            continue
        if user_input.lower().startswith("schema"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                for name in SCHEMAS:
                    print_schema(name)
                    print()
            else:
                print_schema(parts[1].strip())
            print()
            continue

        # Resolve collection
        collection_name = resolve_collection(user_input)
        if not collection_name:
            print("  Could not determine which collection to query.")
            print(f"  Available: {', '.join(SCHEMAS.keys())}")
            print()
            continue

        schema = SCHEMAS[collection_name]
        print(f"  Collection: {collection_name}")

        # Call LoRA inference
        try:
            result = call_inference(schema, user_input)
        except Exception as e:
            print(f"  Inference error: {e}")
            print()
            continue

        latency = result.get("latency_ms", 0)
        query = result.get("query")
        raw = result.get("raw_output", "")

        if not result.get("syntax_valid") or not query:
            print(f"  Invalid query (latency: {latency}ms)")
            print(f"  Raw output: {raw}")
            if result.get("errors"):
                for err in result["errors"]:
                    print(f"  Error: {err}")
            print()
            continue

        print(f"  Query: {json.dumps(query)}")
        print(f"  Latency: {latency}ms")

        # Execute against MongoDB
        try:
            docs = execute_query(db, collection_name, query)
            print(f"  Results: {len(docs)} document(s)")
            print()
            print(json.dumps(docs, indent=2, default=_json_serializer, ensure_ascii=False))
        except Exception as e:
            print(f"  Execution error: {e}")

        print()


if __name__ == "__main__":
    main()
