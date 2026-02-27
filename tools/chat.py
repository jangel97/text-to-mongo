#!/usr/bin/env python3
"""Interactive CLI chat for text-to-MongoDB query generation.

Usage:
    python tools/chat.py demos/dashboard/config.json
    TOOL_CONFIG=demos/dashboard/config.json python tools/chat.py

Environment variables:
    TOOL_CONFIG - path to config JSON (or pass as first CLI argument)
    LORA_URL    - LoRA inference service URL (default: http://192.168.1.138:8080)
    MONGO_URI   - MongoDB connection string (default: mongodb://mongoadmin:mongopassword@localhost:27017)
    MONGO_DB    - Database name (overrides config file)
"""
from __future__ import annotations

import json
import os
import sys

import requests
from pymongo import MongoClient

from core import (
    LORA_URL,
    MONGO_URI,
    call_inference,
    execute_query,
    load_config,
    print_schema,
    resolve_collection,
    _json_serializer,
)


def main() -> None:
    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TOOL_CONFIG")
    if not config_path:
        print("Usage: python tools/chat.py <config.json>")
        print("  or set TOOL_CONFIG env var")
        sys.exit(1)

    config = load_config(config_path)
    schemas = config["schemas"]
    allowed_ops = config["allowed_ops"]
    keywords_map = config["collection_keywords"]
    mongo_db_name = os.getenv("MONGO_DB", config.get("mongo_db", "test"))

    # Connect to MongoDB
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[mongo_db_name]
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

    print(f"Connected to MongoDB ({mongo_db_name}) and LoRA service ({LORA_URL})")
    print(f"Collections: {', '.join(schemas.keys())}")
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
            print(f"  Available: {', '.join(schemas.keys())}")
            print()
            continue
        if user_input.lower().startswith("schema"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                for name in schemas:
                    print_schema(name, schemas)
                    print()
            else:
                print_schema(parts[1].strip(), schemas)
            print()
            continue

        # Resolve collection
        collection_name = resolve_collection(user_input, keywords_map)
        if not collection_name:
            print("  Could not determine which collection to query.")
            print(f"  Available: {', '.join(schemas.keys())}")
            print()
            continue

        schema = schemas[collection_name]
        print(f"  Collection: {collection_name}")

        # Call LoRA inference
        try:
            result = call_inference(schema, allowed_ops, user_input)
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
