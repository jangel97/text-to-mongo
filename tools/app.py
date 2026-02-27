#!/usr/bin/env python3
"""Streamlit frontend for the Text-to-MongoDB LoRA model.

Usage:
    TOOL_CONFIG=demos/dashboard/config.json streamlit run tools/app.py

Environment variables:
    TOOL_CONFIG - path to config JSON (required)
    LORA_URL    - LoRA inference service URL (default: http://192.168.1.138:8080)
    MONGO_URI   - MongoDB connection string (default: mongodb://mongoadmin:mongopassword@localhost:27017)
    MONGO_DB    - Database name (overrides config file)
"""
from __future__ import annotations

import json
import os

import streamlit as st

from core import (
    LORA_URL,
    MAX_RESULTS,
    MONGO_URI,
    _json_serializer,
    call_inference,
    execute_query,
    load_config,
    resolve_collection,
)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

config_path = os.getenv("TOOL_CONFIG")
if not config_path:
    st.error("Set the TOOL_CONFIG environment variable to a config JSON file path.")
    st.stop()

config = load_config(config_path)
schemas = config["schemas"]
allowed_ops = config["allowed_ops"]
keywords_map = config["collection_keywords"]
suggestions = config.get("suggestions", [])
mongo_db_name = os.getenv("MONGO_DB", config.get("mongo_db", "test"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Text to MongoDB",
    page_icon="\U0001f916",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — connection settings & schema browser
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Text to MongoDB")
    st.caption("LoRA-powered natural language to MongoDB queries")

    st.divider()

    # Connection status
    st.subheader("Connections")

    # MongoDB
    mongo_ok = False
    try:
        from pymongo import MongoClient

        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client[mongo_db_name]
        mongo_ok = True
        st.success(f"MongoDB: {mongo_db_name}", icon="\u2705")
    except Exception as e:
        st.error(f"MongoDB: {e}", icon="\u274c")
        db = None

    # LoRA service
    lora_ok = False
    try:
        import requests

        health = requests.get(f"{LORA_URL}/health", timeout=3).json()
        if health.get("status") == "ok":
            lora_ok = True
            model_name = health.get("model", "unknown")
            st.success(f"LoRA: {model_name}", icon="\u2705")
        else:
            st.error("LoRA: not ready", icon="\u274c")
    except Exception as e:
        st.error(f"LoRA: {e}", icon="\u274c")

    st.divider()

    # Schema browser
    st.subheader("Collections")
    selected_collection = st.selectbox(
        "Browse schema",
        options=list(schemas.keys()),
        index=None,
        placeholder="Select a collection...",
    )

    if selected_collection:
        schema = schemas[selected_collection]
        for f in schema["fields"]:
            role_colors = {
                "identifier": ":blue",
                "measure": ":green",
                "timestamp": ":orange",
                "category": ":violet",
                "enum": ":red",
                "boolean": ":gray",
                "text": ":rainbow",
            }
            color = role_colors.get(f["role"], "")
            enum_hint = ""
            if f.get("enum_values"):
                enum_hint = f" `[{', '.join(f['enum_values'][:4])}{'...' if len(f.get('enum_values', [])) > 4 else ''}]`"
            st.markdown(
                f"**{f['name']}** {color}[{f['role']}] "
                f"*{f['type']}*{enum_hint}"
            )

    st.divider()

    # Query suggestions
    if suggestions:
        st.subheader("Try a query")
        for i, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state["_pending_suggestion"] = suggestion
                st.rerun()

        st.divider()

    st.caption(f"Max results: {MAX_RESULTS}")
    st.caption(f"LoRA: `{LORA_URL}`")
    st.caption(f"Mongo: `{MONGO_URI}`")

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("query"):
            st.code(json.dumps(msg["query"], indent=2), language="json")
        if msg.get("results") is not None:
            st.markdown(f"**{msg['collection']}** — {msg['result_count']} document(s) in {msg['latency_ms']}ms")
            st.code(
                json.dumps(msg["results"], indent=2, default=_json_serializer, ensure_ascii=False),
                language="json",
            )
        elif msg.get("error"):
            st.error(msg["error"])
        elif msg.get("content"):
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

# Always call chat_input so Streamlit tracks the widget on every rerun
chat_prompt = st.chat_input("Ask a question about your data...")
prompt = st.session_state.pop("_pending_suggestion", None) or chat_prompt

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process
    with st.chat_message("assistant"):
        if not lora_ok:
            error = "LoRA inference service is not connected."
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "error": error})
        elif not mongo_ok:
            error = "MongoDB is not connected."
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "error": error})
        else:
            # Resolve collection
            collection_name = resolve_collection(prompt, keywords_map)
            if not collection_name:
                error = f"Could not determine which collection to query. Available: {', '.join(schemas.keys())}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "error": error})
            else:
                schema = schemas[collection_name]

                # Call LoRA
                try:
                    with st.spinner(f"Generating query for **{collection_name}**..."):
                        result = call_inference(schema, allowed_ops, prompt)

                    latency = result.get("latency_ms", 0)
                    query = result.get("query")

                    if not result.get("syntax_valid") or not query:
                        raw = result.get("raw_output", "")
                        errors = result.get("errors", [])
                        error = f"Invalid query generated (latency: {latency}ms)\n\nRaw: `{raw}`"
                        if errors:
                            error += "\n\n" + "\n".join(f"- {e}" for e in errors)
                        st.error(error)
                        st.session_state.messages.append({"role": "assistant", "error": error})
                    else:
                        # Show generated query
                        st.code(json.dumps(query, indent=2), language="json")

                        # Execute against MongoDB
                        try:
                            docs = execute_query(db, collection_name, query)
                            st.markdown(
                                f"**{collection_name}** — {len(docs)} document(s) in {latency}ms"
                            )
                            st.code(
                                json.dumps(docs, indent=2, default=_json_serializer, ensure_ascii=False),
                                language="json",
                            )
                            st.session_state.messages.append({
                                "role": "assistant",
                                "query": query,
                                "collection": collection_name,
                                "results": docs,
                                "result_count": len(docs),
                                "latency_ms": latency,
                            })
                        except Exception as e:
                            st.code(json.dumps(query, indent=2), language="json")
                            error = f"Query execution failed: {e}"
                            st.error(error)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "query": query,
                                "error": error,
                            })

                except Exception as e:
                    error = f"Inference error: {e}"
                    st.error(error)
                    st.session_state.messages.append({"role": "assistant", "error": error})
