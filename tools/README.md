# tools/ — Interactive Query Tools

Generic tools for querying MongoDB using the LoRA model. Configure for any database via a JSON config file.

| Tool | File | Description |
|---|---|---|
| **CLI Chat** | `chat.py` | Terminal REPL — type questions, see queries and results |
| **Streamlit UI** | `app.py` | Web UI with chat interface, schema browser, and connection status |
| **Core** | `core.py` | Shared generic functions (inference client, query executor, etc.) |

## Prerequisites

- LoRA inference service running on a GPU machine (see `src/text_to_mongo/serve/`)
- MongoDB instance with your data
- A config JSON file describing your schemas (see [Config Format](#config-format))

## Install

```bash
# CLI only
pip install -e ".[chat]"

# Streamlit UI
pip install -e ".[ui]"
```

## Usage

Both tools require a `TOOL_CONFIG` environment variable pointing to a config JSON file.

```bash
# CLI chat
TOOL_CONFIG=demos/dashboard/config.json python tools/chat.py

# Or pass as CLI argument
python tools/chat.py demos/dashboard/config.json

# Streamlit web UI
TOOL_CONFIG=demos/dashboard/config.json streamlit run tools/app.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TOOL_CONFIG` | — | Path to config JSON (required) |
| `LORA_URL` | `http://192.168.1.138:8080` | LoRA inference service URL |
| `MONGO_URI` | `mongodb://mongoadmin:mongopassword@localhost:27017` | MongoDB connection string |
| `MONGO_DB` | from config file | Database name (overrides config) |

## Config Format

```json
{
  "mongo_db": "my_database",
  "schemas": {
    "collection_name": {
      "collection": "collection_name",
      "domain": "my_domain",
      "fields": [
        {
          "name": "field_name",
          "type": "string",
          "role": "identifier",
          "description": "Short desc"
        }
      ]
    }
  },
  "allowed_ops": {
    "stage_operators": ["$match", "$group", "$sort"],
    "expression_operators": ["$sum", "$avg", "$gt", "$lt"]
  },
  "collection_keywords": {
    "collection_name": ["keyword1", "keyword2"]
  },
  "suggestions": [
    "Example query for the UI"
  ]
}
```

Field descriptions must be **2-5 words** — the LoRA model was trained on short descriptions and hallucinates with longer ones.

See `demos/dashboard/config.json` for a complete example.

## CLI Commands

| Command | Description |
|---|---|
| `collections` | List available collections |
| `schema <name>` | Show the schema for a collection |
| `schema` | Show all schemas |
| `exit` / `quit` / `q` | Exit the chat |

Any other input is treated as a natural language question.

## Starting MongoDB (local)

```bash
podman run -d --name mongodb -p 27017:27017 --rm \
  -v $HOME/mongodb_data:/data/db:z \
  -e MONGODB_INITDB_ROOT_USERNAME=mongoadmin \
  -e MONGODB_INITDB_ROOT_PASSWORD=mongopassword \
  quay.io/mongodb/mongodb-community-server:8.0-ubi9
```
