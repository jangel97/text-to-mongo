# tools/ — Interactive CLI Chat

Test the LoRA model against a live MongoDB instance. Type a question, see the generated query, and get real results.

## Prerequisites

- LoRA inference service running on a GPU machine (see `src/text_to_mongo/serve/`)
- MongoDB instance with the CI/CD dashboard data

```bash
pip install -e ".[chat]"
```

## Usage

```bash
python tools/chat.py
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LORA_URL` | `http://192.168.1.139:8080` | LoRA inference service URL |
| `MONGO_URI` | `mongodb://mongoadmin:mongopassword@localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `cicd_dashboard` | Database name |

```bash
LORA_URL=http://my-gpu:8080 \
MONGO_URI=mongodb://user:pass@host:27017 \
MONGO_DB=my_database \
python tools/chat.py
```

### Starting MongoDB (local)

```bash
podman run -d --name mongodb -p 27017:27017 --rm \
  -v $HOME/mongodb_data:/data/db:z \
  -e MONGODB_INITDB_ROOT_USERNAME=mongoadmin \
  -e MONGODB_INITDB_ROOT_PASSWORD=mongopassword \
  quay.io/mongodb/mongodb-community-server:8.0-ubi9
```

## Example Session

```
Connected to MongoDB (cicd_dashboard) and LoRA service (http://192.168.1.139:8080)
Collections: products, drops, artifacts, git_repositories
Commands: collections, schema <name>, exit

> show latest rhaiis containers
  Collection: artifacts
  Query: {"type": "find", "filter": {"product_key": "rhaiis", "type": "containers"}, "sort": {"created_at": -1}, "limit": 1}
  Latency: 823ms
  Results: 1 document(s)

[
  {
    "key": "...",
    "type": "containers",
    "product_key": "rhaiis",
    ...
  }
]

> how many artifacts per product?
  Collection: artifacts
  Query: {"type": "aggregate", "pipeline": [{"$group": {"_id": "$product_key", "count": {"$sum": 1}}}]}
  Latency: 956ms
  Results: 4 document(s)

[...]

> schema drops
  Collection: drops
  Fields:
    - key (string, identifier): Drop key
    - name (string, text): Drop version name
    - product_key (string, enum): Product key [rhel-ai, rhaiis, base-images, builder-images]
    - product_version (string, text): Semantic version
    - created_at (date, timestamp): Creation time
    - announced_at (date, timestamp): Announcement time
    - published_at (date, timestamp): Publication time

> exit
```

## Commands

| Command | Description |
|---|---|
| `collections` | List available collections |
| `schema <name>` | Show the schema for a collection |
| `schema` | Show all schemas |
| `exit` / `quit` / `q` | Exit the chat |

Any other input is treated as a natural language question. The tool resolves the collection via keyword matching, sends the schema + question to the LoRA model, and executes the returned query against MongoDB.
