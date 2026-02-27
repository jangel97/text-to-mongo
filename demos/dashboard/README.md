# Dashboard Demo

CI/CD dashboard with 4 collections: products, drops, artifacts, git_repositories.

## Run

```bash
# CLI chat
TOOL_CONFIG=demos/dashboard/config.json python tools/chat.py

# Streamlit web UI
TOOL_CONFIG=demos/dashboard/config.json streamlit run tools/app.py
```

## Prerequisites

- LoRA inference service running (see `src/text_to_mongo/serve/`)
- MongoDB with dashboard data (see `tools/README.md` for startup command)
