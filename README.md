---
title: KB RAG Eval Service
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# KB RAG Eval Service

FastAPI RAG endpoint over a 50-passage technical knowledge base.

**Endpoint:** `POST /query`

```json
{"question": "string", "top_k": 3}
