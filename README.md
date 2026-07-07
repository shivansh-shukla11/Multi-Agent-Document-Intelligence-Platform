# Multi-Agent Document Intelligence Platform

A Generative AI system that ingests documents (PDF/TXT), builds a semantic
search index over them, and answers user questions using a **team of
cooperating agents** instead of a single RAG chain:

```
User Query
   │
   ▼
┌─────────────┐     simple lookup      ┌────────────────┐
│ Router Agent│ ─────────────────────► │ Retriever Agent │
└─────────────┘                        └────────┬────────┘
   │ complex / multi-step                        │
   ▼                                              ▼
┌─────────────────┐                     ┌──────────────────┐
│ (future) Planner │                    │ Summarizer Agent  │
└─────────────────┘                     └────────┬──────────┘
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │  Verifier Agent   │
                                         │ (checks claims    │
                                         │  against sources) │
                                         └────────┬──────────┘
                                                  │
                                                  ▼
                                          Final answer + sources
```

## Why this project (for interviews)

Most fresher GenAI portfolios stop at "upload a PDF, ask a question" — a
single retrieval chain. This project adds two things interviewers actually
probe for and most fresher projects don't have:

1. **Agentic routing** — a `RouterAgent` decides whether a query is a
   simple factual lookup or needs multi-step reasoning, instead of always
   running the same fixed chain.
2. **A verification step** — a `VerifierAgent` checks the drafted answer
   against the retrieved chunks and flags claims that aren't supported,
   instead of returning the LLM's output blindly. This is the single
   biggest talking point in an interview: *"how do you know your RAG
   system isn't hallucinating?"* — you can point at working code.

## Tech stack (maps directly to a GenAI resume skill list)

| Skill | Where it's used |
|---|---|
| Python, OOP | `BaseAgent` abstract class, all agents inherit from it |
| LLMs / OpenAI API / Ollama | `app/llm_client.py` — pluggable backend |
| RAG | `app/ingestion.py`, `app/vectorstore.py` |
| Embeddings / Semantic Search | `sentence-transformers` embeddings + Chroma |
| Vector Databases | ChromaDB (local, persistent) |
| AI Agents | `app/agents/*` — Router, Retriever, Summarizer, Verifier |
| Prompt Engineering | Prompt templates in each agent, few-shot in Verifier |
| Fine-tuning | `scripts/finetune_router.py` — LoRA fine-tune of a tiny
  classifier model to do query routing instead of prompting (optional,
  see below) |
| LangChain / LlamaIndex | see `app/ingestion.py` — LlamaIndex-style
  chunking; swap in LangChain's `RecursiveCharacterTextSplitter` if you
  prefer, both are drop-in compatible |
| FastAPI / REST APIs / API Design | `app/main.py` |
| SQL | `app/database.py`, `app/models.py` — SQLite, logs every query and
  document for auditability |
| AWS (EC2, S3) | `app/storage.py` — abstraction with a local-disk backend
  for dev and an S3 backend for prod (`USE_S3=true` in `.env`) |

## Quickstart (runs fully locally, no API key required for retrieval)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Optional: add OPENAI_API_KEY to .env for real LLM answers.
# Without a key, the system falls back to an extractive summarizer
# so you can still demo the full pipeline end-to-end.

uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs for interactive Swagger UI, or:

```bash
# Ingest a document
curl -X POST "http://localhost:8000/upload" -F "file=@sample_docs/sample.txt"

# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the company do?"}'
```

## Running with Docker

```bash
docker build -t doc-intelligence .
docker run -p 8000:8000 --env-file .env doc-intelligence
```

## Running tests

```bash
pytest tests/ -v
```

## Deploying to AWS EC2 (resume bullet: "deployed on AWS EC2 with S3 storage")

1. Launch a small EC2 instance (t2.micro / t3.micro is enough for a demo).
2. Create an S3 bucket, set `USE_S3=true`, `S3_BUCKET=<name>` in `.env`.
3. `git clone` this repo on the instance, install deps, run with
   `uvicorn app.main:app --host 0.0.0.0 --port 8000` (or use the Dockerfile).
4. Open the instance's security group on port 8000.

## Project structure

```
doc-intelligence-platform/
├── app/
│   ├── main.py           # FastAPI app + routes
│   ├── schemas.py         # Pydantic request/response models
│   ├── database.py        # SQLite session setup
│   ├── models.py          # SQLAlchemy ORM models (Document, QueryLog)
│   ├── storage.py         # Local disk / S3 storage abstraction
│   ├── ingestion.py       # Chunking + embedding pipeline
│   ├── vectorstore.py     # ChromaDB wrapper
│   ├── llm_client.py      # OpenAI / Ollama / offline-fallback client
│   └── agents/
│       ├── base_agent.py
│       ├── router_agent.py
│       ├── retriever_agent.py
│       ├── summarizer_agent.py
│       ├── verifier_agent.py
│       └── orchestrator.py   # wires all agents together
├── scripts/
│   ├── ingest_sample.py
│   └── finetune_router.py    # optional LoRA fine-tune, see docstring
├── tests/
├── sample_docs/sample.txt
├── requirements.txt
├── Dockerfile
└── .env.example
```

## What to say in an interview (script for yourself)

- *"Why agents instead of a single chain?"* — Different queries need
  different work. A router avoids paying the cost of a full multi-agent
  pass for a trivial lookup, and it's a pattern that generalizes to
  planner/executor systems.
- *"How do you handle hallucination?"* — The Verifier agent re-checks the
  summarizer's draft against the actual retrieved chunks and flags
  unsupported sentences before the answer is returned.
- *"What would you improve with more time?"* — Add a proper planner for
  multi-hop questions, swap the verifier's heuristic for an LLM-as-judge
  eval harness (RAGAS), and add streaming responses.

## Resume bullet suggestions

- Built a multi-agent GenAI document Q&A platform (FastAPI, LangChain-style
  RAG, ChromaDB) with a router agent for query triage and a verifier agent
  that flags unsupported claims before returning answers.
- Designed a pluggable LLM backend (OpenAI / Ollama) and deployed the
  service on AWS EC2 with S3-backed document storage and SQLite-logged
  query history.
