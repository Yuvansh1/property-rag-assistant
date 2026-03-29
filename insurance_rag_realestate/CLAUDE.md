# CLAUDE.md - Property Knowledge RAG Assistant

## Project Overview

This is a production-grade Retrieval Augmented Generation (RAG) system for real estate property knowledge retrieval.
It enables semantic search over property documents, transaction guides, and market data using Gemini Embeddings,
Gemini LLM (Flash), Pinecone vector database, and FastAPI.

## Architecture

```
User Query -> Gemini Embedding API -> Pinecone Vector DB -> Gemini LLM -> FastAPI -> JSON Response
```

- **Embedding model:** `gemini-embedding-001` (1536-dim, L2 normalized)
- **Generation model:** `gemini-2.5-flash`
- **Vector DB:** Pinecone Serverless, cosine similarity, namespace `property_support`
- **API:** FastAPI, single endpoint `/ask`, Swagger at `/docs`

## Key Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, embedding, retrieval, and generation logic |
| `Dockerfile` | Container build using python:3.11-slim |
| `docker-compose.yml` | Compose config with env_file injection |
| `requirements.txt` | Python dependencies |
| `.env` | API keys (never commit this) |
| `.env.example` | Template for environment variables |
| `CLAUDE.md` | This file - project context for Claude Code |

## Environment Variables

Always required at runtime. Never commit `.env`.

```
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
```

## Local Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally with hot reload
uvicorn main:app --reload

# Run tests
pytest tests/ -v --cov=main --cov-report=term-missing

# Run linting
flake8 main.py --max-line-length=100
black --check main.py
isort --check-only main.py
```

## Docker Commands

```bash
# Build image
docker build -t property-rag .

# Run container
docker run -p 8000:8000 --env-file .env property-rag

# Docker Compose (recommended)
docker compose up --build
```

## API Usage

```bash
# Ask a question
curl "http://localhost:8000/ask?q=What+are+closing+costs"

# Swagger UI
open http://localhost:8000/docs
```

## Coding Conventions

- Python 3.11+
- Type hints required on all functions
- No hardcoded API keys anywhere in source code
- Use `load_dotenv()` for environment variable loading
- Keep embedding and generation logic as separate functions
- FastAPI route handlers should be thin - delegate logic to helper functions

## Pinecone Index Requirements

- Index name: `property-knowledge`
- Metric: cosine
- Dimension: 1536
- Namespace: `property_support`

The index must exist in your Pinecone account before running the app.
Documents are upserted on startup via the `upsert_docs()` startup event.

## CI/CD Notes

- GitHub Actions pipelines are in `.github/workflows/`
- `ci.yml` runs on every push and pull request to `main`
- `cd.yml` runs on successful merge to `main`
- Secrets required in GitHub repo settings: `PINECONE_API_KEY`, `GEMINI_API_KEY`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`
- Do not store secrets in workflow files

## Domain Knowledge Coverage

Current document corpus covers:

- Home inspection process
- Closing costs breakdown
- Buyer and seller agent roles
- Property appraisal process
- Earnest money and deposits
- Contingency clauses
- Title insurance
- Mortgage types (fixed vs adjustable rate)

Extend by adding more documents to the `documents` list in `main.py` and re-running the app to trigger upsert.

## Known Constraints

- `@app.on_event("startup")` is deprecated in newer FastAPI - use `lifespan` context manager for future refactoring
- `res["matches"]` from Pinecone returns a dict-style object; index with string keys
- Embedding normalization is required for cosine similarity alignment with Pinecone
