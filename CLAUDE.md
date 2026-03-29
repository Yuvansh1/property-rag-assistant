# CLAUDE.md - Property Knowledge Agentic RAG Assistant

## Project Overview

Production-grade Agentic RAG system for real estate property knowledge retrieval.
Extends a standard RAG pipeline with two specialized AI agents:
a critic validation agent and a proactive monitoring agent for knowledge gap detection.

Built with Gemini Embeddings, Gemini LLM (Flash), Pinecone, FastAPI, and Docker.

## Architecture

```
User Query
    |
    v
Gemini Embedding API (gemini-embedding-001, 1536-dim, L2 normalized)
    |
    v
Pinecone Vector DB (property-knowledge index, property_support namespace)
    |
    v
Gemini LLM (gemini-2.5-flash) - context-constrained generation
    |
    v
[Critic Agent]  -- validates answer against context, returns confidence score
    |
    v
[Monitor Agent]  -- records query cycle, tracks flagged queries proactively
    |
    v
FastAPI JSON Response (answer + meta: confidence, grounding, flagged)
```

## Key Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, agentic pipeline orchestration |
| `agents/critic_agent.py` | Answer validation and confidence scoring agent |
| `agents/monitor_agent.py` | Proactive query pattern monitoring agent |
| `agents/__init__.py` | Agent exports |
| `Dockerfile` | Container build using python:3.11-slim |
| `docker-compose.yml` | Compose config |
| `requirements.txt` | Python dependencies |
| `.env` | API keys (never commit) |
| `.env.example` | Template |

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /ask?q=...` | Agentic RAG pipeline with critic validation |
| `GET /monitor` | Proactive monitoring report with flagged query analysis |
| `GET /docs` | Swagger UI |

## Environment Variables

```
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
```

## Local Development Commands

```bash
pip install -r requirements.txt
uvicorn main:app --reload
pytest tests/ -v --cov=main --cov-report=term-missing
flake8 main.py agents/ --max-line-length=100
```

## Pinecone Index Requirements

- Index name: `property-knowledge`
- Metric: cosine
- Dimension: 1536
- Namespace: `property_support`

## Agent Design Notes

**Critic Agent (`agents/critic_agent.py`)**
- Runs after answer generation to validate grounding
- Returns: confidence_score (0.0-1.0), grounding_status, reasoning, flagged
- Flags any answer with confidence_score below 0.6
- Fallback: returns flagged=True with score 0.5 on parse failure

**Monitor Agent (`agents/monitor_agent.py`)**
- In-memory session tracker (swap for DB persistence in production)
- Tracks: flag rate, average confidence, grounding distribution
- Exposes proactive recommendations via /monitor endpoint
- Surfaces recent flagged queries for knowledge corpus review

## CI/CD Notes

- `.github/workflows/ci.yml` runs on push/PR to main: lint, test, docker build
- `.github/workflows/cd.yml` runs on merge to main: build and push to Docker Hub
- Secrets: PINECONE_API_KEY, GEMINI_API_KEY, DOCKERHUB_USERNAME, DOCKERHUB_TOKEN
