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

---

## Tasks

### Add Streamlit UI

Add a Streamlit frontend to this project alongside the existing FastAPI backend.

**Files to create:**

`streamlit_app.py`
- Dark themed UI with two tabs: ASK and MONITOR
- ASK tab:
  - Text input for user question
  - On submit, call GET http://api:8000/ask?q={question}
  - Display the answer in a styled card
  - Display confidence_score as a colored percentage (green >= 0.75, yellow >= 0.5, red below)
  - Display grounding_status as a badge (grounded / partially_grounded / ungrounded)
  - Display flagged as a warning badge if true
  - Display critic_reasoning in a muted italic box below
  - Sidebar with 8 sample questions as clickable buttons that populate the input
- MONITOR tab:
  - Call GET http://api:8000/monitor on load
  - Show 4 metric cards: total_queries, flagged_queries, flag_rate, average_confidence_score
  - Show recommendation in a highlighted box
  - Show grounding_distribution as horizontal progress bars
  - Show recent_flagged_queries as a list with score and timestamp
  - Refresh button to reload data
- API base URL: http://api:8000 (Docker) — no env var needed, hardcoded for simplicity
- Hide Streamlit default header, footer, and menu

`Dockerfile.streamlit`
- Base image: python:3.11-slim
- Install streamlit and requests only
- Run: streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0

**Files to update:**

`docker-compose.yml`
- Keep existing api service on port 8000 with healthcheck
- Add streamlit service on port 8501
- streamlit depends_on api with condition: service_healthy
- Both services use restart: unless-stopped

`requirements.txt`
- Add streamlit
- Add requests

`README.md`
- Add a Streamlit UI section documenting the two tabs
- Update Docker Compose instructions to mention both ports
- Add local run instructions: streamlit run streamlit_app.py

**Coding conventions:**
- No em dashes anywhere in code or comments
- Use requests library (not httpx) for API calls in Streamlit
- All Streamlit markdown injected via st.markdown with unsafe_allow_html=True for custom styling
- Keep streamlit_app.py self-contained, no imports from agents/ folder
