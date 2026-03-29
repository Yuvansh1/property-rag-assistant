# Real Estate Knowledge Retrieval System

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production--Ready-green)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-purple)
![Gemini](https://img.shields.io/badge/LLM-Gemini-orange)
![Agentic RAG](https://img.shields.io/badge/Architecture-Agentic--RAG-red)
![Docker](https://img.shields.io/badge/Containerized-Docker-blue)
![CI](https://github.com/Yuvansh1/property-rag-assistant/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/Yuvansh1/property-rag-assistant/actions/workflows/cd.yml/badge.svg)

Production-grade Agentic RAG system for real estate property knowledge retrieval. Extends a standard RAG pipeline with two specialized AI agents that add self-validation and proactive monitoring on top of retrieval and generation.

Built using Gemini Embeddings, Gemini LLM, Pinecone Serverless, FastAPI, and Docker. Includes a full CI/CD pipeline via GitHub Actions.

---

## What Makes This Agentic

Standard RAG: query -> embed -> retrieve -> generate -> respond.

This system adds two agent layers on top:

| Agent | Role |
|---|---|
| **Critic Agent** | Validates every generated answer against its retrieved context, returning a confidence score and grounding status to surface hallucination risk before it reaches the user |
| **Monitor Agent** | Proactively tracks query patterns and confidence scores across sessions, flags low-confidence trends, and generates actionable recommendations without being asked |

---

## Architecture

```
                     User Query
                         |
                         v
         Gemini Embedding API (1536-dim, L2 normalized)
                         |
                         v
            Pinecone Vector Database
            Namespace: property_support
                         |
                     Top-K Results
                         |
                         v
           Gemini LLM (gemini-2.5-flash)
           Context-constrained generation
                         |
                         v
              [Critic Agent]
              - Confidence score (0.0-1.0)
              - Grounding status
              - Flags low-confidence answers
                         |
                         v
              [Monitor Agent]
              - Records query cycle
              - Tracks flagged patterns
              - Surfaces proactive insights
                         |
                         v
              FastAPI JSON Response
              (answer + critic meta)
```

---

## Project Structure

```
property_rag_assistant/
|
|- main.py                          # FastAPI app, agentic pipeline orchestration
|- Dockerfile
|- docker-compose.yml
|- requirements.txt
|- CLAUDE.md
|- README.md
|- .env
|- .env.example
|- .gitignore
|- .dockerignore
|
|- agents/
|   |- __init__.py
|   |- critic_agent.py              # Answer validation + confidence scoring
|   |- monitor_agent.py             # Proactive monitoring + flagged query tracking
|
|- tests/
|   |- conftest.py
|   |- test_main.py                 # Unit + integration tests for all agents
|
|- .github/
    |- workflows/
        |- ci.yml
        |- cd.yml
```

---

## API Endpoints

### `GET /ask?q=your+question`

Full agentic RAG pipeline. Returns the answer plus critic metadata.

```bash
curl "http://localhost:8000/ask?q=What+are+closing+costs"
```

Response:

```json
{
  "response": "Closing costs typically range from 2% to 5% of the purchase price and include lender fees, title insurance, appraisal fees, and prepaid taxes.",
  "meta": {
    "confidence_score": 0.92,
    "grounding_status": "grounded",
    "flagged": false,
    "critic_reasoning": "Answer directly references the cost percentages present in context."
  }
}
```

### `GET /monitor`

Proactive monitoring report. Surfaces flagged query patterns and recommendations without being asked.

```bash
curl "http://localhost:8000/monitor"
```

Response:

```json
{
  "summary": {
    "total_queries": 12,
    "flagged_queries": 2,
    "flag_rate": 0.167,
    "average_confidence_score": 0.84
  },
  "grounding_distribution": {
    "grounded": 9,
    "partially_grounded": 1,
    "ungrounded": 2
  },
  "recent_flagged_queries": [...],
  "recommendation": "Moderate flag rate of 17%. Review recent flagged queries to identify knowledge gaps."
}
```

### `GET /docs`

Swagger UI with full endpoint documentation.

---

## Environment Variables

```
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
```

Copy `.env.example` to `.env` and fill in your keys. Never commit `.env`.

---

## Prerequisites

- Python 3.11+
- Pinecone account with index `property-knowledge` (cosine, 1536 dimensions)
- Google AI Studio API key (Gemini Embedding + Flash access)
- Docker (optional)

---

## Local Setup

```bash
git clone https://github.com/Yuvansh1/property-rag-assistant.git
cd property-rag-assistant

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env

uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API.

---

## Running Tests

```bash
pip install pytest pytest-cov httpx

pytest tests/ -v --cov=main --cov-report=term-missing
```

Test coverage includes:

- Normalize unit tests
- Critic Agent: key validation, fallback on bad JSON, low confidence flagging, grounded answer detection
- Monitor Agent: record tracking, flag rate calculation, average confidence, report structure, empty state, high flag rate recommendation
- `/ask` endpoint: status, response shape, meta fields, missing query validation, response type
- `/monitor` endpoint: status, report keys, reflects actual query history

---

## Streamlit UI

A dark-themed frontend is included alongside the FastAPI backend.

### Tabs

**ASK**
- Text input for any real estate question
- Answer displayed in a styled card
- Confidence score shown as a colored percentage (green >= 75%, yellow >= 50%, red below)
- Grounding status badge: Grounded / Partially Grounded / Ungrounded
- Flagged warning badge when critic flags the answer
- Critic reasoning shown in a muted italic box
- Sidebar with 8 sample questions as clickable buttons

**MONITOR**
- Loads the `/monitor` report on tab open
- Four metric cards: Total Queries, Flagged Queries, Flag Rate, Avg Confidence
- Recommendation displayed in a highlighted box
- Grounding distribution as horizontal progress bars
- Recent flagged queries list with score and timestamp
- Refresh button to reload data

### Local Run

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

---

## Docker Deployment

Both services start together:

```bash
docker compose up --build
```

| Service | Port | URL |
|---|---|---|
| FastAPI | 8000 | http://localhost:8000/docs |
| Streamlit | 8501 | http://localhost:8501 |

The Streamlit container waits for the API healthcheck to pass before starting.

Or run the API directly:

```bash
docker build -t property-rag .
docker run -p 8000:8000 --env-file .env property-rag
```

---

## CI/CD Pipeline

### CI (`ci.yml`) - triggers on push and PR to `main`
1. Lint with flake8, black, isort
2. Run pytest with coverage
3. Docker build validation

### CD (`cd.yml`) - triggers on merge to `main`
1. Build Docker image
2. Push to Docker Hub (`latest` + `sha-` tags)

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `PINECONE_API_KEY` | Pinecone API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

---

## Extending the Knowledge Base

Add documents to the `documents` list in `main.py` and restart. The startup event upserts everything automatically.

```python
{"id": "9", "text": "Zoning regulations determine permitted land use and building types in a given area."}
```

The Monitor Agent will surface if queries about that topic were previously being flagged as low-confidence, confirming the gap was filled.

---

## Portfolio Value

- Agentic pipeline design with two specialized, composable agents
- Self-validating answer generation with confidence scoring and grounding detection
- Proactive monitoring and knowledge gap surfacing
- Production-grade FastAPI microservice architecture
- Containerized deployment with Docker
- CI/CD automation with GitHub Actions
- Comprehensive test coverage including agent unit tests and end-to-end integration
