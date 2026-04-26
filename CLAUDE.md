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
| `streamlit_app.py` | Streamlit frontend UI |
| `Dockerfile` | FastAPI container build using python:3.11-slim |
| `Dockerfile.streamlit` | Streamlit container build |
| `docker-compose.yml` | Runs both FastAPI and Streamlit services |
| `requirements.txt` | FastAPI dependencies |
| `requirements.streamlit.txt` | Streamlit dependencies |
| `.env` | API keys (never commit) |
| `.env.example` | Template |

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /ask?q=...` | Agentic RAG pipeline with critic validation |
| `GET /monitor` | Proactive monitoring report with flagged query analysis |
| `GET /feedback` | Submit thumbs up/down feedback on an answer |
| `GET /metrics` | Structured latency and performance metrics |
| `GET /docs` | Swagger UI |

## Environment Variables

```
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
```

## Local Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI
uvicorn main:app --reload

# Run Streamlit (separate terminal)
streamlit run streamlit_app.py

# Run tests
pytest tests/ -v --cov=main --cov-report=term-missing

# Run linting
flake8 main.py agents/ --max-line-length=100
python -m black main.py agents/
python -m isort main.py agents/
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
- Expects the LLM to return a valid JSON string with no markdown fences

**Monitor Agent (`agents/monitor_agent.py`)**
- In-memory session tracker (swap for DB persistence in production)
- Tracks: flag rate, average confidence, grounding distribution
- Exposes proactive recommendations via /monitor endpoint
- Surfaces recent flagged queries for knowledge corpus review

## CI/CD Notes

- `.github/workflows/ci.yml` runs on push/PR to main: lint, test, docker build
- `.github/workflows/cd.yml` runs on merge to main: docker build validation only (no push)
- Secrets required in GitHub: PINECONE_API_KEY, GEMINI_API_KEY

---

## Tasks

### Fix Failing CI Tests

The CI pipeline is failing on Unit and Integration Tests. The root cause is that
`agents/critic_agent.py` calls `json.loads()` on the Gemini response text, but the
test mocks are not returning valid JSON strings for the second `generate_content` call.

Please do the following in order:

1. Read `tests/conftest.py` in full
2. Read `tests/test_main.py` in full
3. Read `agents/critic_agent.py` to understand exactly what JSON format it expects
4. Read `main.py` to understand the full `/ask` pipeline - note that `generate_content`
   is called TWICE per `/ask` request: once for the answer, once for the critic
5. Fix `tests/conftest.py` to ensure Pinecone and Gemini are fully mocked at module
   level before `main.py` is imported, so `TestNormalize` tests do not hit real APIs
6. Fix `tests/test_main.py` so that every `/ask` endpoint test provides exactly two
   `generate_content` side_effect responses:
   - First: the answer text as a plain string
   - Second: a valid JSON string matching the critic agent schema:
     `{"confidence_score": 0.88, "grounding_status": "grounded", "reasoning": "Matches context.", "flagged": false}`
7. Run `pytest tests/ -v` locally and confirm all tests pass before finishing
8. Do not change any logic in `main.py` or `agents/` - only fix test files

---

### Add MLOps Components

Add three MLOps layers to the project: persistent experiment tracking, a human
feedback loop, and structured logging with latency metrics.

#### 1. Persistent Experiment Tracking (SQLite)

Create `mlops/tracker.py`:
- Use Python's built-in `sqlite3` module, no extra dependencies
- On first run, create a SQLite database at `data/experiments.db`
- Create a table `query_log` with columns:
  id, timestamp, question, answer, confidence_score, grounding_status,
  flagged, embed_latency_ms, retrieve_latency_ms, generate_latency_ms,
  critic_latency_ms, total_latency_ms
- Expose a class `ExperimentTracker` with methods:
  - `log(record: dict)` - insert a row
  - `get_recent(n: int) -> list` - return last n rows as dicts
  - `get_summary() -> dict` - return avg confidence, flag rate, avg latency per step
- The database file should be created automatically if it does not exist
- Add `data/` to `.gitignore`

#### 2. Human Feedback Loop

Add a `POST /feedback` endpoint to `main.py`:
- Accepts JSON body: `{"query_id": int, "rating": "up" or "down", "comment": str (optional)}`
- Creates a `feedback` table in the same SQLite DB with columns:
  id, query_id, rating, comment, timestamp
- Returns `{"status": "recorded", "query_id": int, "rating": str}`
- Add a `GET /feedback/summary` endpoint that returns:
  - total_feedback count
  - thumbs_up count
  - thumbs_down count
  - agreement_rate: % of thumbs_up where critic flagged=False (human agrees with critic)
  - disagreement_rate: % of thumbs_down where critic flagged=False (human disagrees with critic)

Update the `/ask` response to include `query_id` from the database so the frontend
can pass it back to `/feedback`.

#### 3. Structured Logging and Latency Metrics

Create `mlops/logger.py`:
- Use Python's built-in `logging` module configured for JSON output
- Every log entry should be a JSON object with fields:
  timestamp, level, event, query_id, step, latency_ms, extra (dict)
- Expose a function `get_logger(name: str)` that returns a configured logger
- Log events: query_received, embed_complete, retrieve_complete, generate_complete,
  critic_complete, query_complete, feedback_received

Add a `GET /metrics` endpoint to `main.py`:
- Returns real-time aggregate metrics from the SQLite DB:
  - total_queries
  - avg_total_latency_ms
  - avg_embed_latency_ms
  - avg_retrieve_latency_ms
  - avg_generate_latency_ms
  - avg_critic_latency_ms
  - flag_rate
  - avg_confidence_score
  - p95_total_latency_ms (95th percentile)

#### Integration in main.py

Update the `/ask` endpoint to:
- Time each step (embed, retrieve, generate, critic) using `time.perf_counter()`
- Log each step completion using the structured logger
- Insert the full record into the SQLite tracker after each request
- Include `query_id` in the response meta

#### Files to create:
- `mlops/__init__.py`
- `mlops/tracker.py`
- `mlops/logger.py`
- `data/.gitkeep` (empty file to track the data/ folder)

#### Files to update:
- `main.py` - add timing, logging, tracker calls, /feedback and /metrics endpoints
- `requirements.txt` - no new dependencies needed (sqlite3 and logging are stdlib)
- `.gitignore` - add `data/*.db`
- `README.md` - add MLOps section documenting the three new endpoints

#### Coding conventions:
- No em dashes anywhere
- All latency measurements in milliseconds rounded to 2 decimal places
- SQLite connections should use context managers (with sqlite3.connect(...) as conn)
- Logger should output one JSON object per line to stdout
- Do not break any existing tests - add new tests for /feedback and /metrics endpoints

---

### Add Streamlit UI

Add a Streamlit frontend to this project alongside the existing FastAPI backend.

**Files to create:**

`streamlit_app.py`
- Dark themed UI with three tabs: ASK, MONITOR, METRICS
- ASK tab:
  - Text input for user question
  - On submit, call GET http://api:8000/ask?q={question}
  - Display the answer in a styled card
  - Display confidence_score as a colored percentage (green >= 0.75, yellow >= 0.5, red below)
  - Display grounding_status as a badge (grounded / partially_grounded / ungrounded)
  - Display flagged as a warning badge if true
  - Display critic_reasoning in a muted italic box below
  - Show thumbs up / thumbs down buttons after each answer that call POST http://api:8000/feedback
  - Sidebar with 8 sample questions as clickable buttons that populate the input
- MONITOR tab:
  - Call GET http://api:8000/monitor on load
  - Show 4 metric cards: total_queries, flagged_queries, flag_rate, average_confidence_score
  - Show recommendation in a highlighted box
  - Show grounding_distribution as horizontal progress bars
  - Show recent_flagged_queries as a list with score and timestamp
  - Refresh button to reload data
- METRICS tab:
  - Call GET http://api:8000/metrics on load
  - Show latency breakdown: avg embed, retrieve, generate, critic latency as metric cards
  - Show p95 total latency
  - Show feedback summary from GET http://api:8000/feedback/summary
  - Refresh button
- API base URL: http://api:8000 (Docker) -- no env var needed, hardcoded for simplicity
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
- Add a Streamlit UI section documenting the three tabs
- Update Docker Compose instructions to mention both ports
- Add local run instructions: streamlit run streamlit_app.py

**Coding conventions:**
- No em dashes anywhere in code or comments
- Use requests library (not httpx) for API calls in Streamlit
- All Streamlit markdown injected via st.markdown with unsafe_allow_html=True for custom styling
- Keep streamlit_app.py self-contained, no imports from agents/ or mlops/ folders
