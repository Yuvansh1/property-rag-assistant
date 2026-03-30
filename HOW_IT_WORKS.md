# How It Works: Property RAG Assistant

A walkthrough of the full request lifecycle with a concrete example.

---

## What This System Does

You ask a real estate question. The system finds relevant knowledge, generates a grounded answer, audits that answer for hallucinations, and tracks everything for observability. Two specialized agents sit on top of a standard RAG pipeline to make it self-validating.

---

## The Full Pipeline

```
User Question
      |
      v
Gemini Embedding API        (question -> 1536-dim vector)
      |
      v
Pinecone Vector DB          (vector -> top-3 relevant documents)
      |
      v
Gemini Flash                (documents + question -> answer)
      |
      v
Critic Agent                (question + documents + answer -> confidence score)
      |
      v
Monitor Agent               (records result to in-memory log)
      |
      v
SQLite Tracker              (persists row with full latency breakdown)
      |
      v
FastAPI JSON Response       (answer + confidence + grounding + query_id)
      |
      v
Streamlit UI                (renders answer card with badges and feedback buttons)
```

---

## Step-by-Step Example

**Input:** `"What are closing costs?"`

---

### Step 1 -- User submits the question

The Streamlit UI sends a GET request to the FastAPI backend:

```
GET http://api:8000/ask?q=What+are+closing+costs
```

---

### Step 2 -- Question gets embedded

`main.py` calls the Gemini embedding API. The text string is converted into a list of 1,536 numbers that capture its meaning mathematically. The vector is L2-normalized so distances are consistent across queries.

```python
"What are closing costs?"
    |
    v
[0.021, -0.143, 0.087, 0.334, -0.009, ...]   # 1536 numbers
```

**Why this matters:** Two sentences that mean the same thing will produce similar vectors even if they use different words. This is what makes semantic search work.

---

### Step 3 -- Pinecone retrieves the closest documents

The query vector is sent to Pinecone, which compares it against all 8 pre-loaded property documents using cosine similarity. The 3 closest matches are returned as context.

For this question, Pinecone would return:

| Score | Document |
|---|---|
| 0.94 | "Closing costs typically range from 2% to 5% of the purchase price and include lender fees, title insurance, appraisal fees, and prepaid taxes." |
| 0.81 | "Title insurance protects buyers and lenders from financial loss due to defects in the title." |
| 0.74 | "A buyer's agent represents the purchaser's interests and is typically compensated through the seller's commission." |

---

### Step 4 -- Gemini generates a grounded answer

The 3 retrieved documents become the context. Gemini Flash receives a prompt structured like this:

```
Answer using only the context provided. If the context does not
contain enough information, say so clearly.

Context:
Closing costs typically range from 2% to 5%...
Title insurance protects buyers...
A buyer's agent represents...

Question:
What are closing costs?
```

The model is explicitly constrained to the retrieved context. It cannot draw on its general training knowledge. This is what makes the answer grounded rather than hallucinated.

**Answer generated:**

> "Closing costs typically range from 2% to 5% of the purchase price and include lender fees, title insurance, appraisal fees, and prepaid taxes."

---

### Step 5 -- Critic Agent audits the answer

This is a second Gemini call. The Critic Agent receives the question, the retrieved context, and the generated answer, and is asked to evaluate whether the answer is actually supported by the documents.

**Critic prompt (simplified):**

```
You are a critic agent. Evaluate whether the generated answer
is grounded in the retrieved context. Return JSON only.
```

**Critic response:**

```json
{
  "confidence_score": 0.92,
  "grounding_status": "grounded",
  "reasoning": "Answer directly references the cost percentages present in context.",
  "flagged": false
}
```

If the answer had invented something not in the documents, the score would drop below 0.6 and `flagged` would be `true`. The Critic also handles cases where the LLM wraps its response in markdown code fences by stripping them before parsing. If JSON parsing fails entirely, it falls back to a safe default: score 0.5, flagged true.

**Grounding status values:**

| Status | Meaning |
|---|---|
| `grounded` | Answer is fully supported by retrieved context |
| `partially_grounded` | Answer mixes context with unsupported claims |
| `ungrounded` | Answer cannot be traced to the retrieved documents |

---

### Step 6 -- Monitor Agent records the result

The Monitor Agent is a pure Python class with no external API calls. It appends this query to its in-memory log:

```python
{
    "timestamp": "2026-03-29T14:22:01",
    "question": "What are closing costs?",
    "confidence_score": 0.92,
    "grounding_status": "grounded",
    "flagged": False
}
```

Over time this log powers the `/monitor` endpoint, which surfaces patterns like "17% of queries are being flagged" and recommends expanding the knowledge base to cover missing topics.

**Important:** This log is in-memory only. It resets when the server restarts. The SQLite tracker handles persistence.

---

### Step 7 -- SQLite Tracker persists the full record

Every request is written to `data/experiments.db` with a complete latency breakdown:

| Column | Example value |
|---|---|
| `embed_latency_ms` | 312.45 |
| `retrieve_latency_ms` | 87.21 |
| `generate_latency_ms` | 1204.88 |
| `critic_latency_ms` | 980.34 |
| `total_latency_ms` | 2584.88 |
| `confidence_score` | 0.92 |
| `grounding_status` | grounded |
| `flagged` | false |

This powers the `/metrics` endpoint which returns p95 latency, average confidence, and per-step averages.

---

### Step 8 -- Response returned to Streamlit

The full JSON response:

```json
{
  "response": "Closing costs typically range from 2% to 5% of the purchase price and include lender fees, title insurance, appraisal fees, and prepaid taxes.",
  "meta": {
    "query_id": 1,
    "confidence_score": 0.92,
    "grounding_status": "grounded",
    "flagged": false,
    "critic_reasoning": "Answer directly references the cost percentages present in context."
  }
}
```

Streamlit renders:

- The answer in a styled card
- A green **92%** confidence label
- A **Grounded** badge
- The critic reasoning in a muted italic box
- Thumbs up / thumbs down feedback buttons

Clicking a feedback button posts to `POST /feedback` with the `query_id`, which stores the rating in SQLite alongside the original query record.

---

## What Makes It Agentic

A standard RAG system stops after Step 4: retrieve and generate. This system adds two agents that act autonomously without being explicitly instructed for each query.

**Critic Agent** -- reactive, per-query. After every answer is generated, the Critic independently decides whether to trust it. It assigns a confidence score and flags low-confidence answers before they reach the user. This is hallucination detection running automatically on every single request.

**Monitor Agent** -- passive, aggregate. It never makes any API calls. It watches query patterns across the session and proactively surfaces systemic issues. If 8 out of 10 questions about zoning laws all come back ungrounded, the Monitor will flag that pattern in its report and recommend expanding the knowledge corpus. It catches what the Critic cannot: not just whether one answer is bad, but whether a whole category of questions is poorly covered.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /ask?q=...` | Full agentic RAG pipeline, returns answer + critic metadata |
| `GET /monitor` | Proactive monitoring report with flagged query patterns |
| `GET /metrics` | Per-step latency aggregates and p95 from SQLite |
| `POST /feedback` | Submit thumbs up/down for a completed query |
| `GET /feedback/summary` | Aggregate feedback with critic agreement rates |
| `GET /docs` | Swagger UI |

---

## Key Design Decisions

**Why embed on every request?** The query must be in the same vector space as the documents to find matches. There is no shortcut here; every new question needs a fresh embedding.

**Why use the LLM as its own critic?** This is the LLM-as-judge pattern. The same model that generated the answer is asked to evaluate it, but from a different perspective with a different prompt. It works surprisingly well for grounding detection, though it adds an extra API call per request.

**Why two separate observability systems (Monitor + SQLite)?** They serve different purposes. The Monitor is fast and in-memory, designed for real-time pattern detection within a session. SQLite is persistent, designed for long-term metrics and feedback correlation across restarts. In production you would unify these into a single database.

**Why constrain the LLM to context only?** Without this constraint the model answers from its general training knowledge, which may be outdated or incorrect for specific property details. Grounding the answer to retrieved documents makes the system auditable: you can always trace exactly which documents supported each answer.

---

## Running Locally

```bash
# Clone and set up environment
git clone https://github.com/Yuvansh1/property-rag-assistant.git
cd property-rag-assistant
cp .env.example .env
# Add your PINECONE_API_KEY and GEMINI_API_KEY to .env

# Docker (recommended)
docker compose up --build

# Manual
pip install -r requirements.txt
uvicorn main:app --reload
# In a second terminal:
streamlit run streamlit_app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |

---

## Prerequisites

- Python 3.11+
- Pinecone account with an index named `property-knowledge` (cosine, 1536 dimensions)
- Google AI Studio API key with access to Gemini Flash and gemini-embedding-001
- Docker (optional but recommended)
