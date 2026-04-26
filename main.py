import os
import time
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai
from google.genai import types
from pinecone import Pinecone
from pydantic import BaseModel

from agents.critic_agent import validate_answer
from agents.monitor_agent import MonitorAgent
from mlops.logger import get_logger
from mlops.tracker import ExperimentTracker

load_dotenv()

app = FastAPI(
    title="Property Knowledge RAG Assistant",
    description=(
        "Agentic RAG system for real estate property knowledge retrieval. "
        "Features a critic validation agent and a proactive monitoring agent "
        "for knowledge gap detection."
    ),
    version="2.0.0",
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

INDEX_NAME = "property-knowledge"
NAMESPACE = "property_support"

# Deferred clients - populated by ensure_initialized() on first request.
pc = None
index = None
client = None
monitor = None
tracker = None

# Structured JSON logger
logger = get_logger("property_rag")


class FeedbackRequest(BaseModel):
    query_id: int
    rating: Literal["up", "down"]
    comment: str = ""


def normalize(vec: list[float]) -> list[float]:
    v = np.array(vec, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def embed(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=1536),
    )
    emb = result.embeddings[0].values
    return normalize(emb)


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
Answer using only the context provided. If the context does not contain enough information, say so clearly.

Context:
{context}

Question:
{question}
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return resp.text


documents = [
    {
        "id": "1",
        "text": "A home inspection is a critical step before closing. It covers structural integrity, roof condition, plumbing, electrical systems, and HVAC units.",
    },
    {
        "id": "2",
        "text": "Closing costs typically range from 2% to 5% of the purchase price and include lender fees, title insurance, appraisal fees, and prepaid taxes.",
    },
    {
        "id": "3",
        "text": "A buyer's agent represents the purchaser's interests during negotiation and is typically compensated through the seller's commission.",
    },
    {
        "id": "4",
        "text": "An appraisal determines the fair market value of a property and is required by lenders to ensure the loan amount does not exceed the property value.",
    },
    {
        "id": "5",
        "text": "Earnest money is a deposit made by the buyer to demonstrate commitment. It is typically 1% to 3% of the purchase price and applied toward closing costs.",
    },
    {
        "id": "6",
        "text": "A contingency clause in a purchase agreement allows the buyer to withdraw without penalty if conditions such as financing approval or satisfactory inspection are not met.",
    },
    {
        "id": "7",
        "text": "Title insurance protects buyers and lenders from financial loss due to defects in the title, such as liens, encumbrances, or ownership disputes.",
    },
    {
        "id": "8",
        "text": "Fixed-rate mortgages maintain the same interest rate for the life of the loan, while adjustable-rate mortgages (ARMs) fluctuate based on market indices after an initial fixed period.",
    },
]


_initialized = False

def ensure_initialized():
    global _initialized, pc, index, client, monitor, tracker
    if _initialized:
        return

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Proactive monitor - tracks all query cycles in memory
    monitor = MonitorAgent(low_confidence_threshold=0.6)

    # Persistent experiment tracker - logs to SQLite
    tracker = ExperimentTracker()

    vectors = []
    for d in documents:
        vectors.append(
            {
                "id": d["id"],
                "values": embed(d["text"]),
                "metadata": {"text": d["text"]},
            }
        )
    index.upsert(vectors=vectors, namespace=NAMESPACE)
    _initialized = True


@app.get("/ask")
def ask(q: str):
    """
    Agentic RAG pipeline:
    1. Embed query and retrieve from Pinecone
    2. Generate grounded answer via Gemini Flash
    3. Critic Agent - validates answer confidence against context
    4. Monitor Agent - records result for proactive pattern detection
    5. Tracker - persists full record with latency breakdown to SQLite
    """
    ensure_initialized()
    t_start = time.perf_counter()

    logger.info(
        "query_received",
        extra={
            "event": "query_received",
            "query_id": None,
            "step": None,
            "latency_ms": None,
            "log_extra": {"question": q},
        },
    )

    # Step 1: Embed
    t0 = time.perf_counter()
    qvec = embed(q)
    embed_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "embed_complete",
        extra={
            "event": "embed_complete",
            "query_id": None,
            "step": "embed",
            "latency_ms": embed_latency_ms,
            "log_extra": {},
        },
    )

    # Step 1 cont: Retrieve
    t1 = time.perf_counter()
    res = index.query(
        vector=qvec,
        top_k=3,
        include_metadata=True,
        namespace=NAMESPACE,
    )
    retrieve_latency_ms = round((time.perf_counter() - t1) * 1000, 2)
    context = "\n".join([m["metadata"]["text"] for m in res["matches"]])
    logger.info(
        "retrieve_complete",
        extra={
            "event": "retrieve_complete",
            "query_id": None,
            "step": "retrieve",
            "latency_ms": retrieve_latency_ms,
            "log_extra": {},
        },
    )

    # Step 2: Generate answer
    t2 = time.perf_counter()
    answer = generate_answer(q, context)
    generate_latency_ms = round((time.perf_counter() - t2) * 1000, 2)
    logger.info(
        "generate_complete",
        extra={
            "event": "generate_complete",
            "query_id": None,
            "step": "generate",
            "latency_ms": generate_latency_ms,
            "log_extra": {},
        },
    )

    # Step 3: Critic agent - validate answer against context
    t3 = time.perf_counter()
    critique = validate_answer(q, context, answer, client)
    critic_latency_ms = round((time.perf_counter() - t3) * 1000, 2)
    logger.info(
        "critic_complete",
        extra={
            "event": "critic_complete",
            "query_id": None,
            "step": "critic",
            "latency_ms": critic_latency_ms,
            "log_extra": {},
        },
    )

    total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # Step 4: Monitor agent - record this query cycle
    monitor.record(
        question=q,
        confidence_score=critique.get("confidence_score", 0.5),
        grounding_status=critique.get("grounding_status", "unknown"),
        flagged=critique.get("flagged", False),
    )

    # Step 5: Persist record to SQLite
    query_id = tracker.log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": q,
            "answer": answer,
            "confidence_score": critique.get("confidence_score"),
            "grounding_status": critique.get("grounding_status"),
            "flagged": critique.get("flagged", False),
            "embed_latency_ms": embed_latency_ms,
            "retrieve_latency_ms": retrieve_latency_ms,
            "generate_latency_ms": generate_latency_ms,
            "critic_latency_ms": critic_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
    )

    logger.info(
        "query_complete",
        extra={
            "event": "query_complete",
            "query_id": query_id,
            "step": None,
            "latency_ms": total_latency_ms,
            "log_extra": {},
        },
    )

    return {
        "response": answer,
        "meta": {
            "query_id": query_id,
            "confidence_score": critique.get("confidence_score"),
            "grounding_status": critique.get("grounding_status"),
            "flagged": critique.get("flagged"),
            "critic_reasoning": critique.get("reasoning"),
        },
    }


@app.post("/feedback")
def submit_feedback(body: FeedbackRequest):
    """
    Record human thumbs up/down feedback for a completed query.
    Pass the query_id returned by /ask in the request body.
    """
    ensure_initialized()
    feedback_id = tracker.log_feedback(body.query_id, body.rating, body.comment)
    logger.info(
        "feedback_received",
        extra={
            "event": "feedback_received",
            "query_id": body.query_id,
            "step": None,
            "latency_ms": None,
            "log_extra": {"rating": body.rating, "feedback_id": feedback_id},
        },
    )
    return {"status": "recorded", "query_id": body.query_id, "rating": body.rating}


@app.get("/feedback/summary")
def feedback_summary():
    """
    Aggregate feedback statistics with critic agreement and disagreement rates.
    """
    ensure_initialized()
    return tracker.get_feedback_summary()


@app.get("/monitor")
def monitoring_report():
    """
    Proactive monitoring endpoint.
    Returns a real-time report on query patterns, flagged low-confidence
    queries, and actionable recommendations.
    """
    ensure_initialized()
    return monitor.get_report()


@app.get("/metrics")
def metrics():
    """
    Structured latency and performance metrics from the SQLite experiment log.
    Includes per-step average latencies, p95 total latency, flag rate,
    and average confidence score.
    """
    ensure_initialized()
    return tracker.get_metrics()
