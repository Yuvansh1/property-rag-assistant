"""
Query Critic Agent

Validates the generated answer against the retrieved context.
Returns a confidence score and flags whether the answer is grounded,
partially grounded, or hallucinated.
"""

import json

from google import genai

CRITIC_PROMPT = """
You are a critic agent for a real estate knowledge retrieval system.

Your job is to evaluate whether a generated answer is properly grounded in the provided context.

Given:
- The original user question
- The retrieved context documents
- The generated answer

Return a JSON object with:
- "confidence_score": float between 0.0 and 1.0 (1.0 = fully grounded, 0.0 = not grounded at all)
- "grounding_status": one of ["grounded", "partially_grounded", "ungrounded"]
- "reasoning": one sentence explaining your confidence score
- "flagged": boolean - true if confidence_score is below 0.6

Return ONLY valid JSON. No preamble, no explanation, no markdown.

Example output:
{{
  "confidence_score": 0.92,
  "grounding_status": "grounded",
  "reasoning": "The answer directly references closing cost percentages present in the context.",
  "flagged": false
}}
"""


def validate_answer(
    question: str,
    context: str,
    answer: str,
    client: genai.Client,
) -> dict:
    """
    Validates a generated answer against its retrieved context.

    Args:
        question: Original user question
        context: Retrieved context from Pinecone
        answer: Generated answer from Gemini LLM
        client: Initialized Gemini client

    Returns:
        dict with confidence_score, grounding_status, reasoning, flagged
    """
    prompt = f"""
{CRITIC_PROMPT}

User question: {question}

Retrieved context:
{context}

Generated answer:
{answer}
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    raw = resp.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "confidence_score": 0.5,
            "grounding_status": "partially_grounded",
            "reasoning": "Critic agent could not parse response.",
            "flagged": True,
        }

    return result
