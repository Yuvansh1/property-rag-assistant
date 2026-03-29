import os
import numpy as np
from fastapi import FastAPI
from pinecone import Pinecone
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Property Knowledge RAG Assistant",
    description="Semantic retrieval and AI-generated answers over real estate property knowledge.",
    version="1.0.0",
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

INDEX_NAME = "property-knowledge"
NAMESPACE = "property_support"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

client = genai.Client(api_key=GEMINI_API_KEY)


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


@app.on_event("startup")
def upsert_docs():
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


@app.get("/ask")
def ask(q: str):
    qvec = embed(q)
    res = index.query(vector=qvec, top_k=3, include_metadata=True, namespace=NAMESPACE)
    context = "\n".join([m["metadata"]["text"] for m in res["matches"]])
    return {"response": generate_answer(q, context)}
