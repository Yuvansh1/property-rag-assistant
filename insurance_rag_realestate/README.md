# Property Knowledge RAG Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production--Ready-green)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-purple)
![Gemini](https://img.shields.io/badge/LLM-Gemini-orange)
![RAG](https://img.shields.io/badge/Architecture-RAG-red)
![Docker](https://img.shields.io/badge/Containerized-Docker-blue)
![CI](https://github.com/YOUR_USERNAME/property-rag-assistant/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/YOUR_USERNAME/property-rag-assistant/actions/workflows/cd.yml/badge.svg)

Enterprise-grade Retrieval Augmented Generation (RAG) system for real estate property knowledge retrieval. Enables semantic search and context-grounded AI responses over property documents, transaction guides, and market data. Built using Gemini Embeddings, Gemini LLM, Pinecone Serverless Vector Index, FastAPI, and Docker. Includes a full CI/CD pipeline via GitHub Actions.

---

## Problem Statement

Real estate platforms handle high volumes of repetitive, high-stakes queries:

- What are the typical closing costs for a home purchase?
- What does a home inspection cover?
- What is the difference between a fixed and adjustable rate mortgage?
- How does earnest money work?
- What is title insurance and why do I need it?

Manual lookup increases response times and operational costs at scale. This project implements a scalable RAG architecture to deliver context-grounded AI responses from property knowledge documents, reducing latency and improving answer accuracy.

---

## Architecture

```
                  User Query
                      |
                      v
          Gemini Embedding API (gemini-embedding-001)
                      |
                 1536-dim vector (L2 normalized)
                      |
                      v
          Pinecone Vector Database
          Namespace: property_support
                      |
                  Top-K Semantic Results
                      |
                      v
          Gemini LLM (gemini-2.5-flash)
          Context-grounded Generation
                      |
                      v
              FastAPI REST Layer
                      |
                      v
                JSON Response
```

---

## Project Structure

```
property_rag_assistant/
|
|- main.py                          # FastAPI app, RAG logic
|- Dockerfile                       # Container build
|- docker-compose.yml               # Compose config
|- requirements.txt                 # Python dependencies
|- CLAUDE.md                        # Claude Code project context
|- README.md                        # This file
|- .env                             # API keys (never commit)
|- .env.example                     # Environment variable template
|- .gitignore
|- .dockerignore
|
|- tests/
|   |- conftest.py                  # Pytest config
|   |- test_main.py                 # Unit and integration tests
|
|- .github/
    |- workflows/
        |- ci.yml                   # CI: lint, test, docker build check
        |- cd.yml                   # CD: build and push to Docker Hub
```

---

## Environment Variables

Create a `.env` file at the project root before running locally or via Docker:

```
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
```

A `.env.example` template is included. The `.env` file is excluded from version control. Never commit it.

---

## Prerequisites

- Python 3.11+
- A [Pinecone](https://app.pinecone.io) account with an index named `property-knowledge` (cosine metric, 1536 dimensions)
- A [Google AI Studio](https://aistudio.google.com) API key with access to Gemini Embedding and Gemini Flash models
- Docker (for containerized deployment)
- A GitHub account with Actions enabled (for CI/CD)

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Yuvansh1/property-rag-assistant.git
cd property-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your PINECONE_API_KEY and GEMINI_API_KEY
```

### 5. Run the application

```bash
uvicorn main:app --reload
```

### 6. Open Swagger UI

```
http://127.0.0.1:8000/docs
```

### 7. Query the API

```bash
curl "http://127.0.0.1:8000/ask?q=What+are+closing+costs"
```

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run all tests with coverage
pytest tests/ -v --cov=main --cov-report=term-missing

# Run a specific test class
pytest tests/test_main.py::TestAskEndpoint -v
```

## Running Linting

```bash
pip install flake8 black isort

flake8 main.py --max-line-length=100
black --check main.py
isort --check-only main.py
```

---

## Docker Deployment

### Build and run with Docker

```bash
docker build -t property-rag .
docker run -p 8000:8000 --env-file .env property-rag
```

### Build and run with Docker Compose (recommended)

```bash
docker compose up --build
```

Access the API at `http://localhost:8000/docs`.

---

## CI/CD Pipeline (GitHub Actions)

### CI Pipeline (`ci.yml`)

Triggers on every push and pull request to `main` and `develop`.

Steps:
1. Lint with flake8, black, and isort
2. Run pytest with coverage report
3. Build Docker image to validate the Dockerfile

### CD Pipeline (`cd.yml`)

Triggers on every merge to `main` (and can be triggered manually via `workflow_dispatch`).

Steps:
1. Build Docker image
2. Push to Docker Hub with `latest` and `sha-` tags

### Required GitHub Secrets

Add these in your repository under Settings > Secrets and variables > Actions:

| Secret | Description |
|---|---|
| `PINECONE_API_KEY` | Your Pinecone API key |
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token |

### Setting Up the Pipeline

1. Push the repository to GitHub
2. Go to Settings > Secrets and variables > Actions
3. Add all four secrets listed above
4. Push any commit to `main` or open a pull request to trigger CI
5. Update the badge URLs in this README by replacing `YOUR_USERNAME` with your GitHub username

---

## Example Queries

```bash
curl "http://localhost:8000/ask?q=What+does+a+home+inspection+cover"
```

Response:

```json
{
  "response": "A home inspection covers structural integrity, roof condition, plumbing, electrical systems, and HVAC units."
}
```

```bash
curl "http://localhost:8000/ask?q=What+is+earnest+money"
```

Response:

```json
{
  "response": "Earnest money is a deposit made by the buyer to demonstrate commitment to the purchase. It is typically 1% to 3% of the purchase price and is applied toward closing costs at the end of the transaction."
}
```

---

## System Design Details

### Embedding Layer

- Model: `gemini-embedding-001`
- Output dimension: 1536
- L2 normalized for cosine similarity alignment with Pinecone

### Vector Storage

- Pinecone Serverless Index
- Metric: cosine similarity
- Namespace isolation for multi-tenant and multi-domain support
- Metadata stored alongside vectors for context retrieval

### Retrieval Layer

- Top-K semantic similarity search
- Namespace-filtered queries
- Context aggregation from metadata fields

### Generation Layer

- Model: `gemini-2.5-flash`
- Strict context-constrained prompting
- Hallucination minimized through grounding - model explicitly instructed to stay within retrieved context

### API Layer

- FastAPI REST endpoint: `/ask`
- Swagger auto-documentation at `/docs`
- Stateless microservice design

---

## Extending the Knowledge Base

Add new documents to the `documents` list in `main.py`:

```python
{"id": "9", "text": "Your new property knowledge content here."}
```

Restart the app. The startup event automatically upserts all documents into Pinecone on launch.

---

## Portfolio Value

This project demonstrates:

- End-to-end RAG system design and implementation
- Vector database architecture with Pinecone Serverless
- Embedding engineering and L2 normalization
- Context-constrained prompting to minimize hallucination
- FastAPI microservice architecture
- Containerized AI deployment with Docker
- CI/CD automation with GitHub Actions
- Production-safe environment variable management
- Test coverage with mocked external dependencies
