"""
Tests for Property Knowledge RAG Assistant - main.py
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


@pytest.fixture
def mock_pinecone():
    with patch("main.Pinecone") as mock_pc:
        mock_index = MagicMock()
        mock_pc.return_value.Index.return_value = mock_index
        yield mock_index


@pytest.fixture
def mock_genai_client():
    with patch("main.genai.Client") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance
        yield client_instance


@pytest.fixture
def client(mock_pinecone, mock_genai_client):
    """Create a test FastAPI client with mocked dependencies."""
    mock_genai_client.models.embed_content.return_value = MagicMock(
        embeddings=[MagicMock(values=[0.1] * 1536)]
    )
    mock_pinecone.upsert.return_value = None

    from main import app
    return TestClient(app)


# Unit Tests

class TestNormalize:
    def test_normalize_returns_unit_vector(self):
        from main import normalize
        import numpy as np
        vec = [3.0, 4.0]
        result = normalize(vec)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6

    def test_normalize_returns_list(self):
        from main import normalize
        result = normalize([1.0, 0.0, 0.0])
        assert isinstance(result, list)

    def test_normalize_length_preserved(self):
        from main import normalize
        vec = [0.1] * 1536
        result = normalize(vec)
        assert len(result) == 1536


class TestEmbed:
    def test_embed_calls_gemini(self, mock_genai_client):
        mock_genai_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.5] * 1536)]
        )
        from main import embed
        result = embed("what are closing costs")
        assert len(result) == 1536
        mock_genai_client.models.embed_content.assert_called_once()

    def test_embed_returns_normalized_vector(self, mock_genai_client):
        import numpy as np
        mock_genai_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[3.0] + [0.0] * 1535)]
        )
        from main import embed
        result = embed("what is earnest money")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6


class TestGenerateAnswer:
    def test_generate_answer_returns_string(self, mock_genai_client):
        mock_genai_client.models.generate_content.return_value = MagicMock(
            text="Closing costs typically range from 2% to 5% of the purchase price."
        )
        from main import generate_answer
        result = generate_answer(
            "What are closing costs?",
            "Closing costs typically range from 2% to 5% of the purchase price."
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_answer_calls_gemini_flash(self, mock_genai_client):
        mock_genai_client.models.generate_content.return_value = MagicMock(text="Answer.")
        from main import generate_answer
        generate_answer("question", "context")
        call_kwargs = mock_genai_client.models.generate_content.call_args
        assert "gemini-2.5-flash" in str(call_kwargs)


# Integration Tests

class TestAskEndpoint:
    def test_ask_returns_200(self, client, mock_pinecone, mock_genai_client):
        mock_genai_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1] * 1536)]
        )
        mock_pinecone.query.return_value = {
            "matches": [
                {"metadata": {"text": "A home inspection covers structural integrity and roof condition."}}
            ]
        }
        mock_genai_client.models.generate_content.return_value = MagicMock(
            text="A home inspection covers structural integrity, roof condition, plumbing, electrical systems, and HVAC."
        )
        response = client.get("/ask?q=What+does+a+home+inspection+cover")
        assert response.status_code == 200

    def test_ask_returns_response_key(self, client, mock_pinecone, mock_genai_client):
        mock_genai_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1] * 1536)]
        )
        mock_pinecone.query.return_value = {
            "matches": [
                {"metadata": {"text": "Earnest money is typically 1% to 3% of the purchase price."}}
            ]
        }
        mock_genai_client.models.generate_content.return_value = MagicMock(
            text="Earnest money is a deposit of 1% to 3% of the purchase price."
        )
        response = client.get("/ask?q=What+is+earnest+money")
        assert "response" in response.json()

    def test_ask_missing_query_returns_422(self, client):
        response = client.get("/ask")
        assert response.status_code == 422

    def test_docs_endpoint_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_ask_response_is_string(self, client, mock_pinecone, mock_genai_client):
        mock_genai_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1] * 1536)]
        )
        mock_pinecone.query.return_value = {
            "matches": [
                {"metadata": {"text": "Title insurance protects buyers from ownership disputes."}}
            ]
        }
        mock_genai_client.models.generate_content.return_value = MagicMock(
            text="Title insurance protects against liens and ownership disputes."
        )
        response = client.get("/ask?q=What+is+title+insurance")
        assert isinstance(response.json()["response"], str)
