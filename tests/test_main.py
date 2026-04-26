"""
Tests for Property Knowledge Agentic RAG Assistant
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pinecone():
    with patch("main.index") as mock_index:
        yield mock_index


@pytest.fixture
def mock_genai_client():
    with patch("main.client") as mock_client:
        yield mock_client


@pytest.fixture
def mock_tracker():
    with patch("main.tracker") as mt:
        mt.log.return_value = 1
        mt.log_feedback.return_value = 1
        mt.get_metrics.return_value = {
            "total_queries": 0,
            "avg_total_latency_ms": 0.0,
            "avg_embed_latency_ms": 0.0,
            "avg_retrieve_latency_ms": 0.0,
            "avg_generate_latency_ms": 0.0,
            "avg_critic_latency_ms": 0.0,
            "flag_rate": 0.0,
            "avg_confidence_score": 0.0,
            "p95_total_latency_ms": 0.0,
        }
        mt.get_feedback_summary.return_value = {
            "total_feedback": 0,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "agreement_rate": 0.0,
            "disagreement_rate": 0.0,
        }
        yield mt


def make_llm_response(text: str) -> MagicMock:
    return MagicMock(text=text)


CRITIC_JSON = '{"confidence_score": 0.88, "grounding_status": "grounded", "reasoning": "Matches context.", "flagged": false}'


@pytest.fixture
def client(mock_pinecone, mock_genai_client, mock_tracker):
    mock_genai_client.models.embed_content.return_value = MagicMock(
        embeddings=[MagicMock(values=[0.1] * 1536)]
    )
    mock_pinecone.upsert.return_value = None
    import main
    main._initialized = True
    yield TestClient(main.app)
    main._initialized = False


# Unit Tests - Normalize

class TestNormalize:
    def test_returns_unit_vector(self):
        from main import normalize
        import numpy as np
        result = normalize([3.0, 4.0])
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_returns_list(self):
        from main import normalize
        assert isinstance(normalize([1.0, 0.0, 0.0]), list)

    def test_length_preserved(self):
        from main import normalize
        assert len(normalize([0.1] * 1536)) == 1536


# Unit Tests - Critic Agent

class TestCriticAgent:
    def test_validate_returns_required_keys(self):
        from agents.critic_agent import validate_answer
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_llm_response(
            '{"confidence_score": 0.9, "grounding_status": "grounded", "reasoning": "Answer matches context.", "flagged": false}'
        )
        result = validate_answer("question", "context", "answer", mock_client)
        assert "confidence_score" in result
        assert "grounding_status" in result
        assert "reasoning" in result
        assert "flagged" in result

    def test_validate_falls_back_on_bad_json(self):
        from agents.critic_agent import validate_answer
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_llm_response("bad output")
        result = validate_answer("q", "ctx", "ans", mock_client)
        assert result["flagged"] is True
        assert result["confidence_score"] == 0.5

    def test_flagged_true_when_low_confidence(self):
        from agents.critic_agent import validate_answer
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_llm_response(
            '{"confidence_score": 0.3, "grounding_status": "ungrounded", "reasoning": "No match.", "flagged": true}'
        )
        result = validate_answer("q", "ctx", "ans", mock_client)
        assert result["flagged"] is True
        assert result["confidence_score"] < 0.6

    def test_grounded_answer_not_flagged(self):
        from agents.critic_agent import validate_answer
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_llm_response(
            '{"confidence_score": 0.95, "grounding_status": "grounded", "reasoning": "Direct match.", "flagged": false}'
        )
        result = validate_answer("q", "ctx", "ans", mock_client)
        assert result["flagged"] is False
        assert result["confidence_score"] >= 0.6


# Unit Tests - Monitor Agent

class TestMonitorAgent:
    def test_record_adds_to_log(self):
        from agents.monitor_agent import MonitorAgent
        m = MonitorAgent()
        m.record("what are closing costs", 0.9, "grounded", False)
        assert len(m.query_log) == 1

    def test_flagged_query_tracked_separately(self):
        from agents.monitor_agent import MonitorAgent
        m = MonitorAgent()
        m.record("obscure question", 0.3, "ungrounded", True)
        assert len(m.flagged_queries) == 1

    def test_unflagged_query_not_in_flagged(self):
        from agents.monitor_agent import MonitorAgent
        m = MonitorAgent()
        m.record("what is earnest money", 0.9, "grounded", False)
        assert len(m.flagged_queries) == 0

    def test_report_contains_required_keys(self):
        from agents.monitor_agent import MonitorAgent
        m = MonitorAgent()
        m.record("question", 0.85, "grounded", False)
        report = m.get_report()
        assert "summary" in report
        assert "grounding_distribution" in report
        assert "recent_flagged_queries" in report
        assert "recommendation" in report

    def test_flag_rate_calculation(self):
        from agents.monitor_agent import MonitorAgent
        m = MonitorAgent()
        m.record("q1", 0.9, "grounded", False)
        m.record("q2", 0.3, "ungrounded", True)
        report = m.get_report()
        assert report["summary"]["flag_rate"] == 0.5

    def test_average_confidence_calculation(self):
        from agents.monitor_agent import MonitorAgent
        m = MonitorAgent()
        m.record("q1", 0.8, "grounded", False)
        m.record("q2", 0.6, "partially_grounded", False)
        report = m.get_report()
        assert report["summary"]["average_confidence_score"] == 0.7

    def test_empty_monitor_report(self):
        from agents.monitor_agent import MonitorAgent
        m = MonitorAgent()
        report = m.get_report()
        assert report["summary"]["total_queries"] == 0
        assert report["summary"]["flag_rate"] == 0.0

    def test_high_flag_rate_recommendation(self):
        from agents.monitor_agent import MonitorAgent
        m = MonitorAgent()
        for i in range(5):
            m.record(f"q{i}", 0.2, "ungrounded", True)
        report = m.get_report()
        assert "High flag rate" in report["recommendation"]


# Integration Tests - /ask endpoint

class TestAskEndpoint:
    def _setup_mocks(self, mock_pinecone, mock_genai_client, answer_text="Closing costs are 2-5%."):
        mock_genai_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1] * 1536)]
        )
        mock_pinecone.query.return_value = {
            "matches": [{"metadata": {"text": "Closing costs range from 2% to 5%."}}]
        }
        mock_genai_client.models.generate_content.side_effect = [
            make_llm_response(answer_text),
            make_llm_response(CRITIC_JSON),
        ]

    def test_ask_returns_200(self, client, mock_pinecone, mock_genai_client):
        self._setup_mocks(mock_pinecone, mock_genai_client)
        response = client.get("/ask?q=What+are+closing+costs")
        assert response.status_code == 200

    def test_ask_returns_response_and_meta(self, client, mock_pinecone, mock_genai_client):
        self._setup_mocks(mock_pinecone, mock_genai_client)
        data = client.get("/ask?q=What+are+closing+costs").json()
        assert "response" in data
        assert "meta" in data

    def test_meta_contains_critic_fields(self, client, mock_pinecone, mock_genai_client):
        self._setup_mocks(mock_pinecone, mock_genai_client)
        meta = client.get("/ask?q=What+are+closing+costs").json()["meta"]
        assert "confidence_score" in meta
        assert "grounding_status" in meta
        assert "flagged" in meta
        assert "critic_reasoning" in meta

    def test_meta_contains_query_id(self, client, mock_pinecone, mock_genai_client):
        self._setup_mocks(mock_pinecone, mock_genai_client)
        meta = client.get("/ask?q=What+are+closing+costs").json()["meta"]
        assert "query_id" in meta

    def test_ask_missing_query_returns_422(self, client):
        assert client.get("/ask").status_code == 422

    def test_response_is_string(self, client, mock_pinecone, mock_genai_client):
        self._setup_mocks(mock_pinecone, mock_genai_client)
        data = client.get("/ask?q=What+is+title+insurance").json()
        assert isinstance(data["response"], str)


# Integration Tests - /monitor endpoint

class TestMonitorEndpoint:
    def test_monitor_returns_200(self, client):
        assert client.get("/monitor").status_code == 200

    def test_monitor_returns_summary(self, client):
        data = client.get("/monitor").json()
        assert "summary" in data
        assert "recommendation" in data
        assert "grounding_distribution" in data

    def test_monitor_reflects_ask_queries(self, client, mock_pinecone, mock_genai_client):
        mock_genai_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1] * 1536)]
        )
        mock_pinecone.query.return_value = {
            "matches": [{"metadata": {"text": "Earnest money is 1-3% of purchase price."}}]
        }
        mock_genai_client.models.generate_content.side_effect = [
            make_llm_response("Earnest money is a commitment deposit."),
            make_llm_response(CRITIC_JSON),
        ]
        client.get("/ask?q=What+is+earnest+money")
        report = client.get("/monitor").json()
        assert report["summary"]["total_queries"] >= 1


# Integration Tests - /feedback endpoint

class TestFeedbackEndpoint:
    def test_feedback_up_returns_200(self, client):
        response = client.post("/feedback", json={"query_id": 1, "rating": "up"})
        assert response.status_code == 200

    def test_feedback_down_returns_200(self, client):
        response = client.post("/feedback", json={"query_id": 1, "rating": "down"})
        assert response.status_code == 200

    def test_feedback_returns_recorded_status(self, client):
        data = client.post("/feedback", json={"query_id": 1, "rating": "up"}).json()
        assert data["status"] == "recorded"
        assert data["rating"] == "up"
        assert data["query_id"] == 1

    def test_feedback_invalid_rating_returns_422(self, client):
        response = client.post("/feedback", json={"query_id": 1, "rating": "maybe"})
        assert response.status_code == 422

    def test_feedback_missing_query_id_returns_422(self, client):
        response = client.post("/feedback", json={"rating": "up"})
        assert response.status_code == 422

    def test_feedback_summary_returns_200(self, client):
        assert client.get("/feedback/summary").status_code == 200

    def test_feedback_summary_contains_required_keys(self, client):
        data = client.get("/feedback/summary").json()
        assert "total_feedback" in data
        assert "thumbs_up" in data
        assert "thumbs_down" in data
        assert "agreement_rate" in data
        assert "disagreement_rate" in data


# Integration Tests - /metrics endpoint

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_metrics_contains_required_keys(self, client):
        data = client.get("/metrics").json()
        assert "total_queries" in data
        assert "avg_total_latency_ms" in data
        assert "avg_embed_latency_ms" in data
        assert "avg_retrieve_latency_ms" in data
        assert "avg_generate_latency_ms" in data
        assert "avg_critic_latency_ms" in data
        assert "flag_rate" in data
        assert "avg_confidence_score" in data
        assert "p95_total_latency_ms" in data