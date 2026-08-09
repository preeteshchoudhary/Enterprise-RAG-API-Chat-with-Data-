"""
Integration tests for FastAPI REST API endpoints.
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "timestamp" in data


def test_chat_endpoint():
    payload = {
        "query": "What is the capital expenditure breakdown for 2024?",
        "top_k_dense": 10,
        "top_k_sparse": 10,
        "top_k_rerank": 3,
        "apply_rerank": True,
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "latency_metrics" in data
    assert "token_usage" in data
    assert data["query"] == payload["query"]


def test_evaluate_endpoint():
    response = client.post("/api/v1/evaluate")
    assert response.status_code == 200
    data = response.json()
    assert "context_precision" in data
    assert "context_recall" in data
    assert "answer_faithfulness" in data
    assert "overall_ragas_score" in data
