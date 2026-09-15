import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from src.interfaces.api.app import app
from src.interfaces.api.dependencies import get_agentic_workflow, get_embeddings_service
from src.domain.schemas import AgenticAskResponse, SourceCitation


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "database" in data
    assert "vector_store" in data


def test_search_endpoint(client):
    response = client.post("/api/v1/search/", json={"query": "test query", "top_k": 3, "mode": "bm25"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert "results" in data


def test_feedback_endpoint(client):
    response = client.post("/api/v1/feedback", json={"trace_id": "test-trace-123", "score": 1.0, "comment": "Great!"})
    assert response.status_code == 200
    data = response.json()
    assert "success" in data


def test_ask_endpoint_with_override(client):
    mock_workflow = MagicMock()
    mock_response = AgenticAskResponse(
        query="What is RLHF?",
        answer="RLHF aligns LLMs with human preferences.",
        sources=[
            SourceCitation(
                arxiv_id="2203.02155",
                title="Training language models to follow instructions",
                authors="Ouyang et al.",
                chunk_index=0,
                text_snippet="RLHF details...",
                score=0.95,
            )
        ],
        reasoning_steps=["Validated scope", "Retrieved docs", "Generated answer"],
        retrieval_attempts=1,
        guardrail_score=90,
        execution_time=0.45,
    )
    mock_workflow.execute = AsyncMock(return_value=mock_response)

    app.dependency_overrides[get_agentic_workflow] = lambda: mock_workflow

    try:
        response = client.post("/api/v1/ask", json={"query": "What is RLHF?"})
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is RLHF?"
        assert data["answer"] == "RLHF aligns LLMs with human preferences."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["arxiv_id"] == "2203.02155"
    finally:
        app.dependency_overrides.clear()
