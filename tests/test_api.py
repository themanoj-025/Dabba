"""Smoke tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api.main import app

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint."""

    def test_model_info_returns_json(self, client):
        """Model info endpoint should return JSON."""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "rating_model" in data
        assert "eta_model" in data


class TestETAEndpoint:
    """Tests for the /predict-eta endpoint."""

    def test_predict_eta_without_model(self, client):
        """Should return 503 if model is not loaded."""
        response = client.post(
            "/predict-eta",
            json={
                "distance_km": 5.0,
                "traffic_level": 1,
                "is_festival": False,
            },
        )
        # Either 503 (no model) or 200 (model loaded)
        assert response.status_code in [200, 503]

    def test_predict_eta_schema(self, client):
        """Request should be validated by Pydantic schema."""
        response = client.post(
            "/predict-eta",
            json={
                "distance_km": "not_a_number",
            },
        )
        assert response.status_code == 422  # Validation error
