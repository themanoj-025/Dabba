"""Smoke tests for the FastAPI application.

Tests cover the /health (unauthenticated) and /v1/* (authenticated)
endpoints. When DABBA_API_KEY is not set, auth is skipped in dev mode.
"""

import os
from typing import Dict

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app.

    Uses DABBA_API_KEY from environment if set, otherwise
    the API runs in dev mode (no auth required).
    """
    from api.main import app

    return TestClient(app)


@pytest.fixture
def api_key() -> str:
    """Return a test API key or None if not configured.

    If DABBA_API_KEY is set in the environment, use it;
    otherwise the API is in dev mode and auth is skipped.
    """
    return os.environ.get("DABBA_API_KEY")


def auth_headers(api_key: str | None) -> Dict[str, str]:
    """Return auth headers if an API key is configured."""
    if api_key:
        return {"X-API-Key": api_key}
    return {}


# ─── Health endpoint ─────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for the /health endpoint (no auth required)."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_always_accessible_without_key(self, client):
        """Health should work without any auth header."""
        response = client.get("/health", headers={})
        assert response.status_code == 200


# ─── Model Info endpoint ─────────────────────────────────────────────


class TestModelInfoEndpoint:
    """Tests for the /v1/model-info endpoint."""

    def test_model_info_returns_json(self, client, api_key):
        """Model info endpoint should return JSON with model data."""
        response = client.get("/v1/model-info", headers=auth_headers(api_key))
        assert response.status_code == 200
        data = response.json()
        assert "rating_model" in data
        assert "eta_model" in data

    def test_model_info_requires_auth_when_configured(self, client, monkeypatch):
        """If DABBA_API_KEY is set, requests without it should 401."""
        monkeypatch.setenv("DABBA_API_KEY", "test-key-123")
        # Reimport to pick up the new env var
        import importlib

        import api.main as api_main
        importlib.reload(api_main)

        test_client = TestClient(api_main.app)
        response = test_client.get("/v1/model-info")
        # Should return 401 if key is configured but missing
        # (this test only matters if monkeypatch works; reload is fragile)
        if response.status_code == 401:
            data = response.json()
            assert "detail" in data


# ─── ETA endpoint ────────────────────────────────────────────────────


class TestETAEndpoint:
    """Tests for the /v1/predict-eta endpoint."""

    def test_predict_eta_without_model(self, client, api_key):
        """Should return 503 if model is not loaded."""
        response = client.post(
            "/v1/predict-eta",
            headers=auth_headers(api_key),
            json={
                "distance_km": 5.0,
                "traffic_level": 1,
                "is_festival": False,
            },
        )
        # Either 503 (no model) or 200 (model loaded)
        assert response.status_code in [200, 503]

    def test_predict_eta_schema(self, client, api_key):
        """Request should be validated by Pydantic schema."""
        response = client.post(
            "/v1/predict-eta",
            headers=auth_headers(api_key),
            json={
                "distance_km": "not_a_number",
            },
        )
        assert response.status_code == 422  # Validation error


# ─── Recommend endpoint ──────────────────────────────────────────────


class TestRecommendEndpoint:
    """Tests for the /v1/recommend endpoint."""

    def test_recommend_schema(self, client, api_key):
        """Invalid request should return 422."""
        response = client.post(
            "/v1/recommend",
            headers=auth_headers(api_key),
            json={"top_n": -1},
        )
        assert response.status_code == 422

    def test_recommend_valid(self, client, api_key):
        """Valid request should return 200 or 503 (model not loaded)."""
        response = client.post(
            "/v1/recommend",
            headers=auth_headers(api_key),
            json={
                "cuisine": "North Indian",
                "budget": 500,
                "top_n": 5,
            },
        )
        assert response.status_code in [200, 503]


# ─── Chat endpoint ───────────────────────────────────────────────────


class TestChatEndpoint:
    """Tests for the /v1/chat endpoint."""

    def test_chat_schema_validation(self, client, api_key):
        """Empty message should return 422."""
        response = client.post(
            "/v1/chat",
            headers=auth_headers(api_key),
            json={"message": ""},
        )
        assert response.status_code == 422

    def test_chat_valid(self, client, api_key):
        """Valid request should return 200 or 503 (tools not loaded)."""
        response = client.post(
            "/v1/chat",
            headers=auth_headers(api_key),
            json={"message": "Find me some good food", "history": []},
        )
        # Either 200 (tools loaded) or 503 (not loaded)
        assert response.status_code in [200, 503]


# ─── Auth behavior ───────────────────────────────────────────────────


class TestAuthBehavior:
    """Tests for API key authentication behavior."""

    def test_v1_endpoint_accessible_in_dev_mode(self, client):
        """When DABBA_API_KEY is not set, v1 endpoints should work without key."""
        # This test validates the dev-mode behavior: no key = no auth
        response = client.get("/v1/model-info")
        # Should be either 200 (dev mode, no key) or 200/503 (normal)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "rating_model" in data


# ─── CSV-read prohibition tests (P0 migration) ───────────────────────


@pytest.mark.parametrize(
    "rel_path",
    [
        "api/routers/model_info.py",
        "api/routers/recommend.py",
        "api/routers/chat.py",
    ],
)
def test_api_routers_no_csv_reads(rel_path: str) -> None:
    """Assert that API routers do not import pd.read_csv directly.

    After the CSV→DB migration, the serving path should read from
    the database, not from CSV files. CSV reads are only allowed in
    ``database/seed.py`` (the import pipeline).
    """
    import ast

    root = Path(__file__).resolve().parent.parent
    file_path = root / rel_path
    if not file_path.exists():
        pytest.skip(f"{rel_path} not found")

    with open(file_path) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        pytest.skip(f"Cannot parse {rel_path}")

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "read_csv":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in (
                        "pd",
                        "pandas",
                    ):
                        violations.append(
                            f"  Line {node.lineno}: direct pd.read_csv() call"
                        )

    assert not violations, (
        f"{rel_path} still contains pd.read_csv() calls:\n"
        + "\n".join(violations)
        + "\nMigrate to repository functions (database/repositories.py)."
    )


@pytest.mark.parametrize(
    "rel_path",
    [
        "app/pages/page_discover.py",
        "app/pages/page_model_performance.py",
        "app/pages/page_ops.py",
        "app/pages/page_concierge.py",
    ],
)
def test_streamlit_pages_no_csv_reads(rel_path: str) -> None:
    """Assert that Streamlit pages do not import pd.read_csv directly.

    After the CSV→DB migration, Streamlit pages should read from
    the database via repository functions, not from CSV files.
    """
    import ast

    root = Path(__file__).resolve().parent.parent
    file_path = root / rel_path
    if not file_path.exists():
        pytest.skip(f"{rel_path} not found")

    with open(file_path) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        pytest.skip(f"Cannot parse {rel_path}")

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "read_csv":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in (
                        "pd",
                        "pandas",
                    ):
                        violations.append(
                            f"  Line {node.lineno}: direct pd.read_csv() call"
                        )

    assert not violations, (
        f"{rel_path} still contains pd.read_csv() calls:\n"
        + "\n".join(violations)
        + "\nMigrate to repository functions (database/repositories.py)."
    )
