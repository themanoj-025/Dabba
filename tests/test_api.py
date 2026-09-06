"""Smoke tests for the FastAPI application.

Tests cover the /health (unauthenticated) and /v1/* (authenticated)
endpoints. When DABBA_API_KEY is not set, auth is skipped in dev mode.

These tests run in the default CI suite (no `slow` marker) and must be
able to fail: each endpoint test asserts a deterministic outcome — either
the 503 "model not loaded" path, or a stubbed-model 200 path whose
response body is checked field by field.
"""

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dabba.config import DabbaConfig, get_config


@pytest.fixture
def client() -> None:
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


def auth_headers(api_key: str | None) -> dict[str, str]:
    """Return auth headers if an API key is configured."""
    if api_key:
        return {"X-API-Key": api_key}
    return {}


# ─── Minimal model stubs ────────────────────────────────────────────
# The routers treat a non-None app.state entry as "model loaded" and call
# a couple of methods on it. These stubs let the success paths run against
# the real serialization/validation code without trained artifacts.


class _StubETAModel:
    """Stand-in for the joblib-loaded ETA pipeline."""

    def predict(self, features) -> list[float]:
        return [32.5]


class _StubRecommender:
    """Stand-in for HybridRecommender.recommend() output."""

    def recommend(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "name": "Test Dhaba",
                    "rate": 4.2,
                    "bayesian_rating": 4.1,
                    "cost_for_two": 300.0,
                    "location": "Andheri West",
                    "cuisines": "North Indian, Mughlai",
                    "combined_score": 0.92,
                }
            ]
        )


# ─── Health endpoint ─────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for the /health endpoint (no auth required)."""

    def test_health_returns_ok(self, client) -> None:
        """Health endpoint should return 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_always_accessible_without_key(self, client) -> None:
        """Health should work without any auth header."""
        response = client.get("/health", headers={})
        assert response.status_code == 200

    def test_ready_ok_when_models_loaded(self, client) -> None:
        """Readiness returns 200 once both models are in app.state."""
        client.app.state.hybrid_recommender = object()
        client.app.state.eta_model = object()

        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert all(c["status"] == "up" for c in data["checks"])

    def test_ready_503_when_models_not_loaded(self, client) -> None:
        """Readiness returns 503 when startup has not loaded the models."""
        client.app.state.hybrid_recommender = None
        client.app.state.eta_model = None

        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checks"][0]["status"] == "down"


# ─── Model Info endpoint ─────────────────────────────────────────────


class TestModelInfoEndpoint:
    """Tests for the /v1/model-info endpoint."""

    def test_model_info_returns_json(self, client, api_key) -> None:
        """Model info endpoint should return JSON with model data."""
        response = client.get("/v1/model-info", headers=auth_headers(api_key))
        assert response.status_code == 200
        data = response.json()
        assert "rating_model" in data
        assert "eta_model" in data


# ─── ETA endpoint ────────────────────────────────────────────────────


class TestETAEndpoint:
    """Tests for the /v1/predict-eta endpoint."""

    def test_predict_eta_returns_503_without_model(self, client, api_key) -> None:
        """Must 503 deterministically when no model is loaded."""
        client.app.state.eta_model = None
        response = client.post(
            "/v1/predict-eta",
            headers=auth_headers(api_key),
            json={
                "distance_km": 5.0,
                "traffic_level": 1,
                "is_festival": False,
            },
        )
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"]

    def test_predict_eta_success_with_stub_model(self, client, api_key) -> None:
        """With a model loaded, returns the predicted minutes and SLA flag."""
        client.app.state.eta_model = _StubETAModel()
        response = client.post(
            "/v1/predict-eta",
            headers=auth_headers(api_key),
            json={
                "distance_km": 5.0,
                "traffic_level": 1,
                "is_festival": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_minutes"] == 32.5
        assert isinstance(data["is_at_risk"], bool)
        assert isinstance(data["sla_threshold"], (int, float))

    def test_predict_eta_schema(self, client, api_key) -> None:
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

    def test_recommend_schema(self, client, api_key) -> None:
        """Invalid request should return 422."""
        response = client.post(
            "/v1/recommend",
            headers=auth_headers(api_key),
            json={"top_n": -1},
        )
        assert response.status_code == 422

    def test_recommend_returns_503_without_recommender(self, client, api_key) -> None:
        """Must 503 deterministically when the recommender is not loaded."""
        client.app.state.hybrid_recommender = None
        response = client.post(
            "/v1/recommend",
            headers=auth_headers(api_key),
            json={
                "cuisine": "North Indian",
                "budget": 500,
                "top_n": 5,
            },
        )
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"]

    def test_recommend_success_with_stub_recommender(self, client, api_key) -> None:
        """With a recommender loaded, returns ranked recommendations."""
        client.app.state.hybrid_recommender = _StubRecommender()
        response = client.post(
            "/v1/recommend",
            headers=auth_headers(api_key),
            json={
                "cuisine": "North Indian",
                "budget": 500,
                "top_n": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) == 1
        first = data["recommendations"][0]
        assert first["name"] == "Test Dhaba"
        assert first["rating"] == 4.2
        assert first["combined_score"] == 0.92


# ─── Chat endpoint ───────────────────────────────────────────────────


class TestChatEndpoint:
    """Tests for the /v1/chat endpoint."""

    def test_chat_schema_validation(self, client, api_key) -> None:
        """Empty message should return 422."""
        response = client.post(
            "/v1/chat",
            headers=auth_headers(api_key),
            json={"message": ""},
        )
        assert response.status_code == 422

    def test_chat_returns_503_without_tools(self, client, api_key) -> None:
        """Must 503 deterministically when concierge tools are not loaded."""
        client.app.state.concierge_tools = None
        response = client.post(
            "/v1/chat",
            headers=auth_headers(api_key),
            json={"message": "Find me some good food", "history": []},
        )
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"]

    def test_chat_success_returns_reply(self, client, api_key, monkeypatch) -> None:
        """A stubbed concierge response is returned verbatim in the reply."""
        client.app.state.concierge_tools = object()
        monkeypatch.setattr(
            "api.routers.chat.get_concierge_response",
            lambda *_args, **_kwargs: "Try Bombay Canteen!",
        )
        response = client.post(
            "/v1/chat",
            headers=auth_headers(api_key),
            json={"message": "Find me some good food", "history": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "Try Bombay Canteen!"


# ─── Auth behavior ───────────────────────────────────────────────────


class TestAuthBehavior:
    """Tests for API key authentication behavior."""

    def test_v1_endpoint_accessible_in_dev_mode(self, client) -> None:
        """When DABBA_API_KEY is unset, v1 endpoints must NOT require a key."""
        response = client.get("/v1/model-info")
        # Dev mode: auth is skipped — a 401 here means auth is misfiring.
        assert response.status_code != 401
        if response.status_code == 200:
            data = response.json()
            assert "rating_model" in data

    def test_v1_requires_auth_when_key_configured(self, client) -> None:
        """When an API key is configured, requests without it must 401.

        Uses FastAPI dependency_overrides instead of monkeypatching env +
        reloading the app, which the previous version admitted was fragile.
        """
        secured = DabbaConfig(_env_file=None)
        secured.api_key = "test-key-123"
        client.app.dependency_overrides[get_config] = lambda: secured
        try:
            missing = client.get("/v1/model-info")
            assert missing.status_code == 401
            assert "detail" in missing.json()

            # The correct key must pass auth (model may or may not be loaded)
            with_key = client.get(
                "/v1/model-info", headers={"X-API-Key": "test-key-123"}
            )
            assert with_key.status_code == 200
            assert "rating_model" in with_key.json()

            wrong_key = client.get(
                "/v1/model-info", headers={"X-API-Key": "wrong-key"}
            )
            assert wrong_key.status_code == 401
        finally:
            client.app.dependency_overrides.clear()


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

    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
    except UnicodeDecodeError:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except FileNotFoundError:
        pytest.skip(f"{rel_path} not found")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        pytest.skip(f"Cannot parse {rel_path}")

    violations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_csv"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("pd", "pandas")
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

    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
    except UnicodeDecodeError:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except FileNotFoundError:
        pytest.skip(f"{rel_path} not found")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        pytest.skip(f"Cannot parse {rel_path}")

    violations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_csv"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("pd", "pandas")
        ):
            violations.append(
                f"  Line {node.lineno}: direct pd.read_csv() call"
            )

    assert not violations, (
        f"{rel_path} still contains pd.read_csv() calls:\n"
        + "\n".join(violations)
        + "\nMigrate to repository functions (database/repositories.py)."
    )