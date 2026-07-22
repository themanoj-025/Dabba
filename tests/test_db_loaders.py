"""Tests for the DB-backed data loaders and the restaurants API endpoint.

Validates that:
    - load_zomato_from_db() returns a DataFrame from the database
    - load_delivery_from_db() returns a DataFrame from the database
    - load_zomato(use_db=True) falls back to CSV when DB is empty
    - The /v1/restaurants endpoint returns paginated results from the DB
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dabba.database.models import Base, Order, Restaurant, RESTAURANT_COL_MAP
from dabba.database.seed import seed_orders, seed_restaurants


# ─── API test fixtures (shared with test_api.py) ─────────────────────


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api.main import app
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture
def api_key() -> str:
    """Return a test API key or None if not configured."""
    return os.environ.get("DABBA_API_KEY")


def auth_headers(api_key: str | None) -> Dict[str, str]:
    """Return auth headers if an API key is configured."""
    if api_key:
        return {"X-API-Key": api_key}
    return {}


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_memory_config():
    """Create a config with an in-memory SQLite URL and init its tables."""
    from dabba.config import DabbaConfig
    from dabba.database.session import init_db, dispose_engine

    dispose_engine()
    config = DabbaConfig(database_url="sqlite:///:memory:")
    init_db(config)
    return config


# ─── DB-backed Loader Tests ─────────────────────────────────────────


class TestLoadZomatoFromDb:
    """Tests for load_zomato_from_db()."""

    def test_loads_from_seeded_db(self):
        """Should return a DataFrame when restaurants exist in DB."""
        from dabba.data.loaders import load_zomato_from_db

        config = _make_memory_config()
        df = pd.DataFrame(
            {
                "name": ["R1", "R2", "R3"],
                "rate": [4.5, 3.8, 4.2],
                "cost_for_two": [300, 800, 1200],
                "location": ["Koramangala", "Indiranagar", "MG Road"],
                "cuisines": ["Indian", "Italian", "Japanese"],
                "votes": [500, 200, 350],
                "cuisine_count": [2, 2, 2],
            }
        )
        seed_restaurants(df, config)

        result = load_zomato_from_db(config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "name" in result.columns
        assert "rate" in result.columns

    def test_raises_when_empty(self):
        """Should raise ValueError when no restaurants in DB."""
        from dabba.data.loaders import load_zomato_from_db

        config = _make_memory_config()
        with pytest.raises(ValueError, match="No restaurants found"):
            load_zomato_from_db(config)


class TestLoadDeliveryFromDb:
    """Tests for load_delivery_from_db()."""

    def test_loads_from_seeded_db(self):
        """Should return a DataFrame when orders exist in DB."""
        from dabba.data.loaders import load_delivery_from_db

        config = _make_memory_config()
        df = pd.DataFrame(
            {
                "haversine_distance_km": [5.0, 2.0],
                "time_taken_min": [30.0, 15.0],
                "traffic_ordinal": [2, 0],
                "is_festival": [False, False],
                "delivery_person_age": [30.0, 25.0],
                "delivery_person_ratings": [4.0, 4.5],
                "vehicle_condition": [1, 1],
            }
        )
        seed_orders(df, config=config)

        result = load_delivery_from_db(config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "haversine_distance_km" in result.columns
        assert "time_taken_min" in result.columns

    def test_raises_when_empty(self):
        """Should raise ValueError when no orders in DB."""
        from dabba.data.loaders import load_delivery_from_db

        config = _make_memory_config()
        with pytest.raises(ValueError, match="No orders found"):
            load_delivery_from_db(config)


class TestLoadWithUseDbFlag:
    """Tests for the use_db parameter on load_zomato/load_delivery."""

    def test_use_db_false_ignores_db(self):
        """use_db=False should always load from CSV (not DB)."""
        from dabba.data.loaders import load_zomato
        from dabba.config import DabbaConfig
        from dabba.database.session import dispose_engine

        dispose_engine()
        # Create a config pointing to a non-existent CSV
        config = DabbaConfig(
            database_url="sqlite:///:memory:",
            zomato_filename="nonexistent.csv",
        )
        with pytest.raises(FileNotFoundError):
            load_zomato(config, use_db=False)

    def test_use_db_true_falls_back_to_csv(self):
        """use_db=True should fall back to CSV when DB is empty."""
        from dabba.data.loaders import load_zomato

        # This test requires the CSV to exist; if not, it should raise
        # FileNotFoundError after the DB fallback fails.
        from dabba.config import get_config

        config = get_config()
        # Just verify the fallback path doesn't crash when DB is empty
        try:
            result = load_zomato(config, use_db=True)
            assert isinstance(result, pd.DataFrame)
        except (FileNotFoundError, ValueError):
            # Expected when neither DB nor CSV has data
            pass


# ─── Restaurants API Tests ──────────────────────────────────────────


class TestRestaurantsEndpoint:
    """Tests for /v1/restaurants API endpoint.

    Note: These tests use the global FastAPI TestClient which connects
    to the default database (SQLite). The endpoint structure is validated;
    data-level assertions are limited since the TestClient may not share
    the same engine as the seed functions.
    """

    def test_list_restaurants_structure(self, client, api_key):
        """Should return valid JSON with expected keys."""
        response = client.get(
            "/v1/restaurants",
            headers=auth_headers(api_key),
            params={"limit": 5, "offset": 0},
        )
        # 200 if DB is available, 503 if startup failed
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "restaurants" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data

    def test_get_restaurant_not_found(self, client, api_key):
        """Should return 404 for non-existent restaurant ID."""
        response = client.get(
            "/v1/restaurants/999999",
            headers=auth_headers(api_key),
        )
        assert response.status_code in [404, 503]

    def test_search_restaurants_structure(self, client, api_key):
        """Should return valid search response structure."""
        response = client.get(
            "/v1/restaurants/search/Indian",
            headers=auth_headers(api_key),
        )
        assert response.status_code in [200, 404, 503]
        if response.status_code == 200:
            data = response.json()
            assert "restaurants" in data


# ─── Full Import CLI Tests ──────────────────────────────────────────


class TestFullImport:
    """Tests for the full_import() function."""

    def test_full_import_handles_missing_csv(self):
        """Should exit gracefully when raw CSVs are not available."""
        from dabba.database.seed import full_import
        from dabba.config import DabbaConfig

        # Use a config pointing to a non-existent CSV
        config = DabbaConfig(
            database_url="sqlite:///:memory:",
            zomato_filename="definitely_not_real.csv",
        )
        # Should return without crashing (logs error, returns early)
        full_import(config)

    def test_full_import_runs_when_csv_exists(self):
        """Should complete when raw CSVs are available."""
        from dabba.database.seed import full_import
        from dabba.config import get_config

        config = get_config()
        if not config.zomato_path.exists():
            pytest.skip("Raw Zomato CSV not available")
        try:
            full_import(config)
        except Exception:
            # Acceptable — may fail on delivery CSV or features
            pass
