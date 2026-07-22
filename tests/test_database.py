"""Tests for the database layer — seed, repositories, and session management.

Uses an in-memory SQLite database to avoid test pollution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from dabba.database.models import Base, DriftLog, ExperimentResult, Restaurant
from dabba.database.repositories import (
    count_restaurants,
    get_all_orders,
    get_all_restaurants,
    get_drift_summary,
    get_experiment_results,
    get_orders_by_restaurant,
    get_recent_drift_logs,
    get_restaurant_by_id,
    get_restaurant_by_name,
    get_restaurants_by_cuisine,
    get_winning_model,
)
from dabba.database.seed import seed_orders, seed_restaurants


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_memory_config():
    """Create a config with an in-memory SQLite URL and init its tables."""
    from dabba.config import DabbaConfig
    from dabba.database.session import init_db, dispose_engine

    dispose_engine()  # Clear any cached global engine first
    config = DabbaConfig(database_url="sqlite:///:memory:")
    init_db(config)
    return config


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def in_memory_db():
    """Create tables on an in-memory SQLite database and yield a session."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
        db.rollback()
    finally:
        db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_restaurants_df() -> pd.DataFrame:
    """Small restaurant DataFrame for seeding tests."""
    return pd.DataFrame(
        {
            "name": ["Test Biryani House", "Pasta Paradise", "Sushi Spot"],
            "rate": [4.5, 3.8, 4.2],
            "cost_for_two": [300, 800, 1200],
            "location": ["Koramangala", "Indiranagar", "MG Road"],
            "cuisines": ["North Indian, Biryani", "Italian, Pizza", "Japanese, Sushi"],
            "votes": [500, 200, 350],
            "cuisine_count": [2, 2, 2],
        }
    )


@pytest.fixture
def sample_delivery_df() -> pd.DataFrame:
    """Small delivery DataFrame for seeding tests."""
    return pd.DataFrame(
        {
            "haversine_distance_km": [3.5, 7.2, 1.8],
            "time_taken_min": [25.0, 45.0, 18.0],
            "traffic_ordinal": [1, 3, 0],
            "is_festival": [False, True, False],
            "delivery_person_age": [28.0, 35.0, 22.0],
            "delivery_person_ratings": [4.5, 3.2, 4.8],
            "vehicle_condition": [1, 0, 1],
        }
    )


# ─── Seed Tests ──────────────────────────────────────────────────────


class TestSeedRestaurants:
    """Tests for the seed_restaurants function."""

    def test_creates_restaurant_rows(self, sample_restaurants_df):
        """Should create database rows from DataFrame."""
        config = _make_memory_config()
        n = seed_restaurants(sample_restaurants_df, config)
        assert n == 3

    def test_upsert_same_name(self):
        """Seeding the same restaurant name twice should update, not duplicate."""
        df = pd.DataFrame(
            {
                "name": ["Test Place"],
                "rate": [4.0],
                "cost_for_two": [500],
                "location": ["HSR Layout"],
                "cuisines": ["Chinese"],
                "votes": [100],
                "cuisine_count": [1],
            }
        )
        config = _make_memory_config()
        n1 = seed_restaurants(df, config)
        assert n1 == 1

        # Seed with updated rating
        df2 = pd.DataFrame(
            {
                "name": ["Test Place"],
                "rate": [4.5],
                "cost_for_two": [500],
                "location": ["HSR Layout"],
                "cuisines": ["Chinese"],
                "votes": [100],
                "cuisine_count": [1],
            }
        )
        n2 = seed_restaurants(df2, config)
        assert n2 == 1  # Still only one row


class TestSeedOrders:
    """Tests for the seed_orders function."""

    def test_creates_order_rows(self):
        """Should create order rows from a delivery DataFrame."""
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
        config = _make_memory_config()
        n = seed_orders(df, config=config)
        assert n == 2

    def test_accepts_predictions(self):
        """Should use provided predictions array."""
        df = pd.DataFrame(
            {
                "haversine_distance_km": [5.0],
                "time_taken_min": [30.0],
                "traffic_ordinal": [2],
                "is_festival": [False],
                "delivery_person_age": [30.0],
                "delivery_person_ratings": [4.0],
                "vehicle_condition": [1],
            }
        )
        predictions = np.array([25.0])
        config = _make_memory_config()
        n = seed_orders(df, predictions=predictions, config=config)
        assert n == 1


# ─── Repository Tests ────────────────────────────────────────────────


class TestRestaurantRepository:
    """Tests for restaurant repository functions."""

    def test_get_all_restaurants(self, in_memory_db):
        """Should return all restaurants."""
        db = in_memory_db
        for name in ["A", "B", "C"]:
            db.add(Restaurant(name=name, rate=4.0))
        db.commit()

        results = get_all_restaurants(db)
        assert len(results) == 3

    def test_get_all_restaurants_pagination(self, in_memory_db):
        """Should respect limit and offset."""
        db = in_memory_db
        for i in range(10):
            db.add(Restaurant(name=f"R{i:02d}", rate=4.0))
        db.commit()

        page = get_all_restaurants(db, limit=3, offset=0)
        assert len(page) == 3

    def test_get_restaurant_by_id(self, in_memory_db):
        """Should fetch by primary key."""
        db = in_memory_db
        r = Restaurant(name="Found Me", rate=4.5)
        db.add(r)
        db.commit()

        result = get_restaurant_by_id(db, r.id)
        assert result is not None
        assert result.name == "Found Me"

    def test_get_restaurant_by_id_missing(self, in_memory_db):
        """Should return None for missing ID."""
        assert get_restaurant_by_id(in_memory_db, 999) is None

    def test_get_restaurant_by_name(self, in_memory_db):
        """Should find by name (case-insensitive)."""
        db = in_memory_db
        db.add(Restaurant(name="Udupi Palace", rate=4.0))
        db.commit()

        result = get_restaurant_by_name(db, "udupi")
        assert result is not None
        assert result.name == "Udupi Palace"

    def test_get_restaurants_by_cuisine(self, in_memory_db):
        """Should search cuisines text."""
        db = in_memory_db
        db.add(Restaurant(name="A", cuisines="North Indian, Chinese"))
        db.add(Restaurant(name="B", cuisines="Italian, Pizza"))
        db.commit()

        results = get_restaurants_by_cuisine(db, "Chinese")
        assert len(results) == 1
        assert results[0].name == "A"

    def test_count_restaurants(self, in_memory_db):
        """Should return correct count."""
        db = in_memory_db
        for i in range(5):
            db.add(Restaurant(name=f"R{i}", rate=3.0))
        db.commit()

        assert count_restaurants(db) == 5

    def test_count_restaurants_empty(self, in_memory_db):
        """Empty table should return 0."""
        assert count_restaurants(in_memory_db) == 0


class TestExperimentRepository:
    """Tests for experiment result repository functions."""

    def test_get_winning_model(self, in_memory_db):
        """Should return the winning model for a task."""
        db = in_memory_db
        db.add(ExperimentResult(task="rating", model_name="XGBoost", mae=0.3, rmse=0.4, r2=0.8, train_time_s=10.0, is_winner=True))
        db.add(ExperimentResult(task="rating", model_name="CatBoost", mae=0.35, rmse=0.45, r2=0.75, train_time_s=12.0, is_winner=False))
        db.commit()

        winner = get_winning_model(db, task="rating")
        assert winner is not None
        assert winner.model_name == "XGBoost"

    def test_get_winning_model_no_winner(self, in_memory_db):
        """Should return None if no winner flagged."""
        db = in_memory_db
        db.add(ExperimentResult(task="eta", model_name="LightGBM", mae=5.0, rmse=7.0, r2=0.3, train_time_s=5.0, is_winner=False))
        db.commit()

        assert get_winning_model(db, task="eta") is None


class TestDriftLogRepository:
    """Tests for drift log repository functions."""

    def test_get_recent_drift_logs(self, in_memory_db):
        """Should return drift logs ordered by time."""
        db = in_memory_db
        db.add(DriftLog(feature_name="rating", ks_statistic=0.3, p_value=0.01, threshold=0.05, n_reference=100, n_batch=100, alerted=True))
        db.add(DriftLog(feature_name="eta", ks_statistic=0.2, p_value=0.03, threshold=0.05, n_reference=100, n_batch=100, alerted=False))
        db.commit()

        logs = get_recent_drift_logs(db)
        assert len(logs) == 2

    def test_get_drift_summary(self, in_memory_db):
        """Should return aggregate summary."""
        db = in_memory_db
        db.add(DriftLog(feature_name="a", ks_statistic=0.3, p_value=0.01, threshold=0.05, n_reference=100, n_batch=100, alerted=True))
        db.add(DriftLog(feature_name="a", ks_statistic=0.25, p_value=0.02, threshold=0.05, n_reference=100, n_batch=100, alerted=True))
        db.add(DriftLog(feature_name="b", ks_statistic=0.2, p_value=0.04, threshold=0.05, n_reference=100, n_batch=100, alerted=False))
        db.commit()

        summary = get_drift_summary(db)
        assert summary["total_drift_events"] == 3
        assert summary["total_alerted"] == 2
        assert summary["unique_features_monitored"] == 2
