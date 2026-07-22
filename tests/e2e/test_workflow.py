"""End-to-end workflow tests for Dabba.

Tests the full data pipeline on synthetic data:
    1. Cleaning → Feature Engineering (restaurant + delivery)
    2. Feature engineering produces the right columns
    3. Model training produces valid results on small data
    4. Comparison DataFrame has expected structure
    5. Drift detection works end-to-end

These tests are slower than unit tests but don't require
Kaggle datasets — they generate synthetic data inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dabba.data.cleaning import clean_delivery, clean_zomato
from dabba.features.delivery_features import add_delivery_features
from dabba.features.restaurant_features import add_restaurant_features
from dabba.monitoring.drift import DriftDetector, DriftResult


class TestCleaningToFeaturesFlow:
    """E2E test: raw data → cleaned → feature-engineered."""

    def test_zomato_flow_preserves_rows(self):
        """Clean Zomato-like data should survive cleaning + features."""
        raw = pd.DataFrame(
            {
                "name": ["R1", "R2", "R3"],
                "rate": ["4.5/5", "3.8/5", "4.1/5"],
                "approx_cost(for two people)": ["400", "800", "1200"],
                "cuisines": [
                    "North Indian, Chinese",
                    "Italian",
                    "South Indian",
                ],
                "location": ["Koramangala", "Indiranagar", "BTM"],
                "votes": [100, 200, 50],
                "online_order": ["Yes", "No", "Yes"],
                "book_table": ["No", "Yes", "No"],
                "reviews_list": [
                    "[('4.0', 'Great food!')]",
                    "[('3.0', 'Okay')]",
                    "[('5.0', 'Excellent!')]",
                ],
            }
        )
        cleaned = clean_zomato(raw)
        assert len(cleaned) == 3  # All rows valid
        assert cleaned["rate"].iloc[0] == pytest.approx(4.5)

        featured = add_restaurant_features(cleaned)
        assert "votes_log" in featured.columns
        assert "cuisine_count" in featured.columns
        assert "cost_for_two_bucket" in featured.columns

    def test_delivery_flow_adds_all_features(self):
        """Delivery data should get all engineered features."""
        raw = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "Delivery_person_ID": ["A", "B", "C"],
                "Delivery_person_Age": [25, 35, 45],
                "Delivery_person_Ratings": [4.5, 4.0, 3.5],
                "Restaurant_latitude": [12.97, 12.95, 12.91],
                "Restaurant_longitude": [77.61, 77.64, 77.59],
                "Delivery_location_latitude": [12.99, 12.97, 12.93],
                "Delivery_location_longitude": [77.63, 77.66, 77.62],
                "Type_of_order": ["Snack", "Meal", "Snack"],
                "Type_of_vehicle": ["bicycle", "motorcycle", "scooter"],
                "Time_taken(min)": ["25 min", "30 min", "35 min"],
                "order_date": ["2024-01-15", "2024-01-16", "2024-01-17"],
                "Festival": ["No", "Yes", "No"],
                "City": ["Bangalore", "Bangalore", "Bangalore"],
                "Road_traffic_density": ["Low", "High", "Medium"],
                "Weather_conditions": ["Sunny", "Stormy", "Cloudy"],
            }
        )
        cleaned = clean_delivery(raw)
        assert "time_taken_min" in cleaned.columns
        assert len(cleaned) >= 2  # Should survive cleaning

        featured = add_delivery_features(cleaned)
        # All expected feature groups
        assert "haversine_distance_km" in featured.columns
        assert "order_hour" in featured.columns
        assert "day_of_week" in featured.columns
        assert "is_rush_hour" in featured.columns
        assert "hour_sin" in featured.columns
        assert "hour_cos" in featured.columns
        assert "distance_traffic_interaction" in featured.columns
        assert "distance_festival_interaction" in featured.columns
        assert "weather_encoded" in featured.columns
        assert "city_zone" in featured.columns


class TestModelTrainingWorkflow:
    """E2E test: features → train → predict on small data."""

    def test_rating_model_trains_on_small_data(self):
        """Should train all rating models on ~200 synthetic rows."""
        from dabba.models.rating_model import train_and_evaluate_rating_models

        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame(
            {
                "votes_log": rng.uniform(1, 8, n),
                "cost_for_two": rng.uniform(100, 2000, n),
                "online_order_binary": rng.choice([0, 1], n),
                "book_table_binary": rng.choice([0, 1], n),
                "cuisine_count": rng.randint(1, 8, n),
                "avg_sentiment": rng.uniform(-1, 1, n),
            }
        )
        y = pd.Series(rng.uniform(2.5, 5.0, n))

        results, best = train_and_evaluate_rating_models(
            X, y, use_mlflow=False, use_hpo=False
        )
        assert len(results) >= 5  # At least 5 models
        assert best is not None
        assert best.mae >= 0
        assert best.r2 >= -1  # Reasonable R² range

    def test_eta_model_trains_with_new_features(self):
        """Should train ETA models with the expanded feature set."""
        from dabba.models.eta_model import train_and_evaluate_eta_models

        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame(
            {
                "haversine_distance_km": rng.uniform(1, 15, n),
                "traffic_ordinal": rng.choice([0, 1, 2, 3], n),
                "is_festival": rng.choice([0, 1], n),
                "order_hour": rng.randint(0, 24, n),
                "day_of_week": rng.randint(0, 7, n),
                "is_weekend": rng.choice([0, 1], n),
                "is_rush_hour": rng.choice([0, 1], n),
                "hour_sin": np.sin(2 * np.pi * rng.rand(n)),
                "hour_cos": np.cos(2 * np.pi * rng.rand(n)),
                "dow_sin": np.sin(2 * np.pi * rng.rand(n) / 7),
                "dow_cos": np.cos(2 * np.pi * rng.rand(n) / 7),
                "weather_encoded": rng.randint(0, 4, n),
                "distance_traffic_interaction": rng.uniform(0, 50, n),
                "distance_festival_interaction": rng.uniform(0, 15, n),
                "delivery_person_age": rng.uniform(20, 45, n),
                "delivery_person_ratings": rng.uniform(3.0, 5.0, n),
                "vehicle_condition": rng.choice([0, 1, 2], n),
            }
        )
        y = pd.Series(rng.uniform(15, 60, n))

        # Add string columns that ColumnTransformer will one-hot encode
        X["city_zone"] = np.random.choice(["central", "north", "south"], n)

        results, best = train_and_evaluate_eta_models(
            X, y, use_mlflow=False, use_hpo=False
        )
        assert len(results) >= 5
        assert best is not None
        assert best.mae > 0


class TestDriftE2E:
    """E2E test: drift detection → alert result."""

    def test_drift_detection_with_non_drifting_data(self):
        """Same distribution should not trigger drift."""
        rng = np.random.RandomState(42)
        ref = pd.DataFrame({"feat_a": rng.normal(0, 1, 500)})
        detector = DriftDetector(ref)

        # Non-drifting batch (same distribution)
        batch = pd.DataFrame({"feat_a": rng.normal(0, 1, 100)})
        result = detector.detect(batch)
        assert result.has_drift is False
        assert result.drifted_count == 0

    def test_drift_alert_with_drifting_data(self):
        """Shifted distribution should trigger drift and cooldown."""
        rng = np.random.RandomState(42)
        ref = pd.DataFrame(
            {
                "feat_a": rng.normal(0, 1, 500),
                "feat_b": rng.uniform(0, 10, 500),
            }
        )
        detector = DriftDetector(ref)

        # Drifting batch (mean shifted by 3 std)
        shifted = pd.DataFrame(
            {
                "feat_a": rng.normal(3, 1, 100),
                "feat_b": rng.uniform(5, 15, 100),
            }
        )
        result = detector.detect_and_alert(shifted)
        # Should detect drift (features shifted)
        assert result.drifted_count > 0 or result.has_drift

    def test_model_comparison_to_dataframe(self):
        """Model results should convert to clean DataFrame."""
        from dabba.models.base_trainer import ModelResult

        results = [
            ModelResult(name="ModelA", mae=0.5, rmse=0.7, r2=0.8, train_time=1.0),
            ModelResult(name="ModelB", mae=0.3, rmse=0.4, r2=0.9, train_time=0.5),
        ]
        from dabba.models.model_selection import (
            comparison_to_dataframe,
            select_best_model,
        )

        df = comparison_to_dataframe(results)
        assert len(df) == 2
        assert list(df.columns) == ["model", "mae", "rmse", "r2", "train_time_s"]
        # Sorted by MAE ascending → ModelB first
        assert df.iloc[0]["model"] == "ModelB"

        best = select_best_model(results)
        assert best == "ModelB"
