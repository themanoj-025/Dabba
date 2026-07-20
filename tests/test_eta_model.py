"""Tests for ETA model training and evaluation."""

import numpy as np
import pandas as pd

from dabba.models.eta_model import ETAModelResult, train_and_evaluate_eta_models


class TestETAModelResult:
    """Tests for the ETAModelResult dataclass."""

    def test_creation(self):
        """Should create an ETAModelResult with all fields."""
        result = ETAModelResult(
            name="TestModel", mae=5.0, rmse=7.0, r2=0.8, train_time=1.5
        )
        assert result.name == "TestModel"
        assert result.mae == 5.0
        assert result.predictions is None


class TestETAModels:
    """Tests for the ETA model comparison pipeline."""

    def test_returns_results(self):
        """Should return a list of results and a best result."""
        # Create synthetic data
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "haversine_distance_km": rng.uniform(1, 15, n),
            "traffic_ordinal": rng.choice([0, 1, 2, 3], n),
            "is_festival": rng.choice([0, 1], n),
            "delivery_person_age": rng.uniform(20, 45, n),
            "delivery_person_ratings": rng.uniform(3.0, 5.0, n),
            "vehicle_condition": rng.choice([1, 2, 3], n),
        })
        y = pd.Series(rng.uniform(15, 60, n))

        results, best = train_and_evaluate_eta_models(df, y)

        assert len(results) > 0
        assert best is not None
        assert best.mae >= 0

    def test_predictions_shape(self):
        """Predictions should have same length as input."""
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({
            "haversine_distance_km": rng.uniform(1, 15, n),
            "traffic_ordinal": rng.choice([0, 1, 2, 3], n),
        })
        y = pd.Series(rng.uniform(15, 60, n))

        results, _ = train_and_evaluate_eta_models(df, y)

        for result in results:
            if result.predictions is not None:
                assert len(result.predictions) == n

    def test_mae_positive(self):
        """MAE should always be non-negative."""
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({
            "haversine_distance_km": rng.uniform(1, 15, n),
            "traffic_ordinal": rng.choice([0, 1, 2, 3], n),
        })
        y = pd.Series(rng.uniform(15, 60, n))

        results, _ = train_and_evaluate_eta_models(df, y)

        for result in results:
            assert result.mae >= 0
            assert result.rmse >= 0
