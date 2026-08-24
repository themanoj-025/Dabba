"""Tests for ETA model training and evaluation."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from dabba.models.eta_model import ETAModelResult, train_and_evaluate_eta_models


# Lightweight model dict for fast unit tests (avoids training 10+ heavy models)
_LIGHTWEIGHT_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
}


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

    @patch("dabba.models.eta_model.get_eta_models", return_value=_LIGHTWEIGHT_MODELS)
    @patch("dabba.models.eta_model._get_pytorch_nn", return_value=None)
    def test_returns_results(self, _mock_nn, _mock_models):
        """Should return a list of results and a best result."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame(
            {
                "haversine_distance_km": rng.uniform(1, 15, n),
                "traffic_ordinal": rng.choice([0, 1, 2, 3], n),
                "is_festival": rng.choice([0, 1], n),
                "delivery_person_age": rng.uniform(20, 45, n),
                "delivery_person_ratings": rng.uniform(3.0, 5.0, n),
                "vehicle_condition": rng.choice([1, 2, 3], n),
            }
        )
        y = pd.Series(rng.uniform(15, 60, n))

        results, best = train_and_evaluate_eta_models(
            df, y, use_mlflow=False, use_hpo=False
        )

        assert len(results) > 0
        assert best is not None
        assert best.mae >= 0

    @patch("dabba.models.eta_model.get_eta_models", return_value=_LIGHTWEIGHT_MODELS)
    @patch("dabba.models.eta_model._get_pytorch_nn", return_value=None)
    def test_predictions_shape(self, _mock_nn, _mock_models):
        """Predictions should have same length as input."""
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "haversine_distance_km": rng.uniform(1, 15, n),
                "traffic_ordinal": rng.choice([0, 1, 2, 3], n),
            }
        )
        y = pd.Series(rng.uniform(15, 60, n))

        results, _ = train_and_evaluate_eta_models(
            df, y, use_mlflow=False, use_hpo=False
        )

        for result in results:
            if result.predictions is not None:
                assert len(result.predictions) == n

    @patch("dabba.models.eta_model.get_eta_models", return_value=_LIGHTWEIGHT_MODELS)
    @patch("dabba.models.eta_model._get_pytorch_nn", return_value=None)
    def test_mae_positive(self, _mock_nn, _mock_models):
        """MAE should always be non-negative."""
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "haversine_distance_km": rng.uniform(1, 15, n),
                "traffic_ordinal": rng.choice([0, 1, 2, 3], n),
            }
        )
        y = pd.Series(rng.uniform(15, 60, n))

        results, _ = train_and_evaluate_eta_models(
            df, y, use_mlflow=False, use_hpo=False
        )

        for result in results:
            assert result.mae >= 0
            assert result.rmse >= 0
