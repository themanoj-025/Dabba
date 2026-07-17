"""Tests for rating model training, comparison, and full-data fitting."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

import joblib

from dabba.models.rating_model import (
    ModelResult,
    fit_best_rating_model,
    get_rating_models,
    save_model,
    train_and_evaluate_rating_models,
)


class TestModelResult:
    """Tests for the ModelResult dataclass."""

    def test_creation(self):
        """Should create a ModelResult with all fields."""
        result = ModelResult(
            name="TestModel", mae=5.0, rmse=7.0, r2=0.8, train_time=1.5
        )
        assert result.name == "TestModel"
        assert result.mae == 5.0
        assert result.predictions is None

    def test_with_predictions(self):
        """Should store predictions when provided."""
        preds = np.array([1.0, 2.0, 3.0])
        result = ModelResult(
            name="Test", mae=0.5, rmse=0.7, r2=0.9, train_time=0.1, predictions=preds
        )
        assert result.predictions is not None
        assert len(result.predictions) == 3


class TestGetRatingModels:
    """Tests for candidate model dictionary."""

    def test_returns_dict(self):
        """Should return a dictionary of models."""
        models = get_rating_models()
        assert isinstance(models, dict)
        assert len(models) >= 5  # At least 5 models even without xgb/lgbm

    def test_has_required_models(self):
        """Should include the required base models."""
        models = get_rating_models()
        assert "LinearRegression" in models
        assert "Ridge" in models
        assert "Lasso" in models
        assert "DecisionTree" in models
        assert "RandomForest" in models
        assert "GradientBoosting" in models


class TestTrainAndEvaluateRatingModels:
    """Tests for the full model comparison pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create synthetic rating prediction data."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "votes_log": rng.uniform(1, 8, n),
            "cost_for_two": rng.uniform(100, 2000, n),
            "online_order_binary": rng.choice([0, 1], n),
            "book_table_binary": rng.choice([0, 1], n),
            "cuisine_count": rng.randint(1, 8, n),
            "avg_sentiment": rng.uniform(-1, 1, n),
        })
        y = pd.Series(rng.uniform(2.5, 5.0, n))
        return df, y

    def test_returns_results_and_best(self, sample_data):
        """Should return a list of results and a best result."""
        X, y = sample_data
        results, best = train_and_evaluate_rating_models(X, y)
        assert len(results) > 0
        assert best is not None
        assert best.mae >= 0

    def test_predictions_length_matches_input(self, sample_data):
        """Predictions should have same length as input."""
        X, y = sample_data
        results, _ = train_and_evaluate_rating_models(X, y)
        for result in results:
            if result.predictions is not None:
                assert len(result.predictions) == len(y)

    def test_mae_positive(self, sample_data):
        """MAE should always be non-negative."""
        X, y = sample_data
        results, _ = train_and_evaluate_rating_models(X, y)
        for result in results:
            assert result.mae >= 0
            assert result.rmse >= 0


class TestFitBestRatingModel:
    """Tests for fitting the winning model on full data."""

    @pytest.fixture
    def sample_data(self):
        """Create synthetic data."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "votes_log": rng.uniform(1, 8, n),
            "cost_for_two": rng.uniform(100, 2000, n),
            "online_order_binary": rng.choice([0, 1], n),
        })
        y = pd.Series(rng.uniform(2.5, 5.0, n))
        return df, y

    def test_fits_and_saves(self, sample_data, tmp_path):
        """Should fit on full data and save to disk."""
        X, y = sample_data
        save_path = tmp_path / "best_rating_model.pkl"

        fitted = fit_best_rating_model("LinearRegression", X, y, save_path)

        assert fitted is not None
        assert save_path.exists()

        # Verify the saved model can be loaded and makes predictions
        loaded = joblib.load(save_path)
        preds = loaded.predict(X)
        assert len(preds) == len(y)

    def test_invalid_model_name_raises(self, sample_data, tmp_path):
        """Should raise ValueError for unknown model name."""
        X, y = sample_data
        save_path = tmp_path / "model.pkl"

        with pytest.raises(ValueError, match="not found"):
            fit_best_rating_model("NonexistentModel", X, y, save_path)

    def test_saves_to_parent_dirs(self, sample_data, tmp_path):
        """Should create parent directories if they don't exist."""
        X, y = sample_data
        save_path = tmp_path / "nested" / "dir" / "model.pkl"

        fit_best_rating_model("LinearRegression", X, y, save_path)
        assert save_path.exists()


class TestSaveModel:
    """Tests for model persistence."""

    def test_save_and_load(self, tmp_path):
        """Should save and load a model successfully."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        X = np.random.rand(100, 3)
        y = np.random.rand(100)
        model.fit(X, y)

        path = tmp_path / "model.pkl"
        save_model(model, path)
        assert path.exists()

        loaded = joblib.load(path)
        preds = loaded.predict(X)
        assert len(preds) == 100
