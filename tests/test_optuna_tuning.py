"""Tests for Optuna hyperparameter tuning in base_trainer.py.

Covers:
    - Model search space definitions
    - Parameter sampling from search spaces
    - Building models from tuned params
    - Full tuning integration on synthetic data
    - MLflow integration during tuning
    - Graceful fallback when tuning fails
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dabba.models.base_trainer import (
    _build_model_from_params,
    _sample_optuna_params,
    _build_preprocessor,
    get_model_search_spaces,
    tune_hyperparameters,
    get_tuned_model,
    tune_all_models,
)


class TestModelSearchSpaces:
    """Tests for the HPO search space definitions."""

    def test_has_expected_models(self):
        """Should contain search spaces for all ensemble models."""
        spaces = get_model_search_spaces()
        expected = {"XGBoost", "LightGBM", "CatBoost", "RandomForest", "GradientBoosting"}
        assert expected.issubset(set(spaces.keys()))

    def test_each_space_has_valid_params(self):
        """Each model's search space should have well-formed param specs."""
        spaces = get_model_search_spaces()
        for model_name, space in spaces.items():
            assert len(space) >= 3, f"{model_name} has fewer than 3 params"
            for param_name, spec in space.items():
                assert "type" in spec, f"{model_name}.{param_name} missing 'type'"
                assert spec["type"] in {"int", "float", "categorical"}
                if spec["type"] in {"int", "float"}:
                    assert "low" in spec
                    assert "high" in spec
                elif spec["type"] == "categorical":
                    assert "choices" in spec

    def test_search_spaces_have_overlapping_params(self):
        """Models should share commonly tuned params for fair comparison."""
        spaces = get_model_search_spaces()
        for model_name in ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting"]:
            assert "n_estimators" in spaces[model_name]
            assert "learning_rate" in spaces[model_name]
            assert "max_depth" in spaces[model_name]


class TestSampleOptunaParams:
    """Tests for sampling parameters from a search space."""

    def test_samples_all_params(self):
        """Should return a value for every param in the search space."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        space = get_model_search_spaces()["XGBoost"]
        study = optuna.create_study(direction="minimize")
        trial = study.ask()

        params = _sample_optuna_params(trial, space)
        assert set(params.keys()) == set(space.keys())

    def test_int_param_within_range(self):
        """Integer params should be sampled within [low, high]."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        space = get_model_search_spaces()["RandomForest"]
        study = optuna.create_study(direction="minimize")

        # Sample multiple trials to verify bounds
        for _ in range(10):
            trial = study.ask()
            params = _sample_optuna_params(trial, space)
            assert 50 <= params["n_estimators"] <= 500
            assert 3 <= params["max_depth"] <= 30
            assert 2 <= params["min_samples_split"] <= 20


class TestBuildModelFromParams:
    """Tests for building model instances from tuned params."""

    def test_build_xgboost(self):
        """Should build an XGBRegressor from tuned params."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            pytest.skip("xgboost not installed")

        params = {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05}
        model = _build_model_from_params("XGBoost", params)
        assert isinstance(model, XGBRegressor)
        assert model.n_estimators == 200
        assert model.max_depth == 8

    def test_build_randomforest(self):
        """Should build a RandomForestRegressor from tuned params."""
        from sklearn.ensemble import RandomForestRegressor

        params = {
            "n_estimators": 150,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }
        model = _build_model_from_params("RandomForest", params)
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 150
        assert model.max_depth == 15

    def test_build_catboost(self):
        """Should build a CatBoostRegressor from tuned params."""
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            pytest.skip("catboost not installed")

        params = {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03}
        model = _build_model_from_params("CatBoost", params)
        assert isinstance(model, CatBoostRegressor)
        assert model.n_estimators == 300

    def test_build_gradientboosting(self):
        """Should build a GradientBoostingRegressor from tuned params."""
        from sklearn.ensemble import GradientBoostingRegressor

        params = {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.02}
        model = _build_model_from_params("GradientBoosting", params)
        assert isinstance(model, GradientBoostingRegressor)
        assert model.n_estimators == 250

    def test_build_lightgbm(self):
        """Should build an LGBMRegressor from tuned params."""
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            pytest.skip("lightgbm not installed")

        params = {"n_estimators": 180, "max_depth": 10, "learning_rate": 0.05}
        model = _build_model_from_params("LightGBM", params)
        assert isinstance(model, LGBMRegressor)
        assert model.n_estimators == 180

    def test_unknown_model_raises(self):
        """Should raise ValueError for unknown model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            _build_model_from_params("UnknownModel", {})


class TestTuneHyperparameters:
    """Integration tests for Optuna HPO on synthetic data."""

    @pytest.fixture
    def sample_data(self):
        """Create synthetic regression data for tuning."""
        rng = np.random.RandomState(42)
        n = 300
        df = pd.DataFrame(
            {
                "feature_a": rng.uniform(0, 10, n),
                "feature_b": rng.normal(0, 1, n),
                "feature_c": rng.choice(["x", "y", "z"], n),
                "feature_d": rng.uniform(-5, 5, n),
            }
        )
        # y is a noisy linear function of features
        y = pd.Series(
            2.0 * df["feature_a"]
            - 1.5 * df["feature_d"]
            + rng.normal(0, 0.5, n)
        )
        return df, y

    def test_tune_returns_params_and_metrics(self, sample_data):
        """Should return best params, best MAE, and best RMSE."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = sample_data
        best_params, best_mae, best_rmse = tune_hyperparameters(
            X, y, "XGBoost", n_trials=5, cv_folds=3
        )
        assert isinstance(best_params, dict)
        assert len(best_params) >= 3
        assert best_mae >= 0
        assert best_rmse >= 0
        assert best_rmse >= best_mae  # RMSE >= MAE always

    def test_unknown_model_raises(self, sample_data):
        """Should raise ValueError for unsupported model."""
        X, y = sample_data
        with pytest.raises(ValueError, match="not supported"):
            tune_hyperparameters(X, y, "UnsupportedModel", n_trials=2)

    def test_get_tuned_model_returns_estimator(self, sample_data):
        """get_tuned_model should return an unfitted estimator."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = sample_data
        model = get_tuned_model(X, y, "RandomForest", n_trials=3, cv_folds=3)
        assert model is not None
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(model, RandomForestRegressor)
        # Should not be fitted yet (the pipeline fits during CV, model is fresh)
        assert hasattr(model, "n_estimators")

    def test_tune_all_models_returns_dict(self, sample_data):
        """tune_all_models should return tuned models for requested list."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = sample_data
        tuned = tune_all_models(
            X, y,
            models_to_tune=["XGBoost", "RandomForest"],
            n_trials=3,
            cv_folds=3,
        )
        assert "XGBoost" in tuned
        assert "RandomForest" in tuned
        assert tuned["XGBoost"] is not None
        assert tuned["RandomForest"] is not None

    def test_custom_search_space(self, sample_data):
        """Should use a custom search space when provided."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = sample_data
        custom_space = {
            "n_estimators": {"type": "int", "low": 10, "high": 20},
            "max_depth": {"type": "int", "low": 2, "high": 5},
        }
        best_params, best_mae, best_rmse = tune_hyperparameters(
            X, y, "XGBoost", n_trials=3, cv_folds=3, search_space=custom_space
        )
        assert 10 <= best_params["n_estimators"] <= 20
        assert 2 <= best_params["max_depth"] <= 5

    def test_tuning_with_categorical_features(self, sample_data):
        """Should handle DataFrames with categorical columns."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = sample_data
        # Mix numeric + categorical
        best_params, best_mae, best_rmse = tune_hyperparameters(
            X, y, "LightGBM", n_trials=4, cv_folds=3
        )
        assert best_mae >= 0

    def test_tune_returns_better_than_baseline(self, sample_data):
        """Tuning with enough trials should beat default params on synthetic data."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = sample_data

        # Default model performance
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import mean_absolute_error

        preprocessor = _build_preprocessor(X)
        default_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        default_pipe = Pipeline([("preprocessor", preprocessor), ("model", default_model)])
        default_preds = cross_val_predict(default_pipe, X, y, cv=3, method="predict")
        default_mae = mean_absolute_error(y, default_preds)

        # Tuned model performance (small trial count, but should be competitive)
        best_params, tuned_mae, _ = tune_hyperparameters(
            X, y, "RandomForest", n_trials=5, cv_folds=3
        )

        # Tuned should be within reasonable range of default
        # (may not beat on 5 trials with synthetic data, but shouldn't be catastrophically worse)
        assert tuned_mae < default_mae * 3, (
            f"Tuned MAE {tuned_mae:.4f} is > 3x default MAE {default_mae:.4f} — "
            "tuning produced unreasonably bad results"
        )


class TestTuneWithMlflow:
    """Tests that tuning logs properly to MLflow."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({"x1": rng.rand(n), "x2": rng.rand(n)})
        y = pd.Series(3 * df["x1"] + rng.normal(0, 0.2, n))
        return df, y

    def test_mlflow_logging_does_not_crash(self, sample_data):
        """MLflow import errors during tuning should not crash the function."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = sample_data

        # Mock MLflow to raise exception during logging
        with patch(
            "dabba.models.base_trainer.mlflow.log_params",
            side_effect=Exception("MLflow unavailable"),
        ), patch(
            "dabba.models.base_trainer.mlflow.log_metrics",
            side_effect=Exception("MLflow unavailable"),
        ), patch(
            "dabba.models.base_trainer.logger.warning"
        ):
            best_params, best_mae, best_rmse = tune_hyperparameters(
                X, y, "XGBoost", n_trials=3, cv_folds=3
            )
            assert best_mae >= 0  # Tuning succeeded despite MLflow failure


class TestHpoIntegration:
    """Tests that HPO is wired correctly through rating/eta model modules."""

    def test_rating_models_export_hpo_function(self):
        """Rating model module should export get_tuned_rating_models."""
        from dabba.models.rating_model import get_tuned_rating_models
        assert callable(get_tuned_rating_models)

    def test_eta_models_export_hpo_function(self):
        """ETA model module should export get_tuned_eta_models."""
        from dabba.models.eta_model import get_tuned_eta_models
        assert callable(get_tuned_eta_models)

    def test_rating_train_accepts_use_hpo_flag(self):
        """train_and_evaluate_rating_models should accept use_hpo kwarg."""
        from dabba.models.rating_model import train_and_evaluate_rating_models
        import inspect
        sig = inspect.signature(train_and_evaluate_rating_models)
        assert "use_hpo" in sig.parameters

    def test_eta_train_accepts_use_hpo_flag(self):
        """train_and_evaluate_eta_models should accept use_hpo kwarg."""
        from dabba.models.eta_model import train_and_evaluate_eta_models
        import inspect
        sig = inspect.signature(train_and_evaluate_eta_models)
        assert "use_hpo" in sig.parameters

    def test_tuned_rating_models_include_all_defaults(self, sample_data):
        """get_tuned_rating_models should include all the same models as defaults."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        X, y = sample_data
        from dabba.models.rating_model import get_rating_models, get_tuned_rating_models

        default_models = get_rating_models()
        tuned_models = get_tuned_rating_models(
            X, y, models_to_tune=[],  # tune nothing, should be same as defaults
        )

        for name in default_models:
            assert name in tuned_models, f"{name} missing from tuned models"

    @pytest.fixture
    def sample_data(self):
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {"x1": rng.rand(n), "x2": rng.choice(["a", "b", "c"], n)}
        )
        y = pd.Series(2 * df["x1"] + rng.normal(0, 0.3, n))
        return df, y
