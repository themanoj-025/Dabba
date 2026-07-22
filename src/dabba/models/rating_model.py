"""Rating prediction model — trains and compares multiple algorithms.

Uses the shared :mod:`dabba.models.base_trainer` for the CV loop,
MLflow tracking, and persistence. This module only defines the
model dictionary and thin wrappers.

Trains all candidate models on identical features with k-fold CV,
selects the best by lowest MAE, and saves it as the winning model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from dabba.config import DabbaConfig
from dabba.models.base_trainer import (
    ModelResult,
    _get_catboost,
    _get_lightgbm,
    _get_xgboost,
    fit_best_model as _fit_best_model,
    save_model as _save_model,
    train_and_evaluate_models as _train_and_evaluate_models,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "ModelResult",
    "get_rating_models",
    "train_and_evaluate_rating_models",
    "fit_best_rating_model",
    "save_model",
]


def get_rating_models() -> Dict[str, Any]:
    """Return candidate rating prediction models including CatBoost.

    Returns:
        Dict mapping model name to scikit-learn regressor instance.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "XGBoost": _get_xgboost(),
        "LightGBM": _get_lightgbm(),
        "CatBoost": _get_catboost(),
    }
    return {k: v for k, v in models.items() if v is not None}


def train_and_evaluate_rating_models(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[DabbaConfig] = None,
    use_mlflow: bool = True,
) -> Tuple[list, Optional[ModelResult]]:
    """Train all candidate rating models with k-fold CV.

    Delegates to :func:`base_trainer.train_and_evaluate_models`
    with the rating-specific model dictionary and task name.

    Args:
        X: Feature matrix.
        y: Target vector (restaurant ratings).
        config: Project configuration.
        use_mlflow: Whether to log runs to MLflow.

    Returns:
        Tuple of (list of ModelResult, best ModelResult or None).
    """
    models = get_rating_models()
    return _train_and_evaluate_models(
        X, y, models, config=config, use_mlflow=use_mlflow, task="rating"
    )


def save_model(model: Any, path: Any) -> None:
    """Save a trained model to disk using joblib.

    Args:
        model: The fitted Pipeline or estimator.
        path: File path.
    """
    _save_model(model, path)


def fit_best_rating_model(
    best_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    save_path: Any,
    config: Optional[DabbaConfig] = None,
) -> Any:
    """Retrain the winning rating model on full data and save.

    Args:
        best_name: Name of the winning model.
        X: Feature matrix.
        y: Target vector.
        save_path: Path to save the fitted Pipeline.
        config: Project configuration.

    Returns:
        The fitted Pipeline.

    Raises:
        ValueError: If ``best_name`` is not a valid candidate.
    """
    models = get_rating_models()
    return _fit_best_model(
        best_name, X, y, models, save_path, task="rating", config=config
    )
