"""ETA prediction model — trains and compares multiple algorithms.

Trains all candidate models on identical features with k-fold CV,
selects the best by lowest MAE, and saves it as the winning model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from pathlib import Path

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class ETAModelResult:
    """Stores evaluation results for a single ETA model."""

    name: str
    mae: float
    rmse: float
    r2: float
    train_time: float
    predictions: Optional[np.ndarray] = field(default=None, repr=False)


def get_eta_models() -> Dict[str, Any]:
    """Return a dictionary of candidate ETA prediction models.

    Returns:
        Dict mapping model name to scikit-learn regressor instance.
    """
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=10),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": _get_xgboost(),
        "LightGBM": _get_lightgbm(),
    }


def _get_xgboost() -> Any:
    """Safely import and return XGBoost regressor."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        logger.warning("XGBoost not installed — skipping")
        return None


def _get_lightgbm() -> Any:
    """Safely import and return LightGBM regressor."""
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
    except ImportError:
        logger.warning("LightGBM not installed — skipping")
        return None


def train_and_evaluate_eta_models(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[DabbaConfig] = None,
) -> Tuple[List[ETAModelResult], Optional[ETAModelResult]]:
    """Train all candidate ETA models with k-fold CV and return comparison.

    Args:
        X: Feature matrix.
        y: Target vector (delivery time in minutes).
        config: Project configuration.

    Returns:
        Tuple of (list of all ETAModelResult, best ETAModelResult or None).
    """
    config = config or get_config()
    models = get_eta_models()
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)

    # Identify categorical and numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    results: List[ETAModelResult] = []

    for name, model in models.items():
        if model is None:
            continue

        logger.info("Training ETA model: %s...", name)
        start = time.time()

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        try:
            y_pred = cross_val_predict(pipe, X, y, cv=kf, method="predict")
        except Exception as e:
            logger.error("Failed to train %s: %s", name, e)
            continue

        elapsed = time.time() - start

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        result = ETAModelResult(
            name=name, mae=mae, rmse=rmse, r2=r2,
            train_time=elapsed, predictions=y_pred,
        )
        results.append(result)
        logger.info(
            "ETA %s — MAE: %.4f min, RMSE: %.4f, R²: %.4f, Time: %.1fs",
            name, mae, rmse, r2, elapsed,
        )

    # Select best by configured metric
    if not results:
        return results, None

    metric = config.eta_metric
    key_fn = {"mae": lambda r: r.mae, "rmse": lambda r: r.rmse, "r2": lambda r: -r.r2}
    best = min(results, key=key_fn.get(metric, lambda r: r.mae))
    logger.info("Best ETA model: %s (%s=%.4f)", best.name, metric.upper(),
                getattr(best, metric))

    return results, best


def save_eta_model(model: Any, path: Any) -> None:
    """Save a trained ETA model to disk using joblib.

    Args:
        model: Trained scikit-learn model or pipeline.
        path: Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved ETA model to %s", path)


def fit_best_eta_model(
    best_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    save_path: Any,
    config: Optional[DabbaConfig] = None,
) -> Any:
    """Retrain the winning ETA model on the full dataset and save to disk.

    After cross-validation comparison, this function creates a fresh instance
    of the winning model, fits it on ALL available data, and persists it.

    Args:
        best_name: Name of the winning model (key from get_eta_models()).
        X: Full feature matrix.
        y: Full target vector.
        save_path: Path to save the fitted model artifact.
        config: Project configuration.

    Returns:
        The fitted Pipeline object.
    """
    config = config or get_config()
    models = get_eta_models()

    if best_name not in models or models[best_name] is None:
        raise ValueError(f"Model '{best_name}' not found in candidate models")

    # Identify categorical and numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[best_name]),
    ])

    logger.info("Fitting best ETA model '%s' on full data (%d samples)...", best_name, len(X))
    pipe.fit(X, y)

    save_eta_model(pipe, save_path)
    logger.info("Best ETA model '%s' fitted and saved to %s", best_name, save_path)
    return pipe
