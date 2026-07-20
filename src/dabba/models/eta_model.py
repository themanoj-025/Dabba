"""ETA prediction model — trains and compares multiple algorithms
including CatBoost and an optional small neural network. Uses MLflow.

Trains all candidate models on identical features with k-fold CV,
selects the best by lowest MAE, and saves it as the winning model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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
    mlflow_run_id: Optional[str] = field(default=None, repr=False)


def get_eta_models() -> Dict[str, Any]:
    """Return candidate ETA prediction models including CatBoost.

    Returns:
        Dict mapping model name to regressor instance.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=10),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": _get_xgboost(),
        "LightGBM": _get_lightgbm(),
        "CatBoost": _get_catboost(),
    }
    return {k: v for k, v in models.items() if v is not None}


def _get_xgboost() -> Any:
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        logger.warning("XGBoost not installed — skipping")
        return None


def _get_lightgbm() -> Any:
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
    except ImportError:
        logger.warning("LightGBM not installed — skipping")
        return None


def _get_catboost() -> Any:
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            n_estimators=100, random_state=42, verbose=0, allow_writing_files=False,
        )
    except ImportError:
        logger.warning("CatBoost not installed — skipping")
        return None


def _get_pytorch_nn(input_dim: int) -> Any:
    """Create a small feed-forward neural network for tabular data.

    Uses PyTorch with a simple MLP architecture.
    Wrapped in a scikit-learn compatible interface via skorch if available,
    otherwise returns None.
    """
    try:
        import torch
        import torch.nn as nn

        class TabularNN(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x).squeeze(-1)

        # Try skorch for sklearn-compatible wrapper
        from skorch import NeuralNetRegressor
        return NeuralNetRegressor(
            module=lambda: TabularNN(input_dim),
            max_epochs=30,
            batch_size=64,
            lr=0.01,
            device="cpu",
            verbose=0,
        )
    except ImportError:
        logger.info("skorch not installed — skipping neural network ETA model")
        return None


def train_and_evaluate_eta_models(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[DabbaConfig] = None,
    use_mlflow: bool = True,
) -> Tuple[List[ETAModelResult], Optional[ETAModelResult]]:
    """Train all candidate ETA models with k-fold CV and return comparison.

    Args:
        X: Feature matrix.
        y: Target vector (delivery time in minutes).
        config: Project configuration.
        use_mlflow: Whether to log runs to MLflow.

    Returns:
        Tuple of (list of ETAModelResult, best ETAModelResult or None).
    """
    config = config or get_config()
    models = get_eta_models()

    # Add neural network as optional comparison
    nn_model = _get_pytorch_nn(X.shape[1])
    if nn_model is not None:
        models["NeuralNet_MLP"] = nn_model

    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    # MLflow setup (with short timeout to avoid 4min delay if server is down)
    mlflow_run = None
    if use_mlflow:
        try:
            import os
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"
            import mlflow
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment_name + "_eta")
            mlflow_run = mlflow.start_run(run_name="eta_comparison")
            mlflow.log_params({
                "task": "eta",
                "cv_folds": config.cv_folds,
                "test_size": config.test_size,
                "random_seed": config.random_seed,
                "n_features": X.shape[1],
                "n_samples": len(X),
            })
        except Exception as e:
            logger.warning("MLflow tracking disabled for ETA: %s", e)
            mlflow_run = None

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

        # Log to MLflow
        run_id = None
        if mlflow_run:
            try:
                import mlflow
                with mlflow.start_run(nested=True, run_name=name) as child_run:
                    mlflow.log_params({"model": name})
                    mlflow.log_metrics({
                        "mae": mae, "rmse": rmse, "r2": r2, "train_time_s": elapsed,
                    })
                    run_id = child_run.info.run_id
            except Exception as e:
                logger.warning("MLflow logging failed for %s: %s", name, e)

        result = ETAModelResult(
            name=name, mae=mae, rmse=rmse, r2=r2,
            train_time=elapsed, predictions=y_pred, mlflow_run_id=run_id,
        )
        results.append(result)
        logger.info("ETA %s — MAE: %.4f min, RMSE: %.4f, R²: %.4f, Time: %.1fs",
                    name, mae, rmse, r2, elapsed)

    if not results:
        if mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
        return results, None

    metric = config.eta_metric
    key_fn = {"mae": lambda r: r.mae, "rmse": lambda r: r.rmse, "r2": lambda r: -r.r2}
    best = min(results, key=key_fn.get(metric, lambda r: r.mae))
    logger.info("Best ETA model: %s (%s=%.4f)", best.name, metric.upper(),
                getattr(best, metric))

    if mlflow_run and best.mlflow_run_id:
        try:
            import mlflow
            mlflow.set_tag("winning_model", best.name)
            mlflow.log_metrics({f"best_{metric}": getattr(best, metric)})
        except Exception:
            pass

    if mlflow_run:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    return results, best


def save_eta_model(model: Any, path: Any) -> None:
    """Save a trained ETA model to disk using joblib."""
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
    """Retrain the winning ETA model on full data and save."""
    config = config or get_config()
    models = get_eta_models()

    if best_name not in models or models[best_name] is None:
        raise ValueError(f"Model '{best_name}' not found in candidate models")

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

    logger.info("Fitting best ETA model '%s' on full data (%d samples)...",
                best_name, len(X))
    pipe.fit(X, y)
    save_eta_model(pipe, save_path)
    return pipe
