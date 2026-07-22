"""Shared base trainer for Dabba model comparison pipelines.

Extracts the ~150 lines of duplicated code between rating_model.py
and eta_model.py into a single, tested, reusable module.

Responsibilities:
    - Shared model factory helpers (_get_xgboost, _get_lightgbm, _get_catboost)
    - Reusable ModelResult dataclass
    - Generic train_and_evaluate_models() with k-fold CV, MLflow tracking
    - Generic fit_best_model() for full-data retraining + persistence
    - Generic save_model() for joblib persistence
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


# ─── Shared result type ──────────────────────────────────────────────


@dataclass
class ModelResult:
    """Stores evaluation results for a single model.

    Used by both rating and ETA pipelines. The ``predictions`` field
    is excluded from repr to keep logs readable.
    """

    name: str
    mae: float
    rmse: float
    r2: float
    train_time: float
    predictions: Optional[np.ndarray] = field(default=None, repr=False)
    mlflow_run_id: Optional[str] = field(default=None, repr=False)


# ─── Shared model factories (graceful import-error handling) ─────────


def _get_xgboost() -> Any:
    """Return an XGBRegressor or None if xgboost is not installed."""
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        logger.warning("XGBoost not installed — skipping")
        return None


def _get_lightgbm() -> Any:
    """Return an LGBMRegressor or None if lightgbm is not installed."""
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
    except ImportError:
        logger.warning("LightGBM not installed — skipping")
        return None


def _get_catboost() -> Any:
    """Return a CatBoostRegressor or None if catboost is not installed."""
    try:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            n_estimators=100,
            random_state=42,
            verbose=0,
            allow_writing_files=False,
        )
    except ImportError:
        logger.warning("CatBoost not installed — skipping")
        return None


# ─── Core CV training loop ───────────────────────────────────────────


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a standard ColumnTransformer for numeric + categorical features.

    Args:
        X: Feature matrix.

    Returns:
        ColumnTransformer with StandardScaler for numerics and
        OneHotEncoder for categoricals.
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )


def _setup_mlflow(
    config: DabbaConfig,
    task: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> Any:
    """Start an MLflow parent run for a model comparison experiment.

    Args:
        config: Project configuration.
        task: Task name used for experiment suffix ('rating' or 'eta').
        X: Feature matrix (for logging shape).
        y: Target vector (for logging length).

    Returns:
        The MLflow parent run object, or None if MLflow is unavailable.
    """
    try:
        import os

        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"
        import mlflow

        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name + f"_{task}")
        mlflow_run = mlflow.start_run(run_name=f"{task}_comparison")
        mlflow.log_params(
            {
                "task": task,
                "cv_folds": config.cv_folds,
                "test_size": config.test_size,
                "random_seed": config.random_seed,
                "n_features": X.shape[1],
                "n_samples": len(X),
            }
        )
        return mlflow_run
    except Exception as e:
        logger.warning("MLflow tracking disabled for %s: %s", task, e)
        return None


def _log_model_to_mlflow(
    name: str,
    model: Any,
    mae: float,
    rmse: float,
    r2: float,
    elapsed: float,
    parent_run: Any,
    additional_params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Log a single model's metrics to MLflow as a nested run.

    Args:
        name: Model name.
        model: The trained model (for logging type).
        mae: Mean absolute error.
        rmse: Root mean squared error.
        r2: R² score.
        elapsed: Training time in seconds.
        parent_run: The parent MLflow run (or any truthy value).
        additional_params: Extra params to log alongside the standard set.

    Returns:
        The nested run ID, or None if logging failed.
    """
    try:
        import mlflow

        params: Dict[str, Any] = {"model": name}
        if additional_params:
            params.update(additional_params)

        with mlflow.start_run(nested=True, run_name=name) as child_run:
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "train_time_s": elapsed,
                }
            )
            return child_run.info.run_id
    except Exception as e:
        logger.warning("MLflow logging failed for %s: %s", name, e)
        return None


def _end_mlflow_run(mlflow_run: Any) -> None:
    """End an MLflow run, ignoring errors."""
    if mlflow_run:
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Any],
    config: Optional[DabbaConfig] = None,
    use_mlflow: bool = True,
    task: str = "rating",
) -> Tuple[List[ModelResult], Optional[ModelResult]]:
    """Train all candidate models with k-fold CV and return comparison.

    This is the core generic training pipeline used by both the rating
    and ETA tasks. It:

        1. Builds a preprocessor (StandardScaler + OneHotEncoder)
        2. Runs k-fold cross-validation predictions for each model
        3. Logs results to MLflow (if enabled)
        4. Selects the best model by the configured task metric

    Args:
        X: Feature matrix.
        y: Target vector.
        models: Dict of ``{name: estimator}`` candidate models.
        config: Project configuration. If None, a fresh one is created.
        use_mlflow: Whether to log runs to MLflow.
        task: Task name used for experiment naming and metric selection
            ('rating' or 'eta').

    Returns:
        Tuple of (list of all ModelResult, best ModelResult or None).
    """
    config = config or get_config()
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)
    preprocessor = _build_preprocessor(X)

    # Pick the right metric from config based on task
    metric_name = config.rating_metric if task == "rating" else config.eta_metric
    key_fn = {"mae": lambda r: r.mae, "rmse": lambda r: r.rmse, "r2": lambda r: -r.r2}
    sort_key = key_fn.get(metric_name, lambda r: r.mae)

    # MLflow setup
    mlflow_run = _setup_mlflow(config, task, X, y) if use_mlflow else None

    results: List[ModelResult] = []

    for name, model in models.items():
        if model is None:
            continue

        logger.info("Training %s model: %s...", task, name)
        start = time.time()

        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

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
            run_id = _log_model_to_mlflow(
                name,
                model,
                mae,
                rmse,
                r2,
                elapsed,
                mlflow_run,
                additional_params={"model_type": type(model).__name__},
            )

        result = ModelResult(
            name=name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            train_time=elapsed,
            predictions=y_pred,
            mlflow_run_id=run_id,
        )
        results.append(result)
        logger.info(
            "%s %s — MAE: %.4f, RMSE: %.4f, R²: %.4f, Time: %.1fs",
            task.title(),
            name,
            mae,
            rmse,
            r2,
            elapsed,
        )

    if not results:
        _end_mlflow_run(mlflow_run)
        return results, None

    best = min(results, key=sort_key)
    logger.info(
        "Best %s model: %s (%s=%.4f)",
        task,
        best.name,
        metric_name.upper(),
        getattr(best, metric_name),
    )

    # Tag winning run in MLflow
    if mlflow_run and best.mlflow_run_id:
        try:
            import mlflow

            mlflow.set_tag("winning_model", best.name)
            mlflow.log_metrics({f"best_{metric_name}": getattr(best, metric_name)})
        except Exception:
            pass

    _end_mlflow_run(mlflow_run)

    return results, best


# ─── Persistence ──────────────────────────────────────────────────────


def save_model(model: Any, path: Any) -> None:
    """Save a trained model to disk using joblib.

    Parent directories are created automatically.

    Args:
        model: The fitted scikit-learn Pipeline or estimator.
        path: File path (string or PathLike).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def fit_best_model(
    best_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Any],
    save_path: Any,
    task: str = "model",
    config: Optional[DabbaConfig] = None,
) -> Any:
    """Retrain the winning model on full data and save to disk.

    Args:
        best_name: Name of the winning model (key into ``models``).
        X: Feature matrix.
        y: Target vector.
        models: Dict of candidate models (from ``get_*_models()``).
        save_path: Path to save the fitted Pipeline.
        task: Human-readable task name for log messages.
        config: Project configuration.

    Returns:
        The fitted Pipeline.

    Raises:
        ValueError: If ``best_name`` is not found in ``models``.
    """
    config = config or get_config()

    if best_name not in models or models[best_name] is None:
        raise ValueError(
            f"Model '{best_name}' not found in candidate models for task '{task}'"
        )

    preprocessor = _build_preprocessor(X)

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", models[best_name]),
        ]
    )

    logger.info(
        "Fitting best '%s' model '%s' on full data (%d samples)...",
        task,
        best_name,
        len(X),
    )
    pipe.fit(X, y)
    save_model(pipe, save_path)
    return pipe
