"""Shared base trainer for Dabba model comparison pipelines.

Extracts the ~150 lines of duplicated code between rating_model.py
and eta_model.py into a single, tested, reusable module.

Responsibilities:
    - Shared model factory helpers (_get_xgboost, _get_lightgbm, _get_catboost)
    - Reusable ModelResult dataclass
    - Generic train_and_evaluate_models() with k-fold CV, MLflow tracking
    - Generic fit_best_model() for full-data retraining + persistence
    - Generic save_model() for joblib persistence
    - :func:`tune_hyperparameters` for Optuna-based HPO on ensemble models

HPO Search Spaces (default ranges, all tunable via :func:`get_model_search_spaces`):

    XGBoost:
        - n_estimators: int [50, 500]
        - max_depth: int [3, 12]
        - learning_rate: float [1e-3, 0.3] (log)
        - subsample: float [0.6, 1.0]
        - colsample_bytree: float [0.6, 1.0]
        - min_child_weight: int [1, 10]
        - gamma: float [0.0, 5.0]
        - reg_lambda: float [1e-3, 10.0] (log)
        - reg_alpha: float [1e-3, 10.0] (log)

    LightGBM:
        - n_estimators: int [50, 500]
        - max_depth: int [3, 12]
        - learning_rate: float [1e-3, 0.3] (log)
        - num_leaves: int [15, 127]
        - subsample: float [0.6, 1.0]
        - colsample_bytree: float [0.6, 1.0]
        - min_child_samples: int [5, 50]
        - reg_lambda: float [1e-3, 10.0] (log)
        - reg_alpha: float [1e-3, 10.0] (log)

    CatBoost:
        - n_estimators: int [50, 500]
        - max_depth: int [3, 10]
        - learning_rate: float [1e-3, 0.3] (log)
        - l2_leaf_reg: float [1, 10] (log)
        - bagging_temperature: float [0.0, 2.0]
        - random_strength: float [0.0, 2.0]
        - border_count: int [32, 255]

    RandomForest:
        - n_estimators: int [50, 500]
        - max_depth: int [3, 30]
        - min_samples_split: int [2, 20]
        - min_samples_leaf: int [1, 20]
        - max_features: categorical ['sqrt', 'log2', None]

    GradientBoosting:
        - n_estimators: int [50, 500]
        - max_depth: int [3, 12]
        - learning_rate: float [1e-3, 0.3] (log)
        - subsample: float [0.6, 1.0]
        - min_samples_split: int [2, 20]
        - min_samples_leaf: int [1, 20]
        - max_features: categorical ['sqrt', 'log2', None]
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


# ─── Model search spaces for Optuna HPO ──────────────────────────────


def get_model_search_spaces() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return the Optuna search space definitions for each tunable model.

    Each model's entry is a dict of ``{param_name: suggest_kwargs}`` where
    ``suggest_kwargs`` contains the type (one of "int", "float", "categorical")
    and the bounds/choices. Example::

        {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        }

    Returns:
        Dict mapping model name to its parameter search space.
    """
    return {
        "XGBoost": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 10},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
            "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        },
        "LightGBM": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 10},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 127},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "min_child_samples": {"type": "int", "low": 5, "high": 50},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
            "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        },
        "CatBoost": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 10},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "log": True},
            "bagging_temperature": {"type": "float", "low": 0.0, "high": 2.0},
            "random_strength": {"type": "float", "low": 0.0, "high": 2.0},
            "border_count": {"type": "int", "low": 32, "high": 255},
        },
        "RandomForest": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 10},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", None],
            },
        },
        "GradientBoosting": {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 10},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", None],
            },
        },
    }


def _sample_optuna_params(trial: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a set of hyperparameters from a search space using an Optuna trial.

    Args:
        trial: Optuna trial object.
        search_space: Search space dict (from :func:`get_model_search_spaces`).

    Returns:
        Dict of ``{param_name: sampled_value}``.
    """
    params = {}
    for param_name, spec in search_space.items():
        suggest_type = spec["type"]
        if suggest_type == "int":
            params[param_name] = trial.suggest_int(
                param_name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif suggest_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif suggest_type == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name, spec["choices"]
            )
    return params


def _build_model_from_params(model_name: str, params: Dict[str, Any]) -> Any:
    """Build a model instance from a set of tuned hyperparameters.

    Args:
        model_name: One of "XGBoost", "LightGBM", "CatBoost",
            "RandomForest", "GradientBoosting".
        params: Hyperparameter dict from an Optuna trial.

    Returns:
        A scikit-learn compatible regressor instance.
    """
    model_name_lower = model_name.lower()

    if model_name_lower == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params, random_state=42, verbosity=0)

    elif model_name_lower == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**params, random_state=42, verbosity=-1)

    elif model_name_lower == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            **params,
            random_state=42,
            verbose=0,
            allow_writing_files=False,
        )

    elif model_name_lower == "randomforest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params, random_state=42, n_jobs=-1)

    elif model_name_lower == "gradientboosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(**params, random_state=42)

    else:
        raise ValueError(f"Unknown model name for tuning: {model_name}")


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
    timeout_minutes: Optional[int] = None,
    search_space: Optional[Dict[str, Any]] = None,
    progress_bar: bool = True,
    config: Optional[DabbaConfig] = None,
) -> Tuple[Dict[str, Any], float, float]:
    """Run Optuna hyperparameter optimization for a single model.

    Uses :class:`optuna.Study` with ``TPESampler`` to minimize MAE over
    k-fold cross-validated predictions. Logs all trials to stdout and,
    if available, MLflow.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_name: Model type to tune ("XGBoost", "LightGBM", "CatBoost",
            "RandomForest", "GradientBoosting").
        n_trials: Number of Optuna trials.
        cv_folds: Number of CV folds for evaluation.
        random_state: Random seed for reproducibility.
        timeout_minutes: Optional timeout in minutes. ``None`` means no limit.
        search_space: Custom search space. If ``None``, uses the default
            spaces from :func:`get_model_search_spaces`.
        progress_bar: Show Optuna progress bar.
        config: Project configuration (for MLflow logging).

    Returns:
        Tuple of (best_params dict, best_mae, best_rmse).

    Raises:
        ImportError: If optuna is not installed.
        ValueError: If ``model_name`` is not supported.

    Example:
        >>> params, mae, rmse = tune_hyperparameters(X, y, "XGBoost", n_trials=30)
        >>> tuned_model = _build_model_from_params("XGBoost", params)
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. "
            "Install it with: pip install optuna"
        )

    # Resolve search space
    all_spaces = get_model_search_spaces()
    if search_space is not None:
        space = search_space
    elif model_name in all_spaces:
        space = all_spaces[model_name]
    else:
        raise ValueError(
            f"Model '{model_name}' is not supported for tuning. "
            f"Supported models: {list(all_spaces.keys())}"
        )

    # Build preprocessor (shared across all trials for fairness)
    preprocessor = _build_preprocessor(X)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    config = config or get_config()
    timeout_seconds = timeout_minutes * 60 if timeout_minutes else None

    def _objective(trial: optuna.trial.Trial) -> float:
        """Optuna objective: return the CV mean MAE for sampled params."""
        params = _sample_optuna_params(trial, space)

        try:
            model = _build_model_from_params(model_name, params)
        except Exception as e:
            logger.warning("Failed to build %s with params: %s — %s", model_name, params, e)
            return float("inf")

        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        try:
            y_pred = cross_val_predict(pipe, X, y, cv=kf, method="predict")
        except Exception as e:
            logger.warning("CV failed for trial: %s", e)
            return float("inf")

        mae = mean_absolute_error(y, y_pred)
        return mae

    # Create and run Optuna study
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"{model_name}_hpo",
    )

    logger.info(
        "🔬 Starting Optuna HPO for %s — %d trials, %d-fold CV",
        model_name,
        n_trials,
        cv_folds,
    )

    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=progress_bar,
    )

    best_params = study.best_params
    best_mae = study.best_value

    # Evaluate best params for RMSE too
    best_model = _build_model_from_params(model_name, best_params)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
    y_pred = cross_val_predict(pipe, X, y, cv=kf, method="predict")
    best_rmse = np.sqrt(mean_squared_error(y, y_pred))

    logger.info(
        "🏆 Best %s params (MAE=%.4f, RMSE=%.4f): %s",
        model_name,
        best_mae,
        best_rmse,
        best_params,
    )

    # Log to MLflow
    try:
        import mlflow

        mlflow.log_params({f"hpo_{model_name}_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(
            {
                f"hpo_{model_name}_best_mae": best_mae,
                f"hpo_{model_name}_best_rmse": best_rmse,
                f"hpo_{model_name}_n_trials": n_trials,
            }
        )
    except Exception:
        pass

    return best_params, best_mae, best_rmse


def get_tuned_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
    config: Optional[DabbaConfig] = None,
) -> Optional[Any]:
    """Run Optuna tuning and return the best model instance.

    Convenience wrapper around :func:`tune_hyperparameters` that
    returns a ready-to-use model with the best-found parameters.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_name: Model type to tune.
        n_trials: Number of Optuna trials.
        cv_folds: Number of CV folds.
        random_state: Random seed.
        config: Project configuration.

    Returns:
        The tuned model instance, or ``None`` if tuning failed.
    """
    try:
        best_params, best_mae, best_rmse = tune_hyperparameters(
            X,
            y,
            model_name,
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            config=config,
        )
        logger.info(
            "Tuned %s achieved MAE=%.4f / RMSE=%.4f with %d params: %s",
            model_name,
            best_mae,
            best_rmse,
            len(best_params),
            best_params,
        )
        return _build_model_from_params(model_name, best_params)
    except Exception as e:
        logger.warning("Tuning failed for %s: %s — falling back to defaults", model_name, e)
        return None


def tune_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_to_tune: Optional[List[str]] = None,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
    config: Optional[DabbaConfig] = None,
) -> Dict[str, Optional[Any]]:
    """Run Optuna tuning on multiple models and return tuned instances.

    Tunes each model in ``models_to_tune`` and returns a dict of
    ``{model_name: tuned_model_or_None}``. Models that fail tuning
    fall back to ``None`` (callers should use defaults as fallback).

    Args:
        X: Feature matrix.
        y: Target vector.
        models_to_tune: List of model names to tune. If ``None``,
            defaults to all models with defined search spaces.
        n_trials: Number of trials per model.
        cv_folds: Number of CV folds.
        random_state: Random seed.
        config: Project configuration.

    Returns:
        Dict mapping model name to tuned model (or ``None`` on failure).
    """
    config = config or get_config()
    if models_to_tune is None:
        models_to_tune = config.optuna_models_to_tune

    tuned_models: Dict[str, Optional[Any]] = {}

    for model_name in models_to_tune:
        tuned = get_tuned_model(
            X,
            y,
            model_name,
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            config=config,
        )
        tuned_models[model_name] = tuned

        if tuned is None:
            logger.warning(
                "⚠️  Tuning for %s returned None — defaults will be used",
                model_name,
            )

    return tuned_models


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
