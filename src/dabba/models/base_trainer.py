"""Backward-compatibility shim for the split ``base_trainer`` monolith.

The original ~890-line monolith was split into ``trainer_pkg/``
(``types`` / ``model_builders`` / ``training``) plus ``model_selection``.
This module re-exports the original public and private API names so
existing imports (``from dabba.models.base_trainer import ...``) keep
working unchanged.
"""

from dabba.models.model_selection import comparison_to_dataframe, select_best_model
from dabba.models.trainer_pkg.model_builders import (
    _build_preprocessor,
    _end_mlflow_run,
    _get_catboost,
    _get_lightgbm,
    _get_xgboost,
    _log_model_to_mlflow,
    _setup_mlflow,
)
from dabba.models.trainer_pkg.training import (
    fit_best_model,
    save_model,
    train_and_evaluate_models,
)
from dabba.models.trainer_pkg.types import (
    ModelResult,
    _build_model_from_params,
    _sample_optuna_params,
    get_model_search_spaces,
    get_tuned_model,
    tune_all_models,
    tune_hyperparameters,
)

__all__ = [
    "ModelResult",
    "comparison_to_dataframe",
    "fit_best_model",
    "get_model_search_spaces",
    "get_tuned_model",
    "save_model",
    "select_best_model",
    "train_and_evaluate_models",
    "tune_all_models",
    "tune_hyperparameters",
]
