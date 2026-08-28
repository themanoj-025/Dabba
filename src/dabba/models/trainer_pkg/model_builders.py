# Model builder utilities

from __future__ import annotations


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
    except (ImportError, OSError, AttributeError) as e:
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
    additional_params: dict[str, Any] | None = None,
) -> str | None:
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

        params: dict[str, Any] = {"model": name}
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
    except (ImportError, OSError, AttributeError) as e:
        logger.warning("MLflow logging failed for %s: %s", name, e)
        return None


def _end_mlflow_run(mlflow_run: Any) -> None:
    """End an MLflow run, ignoring errors."""
    if mlflow_run:
        try:
            import mlflow

            mlflow.end_run()
        except (ImportError, OSError):
            pass


