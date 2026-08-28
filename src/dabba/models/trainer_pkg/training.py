# Training, evaluation, and model persistence

from __future__ import annotations


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Any],
    config: DabbaConfig | None = None,
    use_mlflow: bool = True,
    task: str = "rating",
) -> tuple[list[ModelResult], ModelResult | None]:
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

    results: list[ModelResult] = []

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
        except (ValueError, TypeError) as e:
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
        except (ImportError, OSError, AttributeError):
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
    models: dict[str, Any],
    save_path: Any,
    task: str = "model",
    config: DabbaConfig | None = None,
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
