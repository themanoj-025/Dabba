"""Model evaluation metrics and utilities.

Provides standardized metric computation for regression tasks.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        prefix: Optional prefix for metric names (e.g., 'rating_', 'eta_').

    Returns:
        Dict with keys: {prefix}mae, {prefix}rmse, {prefix}r2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        f"{prefix}mae": round(mae, 4),
        f"{prefix}rmse": round(rmse, 4),
        f"{prefix}r2": round(r2, 4),
    }

    logger.info("Metrics %s: MAE=%.4f, RMSE=%.4f, R²=%.4f", prefix, mae, rmse, r2)
    return metrics
