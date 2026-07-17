"""Model selection module — trains, compares, and persists the best models.

This is the central module that:
1. Trains all candidate models with identical features and CV
2. Produces comparison CSVs and charts
3. Automatically selects the best by lowest MAE (overridable)
4. Saves only the winning model to disk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


def comparison_to_dataframe(results: list, task: str = "rating") -> pd.DataFrame:
    """Convert a list of ModelResult/ETAModelResult to a comparison DataFrame.

    Args:
        results: List of result dataclass instances.
        task: Task name ('rating' or 'eta') for logging.

    Returns:
        pd.DataFrame: Comparison table with columns:
            model, mae, rmse, r2, train_time_s.
    """
    records = []
    for r in results:
        records.append({
            "model": r.name,
            "mae": round(r.mae, 4),
            "rmse": round(r.rmse, 4),
            "r2": round(r.r2, 4),
            "train_time_s": round(r.train_time, 2),
        })
    df = pd.DataFrame(records).sort_values("mae").reset_index(drop=True)
    logger.info("Comparison table for %s:\n%s", task, df.to_string(index=False))
    return df


def save_comparison_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a comparison DataFrame to CSV.

    Args:
        df: Comparison DataFrame.
        path: Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved comparison CSV to %s", path)


def select_best_model(
    results: list,
    metric: str = "mae",
    task: str = "rating",
) -> Optional[str]:
    """Select the best model by the given metric.

    Args:
        results: List of ModelResult/ETAModelResult instances.
        metric: Metric to optimize ('mae', 'rmse', 'r2'). For R², higher is better.
        task: Task name for logging.

    Returns:
        The winning model name, or None if no results.
    """
    if not results:
        logger.warning("No results to select from for task '%s'", task)
        return None

    key_fn = {"mae": lambda r: r.mae, "rmse": lambda r: r.rmse, "r2": lambda r: -r.r2}
    best = min(results, key=key_fn.get(metric, lambda r: r.mae))

    logger.info(
        "Best %s model: %s (%s=%.4f)",
        task, best.name, metric.upper(), getattr(best, metric),
    )

    return best.name
