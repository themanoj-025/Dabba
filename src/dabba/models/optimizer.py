"""Delivery partner assignment optimizer.

Uses the Hungarian algorithm (linear_sum_assignment) to assign
delivery partners to orders minimizing total predicted delivery time.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


def optimize_assignments(
    cost_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Assign delivery partners to orders using the Hungarian algorithm.

    Args:
        cost_matrix: N×M matrix where entry (i, j) is the predicted
            delivery time for order i assigned to partner j.

    Returns:
        Tuple of (assignment array, total predicted time).
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_time = cost_matrix[row_ind, col_ind].sum()
    logger.info(
        "Optimized %d assignments — total predicted time: %.1f min",
        len(row_ind), total_time,
    )
    return col_ind, total_time


def naive_assignments(cost_matrix: np.ndarray) -> float:
    """Compute total delivery time using naive first-available assignment.

    Each order is assigned to the partner with the lowest index available.

    Args:
        cost_matrix: N×M cost matrix.

    Returns:
        Total predicted delivery time under naive assignment.
    """
    n_orders, n_partners = cost_matrix.shape
    total = 0.0
    partner_load = np.zeros(n_partners)

    for i in range(n_orders):
        # Assign to partner with current lowest total load
        j = np.argmin(partner_load)
        total += cost_matrix[i, j]
        partner_load[j] += cost_matrix[i, j]

    logger.info("Naive assignment total: %.1f min", total)
    return total


def compare_assignment_strategies(
    orders_df: pd.DataFrame,
    eta_model: object,
    feature_cols: list[str],
) -> dict[str, float]:
    """Compare optimized vs naive assignment on a set of simulated orders.

    Args:
        orders_df: DataFrame of pending orders with features.
        eta_model: Trained ETA prediction model.
        feature_cols: Feature columns used by the model.

    Returns:
        Dict with 'optimized_total_min' and 'naive_total_min'.
    """
    if len(orders_df) == 0:
        return {"optimized_total_min": 0, "naive_total_min": 0}

    # Build cost matrix: predict delivery time for each (order, partner) pair
    # For simplicity, we assume all partners are identical and use order features
    # In production, partner-specific features (age, rating, vehicle) would vary
    X = orders_df[feature_cols].fillna(0)
    predictions = eta_model.predict(X)

    # Create a synthetic cost matrix: orders × simulated partners
    n_orders = len(predictions)
    n_partners = min(n_orders, 10)  # simulate up to 10 partners

    # Add small random variation per partner to make assignment non-trivial
    rng = np.random.RandomState(42)
    partner_variations = rng.uniform(0.8, 1.2, size=(1, n_partners))
    cost_matrix = np.outer(predictions, partner_variations)

    opt_assignment, opt_total = optimize_assignments(cost_matrix)
    naive_total = naive_assignments(cost_matrix)

    improvement = (naive_total - opt_total) / naive_total * 100 if naive_total > 0 else 0

    logger.info(
        "Assignment comparison: optimized=%.1f min, naive=%.1f min, improvement=%.1f%%",
        opt_total, naive_total, improvement,
    )

    return {
        "optimized_total_min": round(float(opt_total), 1),
        "naive_total_min": round(float(naive_total), 1),
        "improvement_pct": round(float(improvement), 1),
    }
