"""Business cost analysis for delivery SLA violations.

Computes the cost impact of late deliveries, model performance
in business-interpretable terms, and the Reliability Score.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


def compute_sla_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sla_threshold: float | None = None,
    config: DabbaConfig | None = None,
) -> Dict[str, float]:
    """Analyze SLA violation rates using predicted vs actual delivery times.

    Args:
        y_true: Actual delivery times in minutes.
        y_pred: Predicted delivery times in minutes.
        sla_threshold: SLA threshold in minutes. Uses config default if None.
        config: Project configuration.

    Returns:
        Dict with SLA metrics:
            - actual_on_time_rate: fraction of actual deliveries within SLA
            - predicted_on_time_rate: fraction predicted as on-time
            - true_positives: correctly predicted on-time
            - false_positives: predicted on-time but actually late
            - false_negatives: predicted late but actually on-time
            - true_negatives: correctly predicted late
            - precision: of the "on-time" predictions, how many are correct
            - recall: of the actually on-time, how many were flagged
    """
    config = config or get_config()
    threshold = sla_threshold or config.sla_threshold_minutes

    actual_late = y_true > threshold
    predicted_late = y_pred > threshold

    tp = int((~actual_late) & (~predicted_late))  # both on-time
    fp = int((actual_late) & (~predicted_late))    # predicted on-time but actually late
    fn = int((~actual_late) & (predicted_late))    # predicted late but actually on-time
    tn = int((actual_late) & (predicted_late))      # both late

    total = len(y_true)
    actual_ontime = total - actual_late.sum()
    predicted_ontime = total - predicted_late.sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "sla_threshold_min": threshold,
        "total_orders": total,
        "actual_on_time_rate": round(float(actual_ontime / total), 4),
        "predicted_on_time_rate": round(float(predicted_ontime / total), 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    logger.info("SLA analysis (threshold=%.0f min): on-time rate=%.2f%%",
                threshold, metrics["actual_on_time_rate"] * 100)
    return metrics


def compute_reliability_score(
    rating: float | np.ndarray,
    sentiment: float | np.ndarray,
    delay_risk: float | np.ndarray,
    config: Optional[DabbaConfig] = None,
) -> float | np.ndarray:
    """Compute the Reliability Score for one or more restaurants.

    Formula:
        reliability_score = w1 * norm(rating) + w2 * norm(sentiment) - w3 * norm(delay_risk)

    Where norm() min-max normalizes each component to [0, 1].
    Weights w1, w2, w3 come from config.py (defaults: 0.4, 0.3, 0.3).

    Args:
        rating: Restaurant rating(s), typically in [0, 5] range.
        sentiment: Sentiment score(s), typically in [-1, 1] range.
        delay_risk: Predicted delay risk(s), e.g. predicted ETA in minutes
            or a probability in [0, 1].
        config: Project configuration for weights.

    Returns:
        Reliability score in [0, 1] range. Higher = more reliable.
    """
    config = config or get_config()

    rating = np.asarray(rating, dtype=np.float64)
    sentiment = np.asarray(sentiment, dtype=np.float64)
    delay_risk = np.asarray(delay_risk, dtype=np.float64)

    # Min-max normalization helper
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    norm_rating = _norm(rating)
    norm_sentiment = _norm(sentiment)
    norm_delay = _norm(delay_risk)

    score = (
        config.reliability_w_rating * norm_rating
        + config.reliability_w_sentiment * norm_sentiment
        - config.reliability_w_delay * norm_delay
    )

    # Clip to [0, 1]
    score = np.clip(score, 0.0, 1.0)

    logger.info(
        "Reliability score computed — mean=%.3f, std=%.3f",
        float(np.mean(score)), float(np.std(score)),
    )
    return float(score) if score.ndim == 0 else score
