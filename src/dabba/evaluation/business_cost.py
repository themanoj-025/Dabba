"""Business cost analysis for delivery SLA violations, Reliability Score
computation, and A/B weight-scenario simulations.

Computes the cost impact of late deliveries, model performance
in business-interpretable terms, and the Reliability Score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)

# ─── Weight profiles for A/B scenario simulation ───────────────────────

WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {
        "w_rating": 0.4,
        "w_sentiment": 0.3,
        "w_delay": 0.3,
        "description": "Default: equal balance of quality, sentiment, and delivery speed",
    },
    "quality_first": {
        "w_rating": 0.5,
        "w_sentiment": 0.3,
        "w_delay": 0.2,
        "description": "Quality-first: prioritizes food rating and sentiment over speed",
    },
    "speed_first": {
        "w_rating": 0.2,
        "w_sentiment": 0.2,
        "w_delay": 0.6,
        "description": "Speed-first: heavily weights on-time delivery reliability",
    },
}


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
        Dict with SLA metrics (on_time_rate, precision, recall, etc.).
    """
    config = config or get_config()
    threshold = sla_threshold or config.sla_threshold_minutes

    actual_late = y_true > threshold
    predicted_late = y_pred > threshold

    tp = int(((~actual_late) & (~predicted_late)).sum())
    fp = int(((actual_late) & (~predicted_late)).sum())
    fn = int(((~actual_late) & (predicted_late)).sum())
    tn = int(((actual_late) & (predicted_late)).sum())

    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "sla_threshold_min": threshold,
        "total_orders": total,
        "actual_on_time_rate": round(float((total - actual_late.sum()) / total), 4),
        "predicted_on_time_rate": round(
            float((total - predicted_late.sum()) / total), 4
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
    logger.info(
        "SLA analysis (threshold=%.0f min): on-time rate=%.2f%%",
        threshold,
        metrics["actual_on_time_rate"] * 100,
    )
    return metrics


def _min_max_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def compute_reliability_score(
    rating: float | np.ndarray,
    sentiment: float | np.ndarray,
    delay_risk: float | np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    config: Optional[DabbaConfig] = None,
) -> float | np.ndarray:
    """Compute the Reliability Score for one or more restaurants.

    Formula:
        reliability_score = w1 * norm(rating) + w2 * norm(sentiment) - w3 * norm(delay_risk)

    Where norm() min-max normalizes each component to [0, 1].

    Args:
        rating: Restaurant rating(s), typically in [0, 5] range.
        sentiment: Sentiment score(s), typically in [-1, 1] range.
        delay_risk: Predicted delay risk(s).
        weights: Dict with 'w_rating', 'w_sentiment', 'w_delay' keys.
            Uses config defaults if None.
        config: Project configuration.

    Returns:
        Reliability score in [0, 1]. Higher = more reliable.
    """
    config = config or get_config()

    if weights is None:
        w_rating = config.reliability_w_rating
        w_sentiment = config.reliability_w_sentiment
        w_delay = config.reliability_w_delay
    else:
        w_rating = weights.get("w_rating", config.reliability_w_rating)
        w_sentiment = weights.get("w_sentiment", config.reliability_w_sentiment)
        w_delay = weights.get("w_delay", config.reliability_w_delay)

    rating = np.asarray(rating, dtype=np.float64)
    sentiment = np.asarray(sentiment, dtype=np.float64)
    delay_risk = np.asarray(delay_risk, dtype=np.float64)

    norm_rating = _min_max_norm(rating)
    norm_sentiment = _min_max_norm(sentiment)
    norm_delay = _min_max_norm(delay_risk)

    score = w_rating * norm_rating + w_sentiment * norm_sentiment - w_delay * norm_delay
    score = np.clip(score, 0.0, 1.0)

    return float(score) if score.ndim == 0 else score


def run_ab_scenario_simulation(
    df: pd.DataFrame,
    rating_col: str = "rate",
    sentiment_col: str = "avg_sentiment",
    delay_col: str = "delay_risk",
    top_n: int = 10,
) -> Dict[str, Any]:
    """Run Reliability Score ranking under multiple weight profiles.

    Simulates what a product team would A/B test: how does the
    top-N restaurant list change when you prioritize speed vs. quality?

    Args:
        df: DataFrame with rating, sentiment, delay_risk columns.
        rating_col: Column name for ratings.
        sentiment_col: Column name for sentiment scores.
        delay_col: Column name for delay risk.
        top_n: Number of top restaurants to show per profile.

    Returns:
        Dict mapping profile name to {scores, top_restaurants, description}.
    """
    if rating_col not in df.columns:
        logger.warning("Column '%s' not found — using 0", rating_col)
        ratings = np.zeros(len(df))
    else:
        ratings = df[rating_col].fillna(0).values

    if sentiment_col not in df.columns:
        logger.warning("Column '%s' not found — using 0", sentiment_col)
        sentiments = np.zeros(len(df))
    else:
        sentiments = df[sentiment_col].fillna(0).values

    if delay_col not in df.columns:
        logger.warning("Column '%s' not found — using 0.5", delay_col)
        delays = np.full(len(df), 0.5)
    else:
        delays = df[delay_col].fillna(0.5).values

    results: Dict[str, Any] = {}

    for profile_name, weights in WEIGHT_PROFILES.items():
        scores = compute_reliability_score(ratings, sentiments, delays, weights)
        df_copy = df.copy()
        df_copy["ab_score"] = scores

        # Get top N
        top_idx = np.argsort(scores)[::-1][:top_n]
        top_restaurants = []
        for idx in top_idx:
            rest = {
                "name": df_copy.iloc[idx].get("name", f"Restaurant {idx}"),
                "score": float(scores[idx]),
            }
            for col in ["rate", "cost_for_two", "location", "cuisines"]:
                if col in df_copy.columns:
                    rest[col] = df_copy.iloc[idx][col]
            top_restaurants.append(rest)

        results[profile_name] = {
            "weights": weights,
            "description": weights["description"],
            "mean_score": float(np.mean(scores)),
            "top_restaurants": top_restaurants,
        }

    # Compute overlap between profiles
    balanced_names = {
        r["name"] for r in results.get("balanced", {}).get("top_restaurants", [])
    }
    quality_names = {
        r["name"] for r in results.get("quality_first", {}).get("top_restaurants", [])
    }
    speed_names = {
        r["name"] for r in results.get("speed_first", {}).get("top_restaurants", [])
    }

    results["_meta"] = {
        "balanced_vs_quality_overlap": len(balanced_names & quality_names),
        "balanced_vs_speed_overlap": len(balanced_names & speed_names),
        "quality_vs_speed_overlap": len(quality_names & speed_names),
    }

    logger.info(
        "A/B scenario simulation complete: %d profiles, %d restaurants",
        len(WEIGHT_PROFILES),
        len(df),
    )
    return results
