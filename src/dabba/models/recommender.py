"""Hybrid restaurant recommender combining content-based and popularity signals.

Uses cosine similarity over features, Bayesian-adjusted ratings,
and the winning rating model for imputation / reliability scoring.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


def bayesian_average(
    ratings: pd.Series,
    vote_counts: pd.Series,
    C: float | None = None,
    m: float | None = None,
) -> pd.Series:
    """Compute Bayesian average rating to prevent low-vote restaurants from ranking high.

    Formula: weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C
    where R = restaurant rating, v = vote count, C = mean rating across all,
    m = minimum votes threshold (e.g., 25th percentile of vote counts).

    Args:
        ratings: Series of restaurant ratings.
        vote_counts: Series of vote counts per restaurant.
        C: Global mean rating. Computed from data if None.
        m: Minimum votes threshold. Uses 25th percentile if None.

    Returns:
        pd.Series: Bayesian-adjusted ratings.
    """
    if C is None:
        C = ratings.mean()
    if m is None:
        m = vote_counts.quantile(0.25)

    v = vote_counts.fillna(0)
    R = ratings.fillna(C)

    weighted = (v / (v + m)) * R + (m / (v + m)) * C
    return weighted


class RestaurantRecommender:
    """Hybrid recommender blending content similarity with popularity signals.

    Attributes:
        df: DataFrame with restaurant features.
        feature_matrix: Numerical feature matrix for similarity.
        rating_model: Loaded winning rating model (for prediction).
        config: Project configuration.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        config: Optional[DabbaConfig] = None,
    ):
        """Initialize the recommender.

        Args:
            df: DataFrame with restaurant data and features.
            feature_cols: Column names to use for similarity computation.
            config: Project configuration.
        """
        self.config = config or get_config()
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.rating_model = None

        # Build feature matrix (numeric only for cosine similarity)
        num_cols = [c for c in feature_cols if c in df.columns]
        self.feature_matrix = self.df[num_cols].fillna(0).values

        # Compute Bayesian-adjusted ratings
        vote_col = "votes" if "votes" in df.columns else None
        rate_col = "rate" if "rate" in df.columns else None
        if rate_col and vote_col:
            self.df["bayesian_rating"] = bayesian_average(
                self.df[rate_col], self.df[vote_col]
            )
        elif rate_col:
            self.df["bayesian_rating"] = self.df[rate_col]
        else:
            self.df["bayesian_rating"] = 3.5  # fallback

        logger.info("Recommender initialized with %d restaurants", len(self.df))

    def load_rating_model(self, model_path: Optional[str] = None) -> None:
        """Load the winning rating model for prediction.

        Args:
            model_path: Path to the saved model. Uses config default if None.
        """
        path = model_path or str(self.config.best_rating_model_path)
        try:
            self.rating_model = joblib.load(path)
            logger.info("Loaded rating model from %s", path)
        except FileNotFoundError:
            logger.warning(
                "Rating model not found at %s — predictions will use bayesian rating",
                path,
            )

    def _compute_similarity(self, query_features: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all restaurants.

        Args:
            query_features: 1D array of query feature values.

        Returns:
            np.ndarray: Similarity scores for each restaurant.
        """
        query = query_features.reshape(1, -1)
        return cosine_similarity(query, self.feature_matrix).flatten()

    def recommend(
        self,
        cuisine: Optional[str] = None,
        budget: Optional[float] = None,
        area: Optional[str] = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Generate ranked restaurant recommendations with explanations.

        Args:
            cuisine: Preferred cuisine filter (partial match).
            budget: Maximum cost for two.
            area: Area/neighborhood filter.
            top_n: Number of recommendations to return.

        Returns:
            pd.DataFrame: Top-N restaurants with rating, bayesian_rating,
                similarity_score, and recommendation_explanation columns.
        """
        mask = pd.Series(True, index=self.df.index)

        # Apply filters
        if cuisine and "cuisines" in self.df.columns:
            mask &= self.df["cuisines"].str.contains(cuisine, case=False, na=False)

        if budget and "cost_for_two" in self.df.columns:
            mask &= self.df["cost_for_two"] <= budget

        if area and "location" in self.df.columns:
            mask &= self.df["location"].str.contains(area, case=False, na=False)

        candidates = self.df[mask].copy()

        if candidates.empty:
            logger.warning("No restaurants match the given filters")
            return pd.DataFrame()

        # Compute content similarity (use mean values as query if no specific query)
        if len(self.feature_cols) > 0:
            query_features = candidates[self.feature_cols].mean().values
            similarities = self._compute_similarity(query_features)
            candidates["similarity_score"] = similarities[mask.values]
        else:
            candidates["similarity_score"] = 1.0

        # Combine: similarity + bayesian rating
        candidates["combined_score"] = 0.5 * candidates["similarity_score"] + 0.5 * (
            candidates["bayesian_rating"] / 5.0
        )

        # Generate explanations
        def _explain(row: pd.Series) -> str:
            parts = []
            if "rate" in row and pd.notna(row["rate"]):
                parts.append(f"Rated {row['rate']}/5")
            if "bayesian_rating" in row:
                parts.append(f"(adj: {row['bayesian_rating']:.2f})")
            if "cost_for_two" in row and pd.notna(row["cost_for_two"]):
                parts.append(f"₹{int(row['cost_for_two'])} for two")
            if "cuisines" in row and pd.notna(row["cuisines"]):
                parts.append(f"Cuisines: {row['cuisines'][:60]}")
            return " | ".join(parts) if parts else "Good match for your preferences"

        candidates["recommendation_explanation"] = candidates.apply(_explain, axis=1)

        # Sort and return top N
        result = candidates.sort_values("combined_score", ascending=False).head(top_n)
        display_cols = [
            "name",
            "rate",
            "bayesian_rating",
            "cost_for_two",
            "location",
            "cuisines",
            "similarity_score",
            "combined_score",
            "recommendation_explanation",
        ]
        display_cols = [c for c in display_cols if c in result.columns]
        return result[display_cols].reset_index(drop=True)
