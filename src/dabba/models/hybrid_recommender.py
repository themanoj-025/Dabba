"""Hybrid recommender — blends content-based similarity, collaborative
filtering scores, and the Reliability Score into one ranked output.

Blend weights are configurable in config.py (same pattern as the
Reliability Score weights), enabling A/B testing of different
recommendation strategies.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from dabba.config import DabbaConfig, get_config
from dabba.models.recommender import bayesian_average
from dabba.models.collaborative_recommender import (
    MatrixFactorization,
    get_collaborative_scores,
)

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Recommender combining content-based, collaborative, and reliability signals.

    Args:
        restaurants_df: Processed restaurant DataFrame.
        content_feature_cols: Feature columns for content-based similarity.
        collaborative_model: Trained collaborative filtering model (optional).
        config: Project configuration.
    """

    def __init__(
        self,
        restaurants_df: pd.DataFrame,
        content_feature_cols: List[str],
        collaborative_model: Optional[MatrixFactorization] = None,
        config: Optional[DabbaConfig] = None,
    ):
        self.config = config or get_config()
        self.df = restaurants_df.copy()
        self.content_feature_cols = content_feature_cols
        self.collaborative_model = collaborative_model
        self.rng = np.random.RandomState(self.config.random_seed)

        # Compute Bayesian-adjusted ratings (similarity computed on-the-fly in recommend())
        num_cols = [c for c in content_feature_cols if c in self.df.columns]
        self.feature_matrix = self.df[num_cols].fillna(0).values
        vote_col = "votes" if "votes" in self.df.columns else None
        rate_col = "rate" if "rate" in self.df.columns else None
        if rate_col and vote_col:
            self.df["bayesian_rating"] = bayesian_average(
                self.df[rate_col], self.df[vote_col]
            )
        elif rate_col:
            self.df["bayesian_rating"] = self.df[rate_col]
        else:
            self.df["bayesian_rating"] = 3.5

        # Compute collaborative scores if model is available
        self.collaborative_scores: Optional[np.ndarray] = None
        if collaborative_model is not None:
            n_users = collaborative_model.user_embeddings.weight.shape[0]
            restaurant_ids = self.df.index.values
            try:
                raw_scores = get_collaborative_scores(
                    collaborative_model, n_users, restaurant_ids
                )
                # Normalize to [0, 1]
                score_min, score_max = raw_scores.min(), raw_scores.max()
                if score_max > score_min:
                    self.collaborative_scores = (raw_scores - score_min) / (score_max - score_min)
                else:
                    self.collaborative_scores = np.full_like(raw_scores, 0.5)
                logger.info("Collaborative scores computed for %d restaurants",
                            len(self.collaborative_scores))
            except Exception as e:
                logger.warning("Failed to compute collaborative scores: %s", e)

        logger.info("HybridRecommender initialized with %d restaurants", len(self.df))

    def recommend(
        self,
        cuisine: Optional[str] = None,
        budget: Optional[float] = None,
        area: Optional[str] = None,
        prioritize: str = "balanced",
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Generate hybrid restaurant recommendations.

        Args:
            cuisine: Preferred cuisine filter.
            budget: Maximum cost for two.
            area: Area/neighborhood filter.
            prioritize: 'balanced', 'speed', or 'quality' — selects weight profile.
            top_n: Number of recommendations to return.

        Returns:
            pd.DataFrame: Ranked recommendations with scores.
        """
        # Apply filters
        mask = pd.Series(True, index=self.df.index)

        if cuisine and "cuisines" in self.df.columns:
            mask &= self.df["cuisines"].str.contains(cuisine, case=False, na=False)
        if budget and "cost_for_two" in self.df.columns:
            mask &= self.df["cost_for_two"] <= budget
        if area and "location" in self.df.columns:
            mask &= self.df["location"].str.contains(area, case=False, na=False)

        candidates = self.df[mask].copy()
        if candidates.empty:
            return pd.DataFrame()

        # 1. Content-based similarity score
        if len(self.content_feature_cols) > 0 and len(candidates) > 0:
            # Use mean features as query
            query_features = self.df[self.content_feature_cols].mean().values.reshape(1, -1)
            candidate_features = candidates[self.content_feature_cols].fillna(0).values
            sim_scores_raw = cosine_similarity(query_features, candidate_features)[0]
            # Normalize
            s_min, s_max = sim_scores_raw.min(), sim_scores_raw.max()
            if s_max > s_min:
                content_scores = (sim_scores_raw - s_min) / (s_max - s_min)
            else:
                content_scores = np.full_like(sim_scores_raw, 0.5)
        else:
            content_scores = np.ones(len(candidates)) * 0.5

        # 2. Collaborative filtering score
        if self.collaborative_scores is not None:
            cf_scores = np.array([
                self.collaborative_scores[self.df.index.get_loc(idx)]
                for idx in candidates.index
            ])
        else:
            cf_scores = np.ones(len(candidates)) * 0.5

        # 3. Reliability score
        if "reliability_score" in candidates.columns:
            rel_scores = candidates["reliability_score"].fillna(0.5).values
        else:
            rel_scores = np.ones(len(candidates)) * 0.5

        # 4. Bayesian rating (normalized to [0, 1])
        bayes_scores = candidates["bayesian_rating"].values / 5.0

        # Select weight profile
        weights = self._get_weight_profile(prioritize)

        # Compute combined score
        candidates["content_score"] = content_scores
        candidates["cf_score"] = cf_scores
        candidates["reliability_score_display"] = rel_scores
        candidates["bayesian_norm"] = bayes_scores

        candidates["combined_score"] = (
            weights["content"] * content_scores
            + weights["collaborative"] * cf_scores
            + weights["reliability"] * rel_scores
            + weights["bayesian"] * bayes_scores
        )

        # Generate explanations
        def _explain(row: pd.Series) -> str:
            parts = []
            if row.get("rate") and pd.notna(row["rate"]):
                parts.append(f"Rated {row['rate']}/5")
            if "reliability_score_display" in row:
                rs = row["reliability_score_display"]
                badge = "🟢" if rs >= 0.7 else "🟡" if rs >= 0.4 else "🔴"
                parts.append(f"{badge} Rel:{rs:.2f}")
            if row.get("cost_for_two") and pd.notna(row["cost_for_two"]):
                parts.append(f"₹{int(row['cost_for_two'])}")
            return " | ".join(parts) if parts else "Good match for your preferences"

        candidates["explanation"] = candidates.apply(_explain, axis=1)

        result = candidates.sort_values("combined_score", ascending=False).head(top_n)
        display_cols = [
            "name", "rate", "bayesian_rating", "cost_for_two",
            "location", "cuisines", "reliability_score_display",
            "cf_score", "combined_score", "explanation",
        ]
        display_cols = [c for c in display_cols if c in result.columns]
        return result[display_cols].reset_index(drop=True)

    def _get_weight_profile(self, prioritize: str) -> Dict[str, float]:
        """Get weight profile for the given prioritization.

        Args:
            prioritize: 'balanced', 'speed', or 'quality'.

        Returns:
            Dict with content, collaborative, reliability, bayesian weights.
        """
        c = self.config
        profiles = {
            "balanced": {
                "content": c.hybrid_weight_content,
                "collaborative": c.hybrid_weight_collaborative,
                "reliability": c.hybrid_weight_reliability,
                "bayesian": 0.0,
            },
            "speed": {
                "content": 0.2,
                "collaborative": 0.2,
                "reliability": 0.5,  # reliability heavily weights delivery speed
                "bayesian": 0.1,
            },
            "quality": {
                "content": 0.25,
                "collaborative": 0.25,
                "reliability": 0.1,
                "bayesian": 0.4,
            },
        }
        return profiles.get(prioritize, profiles["balanced"])
