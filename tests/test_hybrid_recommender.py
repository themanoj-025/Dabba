"""Tests for the HybridRecommender — blending content, collaborative, and reliability signals."""

import numpy as np
import pandas as pd
import pytest

from dabba.models.hybrid_recommender import HybridRecommender


class TestHybridRecommenderInit:
    """Tests for HybridRecommender initialization."""

    @pytest.fixture
    def sample_df(self):
        """Create a small restaurant DataFrame for testing."""
        return pd.DataFrame({
            "name": ["A", "B", "C", "D"],
            "rate": [4.5, 3.8, 4.2, 3.5],
            "votes": [500, 50, 300, 10],
            "cost_for_two": [500, 300, 800, 200],
            "location": ["Koramangala", "Indiranagar", "Koramangala", "HSR Layout"],
            "cuisines": ["North Indian, Chinese", "Italian", "North Indian", "South Indian"],
            "votes_log": [6.2, 3.9, 5.7, 2.4],
            "online_order_binary": [1, 1, 0, 1],
        })

    def test_initializes_with_dataframe(self, sample_df):
        """Should initialize with a DataFrame and feature columns."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        assert len(rec.df) == 4
        assert "bayesian_rating" in rec.df.columns

    def test_bayesian_rating_computed(self, sample_df):
        """Bayesian rating should be computed during init."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        assert "bayesian_rating" in rec.df.columns
        # Low-vote restaurant should be pulled toward global mean
        assert rec.df["bayesian_rating"].iloc[3] < rec.df["rate"].iloc[3]

    def test_collaborative_model_none_by_default(self, sample_df):
        """Collaborative model should be None if not provided."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        assert rec.collaborative_model is None
        assert rec.collaborative_scores is None

    def test_random_seed_from_config(self, sample_df):
        """Random state should use config's random_seed."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        assert rec.rng is not None

    def test_empty_feature_cols(self, sample_df):
        """Should handle empty feature columns gracefully."""
        rec = HybridRecommender(sample_df, [])
        assert len(rec.df) == 4

    def test_missing_vote_column(self):
        """Should fall back to raw rating when vote column is missing."""
        df = pd.DataFrame({
            "name": ["A", "B"],
            "rate": [4.0, 3.5],
            "cost_for_two": [500, 300],
        })
        rec = HybridRecommender(df, ["cost_for_two"])
        assert rec.df["bayesian_rating"].iloc[0] == 4.0


class TestHybridRecommenderRecommend:
    """Tests for HybridRecommender.recommend()."""

    @pytest.fixture
    def sample_df(self):
        """Create a restaurant DataFrame with feature columns."""
        return pd.DataFrame({
            "name": ["A", "B", "C", "D", "E"],
            "rate": [4.5, 3.8, 4.2, 3.5, 4.8],
            "votes": [500, 50, 300, 10, 1000],
            "cost_for_two": [500, 300, 800, 200, 1200],
            "location": ["Koramangala", "Indiranagar", "Koramangala", "HSR Layout", "MG Road"],
            "cuisines": ["North Indian, Chinese", "Italian", "North Indian, Mughlai", "South Indian", "Japanese"],
            "votes_log": [6.2, 3.9, 5.7, 2.4, 6.9],
            "online_order_binary": [1, 1, 0, 1, 1],
        })

    def test_recommend_returns_dataframe(self, sample_df):
        """recommend() should return a DataFrame."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(cuisine="North Indian", top_n=3)
        assert isinstance(result, pd.DataFrame)

    def test_recommend_respects_budget(self, sample_df):
        """Recommendations should respect the budget filter."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(budget=400, top_n=10)
        if not result.empty and "cost_for_two" in result.columns:
            assert (result["cost_for_two"] <= 400).all()

    def test_recommend_respects_cuisine(self, sample_df):
        """Recommendations should filter by cuisine."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(cuisine="Italian", top_n=5)
        if not result.empty and "cuisines" in result.columns:
            assert all("Italian" in str(c) for c in result["cuisines"])

    def test_recommend_respects_area(self, sample_df):
        """Recommendations should filter by area."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(area="Koramangala", top_n=5)
        if not result.empty and "location" in result.columns:
            assert all("Koramangala" in str(l) for l in result["location"])

    def test_recommend_empty_result(self, sample_df):
        """Should return empty DataFrame when no matches."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(cuisine="Thai", area="Nonexistent")
        assert result.empty

    def test_recommend_has_explanation_column(self, sample_df):
        """Result should include an explanation column."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(cuisine="North Indian", top_n=3)
        if not result.empty:
            assert "explanation" in result.columns

    def test_recommend_has_combined_score(self, sample_df):
        """Result should include a combined_score column."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(cuisine="North Indian", top_n=3)
        if not result.empty:
            assert "combined_score" in result.columns

    def test_top_n_respected(self, sample_df):
        """Should return at most top_n results."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(top_n=2)
        assert len(result) <= 2

    def test_different_prioritize_modes(self, sample_df):
        """Different prioritize modes should return results."""
        rec = HybridRecommender(sample_df, ["votes_log", "cost_for_two"])
        for mode in ["balanced", "speed", "quality"]:
            result = rec.recommend(prioritize=mode, top_n=3)
            assert isinstance(result, pd.DataFrame)


class TestGetWeightProfile:
    """Tests for _get_weight_profile()."""

    @pytest.fixture
    def recommender(self):
        """Create a minimal HybridRecommender."""
        df = pd.DataFrame({
            "name": ["Test"],
            "rate": [4.0],
            "votes": [100],
            "cost_for_two": [500],
        })
        return HybridRecommender(df, ["cost_for_two"])

    def test_balanced_has_all_weights(self, recommender):
        """Balanced profile should have content, collaborative, reliability, bayesian keys."""
        weights = recommender._get_weight_profile("balanced")
        for key in ["content", "collaborative", "reliability", "bayesian"]:
            assert key in weights

    def test_speed_profile_higher_reliability(self, recommender):
        """Speed profile should weight reliability higher than balanced."""
        balanced = recommender._get_weight_profile("balanced")
        speed = recommender._get_weight_profile("speed")
        assert speed["reliability"] > balanced["reliability"]

    def test_quality_profile_higher_bayesian(self, recommender):
        """Quality profile should weight bayesian higher than speed."""
        quality = recommender._get_weight_profile("quality")
        speed = recommender._get_weight_profile("speed")
        assert quality["bayesian"] > speed["bayesian"]

    def test_unknown_profile_falls_back_to_balanced(self, recommender):
        """Unknown prioritize mode should fall back to balanced."""
        balanced = recommender._get_weight_profile("balanced")
        unknown = recommender._get_weight_profile("unknown_mode")
        assert balanced == unknown
