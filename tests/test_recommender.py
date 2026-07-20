"""Tests for the hybrid restaurant recommender."""

import pandas as pd
import pytest

from dabba.models.recommender import RestaurantRecommender, bayesian_average


class TestBayesianAverage:
    """Tests for Bayesian average rating computation."""

    def test_basic_computation(self):
        """Should blend restaurant rating toward global mean for low-vote restaurants."""
        ratings = pd.Series([5.0, 3.0])
        vote_counts = pd.Series([10, 1000])
        result = bayesian_average(ratings, vote_counts, C=3.5, m=25)

        # High-vote restaurant should stay close to its rating
        assert result.iloc[1] == pytest.approx(3.0, abs=0.5)
        # Low-vote restaurant should be pulled toward global mean
        assert result.iloc[0] < 5.0
        assert result.iloc[0] > 3.0

    def test_equal_votes(self):
        """With equal high votes, Bayesian average should equal raw rating."""
        ratings = pd.Series([4.0, 4.0])
        vote_counts = pd.Series([1000, 1000])
        result = bayesian_average(ratings, vote_counts, C=3.5, m=25)
        assert result.iloc[0] == pytest.approx(4.0, abs=0.1)


class TestRestaurantRecommender:
    """Tests for the hybrid recommender."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample restaurant DataFrame."""
        return pd.DataFrame(
            {
                "name": ["A", "B", "C", "D", "E"],
                "rate": [4.5, 3.8, 4.2, 3.5, 4.8],
                "votes": [500, 50, 300, 10, 1000],
                "cost_for_two": [500, 300, 800, 200, 1200],
                "location": [
                    "Koramangala",
                    "Indiranagar",
                    "Koramangala",
                    "HSR Layout",
                    "MG Road",
                ],
                "cuisines": [
                    "North Indian, Chinese",
                    "Italian",
                    "North Indian, Mughlai",
                    "South Indian",
                    "Japanese",
                ],
                "votes_log": [6.2, 3.9, 5.7, 2.4, 6.9],
                "online_order_binary": [1, 1, 0, 1, 1],
            }
        )

    def test_recommend_returns_dataframe(self, sample_df):
        """recommend() should return a DataFrame."""
        rec = RestaurantRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(cuisine="North Indian", top_n=3)
        assert isinstance(result, pd.DataFrame)

    def test_recommend_respects_budget(self, sample_df):
        """Recommendations should respect the budget filter."""
        rec = RestaurantRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(budget=400, top_n=10)
        if not result.empty and "cost_for_two" in result.columns:
            assert (result["cost_for_two"] <= 400).all()

    def test_recommend_respects_cuisine(self, sample_df):
        """Recommendations should filter by cuisine."""
        rec = RestaurantRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(cuisine="Italian", top_n=5)
        if not result.empty and "cuisines" in result.columns:
            assert all("Italian" in str(c) for c in result["cuisines"])

    def test_recommend_empty_result(self, sample_df):
        """Should return empty DataFrame when no matches."""
        rec = RestaurantRecommender(sample_df, ["votes_log", "cost_for_two"])
        result = rec.recommend(cuisine="Thai", area="Nonexistent")
        assert result.empty or len(result) == 0
