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


class TestRecommenderSimilarity:
    """Tests for the _compute_similarity method."""

    @pytest.fixture
    def recommender(self):
        """Create a recommender with a small DataFrame."""
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "rate": [4.0, 3.5, 4.5],
                "votes": [100, 50, 200],
                "cost_for_two": [500, 300, 800],
                "location": ["X", "Y", "Z"],
                "votes_log": [4.6, 3.9, 5.3],
                "online_order_binary": [1, 0, 1],
            }
        )
        return RestaurantRecommender(df, ["votes_log", "cost_for_two"])

    def test_returns_numpy_array(self, recommender):
        """Should return a numpy array."""
        query = np.array([5.0, 600.0])
        sim = recommender._compute_similarity(query)
        assert isinstance(sim, np.ndarray)

    def test_length_matches_dataframe(self, recommender):
        """Should return one score per restaurant."""
        query = np.array([5.0, 600.0])
        sim = recommender._compute_similarity(query)
        assert len(sim) == len(recommender.df)

    def test_similarity_between_zero_and_one(self, recommender):
        """Cosine similarity should be in [0, 1] for non-negative features."""
        query = np.array([5.0, 600.0])
        sim = recommender._compute_similarity(query)
        assert all(-0.01 <= s <= 1.01 for s in sim)

    def test_identical_query_returns_one(self, recommender):
        """Query equal to a restaurant should yield similarity ~1."""
        query = recommender.feature_matrix[0]
        sim = recommender._compute_similarity(query)
        assert sim[0] == pytest.approx(1.0, abs=1e-5)


class TestRecommenderLoadModel:
    """Tests for load_rating_model()."""

    @pytest.fixture
    def recommender(self):
        """Create a basic recommender."""
        df = pd.DataFrame({"name": ["A"], "rate": [4.0], "votes": [100], "cost_for_two": [500]})
        return RestaurantRecommender(df, ["cost_for_two"])

    def test_model_none_initially(self, recommender):
        """Model should be None before loading."""
        assert recommender.rating_model is None

    def test_load_nonexistent_path(self, recommender):
        """Loading a nonexistent path should not crash."""
        recommender.load_rating_model(model_path="/nonexistent/path/model.pkl")
        assert recommender.rating_model is None
