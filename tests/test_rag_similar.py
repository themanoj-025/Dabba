"""Tests for RAG similar-restaurant retrieval — FAISS and sklearn fallback."""

import numpy as np
import pandas as pd
import pytest

from dabba.llm.rag_similar_restaurants import (
    build_restaurant_embeddings,
    find_similar_restaurants,
)


class TestBuildRestaurantEmbeddings:
    """Tests for build_restaurant_embeddings()."""

    @pytest.fixture
    def sample_df(self):
        """Create a small restaurant DataFrame with feature columns."""
        return pd.DataFrame({
            "name": ["Rest A", "Rest B", "Rest C", "Rest D"],
            "rate": [4.5, 3.8, 4.2, 3.5],
            "cost_for_two": [500, 300, 800, 200],
            "votes_log": [6.2, 3.9, 5.7, 2.4],
            "online_order_binary": [1, 1, 0, 1],
        })

    def test_returns_numpy_array(self, sample_df):
        """Should return a numpy array."""
        embeddings = build_restaurant_embeddings(
            sample_df, ["rate", "cost_for_two", "votes_log"]
        )
        assert isinstance(embeddings, np.ndarray)

    def test_shape_matches_rows_and_features(self, sample_df):
        """Shape should be (n_restaurants, n_features)."""
        n_features = 3
        embeddings = build_restaurant_embeddings(
            sample_df, ["rate", "cost_for_two", "votes_log"]
        )
        assert embeddings.shape == (len(sample_df), n_features)

    def test_normalized_vectors(self, sample_df):
        """L2 norm of each embedding should be approximately 1."""
        embeddings = build_restaurant_embeddings(
            sample_df, ["rate", "cost_for_two", "votes_log"]
        )
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_empty_feature_cols(self, sample_df):
        """Empty feature columns should return a zero matrix."""
        embeddings = build_restaurant_embeddings(sample_df, [])
        assert embeddings.shape == (len(sample_df), 1)

    def test_partial_feature_cols(self, sample_df):
        """Only existing columns should be used."""
        embeddings = build_restaurant_embeddings(
            sample_df, ["rate", "cost_for_two", "nonexistent_col"]
        )
        assert embeddings.shape == (len(sample_df), 2)

    def test_deterministic(self, sample_df):
        """Same input should produce same embeddings."""
        e1 = build_restaurant_embeddings(
            sample_df, ["rate", "cost_for_two"]
        )
        e2 = build_restaurant_embeddings(
            sample_df, ["rate", "cost_for_two"]
        )
        assert np.allclose(e1, e2)


class TestFindSimilarRestaurants:
    """Tests for find_similar_restaurants()."""

    @pytest.fixture
    def sample_df(self):
        """Create a restaurant DataFrame with distinct restaurants."""
        rng = np.random.RandomState(42)
        n = 10
        return pd.DataFrame({
            "name": [f"Restaurant {i}" for i in range(n)],
            "rate": rng.uniform(3.0, 5.0, n),
            "cost_for_two": rng.randint(200, 1500, n),
            "location": rng.choice(["Koramangala", "Indiranagar", "MG Road"], n),
            "cuisines": rng.choice(["North Indian", "Chinese", "Italian"], n),
        })

    @pytest.fixture
    def embeddings(self, sample_df):
        """Build embeddings for the sample DataFrame."""
        return build_restaurant_embeddings(
            sample_df, ["rate", "cost_for_two"]
        )

    def test_returns_dataframe(self, sample_df, embeddings):
        """Should return a DataFrame."""
        result = find_similar_restaurants(0, sample_df, embeddings, top_k=3)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_number(self, sample_df, embeddings):
        """Should return top_k similar restaurants (excluding itself)."""
        result = find_similar_restaurants(0, sample_df, embeddings, top_k=3)
        assert len(result) == 3

    def test_does_not_include_self(self, sample_df, embeddings):
        """The query restaurant should not appear in results."""
        result = find_similar_restaurants(0, sample_df, embeddings, top_k=5)
        assert sample_df.iloc[0]["name"] not in result["name"].values

    def test_has_similarity_score_column(self, sample_df, embeddings):
        """Result should have a similarity_score column."""
        result = find_similar_restaurants(0, sample_df, embeddings, top_k=3)
        assert "similarity_score" in result.columns

    def test_scores_in_zero_to_one(self, sample_df, embeddings):
        """Similarity scores should be in (0, 1] range."""
        result = find_similar_restaurants(0, sample_df, embeddings, top_k=3)
        for score in result["similarity_score"]:
            assert 0 < score <= 1.0

    def test_sorted_by_score_descending(self, sample_df, embeddings):
        """Results should be sorted by similarity score descending."""
        result = find_similar_restaurants(0, sample_df, embeddings, top_k=5)
        scores = result["similarity_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_different_query_different_results(self, sample_df, embeddings):
        """Different query indices should return different results (usually)."""
        result_0 = find_similar_restaurants(0, sample_df, embeddings, top_k=3)
        result_1 = find_similar_restaurants(1, sample_df, embeddings, top_k=3)
        # At most 1 overlap between the two result sets (unlikely but possible)
        names_0 = set(result_0["name"].values)
        names_1 = set(result_1["name"].values)
        # They might share some but shouldn't be identical
        assert names_0 != names_1 or len(names_0) < 3

    def test_top_k_greater_than_available(self, sample_df, embeddings):
        """When top_k > available restaurants, should not crash."""
        small_df = sample_df.head(3)
        small_emb = embeddings[:3]
        result = find_similar_restaurants(0, small_df, small_emb, top_k=10)
        assert len(result) == 2  # 3 restaurants - 1 (self) = 2 max
