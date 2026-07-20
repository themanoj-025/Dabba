"""Tests for feature engineering utilities."""

import numpy as np
import pandas as pd
import pytest

from dabba.features.geo import (
    BANGALORE_CENTROIDS,
    compare_clustering_methods,
    geocode_location,
    haversine_distance,
)
from dabba.features.restaurant_features import (
    add_restaurant_features,
    encode_cuisines,
)
from dabba.features.delivery_features import add_delivery_features


class TestHaversineDistance:
    """Tests for the haversine distance calculation."""

    def test_same_point(self):
        """Distance from a point to itself should be zero."""
        result = haversine_distance(12.97, 77.59, 12.97, 77.59)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self):
        """Test a known distance (Koramangala to MG Road ~3 km)."""
        lat1, lon1 = BANGALORE_CENTROIDS["Koramangala"]
        lat2, lon2 = BANGALORE_CENTROIDS["MG Road"]
        result = haversine_distance(lat1, lon1, lat2, lon2)
        # Should be roughly 3-4 km
        assert 2.0 < result < 5.0

    def test_array_input(self):
        """Should work with numpy arrays."""
        lats = np.array([12.97, 12.97])
        lons = np.array([77.59, 77.64])
        result = haversine_distance(lats, lons, lats, lons)
        assert len(result) == 2
        assert all(r == pytest.approx(0.0, abs=0.01) for r in result)


class TestGeocodeLocation:
    """Tests for Bangalore neighborhood geocoding."""

    def test_exact_match(self):
        """Exact neighborhood name should return coordinates."""
        result = geocode_location("Koramangala")
        assert result is not None
        assert len(result) == 2
        assert 6 < result[0] < 37  # Valid lat range for India

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        result = geocode_location("koramangala")
        assert result is not None

    def test_unknown_location(self):
        """Unknown location should return None."""
        result = geocode_location("Atlantis")
        assert result is None


class TestClusteringMethods:
    """Tests for clustering comparison."""

    def test_returns_all_methods(self):
        """Should return results for KMeans, DBSCAN, and Agglomerative."""
        X = np.random.RandomState(42).rand(100, 2)
        results = compare_clustering_methods(X, k_range=range(2, 6))
        assert "KMeans" in results
        assert "Agglomerative" in results
        # DBSCAN may or may not find clusters depending on data

    def test_silhouette_scores(self):
        """Silhouette scores should be between -1 and 1."""
        X = np.random.RandomState(42).rand(100, 2)
        results = compare_clustering_methods(X, k_range=range(2, 6))
        for method, info in results.items():
            if "silhouette_score" in info:
                assert -1 <= info["silhouette_score"] <= 1


class TestRestaurantFeatures:
    """Tests for restaurant feature engineering."""

    def test_encode_cuisines(self):
        """Should create binary columns for top cuisines."""
        df = pd.DataFrame(
            {
                "cuisines": ["North Indian, Chinese", "Italian", "North Indian"],
            }
        )
        result = encode_cuisines(df, top_n=5)
        assert "cuisine_north_indian" in result.columns
        assert result["cuisine_north_indian"].iloc[0] == 1

    def test_add_restaurant_features(self):
        """Should add all expected feature columns."""
        df = pd.DataFrame(
            {
                "cost_for_two": [500, 1200, 300],
                "cuisines": ["North Indian, Chinese", "Italian", "Biryani"],
                "online_order": ["Yes", "No", "Yes"],
                "book_table": ["Yes", "No", "No"],
                "votes": [100, 500, 10],
            }
        )
        result = add_restaurant_features(df)
        assert "cost_for_two_bucket" in result.columns
        assert "cuisine_count" in result.columns
        assert "online_order_binary" in result.columns
        assert "book_table_binary" in result.columns
        assert "votes_log" in result.columns


class TestDeliveryFeatures:
    """Tests for delivery feature engineering."""

    def test_add_delivery_features(self):
        """Should add expected feature columns."""
        df = pd.DataFrame(
            {
                "time_taken_min": [25.0, 30.0, 35.0],
                "restaurant_latitude": [12.9, 12.95, 12.91],
                "restaurant_longitude": [77.6, 77.65, 77.59],
                "delivery_location_latitude": [12.95, 13.0, 12.93],
                "delivery_location_longitude": [77.65, 77.7, 77.62],
                "order_date": ["2024-01-15", "2024-01-16", "2024-01-17"],
                "festival": ["No", "Yes", "No"],
                "road_traffic_density": ["Low", "High", "Medium"],
            }
        )
        result = add_delivery_features(df)
        assert "haversine_distance_km" in result.columns
        assert "is_festival" in result.columns
        assert "traffic_ordinal" in result.columns
