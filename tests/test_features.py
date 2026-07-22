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


class TestCyclicalEncode:
    """Tests for cyclical encoding helper."""

    def test_hour_encoding(self):
        """Hours 0 and 24 should produce same sin/cos values."""
        from dabba.features.delivery_features import cyclical_encode
        sin_0, cos_0 = cyclical_encode(np.array([0]), period=24)
        sin_24, cos_24 = cyclical_encode(np.array([24]), period=24)
        assert sin_0[0] == pytest.approx(sin_24[0], abs=1e-10)
        assert cos_0[0] == pytest.approx(cos_24[0], abs=1e-10)

    def test_hour_6_and_18_opposite(self):
        """Hours 6 and 18 should have opposite sin values."""
        from dabba.features.delivery_features import cyclical_encode
        sin_6, _ = cyclical_encode(np.array([6]), period=24)
        sin_18, _ = cyclical_encode(np.array([18]), period=24)
        assert sin_6[0] == pytest.approx(-sin_18[0], abs=1e-10)


class TestCityZone:
    """Tests for Bangalore city zone assignment."""

    def test_central_zone(self):
        """MG Road area should be central."""
        from dabba.features.delivery_features import _assign_city_zone
        zone = _assign_city_zone(12.97, 77.61)
        assert zone == "central"

    def test_north_zone(self):
        """Yelahanka area should be north."""
        from dabba.features.delivery_features import _assign_city_zone
        zone = _assign_city_zone(13.10, 77.57)
        assert zone == "north"

    def test_unknown_zone(self):
        """NaN coordinates should return unknown."""
        from dabba.features.delivery_features import _assign_city_zone
        zone = _assign_city_zone(float("nan"), 77.6)
        assert zone == "unknown"


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
        # Existing features
        assert "haversine_distance_km" in result.columns
        assert "is_festival" in result.columns
        assert "traffic_ordinal" in result.columns
        # New temporal features
        assert "order_hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "is_rush_hour" in result.columns
        # New cyclical encoding
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns
        # New interaction features
        assert "distance_traffic_interaction" in result.columns
        assert "distance_festival_interaction" in result.columns
        # New spatial features
        assert "city_zone" in result.columns

    def test_rush_hour_flag(self):
        """Hour 9 should be rush hour, hour 3 should not."""
        df = pd.DataFrame(
            {
                "time_taken_min": [30.0, 30.0],
                "restaurant_latitude": [12.97, 12.97],
                "restaurant_longitude": [77.61, 77.61],
                "delivery_location_latitude": [12.99, 12.99],
                "delivery_location_longitude": [77.63, 77.63],
                "order_date": ["2024-01-15 09:00", "2024-01-16 03:00"],
                "road_traffic_density": ["Low", "Low"],
            }
        )
        result = add_delivery_features(df)
        assert result["is_rush_hour"].iloc[0] == 1  # 9 AM = rush
        assert result["is_rush_hour"].iloc[1] == 0  # 3 AM = not rush

    def test_traffic_interaction(self):
        """Distance × traffic should be higher with more traffic."""
        df = pd.DataFrame(
            {
                "time_taken_min": [30.0, 30.0],
                "restaurant_latitude": [12.97, 12.97],
                "restaurant_longitude": [77.61, 77.61],
                "delivery_location_latitude": [12.99, 12.99],
                "delivery_location_longitude": [77.63, 77.63],
                "order_date": ["2024-01-15", "2024-01-16"],
                "road_traffic_density": ["Low", "Jam"],
            }
        )
        result = add_delivery_features(df)
        # Same distance, more traffic → higher interaction value
        assert (
            result["distance_traffic_interaction"].iloc[1]
            > result["distance_traffic_interaction"].iloc[0]
        )
