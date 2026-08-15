"""Tests for data cleaning utilities."""

import numpy as np
import pandas as pd
import pytest

from dabba.data.cleaning import (
    clean_delivery,
    clean_zomato,
    clean_zomato_cost,
    clean_zomato_rating,
)


class TestCleanZomatoRating:
    """Tests for the Zomato rating parser."""

    def test_valid_ratings(self):
        """Standard 'X.X/5' format should parse correctly."""
        series = pd.Series(["4.1/5", "3.5/5", "5.0/5"])
        result = clean_zomato_rating(series)
        assert result.tolist() == [4.1, 3.5, 5.0]

    def test_new_and_dash_sentinels(self):
        """Non-numeric sentinels should become NaN."""
        series = pd.Series(["NEW", "-", "4.1/5"])
        result = clean_zomato_rating(series)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 4.1

    def test_null_input(self):
        """NaN input should remain NaN."""
        series = pd.Series([np.nan, None])
        result = clean_zomato_rating(series)
        assert result.isna().all()

    def test_whitespace_handling(self):
        """Extra whitespace should be handled gracefully."""
        series = pd.Series(["  4.1 / 5  ", "3.5/5"])
        result = clean_zomato_rating(series)
        assert result.iloc[0] == pytest.approx(4.1)
        assert result.iloc[1] == pytest.approx(3.5)


class TestCleanZomatoCost:
    """Tests for the Zomato cost parser."""

    def test_comma_separated(self):
        """Cost with commas should parse correctly."""
        series = pd.Series(["1,200", "300", "5,500"])
        result = clean_zomato_cost(series)
        assert result.tolist() == [1200.0, 300.0, 5500.0]

    def test_rupee_symbol(self):
        """Cost with ₹ symbol should parse correctly."""
        series = pd.Series(["₹400", "₹1,500"])
        result = clean_zomato_cost(series)
        assert result.iloc[0] == 400.0
        assert result.iloc[1] == 1500.0

    def test_null_and_invalid(self):
        """Null and unparseable values should become NaN."""
        series = pd.Series([np.nan, "abc", ""])
        result = clean_zomato_cost(series)
        assert result.isna().all()


class TestCleanZomato:
    """Tests for the full Zomato cleaning pipeline."""

    def test_removes_duplicates(self):
        """Duplicate rows should be removed."""
        df = pd.DataFrame(
            {
                "rate": ["4.1/5", "4.1/5", "3.0/5"],
                "approx_cost(for two people)": ["500", "500", "300"],
                "name": ["A", "A", "B"],
            }
        )
        result = clean_zomato(df)
        assert len(result) == 2

    def test_drops_missing_target(self):
        """Rows with missing rate should be dropped."""
        df = pd.DataFrame(
            {
                "rate": ["4.1/5", None, "3.0/5"],
                "approx_cost(for two people)": ["500", "300", "300"],
                "name": ["A", "B", "C"],
            }
        )
        result = clean_zomato(df)
        assert len(result) == 2

    def test_snake_case_columns(self):
        """Column names should be normalized to snake_case."""
        df = pd.DataFrame(
            {
                "rate": ["4.1/5"],
                "approx_cost(for two people)": ["500"],
                "Online Order": ["Yes"],
            }
        )
        result = clean_zomato(df)
        assert "online_order" in result.columns
        assert "cost_for_two" in result.columns


class TestCleanDelivery:
    """Tests for the delivery data cleaning pipeline."""

    def test_parses_time_taken(self):
        """Time_taken column should be parsed to float."""
        df = pd.DataFrame(
            {
                "Time_taken(min)": ["25 min", "30 min", "NAN"],
                "Restaurant_latitude": [12.9, 12.9, 12.9],
                "Restaurant_longitude": [77.6, 77.6, 77.6],
                "Delivery_location_latitude": [12.95, 12.95, 12.95],
                "Delivery_location_longitude": [77.65, 77.65, 77.65],
            }
        )
        result = clean_delivery(df)
        assert "time_taken_min" in result.columns
        assert result["time_taken_min"].iloc[0] == 25.0

    def test_removes_invalid_latlong(self):
        """Rows with invalid lat/long should be removed."""
        df = pd.DataFrame(
            {
                "Time_taken(min)": ["25", "30", "35"],
                "Restaurant_latitude": [12.9, 999.0, 12.9],
                "Restaurant_longitude": [77.6, 77.6, 77.6],
                "Delivery_location_latitude": [12.95, 12.95, 12.95],
                "Delivery_location_longitude": [77.65, 77.65, 77.65],
            }
        )
        result = clean_delivery(df)
        assert len(result) == 2

    def test_removes_extreme_times(self):
        """Delivery times > 120 min should be removed as outliers."""
        df = pd.DataFrame(
            {
                "Time_taken(min)": ["25", "150", "30"],
                "Restaurant_latitude": [12.9, 12.9, 12.9],
                "Restaurant_longitude": [77.6, 77.6, 77.6],
                "Delivery_location_latitude": [12.95, 12.95, 12.95],
                "Delivery_location_longitude": [77.65, 77.65, 77.65],
            }
        )
        result = clean_delivery(df)
        assert len(result) == 2
