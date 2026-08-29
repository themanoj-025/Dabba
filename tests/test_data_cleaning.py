"""Tests for dabba.data.cleaning — Zomato rating and cost parsing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestCleanZomatoRating:
    """clean_zomato_rating parses messy rating strings."""

    def test_clean_valid_ratings(self) -> None:
        from dabba.data.cleaning import clean_zomato_rating

        series = pd.Series(["4.1/5", "3.5 /5", "4.8/5"])
        result = clean_zomato_rating(series)
        assert result.iloc[0] == pytest.approx(4.1)
        assert result.iloc[1] == pytest.approx(3.5)
        assert result.iloc[2] == pytest.approx(4.8)

    def test_new_rating_becomes_nan(self) -> None:
        from dabba.data.cleaning import clean_zomato_rating

        series = pd.Series(["NEW", "4.1/5"])
        result = clean_zomato_rating(series)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(4.1)

    def test_dash_becomes_nan(self) -> None:
        from dabba.data.cleaning import clean_zomato_rating

        series = pd.Series(["-", "4.1/5"])
        result = clean_zomato_rating(series)
        assert pd.isna(result.iloc[0])

    def test_nan_input_stays_nan(self) -> None:
        from dabba.data.cleaning import clean_zomato_rating

        series = pd.Series([np.nan, "4.1/5"])
        result = clean_zomato_rating(series)
        assert pd.isna(result.iloc[0])

    def test_numeric_input_becomes_nan(self) -> None:
        from dabba.data.cleaning import clean_zomato_rating

        series = pd.Series([4.1, "4.1/5"])
        result = clean_zomato_rating(series)
        # numeric 4.1 -> str "4.1" doesn't match "X/5" pattern -> NaN
        assert pd.isna(result.iloc[0])

    def test_empty_string_becomes_nan(self) -> None:
        from dabba.data.cleaning import clean_zomato_rating

        series = pd.Series(["", "4.1/5"])
        result = clean_zomato_rating(series)
        assert pd.isna(result.iloc[0])

    def test_all_valid(self) -> None:
        from dabba.data.cleaning import clean_zomato_rating

        series = pd.Series(["1.0/5", "2.5/5", "5.0/5"])
        result = clean_zomato_rating(series)
        assert result.isna().sum() == 0
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[2] == pytest.approx(5.0)


class TestCleanZomatoCost:
    """clean_zomato_cost parses cost-for-two strings."""

    def test_clean_valid_cost(self) -> None:
        from dabba.data.cleaning import clean_zomato_cost

        series = pd.Series(["1,200", "300", "₹500"])
        result = clean_zomato_cost(series)
        assert result.iloc[0] == pytest.approx(1200.0)
        assert result.iloc[1] == pytest.approx(300.0)

    def test_cost_with_rupee_symbol(self) -> None:
        from dabba.data.cleaning import clean_zomato_cost

        series = pd.Series(["₹1,500"])
        result = clean_zomato_cost(series)
        assert result.iloc[0] == pytest.approx(1500.0)

    def test_non_numeric_becomes_nan(self) -> None:
        from dabba.data.cleaning import clean_zomato_cost

        series = pd.Series(["abc", "300"])
        result = clean_zomato_cost(series)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(300.0)
