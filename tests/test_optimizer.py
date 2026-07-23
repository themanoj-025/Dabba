"""Tests for delivery partner assignment optimizer."""

import numpy as np
import pandas as pd
import pytest

from dabba.models.optimizer import (
    compare_assignment_strategies,
    naive_assignments,
    optimize_assignments,
)


class TestOptimizeAssignments:
    """Tests for the Hungarian algorithm optimizer."""

    def test_returns_assignment_and_total(self):
        """Should return an assignment array and total time."""
        cost = np.array([[5.0, 9.0], [10.0, 6.0]])
        assignment, total = optimize_assignments(cost)
        assert isinstance(assignment, np.ndarray)
        assert isinstance(total, float)

    def test_optimal_assignment_2x2(self):
        """Should find the optimal assignment for a 2x2 matrix."""
        cost = np.array([[5.0, 9.0], [10.0, 6.0]])
        assignment, total = optimize_assignments(cost)
        # Optimal: order0→partner0 (5), order1→partner1 (6) = 11
        assert total == pytest.approx(11.0)

    def test_optimal_assignment_3x3(self):
        """Should find optimal assignment for a 3x3 matrix."""
        cost = np.array([
            [10.0, 19.0, 8.0],
            [10.0, 7.0, 10.0],
            [13.0, 13.0, 14.0],
        ])
        assignment, total = optimize_assignments(cost)
        assert total > 0
        assert len(assignment) == 3

    def test_assignment_count_matches_orders(self):
        """Number of assignments should match min(orders, partners)."""
        cost = np.random.RandomState(42).rand(5, 3)
        assignment, total = optimize_assignments(cost)
        assert len(assignment) == min(*cost.shape)

    def test_single_order(self):
        """Single order should return the cheapest partner."""
        cost = np.array([[10.0, 5.0, 8.0]])
        assignment, total = optimize_assignments(cost)
        assert assignment[0] == 1  # partner 1 is cheapest at 5.0
        assert total == pytest.approx(5.0)

    def test_square_matrix(self):
        """Square matrix should match each partner to exactly one order."""
        cost = np.array([
            [5.0, 9.0, 9.0],
            [9.0, 5.0, 9.0],
            [9.0, 9.0, 5.0],
        ])  # diagonal cheapest at 5, off-diagonals at 9
        assignment, total = optimize_assignments(cost)
        assert total == pytest.approx(15.0)  # all three on diagonal
        assert len(set(assignment)) == 3  # each partner assigned once


class TestNaiveAssignments:
    """Tests for the naive (first-available) assignment strategy."""

    def test_returns_total_time(self):
        """Should return a float total time."""
        cost = np.array([[5.0, 9.0], [10.0, 6.0]])
        total = naive_assignments(cost)
        assert isinstance(total, float)

    def test_naive_worse_or_equal_to_optimal(self):
        """Naive assignment should be >= optimal assignment."""
        cost = np.random.RandomState(42).rand(4, 3)
        _, optimal = optimize_assignments(cost)
        naive = naive_assignments(cost)
        assert naive >= optimal - 1e-10

    def test_single_order_single_partner(self):
        """Single order with one partner should return that cost."""
        cost = np.array([[15.0]])
        total = naive_assignments(cost)
        assert total == pytest.approx(15.0)

    def test_orders_fewer_than_partners(self):
        """Should handle more partners than orders."""
        cost = np.random.RandomState(42).rand(2, 5)
        total = naive_assignments(cost)
        assert total > 0


class TestCompareAssignmentStrategies:
    """Tests for compare_assignment_strategies()."""

    class MockModel:
        """A mock ETA model that returns predictions from a feature column."""

        def predict(self, X):
            return X.iloc[:, 0].values * 2 + 5

    @pytest.fixture
    def orders_df(self):
        """Create a sample orders DataFrame."""
        return pd.DataFrame({
            "distance_km": [5.0, 10.0, 3.0],
            "traffic_level": [1, 2, 0],
        })

    def test_returns_dict_with_keys(self, orders_df):
        """Should return dict with expected keys."""
        model = self.MockModel()
        result = compare_assignment_strategies(
            orders_df, model, ["distance_km", "traffic_level"]
        )
        assert isinstance(result, dict)
        assert "optimized_total_min" in result
        assert "naive_total_min" in result
        assert "improvement_pct" in result

    def test_optimized_less_than_or_equal_naive(self, orders_df):
        """Optimized total should be <= naive total."""
        model = self.MockModel()
        result = compare_assignment_strategies(
            orders_df, model, ["distance_km", "traffic_level"]
        )
        assert result["optimized_total_min"] <= result["naive_total_min"] + 0.01

    def test_improvement_positive_or_zero(self, orders_df):
        """Improvement should be >= 0."""
        model = self.MockModel()
        result = compare_assignment_strategies(
            orders_df, model, ["distance_km", "traffic_level"]
        )
        assert result["improvement_pct"] >= 0

    def test_empty_dataframe(self):
        """Empty DataFrame should return zeros."""
        model = self.MockModel()
        df_empty = pd.DataFrame(columns=["distance_km"])
        result = compare_assignment_strategies(df_empty, model, ["distance_km"])
        assert result["optimized_total_min"] == 0
        assert result["naive_total_min"] == 0
