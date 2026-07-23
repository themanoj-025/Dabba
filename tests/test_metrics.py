"""Tests for model evaluation metrics."""

import numpy as np
import pytest

from dabba.evaluation.metrics import compute_regression_metrics


class TestComputeRegressionMetrics:
    """Tests for the compute_regression_metrics function."""

    def test_returns_dict(self):
        """Should return a dict with mae, rmse, r2 keys."""
        y_true = np.array([3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 4.0, 5.0])
        result = compute_regression_metrics(y_true, y_pred)
        assert isinstance(result, dict)
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result

    def test_perfect_prediction(self):
        """Perfect predictions should yield MAE=0, RMSE=0, R²=1."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["r2"] == 1.0

    def test_constant_prediction(self):
        """R² should be 0 when predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())
        result = compute_regression_metrics(y_true, y_pred)
        assert result["r2"] == 0.0

    def test_prefix_applied(self):
        """Prefix should be prepended to metric names."""
        y_true = np.array([3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 4.0, 5.0])
        result = compute_regression_metrics(y_true, y_pred, prefix="eta_")
        assert "eta_mae" in result
        assert "eta_rmse" in result
        assert "eta_r2" in result

    def test_mae_non_negative(self):
        """MAE should always be non-negative."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["mae"] >= 0

    def test_rmse_greater_than_mae(self):
        """RMSE should be >= MAE for non-constant errors."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 2.0, 4.5, 5.5])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["rmse"] >= result["mae"]

    def test_negative_r2(self):
        """R² can be negative for models worse than the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["r2"] < 0

    def test_rounding_precision(self):
        """Metrics should be rounded to 4 decimal places."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_regression_metrics(y_true, y_pred)
        for val in result.values():
            string = str(val)
            # Check at most 4 decimal places
            if "." in string:
                decimals = len(string.split(".")[1])
                assert decimals <= 4, f"Too many decimals: {val}"
