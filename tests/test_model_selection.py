"""Unit tests for model_selection.py — critical test protecting the automatic
model selection logic from silently breaking.
"""

import pandas as pd

from dabba.models.model_selection import (
    comparison_to_dataframe,
    select_best_model,
    save_comparison_csv,
)


class MockResult:
    """Lightweight mock for ModelResult/ETAModelResult."""

    def __init__(self, name, mae, rmse, r2, train_time=1.0):
        self.name = name
        self.mae = mae
        self.rmse = rmse
        self.r2 = r2
        self.train_time = train_time
        self.predictions = None


class TestComparisonToDataframe:
    """Tests for converting results to a comparison DataFrame."""

    def test_creates_dataframe(self):
        """Should return a DataFrame with expected columns."""
        results = [
            MockResult("A", 5.0, 7.0, 0.8),
            MockResult("B", 3.0, 4.0, 0.9),
        ]
        df = comparison_to_dataframe(results, task="test")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"model", "mae", "rmse", "r2", "train_time_s"}

    def test_sorted_by_mae(self):
        """Results should be sorted by MAE ascending."""
        results = [
            MockResult("A", 5.0, 7.0, 0.8),
            MockResult("B", 3.0, 4.0, 0.9),
            MockResult("C", 4.0, 5.5, 0.85),
        ]
        df = comparison_to_dataframe(results)
        assert df["model"].iloc[0] == "B"
        assert df["model"].iloc[1] == "C"
        assert df["model"].iloc[2] == "A"


class TestSelectBestModel:
    """Tests for automatic best model selection."""

    def test_selects_lowest_mae(self):
        """Should select the model with lowest MAE by default."""
        results = [
            MockResult("LinearRegression", 5.0, 7.0, 0.8),
            MockResult("XGBoost", 2.5, 3.5, 0.95),
            MockResult("RandomForest", 3.0, 4.0, 0.9),
        ]
        best = select_best_model(results, metric="mae")
        assert best == "XGBoost"

    def test_selects_lowest_rmse(self):
        """Should select by RMSE when metric='rmse'."""
        results = [
            MockResult("A", 5.0, 8.0, 0.8),
            MockResult("B", 4.0, 3.0, 0.9),
            MockResult("C", 3.0, 5.0, 0.85),
        ]
        best = select_best_model(results, metric="rmse")
        assert best == "B"

    def test_selects_highest_r2(self):
        """Should select highest R² when metric='r2'."""
        results = [
            MockResult("A", 5.0, 7.0, 0.7),
            MockResult("B", 4.0, 6.0, 0.95),
            MockResult("C", 3.0, 5.0, 0.85),
        ]
        best = select_best_model(results, metric="r2")
        assert best == "B"

    def test_empty_results(self):
        """Should return None for empty results."""
        best = select_best_model([])
        assert best is None

    def test_single_model(self):
        """Should select the only model when there's just one."""
        results = [MockResult("OnlyModel", 5.0, 7.0, 0.8)]
        best = select_best_model(results)
        assert best == "OnlyModel"


class TestSaveComparisonCsv:
    """Tests for saving comparison CSV."""

    def test_saves_csv(self, tmp_path):
        """Should save a valid CSV file."""
        df = pd.DataFrame({
            "model": ["A", "B"],
            "mae": [3.0, 4.0],
            "rmse": [4.0, 5.0],
            "r2": [0.9, 0.8],
            "train_time_s": [1.0, 0.5],
        })
        path = tmp_path / "comparison.csv"
        save_comparison_csv(df, path)
        assert path.exists()

        loaded = pd.read_csv(path)
        assert len(loaded) == 2
        assert loaded["model"].iloc[0] == "A"
