"""Tests for drift detection module."""

import numpy as np
import pandas as pd
import pytest

from dabba.monitoring.drift import DriftDetector


@pytest.fixture
def reference_data():
    """Create reference data with known distribution."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 500),
            "feature_b": rng.uniform(0, 10, 500),
            "feature_c": rng.poisson(5, 500),
        }
    )


class TestDriftDetector:
    """Tests for the DriftDetector class."""

    def test_fit_extracts_stats(self, reference_data):
        """fit() should extract reference statistics."""
        detector = DriftDetector(reference_data)
        assert len(detector.reference_stats) == 3
        for col in ["feature_a", "feature_b", "feature_c"]:
            assert col in detector.reference_stats
            assert "mean" in detector.reference_stats[col]
            assert "std" in detector.reference_stats[col]

    def test_no_drift_on_same_distribution(self, reference_data):
        """Same distribution should not trigger drift."""
        detector = DriftDetector(reference_data)
        batch = reference_data.sample(100, random_state=42).reset_index(drop=True)
        result = detector.detect(batch)
        assert result.has_drift is False
        assert result.drifted_count == 0

    def test_drift_on_shifted_distribution(self, reference_data):
        """Shifted distribution should trigger drift."""
        detector = DriftDetector(reference_data)
        rng = np.random.RandomState(42)
        shifted = pd.DataFrame(
            {
                "feature_a": rng.normal(5, 1, 100),  # shifted mean by 5 std
                "feature_b": rng.uniform(5, 15, 100),
                "feature_c": rng.poisson(15, 100),
            }
        )
        result = detector.detect(shifted)
        # At least one feature should drift
        assert result.drifted_count > 0

    def test_drift_generates_message(self, reference_data):
        """Drift result should include a human-readable message."""
        detector = DriftDetector(reference_data)
        rng = np.random.RandomState(42)
        shifted = pd.DataFrame(
            {
                "feature_a": rng.normal(5, 1, 100),
                "feature_b": rng.uniform(5, 15, 100),
                "feature_c": rng.poisson(15, 100),
            }
        )
        result = detector.detect(shifted)
        assert len(result.message) > 0
        assert result.has_drift == (result.drifted_count > 0)

    def test_generate_drift_batch(self, reference_data):
        """generate_drift_batch should produce shifted data."""
        detector = DriftDetector(reference_data)
        batch = detector.generate_drift_batch(n_samples=50, shift_scale=3.0)
        assert len(batch) == 50
        result = detector.detect(batch)
        assert result.has_drift

    def test_empty_batch_does_not_crash(self, reference_data):
        """Empty batch should not crash."""
        detector = DriftDetector(reference_data)
        empty = pd.DataFrame()
        result = detector.detect(empty)
        assert result.total_features == 0
        assert result.has_drift is False

    def test_partial_column_overlap(self, reference_data):
        """Batch with subset of reference columns should work."""
        detector = DriftDetector(reference_data)
        partial = reference_data[["feature_a"]].head(50)
        result = detector.detect(partial)
        assert result.total_features == 1
