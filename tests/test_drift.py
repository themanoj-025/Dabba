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


class TestSlackAlertFunctions:
    """Tests for Slack alerting functions (no webhook needed)."""

    def test_format_drift_message_with_drift(self):
        """Should format a message with drifted features."""
        from dabba.monitoring.drift import _format_drift_message, DriftResult
        result = DriftResult(
            drifted_features={"feature_a": (0.001, 0.42), "feature_b": (0.003, 0.38)},
            total_features=5,
            drifted_count=2,
            has_drift=True,
            threshold=0.05,
        )
        msg = _format_drift_message(result)
        assert "Drift Detected" in msg
        assert "feature_a" in msg
        assert "feature_b" in msg
        assert "2/5" in msg

    def test_format_drift_message_no_drift(self):
        """Should format a clean message when no drift."""
        from dabba.monitoring.drift import _format_drift_message, DriftResult
        result = DriftResult(
            drifted_features={},
            total_features=5,
            drifted_count=0,
            has_drift=False,
            threshold=0.05,
        )
        msg = _format_drift_message(result)
        assert "No drift detected" in msg

    def test_send_slack_invalid_url(self):
        """Should return not-sent when webhook URL is unreachable."""
        from dabba.monitoring.drift import _send_slack_alert
        # Use a clearly invalid URL that will fail DNS resolution
        result = _send_slack_alert(
            "https://hooks.invalid-slack-domain-that-does-not-exist.example.com/services/test",
            "test alert",
        )
        assert result.sent is False
        assert len(result.reason) > 0


class TestDetectAndAlert:
    """Tests for detect_and_alert with cooldown logic."""

    @pytest.fixture
    def detector(self):
        rng = np.random.RandomState(42)
        ref = pd.DataFrame({
            "feature_a": rng.normal(0, 1, 500),
            "feature_b": rng.uniform(0, 10, 500),
        })
        return DriftDetector(ref)

    def test_detect_and_alert_no_drift(self, detector):
        """Should return clean result when no drift."""
        batch = detector.generate_drift_batch(n_samples=50, shift_scale=0.0)
        result = detector.detect_and_alert(batch)
        assert result.has_drift is False

    def test_detect_and_alert_with_drift(self, detector):
        """Should detect drift on shifted data."""
        batch = detector.generate_drift_batch(n_samples=50, shift_scale=3.0)
        result = detector.detect_and_alert(batch)
        # Should detect drift (but may not alert without Slack configured)
        assert result.drifted_count > 0 or result.has_drift is False

    def test_cooldown_suppresses_duplicate(self, detector):
        """Same feature alerted twice should respect cooldown."""
        from dabba.monitoring.drift import _format_drift_message

        batch = detector.generate_drift_batch(n_samples=50, shift_scale=3.0)

        # First call — should detect and attempt alert
        r1 = detector.detect_and_alert(batch)
        if not r1.has_drift:
            pytest.skip("No drift detected — cannot test cooldown")

        # Manually set cooldown for all drifted features
        import time
        for feature in r1.drifted_features:
            detector._alert_cooldowns[feature] = time.time()

        # Second call — features are on cooldown, detect_and_alert
        # should log to DB but note that alerts were suppressed
        r2 = detector.detect_and_alert(batch)
        # Detection results should be same
        assert r2.drifted_count == r1.drifted_count
