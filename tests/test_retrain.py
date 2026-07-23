"""Tests for the drift-triggered retraining hook (``dabba.monitoring.retrain``)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dabba.monitoring.drift import DriftResult
from dabba.monitoring.retrain import maybe_trigger_retraining


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def severe_drift_result() -> DriftResult:
    """A DriftResult with >30% of features drifted."""
    return DriftResult(
        drifted_features={
            "distance_km": (0.001, 0.45),
            "traffic_level": (0.002, 0.40),
            "rush_hour": (0.003, 0.38),
            "order_frequency": (0.01, 0.35),
        },
        total_features=10,
        drifted_count=4,
        has_drift=True,
        threshold=0.05,
        message="Drift detected in 4/10 features",
    )


@pytest.fixture
def mild_drift_result() -> DriftResult:
    """A DriftResult with <30% of features drifted (below threshold)."""
    return DriftResult(
        drifted_features={
            "distance_km": (0.001, 0.45),
        },
        total_features=10,
        drifted_count=1,
        has_drift=True,
        threshold=0.05,
        message="Drift detected in 1/10 features",
    )


@pytest.fixture
def no_drift_result() -> DriftResult:
    """A DriftResult with no drift."""
    return DriftResult(
        drifted_features={},
        total_features=10,
        drifted_count=0,
        has_drift=False,
        threshold=0.05,
        message="No drift detected",
    )


# ─── Happy path ───────────────────────────────────────────────────────────


class TestMaybeTriggerRetrainingHappyPath:
    """The ``maybe_trigger_retraining`` function — successful triggers."""

    def test_triggers_on_severe_drift(self, severe_drift_result: DriftResult) -> None:
        """Should return True and spawn a subprocess when drift exceeds threshold."""
        from dabba.monitoring.retrain import _last_retrain_time

        # Reset cooldown
        import dabba.monitoring.retrain as retrain_mod
        retrain_mod._last_retrain_time = 0.0

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.pid = 12345
            result = maybe_trigger_retraining(
                severe_drift_result,
                drift_threshold=0.3,
                dry_run=False,
            )
            assert result is True
            mock_popen.assert_called_once()

    def test_dry_run_logs_only(self, severe_drift_result: DriftResult) -> None:
        """With dry_run=True, should return True but NOT spawn a subprocess."""
        from dabba.monitoring.retrain import _last_retrain_time
        import dabba.monitoring.retrain as retrain_mod
        retrain_mod._last_retrain_time = 0.0

        with patch("subprocess.Popen") as mock_popen:
            result = maybe_trigger_retraining(
                severe_drift_result,
                drift_threshold=0.3,
                dry_run=True,
            )
            assert result is True
            mock_popen.assert_not_called()


# ─── Edge cases / no-ops ──────────────────────────────────────────────────


class TestMaybeTriggerRetrainingEdgeCases:
    """Cases where retraining should NOT be triggered."""

    def test_no_drift_no_trigger(self, no_drift_result: DriftResult) -> None:
        """Should return False when there's no drift."""
        result = maybe_trigger_retraining(no_drift_result, drift_threshold=0.3)
        assert result is False

    def test_mild_drift_below_threshold(self, mild_drift_result: DriftResult) -> None:
        """Should return False when drift fraction is below the threshold."""
        result = maybe_trigger_retraining(mild_drift_result, drift_threshold=0.3)
        assert result is False

    def test_empty_drifted_features(self) -> None:
        """Should return False when drifted_features is empty but has_drift is True."""
        result = maybe_trigger_retraining(
            DriftResult(
                drifted_features={},
                total_features=10,
                drifted_count=0,
                has_drift=True,  # inconsistent but we handle gracefully
                threshold=0.05,
                message="",
            ),
        )
        assert result is False

    def test_zero_total_features(self) -> None:
        """Should return False when total_features is 0 (division guard)."""
        result = maybe_trigger_retraining(
            DriftResult(
                drifted_features={"a": (0.001, 0.5)},
                total_features=0,
                drifted_count=1,
                has_drift=True,
                threshold=0.05,
                message="",
            ),
        )
        assert result is False

    def test_cooldown_respected(self, severe_drift_result: DriftResult) -> None:
        """Should return False during cooldown period."""
        from dabba.monitoring.retrain import _last_retrain_time
        import dabba.monitoring.retrain as retrain_mod

        # Set cooldown to the recent past (5 seconds ago)
        import time
        retrain_mod._last_retrain_time = time.time() - 5

        with patch("subprocess.Popen") as mock_popen:
            result = maybe_trigger_retraining(
                severe_drift_result,
                drift_threshold=0.3,
                retrain_cooldown_hours=6.0,  # 5 sec < 6 hours → on cooldown
                dry_run=True,
            )
            assert result is False
            mock_popen.assert_not_called()


# ─── Graceful failure ─────────────────────────────────────────────────────


class TestMaybeTriggerRetrainingFailures:
    """Graceful failure paths when retrain can't proceed."""

    def test_no_project_root_graceful(self, severe_drift_result: DriftResult) -> None:
        """Should return False gracefully when project_root doesn't have pyproject.toml."""
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = maybe_trigger_retraining(
                severe_drift_result,
                drift_threshold=0.3,
                project_root=Path(tmpdir),  # No pyproject.toml here
                dry_run=False,
            )
            assert result is False

    def test_subprocess_exception_graceful(self, severe_drift_result: DriftResult) -> None:
        """Should return False gracefully when subprocess.Popen raises."""
        from dabba.monitoring.retrain import _last_retrain_time
        import dabba.monitoring.retrain as retrain_mod
        retrain_mod._last_retrain_time = 0.0

        with patch("subprocess.Popen", side_effect=OSError("No such file")):
            result = maybe_trigger_retraining(
                severe_drift_result,
                drift_threshold=0.3,
                dry_run=False,
            )
            # Should return False on failure
            assert result is False
