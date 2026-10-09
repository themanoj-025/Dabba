"""Tests for dabba.pipeline — pipeline orchestration logic."""

from __future__ import annotations

import numpy as np
import pytest


class TestPipelineHelpers:
    """Test helper functions used by the pipeline."""

    def test_reliability_score_calculation(self) -> None:
        """Verify the reliability score formula matches config weights."""
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        rating_score = 4.0 / 5.0
        sentiment_score = 0.8
        delay_risk = 0.2

        reliability = (
            cfg.reliability_w_rating * rating_score
            + cfg.reliability_w_sentiment * sentiment_score
            + cfg.reliability_w_delay * (1 - delay_risk)
        )
        assert reliability == pytest.approx(0.8)

    def test_reliability_score_zero_ratings(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        reliability = (
            cfg.reliability_w_rating * 0.0
            + cfg.reliability_w_sentiment * 0.0
            + cfg.reliability_w_delay * 1.0
        )
        assert 0.0 <= reliability <= 1.0

    def test_reliability_score_perfect(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        reliability = (
            cfg.reliability_w_rating * 1.0
            + cfg.reliability_w_sentiment * 1.0
            + cfg.reliability_w_delay * 1.0
        )
        assert reliability == pytest.approx(1.0)


class TestCyclicalEncode:
    """Test cyclical encoding of temporal features."""

    def test_returns_tuple_of_arrays(self) -> None:
        from dabba.features.delivery_features import cyclical_encode

        hours = np.array([0, 6, 12, 18, 24])
        sin_arr, cos_arr = cyclical_encode(hours)
        assert isinstance(sin_arr, np.ndarray)
        assert isinstance(cos_arr, np.ndarray)
        assert len(sin_arr) == 5
        assert len(cos_arr) == 5

    def test_six_am_is_max_sin(self) -> None:
        from dabba.features.delivery_features import cyclical_encode

        hours = np.array([0, 6, 12, 18])
        sin_arr, cos_arr = cyclical_encode(hours)
        # angle = 2π*6/24 = π/2, so sin(π/2) = 1 at hour 6
        assert sin_arr[1] == pytest.approx(1.0, abs=0.01)
        # angle = 2π*12/24 = π, so sin(π) ≈ 0 at hour 12
        assert sin_arr[2] == pytest.approx(0.0, abs=0.01)

    def test_midnight_is_zero_sin(self) -> None:
        from dabba.features.delivery_features import cyclical_encode

        hours = np.array([0, 24])
        sin_arr, _ = cyclical_encode(hours)
        # sin(0) = 0 and sin(2π) = 0
        assert sin_arr[0] == pytest.approx(0.0, abs=0.01)
        assert sin_arr[1] == pytest.approx(0.0, abs=0.01)

    def test_custom_period(self) -> None:
        from dabba.features.delivery_features import cyclical_encode

        days = np.array([0, 3, 7])
        sin_arr, cos_arr = cyclical_encode(days, period=7)
        assert len(sin_arr) == 3
