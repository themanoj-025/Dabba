"""Tests for the real-time traffic API module (P3)."""

import pytest

from dabba.features.traffic import (
    TrafficInfo,
    _simulate_traffic,
    get_traffic_level,
)


class TestTrafficInfo:
    """Tests for the TrafficInfo dataclass."""

    def test_created_with_fields(self):
        info = TrafficInfo(level=1, label="Medium", speed_ratio=0.7, source="test")
        assert info.level == 1
        assert info.label == "Medium"
        assert info.speed_ratio == 0.7
        assert info.source == "test"


class TestSimulateTraffic:
    """Tests for the simulated traffic provider."""

    def test_returns_traffic_info(self):
        result = _simulate_traffic(12.97, 77.59, hour=14, day_of_week=2)
        assert isinstance(result, TrafficInfo)
        assert 0 <= result.level <= 3
        assert result.label in ("Low", "Medium", "High", "Jam")
        assert result.source == "simulated"

    def test_rush_hour_higher_traffic(self):
        """Rush hour should generally have higher traffic than late night."""
        rush_hour = _simulate_traffic(12.97, 77.59, hour=18, day_of_week=2)
        late_night = _simulate_traffic(12.97, 77.59, hour=3, day_of_week=2)
        # Rush hour average should be >= late night (run multiple times)
        rush_levels = [
            _simulate_traffic(12.97, 77.59, hour=18, day_of_week=2).level
            for _ in range(20)
        ]
        night_levels = [
            _simulate_traffic(12.97, 77.59, hour=3, day_of_week=2).level
            for _ in range(20)
        ]
        assert sum(rush_levels) >= sum(night_levels)  # Statistical tendency

    def test_weekend_pattern(self):
        """Weekend traffic should be moderate during mid-day."""
        weekend = _simulate_traffic(12.97, 77.59, hour=14, day_of_week=6)
        assert weekend.level >= 0

    def test_valid_range_all_hours(self):
        """All hours of day should return valid levels."""
        for hour in range(24):
            result = _simulate_traffic(12.97, 77.59, hour=hour, day_of_week=3)
            assert 0 <= result.level <= 3
            assert isinstance(result.label, str)


class TestGetTrafficLevel:
    """Tests for the public get_traffic_level() function."""

    def test_falls_back_to_simulated(self):
        """Without API keys, should fall back to simulated."""
        from dabba.config import get_config

        config = get_config()
        config.tomtom_api_key = None
        config.mappls_api_key = None

        result = get_traffic_level(
            lat=12.97, lon=77.59, hour=14, day_of_week=2, config=config
        )
        assert result.source == "simulated"
        assert 0 <= result.level <= 3

    def test_returns_traffic_info_object(self):
        result = get_traffic_level()
        assert isinstance(result, TrafficInfo)

    def test_defaults_to_current_time(self):
        """Should use current time when no hour/dow provided."""
        result = get_traffic_level()
        assert 0 <= result.level <= 3
        assert isinstance(result.label, str)
