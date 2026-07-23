"""Real-time traffic data integration for ETA prediction.

Provides a unified interface for multiple traffic data providers:
    - **TomTom Traffic API** (recommended, generous free tier)
    - **Mappls/MapmyIndia** (India-specific)
    - **OSRM** (open-source, no real-time data but useful for routing)
    - **Simulated** (deterministic mock when no API key is configured)

The module auto-selects the backend based on available config and
always falls back to a simulated (but time-of-day-aware) estimate
so the app never breaks without an API key.

Usage:
    >>> from dabba.features.traffic import get_traffic_level
    >>> level = get_traffic_level(lat=12.97, lon=77.59, hour=18)
    >>> level
    2  # High traffic (rush hour)
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import random
from typing import Optional

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


# ─── Data types ──────────────────────────────────────────────────────


@dataclasses.dataclass
class TrafficInfo:
    """Normalised traffic information from any provider.

    Attributes:
        level: Traffic ordinal (0=Low, 1=Medium, 2=High, 3=Jam).
        label: Human-readable label.
        speed_ratio: Current speed / free-flow speed (0.0 = stopped, 1.0 = free flow).
        source: Provider name for traceability.
    """

    level: int
    label: str
    speed_ratio: float
    source: str


# ─── Provider implementations ────────────────────────────────────────


def _simulate_traffic(
    lat: float,
    lon: float,
    hour: int,
    day_of_week: int,
) -> TrafficInfo:
    """Simulate traffic level based on time-of-day heuristics.

    Produces realistic patterns:
        - Rush hours (8-10am, 6-9pm weekdays): High traffic
        - Mid-day (11am-4pm): Moderate
        - Late night (10pm-6am): Low
        - Weekends: lighter overall

    Args:
        lat: Latitude (unused in simulation).
        lon: Longitude (unused).
        hour: Hour of day (0-23).
        day_of_week: Day of week (0=Monday, 6=Sunday).

    Returns:
        TrafficInfo with simulated level.
    """
    is_weekend = day_of_week >= 5

    # Base probability of higher traffic
    if is_weekend:
        if 11 <= hour <= 20:
            base_level = 1  # Medium (shopping hours)
        else:
            base_level = 0  # Low
    elif hour in {8, 9, 10, 18, 19, 20}:  # Rush hours
        base_level = 2  # High
    elif 11 <= hour <= 17:  # Mid-day
        base_level = 1  # Medium
    else:
        base_level = 0  # Low

    # Add randomness (±1)
    level = max(0, min(3, base_level + random.choice([-1, 0, 0, 1])))

    labels = {0: "Low", 1: "Medium", 2: "High", 3: "Jam"}
    speed_ratios = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}

    return TrafficInfo(
        level=level,
        label=labels[level],
        speed_ratio=speed_ratios[level],
        source="simulated",
    )


def _tomtom_traffic(
    lat: float,
    lon: float,
    api_key: str,
) -> Optional[TrafficInfo]:
    """Fetch real-time traffic flow data from TomTom Traffic API.

    Uses the TomTom Traffic Flow API to get current traffic conditions
    near a given coordinate.

    Args:
        lat: Latitude.
        lon: Longitude.
        api_key: TomTom API key.

    Returns:
        TrafficInfo or None if the request fails.
    """
    try:
        import requests

        url = (
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
            f"absolute/10/json?point={lat},{lon}"
            f"&key={api_key}"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        flow = data.get("flowSegmentData", {})
        current_speed = flow.get("currentSpeed", 0)
        free_flow_speed = flow.get("freeFlowSpeed", 1)

        if free_flow_speed > 0:
            speed_ratio = current_speed / free_flow_speed
        else:
            speed_ratio = 0.5

        if speed_ratio >= 0.8:
            level, label = 0, "Low"
        elif speed_ratio >= 0.6:
            level, label = 1, "Medium"
        elif speed_ratio >= 0.4:
            level, label = 2, "High"
        else:
            level, label = 3, "Jam"

        return TrafficInfo(
            level=level,
            label=label,
            speed_ratio=round(speed_ratio, 2),
            source="tomtom",
        )
    except Exception as e:
        logger.warning("TomTom traffic request failed: %s", e)
        return None


def _mappls_traffic(
    lat: float,
    lon: float,
    api_key: str,
) -> Optional[TrafficInfo]:
    """Fetch traffic data from Mappls (MapmyIndia) Traffic API.

    Args:
        lat: Latitude.
        lon: Longitude.
        api_key: Mappls API key.

    Returns:
        TrafficInfo or None.
    """
    try:
        import requests

        url = (
            f"https://apis.mappls.com/advancedmaps/v1/{api_key}/"
            f"traffic/flow?location={lon},{lat}"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # Parse Mappls response format
        flow_data = data.get("results", [{}])[0]
        congestion = flow_data.get("congestion_level", "low")

        level_map = {"low": 0, "medium": 1, "high": 2, "jam": 3}
        level = level_map.get(congestion.lower(), 1)
        labels = {0: "Low", 1: "Medium", 2: "High", 3: "Jam"}

        return TrafficInfo(
            level=level,
            label=labels[level],
            speed_ratio=1.0 - (level * 0.25),
            source="mappls",
        )
    except Exception as e:
        logger.warning("Mappls traffic request failed: %s", e)
        return None


# ─── Public API ──────────────────────────────────────────────────────


def get_traffic_level(
    lat: float = 12.97,
    lon: float = 77.59,
    hour: Optional[int] = None,
    day_of_week: Optional[int] = None,
    config: Optional[DabbaConfig] = None,
) -> TrafficInfo:
    """Get the current traffic level from the best available provider.

    Resolution order:
        1. TomTom Traffic API (if ``DABBA_TOMTOM_API_KEY`` is set)
        2. Mappls Traffic API (if ``DABBA_MAPPLS_API_KEY`` is set)
        3. Time-based simulation (always available)

    Args:
        lat: Origin/delivery-point latitude.
        lon: Origin/delivery-point longitude.
        hour: Hour of day (0-23). Defaults to current hour.
        day_of_week: Day of week (0=Monday). Defaults to today.
        config: Project configuration.

    Returns:
        TrafficInfo with level, label, speed_ratio, and source.
    """
    config = config or get_config()
    now = datetime.datetime.now()
    hour = hour if hour is not None else now.hour
    day_of_week = day_of_week if day_of_week is not None else now.weekday()

    # Try real providers first (fields always exist on Pydantic config, default to None)
    tomtom_key = config.tomtom_api_key
    mappls_key = config.mappls_api_key

    if tomtom_key:
        result = _tomtom_traffic(lat, lon, tomtom_key)
        if result is not None:
            return result

    if mappls_key:
        result = _mappls_traffic(lat, lon, mappls_key)
        if result is not None:
            return result

    # Fallback: simulate based on time-of-day
    logger.debug(
        "No traffic API key configured — using simulated traffic (hour=%d, dow=%d)",
        hour,
        day_of_week,
    )
    return _simulate_traffic(lat, lon, hour, day_of_week)
