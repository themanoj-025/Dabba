"""Feature engineering for the delivery ETA prediction module.

Transforms cleaned delivery data into model-ready features.

Features created (existing + new):
    - haversine_distance_km: distance from restaurant to delivery point
    - order_hour: hour of day (0-23)
    - order_hour_bucket: breakfast/lunch/evening/late-night
    - day_of_week: 0=Monday, 6=Sunday
    - is_weekend: binary flag
    - is_rush_hour: binary — peak traffic windows (8-10am, 1-2pm, 6-9pm)
    - hour_sin / hour_cos: cyclical encoding of hour for tree/linear models
    - dow_sin / dow_cos: cyclical encoding of day_of_week
    - is_festival: binary (Yes=1, No=0)
    - traffic_ordinal: ordinal encoding of traffic density
    - distance_traffic_interaction: haversine × traffic (high distance + jam = non-linear delay)
    - distance_festival_interaction: haversine × festival flag
    - weather_encoded: ordinal encoding of weather conditions
    - delivery_person_age_bucket: age group
    - city_zone: simple N/S/E/W/Central zone from restaurant lat/long
    - speed_kmh: distance / time (informational, for outlier detection)

Usage in pipeline:
    See pipeline.py's ``eta_feature_cols`` list — this module only
    creates the columns; the pipeline selects which to use.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config
from dabba.features.geo import haversine_distance

logger = logging.getLogger(__name__)

# Cyclical encoding helpers (exported for reuse)


def cyclical_encode(hours: np.ndarray, period: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Convert a cyclical feature (hour, day) to sin/cos components.

    Args:
        hours: Raw values (e.g. 0-23 for hour, 0-6 for day_of_week).
        period: The cycle length (24 for hours, 7 for days).

    Returns:
        Tuple of (sin_component, cos_component) arrays.
    """
    angle = 2 * np.pi * np.asarray(hours, dtype=float) / period
    return np.sin(angle), np.cos(angle)


def _assign_city_zone(lat: float, lon: float) -> str:
    """Assign a broad Bangalore zone from lat/long coordinates.

    Uses approximate centroids:
        - Central: near MG Road / Vidhana Soudha
        - North: beyond Malleshwaram
        - South: beyond JP Nagar / BTM
        - East: Whitefield / Marathahalli side
        - West: Rajajinagar / Vijayanagar side

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        Zone name string.
    """
    if pd.isna(lat) or pd.isna(lon):
        return "unknown"
    if 12.96 <= lat <= 13.01 and 77.58 <= lon <= 77.65:
        return "central"
    elif lat > 13.01:
        return "north"
    elif lat < 12.91:
        return "south"
    elif lon > 77.68:
        return "east"
    else:
        return "west"


# ─── Weather encoding map ─────────────────────────────────────────────

WEATHER_ENCODING = {
    "sunny": 0,
    "cloudy": 1,
    "fog": 2,
    "sandstorms": 2,
    "windy": 3,
    "stormy": 4,
    "rainy": 4,
}


def add_delivery_features(
    df: pd.DataFrame, config: Optional[DabbaConfig] = None
) -> pd.DataFrame:
    """Engineer features for the ETA prediction model.

    Features created fall into five groups:
        1. **Spatial** — haversine_distance_km, city_zone
        2. **Temporal** — order_hour, day_of_week, is_weekend, is_rush_hour,
           hour_sin/cos, dow_sin/cos, order_hour_bucket
        3. **Environmental** — traffic_ordinal, weather_encoded, is_festival
        4. **Interaction** — distance_traffic_interaction, distance_festival_interaction
        5. **Profile** — delivery_person_age, delivery_person_ratings,
           vehicle_condition, delivery_person_age_bucket

    Args:
        df: Cleaned delivery DataFrame.
        config: Project configuration.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    config = config or get_config()
    df = df.copy()
    logger.info("Engineering delivery features — input shape: %s", df.shape)

    # ── 1. Spatial features ────────────────────────────────────────
    rest_lat = "restaurant_latitude" if "restaurant_latitude" in df.columns else None
    rest_lon = "restaurant_longitude" if "restaurant_longitude" in df.columns else None
    del_lat = (
        "delivery_location_latitude"
        if "delivery_location_latitude" in df.columns
        else None
    )
    del_lon = (
        "delivery_location_longitude"
        if "delivery_location_longitude" in df.columns
        else None
    )

    if rest_lat and rest_lon and del_lat and del_lon:
        df["haversine_distance_km"] = haversine_distance(
            df[rest_lat].values,
            df[rest_lon].values,
            df[del_lat].values,
            df[del_lon].values,
        )
        logger.info(
            "Computed haversine distances — mean=%.2f km",
            df["haversine_distance_km"].mean(),
        )

        # City zone from restaurant location
        df["city_zone"] = df.apply(
            lambda row: _assign_city_zone(
                row.get(rest_lat, np.nan), row.get(rest_lon, np.nan)
            ),
            axis=1,
        )
        logger.info(
            "City zone distribution: %s",
            df["city_zone"].value_counts().to_dict(),
        )

    # ── 2. Temporal features ────────────────────────────────────────
    if "order_date" in df.columns:
        dt = pd.to_datetime(df["order_date"], errors="coerce")
        df["order_hour"] = dt.dt.hour.fillna(0).astype(int)
        df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

        # Rush hour: morning (8-10), lunch (1-2), evening (6-9)
        df["is_rush_hour"] = df["order_hour"].apply(
            lambda h: 1 if h in {8, 9, 10, 13, 14, 18, 19, 20} else 0
        ).astype(int)
        logger.info(
            "Rush hour orders: %d/%d (%.1f%%)",
            df["is_rush_hour"].sum(),
            len(df),
            df["is_rush_hour"].mean() * 100,
        )

        # Cyclical encoding (helps linear models & tree splits)
        hour_sin, hour_cos = cyclical_encode(df["order_hour"].values, period=24)
        df["hour_sin"] = hour_sin
        df["hour_cos"] = hour_cos

        dow_sin, dow_cos = cyclical_encode(df["day_of_week"].values, period=7)
        df["dow_sin"] = dow_sin
        df["dow_cos"] = dow_cos

        # Hour buckets for one-hot (preserves existing pipeline behavior)
        def _hour_bucket(h: int) -> str:
            if pd.isna(h):
                return "unknown"
            if 5 <= h < 11:
                return "breakfast"
            elif 11 <= h < 16:
                return "lunch"
            elif 16 <= h < 21:
                return "evening"
            else:
                return "late_night"

        df["order_hour_bucket"] = df["order_hour"].map(_hour_bucket)

    # ── 3. Environmental features ───────────────────────────────────
    if "festival" in df.columns:
        df["is_festival"] = (
            df["festival"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        )

    if "road_traffic_density" in df.columns:
        traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
        df["traffic_ordinal"] = (
            df["road_traffic_density"].map(traffic_map).fillna(1).astype(int)
        )

    # Weather encoding (raw column name after cleaning normalization)
    weather_col_candidates = [
        c for c in df.columns if "weather" in c.lower()
    ]
    if weather_col_candidates:
        weather_col = weather_col_candidates[0]
        df["weather_encoded"] = (
            df[weather_col]
            .str.lower()
            .map(WEATHER_ENCODING)
            .fillna(1)  # default to "cloudy" equivalent
            .astype(int)
        )
        logger.info(
            "Weather encoding applied from column '%s' — values: %s",
            weather_col,
            df["weather_encoded"].value_counts().to_dict(),
        )

    # ── 4. Interaction features ─────────────────────────────────────
    if "haversine_distance_km" in df.columns and "traffic_ordinal" in df.columns:
        df["distance_traffic_interaction"] = (
            df["haversine_distance_km"] * (df["traffic_ordinal"] + 1)
        )
        logger.info(
            "Distance×traffic interaction — mean=%.2f",
            df["distance_traffic_interaction"].mean(),
        )

    if "haversine_distance_km" in df.columns and "is_festival" in df.columns:
        df["distance_festival_interaction"] = (
            df["haversine_distance_km"] * df["is_festival"]
        )
        logger.info(
            "Distance×festival interaction — %d non-zero rows",
            (df["distance_festival_interaction"] > 0).sum(),
        )

    # ── 5. Profile features ─────────────────────────────────────────
    if "delivery_person_age" in df.columns:
        df["delivery_person_age"] = pd.to_numeric(
            df["delivery_person_age"], errors="coerce"
        )
        bins = [0, 25, 35, 45, 100]
        labels = ["young", "mid", "senior", "veteran"]
        df["delivery_person_age_bucket"] = pd.cut(
            df["delivery_person_age"], bins=bins, labels=labels
        )

    # ── 6. Speed for outlier detection (informational) ──────────────
    if "haversine_distance_km" in df.columns and "time_taken_min" in df.columns:
        df["speed_kmh"] = (
            df["haversine_distance_km"] / (df["time_taken_min"] / 60)
        ).replace([np.inf, -np.inf], np.nan)

    # ── Drop rows where key features are missing ────────────────────
    key_features = ["time_taken_min"]
    if "haversine_distance_km" in df.columns:
        key_features.append("haversine_distance_km")

    n_before = len(df)
    df = df.dropna(subset=key_features)
    logger.info("Dropped %d rows with missing key features", n_before - len(df))

    logger.info("Delivery features complete — output shape: %s", df.shape)
    return df.reset_index(drop=True)
