"""Feature engineering for the delivery ETA prediction module.

Transforms cleaned delivery data into model-ready features and provides
:func:`build_eta_features_for_api` for serving-side feature construction.

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
    See ``ETA_FEATURE_COLS`` — a shared constant that serves as the single
    source of truth for which columns the model expects. Both ``pipeline.py``
    and the API serving endpoint import from here to prevent feature drift.
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config
from dabba.features.geo import haversine_distance

logger = logging.getLogger(__name__)

# Cyclical encoding helpers (exported for reuse)


def _hour_bucket(h: int) -> str:
    """Assign an order hour to a meal-time bucket.

    Args:
        h: Hour of day (0-23).

    Returns:
        Bucket name string.
    """
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


def _age_bucket(age: float) -> str:
    """Assign a delivery person age to an age group bucket.

    Args:
        age: Age in years.

    Returns:
        Bucket label string.
    """
    if pd.isna(age) or age <= 0:
        return "mid"
    if age <= 25:
        return "young"
    elif age <= 35:
        return "mid"
    elif age <= 45:
        return "senior"
    else:
        return "veteran"


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
        df["delivery_person_age_bucket"] = df["delivery_person_age"].apply(_age_bucket)

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


# ─── Feature builder for API requests ─────────────────────────────────

def build_eta_features_for_api(
    *,
    distance_km: float,
    traffic_level: int = 1,
    is_festival: bool = False,
    delivery_person_age: float = 30.0,
    delivery_person_rating: float = 4.0,
    vehicle_condition: int = 1,
    order_hour: Optional[int] = None,
    day_of_week: Optional[int] = None,
    weather_encoded: int = 1,
    city_zone: str = "unknown",
    config: Optional[DabbaConfig] = None,
) -> pd.DataFrame:
    """Build the full ETA feature vector from API-level inputs.

    Constructs all features that the winning ETA model expects
    (including derived temporal, interaction, and categorical features)
    from the limited set of fields available via API requests.

    Temporal fields (``order_hour``, ``day_of_week``) default to the
    current time if not provided, so the API works without requiring
    the caller to specify them.

    Args:
        distance_km: Haversine distance in km.
        traffic_level: Traffic ordinal (0=Low, 1=Medium, 2=High, 3=Jam).
        is_festival: Whether it's a festival day.
        delivery_person_age: Delivery person age.
        delivery_person_rating: Delivery person rating (1-5).
        vehicle_condition: Vehicle condition score (0-3).
        order_hour: Hour of day (0-23). If None, uses current hour.
        day_of_week: Day of week (0=Monday, 6=Sunday). If None, uses today.
        weather_encoded: Weather ordinal (0=sunny, 1=cloudy, ..., 4=stormy).
        city_zone: Bangalore zone string.
        config: Project configuration.

    Returns:
        pd.DataFrame: Single-row DataFrame with all model-required columns.
    """
    config = config or get_config()
    now = datetime.datetime.now()
    hour = order_hour if order_hour is not None else now.hour
    dow = day_of_week if day_of_week is not None else now.weekday()

    hour_sin, hour_cos = cyclical_encode(np.array([hour]), period=24)
    dow_sin, dow_cos = cyclical_encode(np.array([dow]), period=7)

    is_festival_int = int(is_festival)
    festival_flag = 1 if is_festival else 0  # For interaction

    data = {
        "haversine_distance_km": distance_km,
        "city_zone": city_zone,
        "order_hour": hour,
        "day_of_week": dow,
        "is_weekend": 1 if dow >= 5 else 0,
        "is_rush_hour": 1 if hour in {8, 9, 10, 13, 14, 18, 19, 20} else 0,
        "hour_sin": float(hour_sin[0]),
        "hour_cos": float(hour_cos[0]),
        "dow_sin": float(dow_sin[0]),
        "dow_cos": float(dow_cos[0]),
        "order_hour_bucket": _hour_bucket(hour),
        "traffic_ordinal": traffic_level,
        "is_festival": is_festival_int,
        "weather_encoded": weather_encoded,
        "distance_traffic_interaction": distance_km * (traffic_level + 1),
        "distance_festival_interaction": distance_km * festival_flag,
        "delivery_person_age": delivery_person_age,
        "delivery_person_ratings": delivery_person_rating,
        "vehicle_condition": vehicle_condition,
        "delivery_person_age_bucket": _age_bucket(delivery_person_age),
    }

    return pd.DataFrame([data])


ETA_FEATURE_COLS: list[str] = [
    # Spatial
    "haversine_distance_km",
    "city_zone",  # string → one-hot encoded by ColumnTransformer
    # Temporal
    "order_hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "order_hour_bucket",  # string → one-hot
    # Environmental
    "traffic_ordinal",
    "is_festival",
    "weather_encoded",
    # Interactions
    "distance_traffic_interaction",
    "distance_festival_interaction",
    # Profile
    "delivery_person_age",
    "delivery_person_ratings",
    "vehicle_condition",
    "delivery_person_age_bucket",  # string → one-hot
]
