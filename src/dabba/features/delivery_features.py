"""Feature engineering for the delivery ETA prediction module.

Transforms cleaned delivery data into model-ready features.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config
from dabba.features.geo import haversine_distance

logger = logging.getLogger(__name__)


def add_delivery_features(df: pd.DataFrame, config: Optional[DabbaConfig] = None) -> pd.DataFrame:
    """Engineer features for the ETA prediction model.

    Features created:
        - haversine_distance_km: distance from restaurant to delivery point
        - order_hour: hour of day (0-23)
        - order_hour_bucket: breakfast/lunch/evening/late-night
        - day_of_week: 0=Monday, 6=Sunday
        - is_weekend: binary
        - is_festival: binary (Yes=1, No=0)
        - traffic_ordinal: ordinal encoding of traffic density
        - weather_category: standardized weather label
        - delivery_person_age_bucket: age group
        - speed_kmh: distance / time (for outlier detection)

    Args:
        df: Cleaned delivery DataFrame.
        config: Project configuration.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    config = config or get_config()
    df = df.copy()
    logger.info("Engineering delivery features — input shape: %s", df.shape)

    # --- Haversine distance ---
    rest_lat = "restaurant_latitude" if "restaurant_latitude" in df.columns else None
    rest_lon = "restaurant_longitude" if "restaurant_longitude" in df.columns else None
    del_lat = "delivery_location_latitude" if "delivery_location_latitude" in df.columns else None
    del_lon = "delivery_location_longitude" if "delivery_location_longitude" in df.columns else None

    if rest_lat and rest_lon and del_lat and del_lon:
        df["haversine_distance_km"] = haversine_distance(
            df[rest_lat].values, df[rest_lon].values,
            df[del_lat].values, df[del_lon].values,
        )
        logger.info("Computed haversine distances — mean=%.2f km", df["haversine_distance_km"].mean())

    # --- Time features ---
    if "order_date" in df.columns:
        dt = pd.to_datetime(df["order_date"], errors="coerce")
        df["order_hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

        # Hour buckets
        def _hour_bucket(h: float) -> str:
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

    # --- Festival flag ---
    if "festival" in df.columns:
        df["is_festival"] = df["festival"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # --- Traffic ordinal encoding ---
    if "road_traffic_density" in df.columns:
        traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
        df["traffic_ordinal"] = df["road_traffic_density"].map(traffic_map).fillna(1).astype(int)

    # --- Delivery person age bucket ---
    if "delivery_person_age" in df.columns:
        df["delivery_person_age"] = pd.to_numeric(df["delivery_person_age"], errors="coerce")
        bins = [0, 25, 35, 45, 100]
        labels = ["young", "mid", "senior", "veteran"]
        df["delivery_person_age_bucket"] = pd.cut(
            df["delivery_person_age"], bins=bins, labels=labels
        )

    # --- Speed for outlier detection (informational) ---
    if "haversine_distance_km" in df.columns and "time_taken_min" in df.columns:
        df["speed_kmh"] = (df["haversine_distance_km"] / (df["time_taken_min"] / 60)).replace(
            [np.inf, -np.inf], np.nan
        )

    # --- Drop rows where key features are missing ---
    key_features = ["time_taken_min"]
    if "haversine_distance_km" in df.columns:
        key_features.append("haversine_distance_km")

    n_before = len(df)
    df = df.dropna(subset=key_features)
    logger.info("Dropped %d rows with missing key features", n_before - len(df))

    logger.info("Delivery features complete — output shape: %s", df.shape)
    return df.reset_index(drop=True)
