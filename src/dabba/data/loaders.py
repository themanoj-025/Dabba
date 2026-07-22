"""Data loading utilities for the Dabba project.

Provides functions to load raw CSV files with schema verification.
Also provides DB-backed alternatives for production use.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


def load_zomato(
    config: Optional[DabbaConfig] = None,
    use_db: bool = False,
) -> pd.DataFrame:
    """Load the raw Zomato Bangalore Restaurants dataset.

    When ``use_db=True``, tries to load from the database first;
    falls back to CSV if no DB data exists.

    Args:
        config: Project configuration. Uses default if None.
        use_db: If True, attempt DB-backed load with CSV fallback.

    Returns:
        pd.DataFrame: Raw Zomato data.

    Raises:
        FileNotFoundError: If neither DB nor CSV has data.
    """
    config = config or get_config()

    if use_db:
        try:
            return load_zomato_from_db(config)
        except Exception as e:
            logger.warning(
                "DB load failed for Zomato data (%s) — falling back to CSV", e
            )

    path = config.zomato_path
    logger.info("Loading Zomato dataset from %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Zomato dataset not found at {path}. "
            "Please download it first — see README for instructions."
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info(
        "Loaded Zomato dataset: shape=%s, columns=%s", df.shape, list(df.columns)
    )
    return df


def load_zomato_from_db(config: Optional[DabbaConfig] = None) -> pd.DataFrame:
    """Load restaurant data from the database as a DataFrame.

    Converts the ``Restaurant`` ORM rows back to a DataFrame with the
    same column schema that the cleaning/feature pipeline expects.

    Args:
        config: Project configuration.

    Returns:
        pd.DataFrame: Restaurant data with schema matching
            ``clean_zomato`` + ``add_restaurant_features`` output.

    Raises:
        ValueError: If no restaurants exist in the database.
    """
    from dabba.database.session import get_db
    from dabba.database.repositories import count_restaurants, get_all_restaurants

    config = config or get_config()

    with get_db() as db:
        if count_restaurants(db) == 0:
            raise ValueError("No restaurants found in database — run seed first")

        restaurants = get_all_restaurants(db, limit=10000)

    records = []
    for r in restaurants:
        records.append(
            {
                "name": r.name,
                "rate": r.rate,
                "bayesian_rating": r.bayesian_rating,
                "cost_for_two": r.cost_for_two,
                "location": r.location,
                "cuisines": r.cuisines,
                "votes": r.votes,
                "votes_log": r.votes_log,
                "online_order_binary": r.online_order_binary,
                "book_table_binary": r.book_table_binary,
                "cuisine_count": r.cuisine_count,
                "avg_sentiment": r.avg_sentiment,
                "reliability_score": r.reliability_score,
                "latitude": r.latitude,
                "longitude": r.longitude,
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d restaurants from database", len(df)
    )
    return df


def load_delivery(
    config: Optional[DabbaConfig] = None,
    use_db: bool = False,
) -> pd.DataFrame:
    """Load the raw food delivery time dataset.

    When ``use_db=True``, tries to load from the database first;
    falls back to CSV if no DB data exists.

    Args:
        config: Project configuration. Uses default if None.
        use_db: If True, attempt DB-backed load with CSV fallback.

    Returns:
        pd.DataFrame: Raw delivery data.

    Raises:
        FileNotFoundError: If neither DB nor CSV has data.
    """
    config = config or get_config()

    if use_db:
        try:
            return load_delivery_from_db(config)
        except Exception as e:
            logger.warning(
                "DB load failed for delivery data (%s) — falling back to CSV", e
            )

    path = config.delivery_path
    logger.info("Loading delivery dataset from %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Delivery dataset not found at {path}. "
            "Please download it first — see README for instructions."
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info(
        "Loaded delivery dataset: shape=%s, columns=%s", df.shape, list(df.columns)
    )
    return df


def load_delivery_from_db(config: Optional[DabbaConfig] = None) -> pd.DataFrame:
    """Load delivery order data from the database as a DataFrame.

    Converts the ``Order`` ORM rows back to a DataFrame with the
    schema that the feature engineering pipeline expects.

    Args:
        config: Project configuration.

    Returns:
        pd.DataFrame: Order data with key delivery features.

    Raises:
        ValueError: If no orders exist in the database.
    """
    from dabba.database.session import get_db
    from dabba.database.repositories import get_all_orders

    config = config or get_config()

    with get_db() as db:
        orders = get_all_orders(db, limit=50000)

    if not orders:
        raise ValueError("No orders found in database — run seed first")

    records = []
    for o in orders:
        records.append(
            {
                "haversine_distance_km": o.distance_km,
                "traffic_ordinal": o.traffic_level,
                "is_festival": o.is_festival,
                "delivery_person_age": o.delivery_person_age,
                "delivery_person_ratings": o.delivery_person_rating,
                "vehicle_condition": o.vehicle_condition,
                "time_taken_min": o.actual_eta,
                "predicted_eta": o.predicted_eta,
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d orders from database", len(df)
    )
    return df


def describe_dataset(df: pd.DataFrame, name: str) -> None:
    """Print schema summary for a loaded DataFrame.

    Args:
        df: The DataFrame to describe.
        name: Human-readable name for logging.
    """
    logger.info("=== %s Schema Summary ===", name)
    logger.info("Shape: %s", df.shape)
    logger.info("Columns:\n%s", df.dtypes.to_string())
    logger.info("First 5 rows:\n%s", df.head().to_string())
    logger.info("Null counts:\n%s", df.isnull().sum().to_string())
    logger.info("============================")
