"""Database seeding — import CSV data into SQLAlchemy tables.

Provides functions to load the processed restaurant and delivery
DataFrames into the Dabba database (SQLite locally, Postgres in
production). Designed to be called from the pipeline or as a
standalone CLI::

    python -m dabba.database.seed                     # uses defaults
    python -m dabba.database.seed --restaurants-csv ... --delivery-csv ...
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config
from dabba.database.models import Order, Restaurant, RESTAURANT_COL_MAP
from dabba.database.session import get_db, init_db

logger = logging.getLogger(__name__)


def seed_restaurants(
    df: pd.DataFrame,
    config: Optional[DabbaConfig] = None,
) -> int:
    """Upsert all restaurants from a DataFrame into the database.

    Each row is matched by ``name`` — existing rows are updated,
    new rows are inserted. Missing/malformed coordinates are treated
    as ``NULL``.

    Args:
        df: Processed restaurant DataFrame (from :func:`add_restaurant_features`).
        config: Project configuration.

    Returns:
        Number of restaurants upserted.
    """
    config = config or get_config()
    init_db(config)

    col_map = RESTAURANT_COL_MAP
    lat_col = next(
        (c for c in ["restaurant_latitude", "latitude", "lat"] if c in df.columns),
        None,
    )
    lon_col = next(
        (c for c in ["restaurant_longitude", "longitude", "lon", "lng"] if c in df.columns),
        None,
    )

    count = 0
    with get_db() as db:
        for _, row in df.iterrows():
            name = row.get("name")
            if pd.isna(name):
                continue

            existing = db.query(Restaurant).filter(Restaurant.name == str(name)).first()
            restaurant = existing or Restaurant(name=str(name))
            if not existing:
                db.add(restaurant)

            for df_col, db_col in col_map.items():
                if df_col in row and not pd.isna(row[df_col]):
                    setattr(restaurant, db_col, row[df_col])

            if lat_col and lon_col and not pd.isna(row.get(lat_col)) and not pd.isna(row.get(lon_col)):
                restaurant.latitude = float(row[lat_col])
                restaurant.longitude = float(row[lon_col])

            count += 1
            if count % 100 == 0:
                db.flush()

    logger.info("Seeded %d restaurants into database", count)
    return count


def seed_orders(
    df: pd.DataFrame,
    predictions: Optional[np.ndarray] = None,
    config: Optional[DabbaConfig] = None,
) -> int:
    """Insert delivery orders into the database.

    Args:
        df: Delivery features DataFrame (from :func:`add_delivery_features`).
        predictions: Optional array of ETA predictions (same length as df).
        config: Project configuration.

    Returns:
        Number of orders inserted.
    """
    config = config or get_config()
    init_db(config)

    sla_threshold = config.sla_threshold_minutes
    count = 0
    with get_db() as db:
        for i, (_, row) in enumerate(df.iterrows()):
            pred_eta = float(predictions[i]) if predictions is not None else row.get("time_taken_min", 30.0)
            actual_eta = float(row.get("time_taken_min", pred_eta))

            order = Order(
                distance_km=float(row.get("haversine_distance_km", 0)),
                traffic_level=int(row.get("traffic_ordinal", 0)),
                is_festival=bool(row.get("is_festival", False)),
                delivery_person_age=float(row["delivery_person_age"]) if not pd.isna(row.get("delivery_person_age")) else None,
                delivery_person_rating=float(row["delivery_person_ratings"]) if not pd.isna(row.get("delivery_person_ratings")) else None,
                vehicle_condition=int(row["vehicle_condition"]) if not pd.isna(row.get("vehicle_condition")) else None,
                predicted_eta=pred_eta,
                actual_eta=actual_eta,
                sla_threshold=sla_threshold,
                is_at_risk=bool(pred_eta > sla_threshold),
                actual_late=bool(actual_eta > sla_threshold) if not pd.isna(actual_eta) else None,
            )
            db.add(order)
            count += 1
            if count % 100 == 0:
                db.flush()

    logger.info("Seeded %d orders into database", count)
    return count


def clear_all(config: Optional[DabbaConfig] = None) -> None:
    """Delete all rows from all tables in a single transaction.

    Uses one ``with get_db():`` block so that if the Restaurant delete
    fails, the Order delete is rolled back too — no inconsistent state.
    """
    config = config or get_config()
    init_db(config)
    with get_db() as db:
        db.query(Order).delete()
        db.query(Restaurant).delete()
    logger.info("Cleared all data from database")


# ─── CLI ─────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the Dabba database from CSV files",
    )
    parser.add_argument(
        "--restaurants-csv",
        default=None,
        help="Path to processed restaurants CSV (default: config.data_processed_dir/restaurants_processed.csv)",
    )
    parser.add_argument(
        "--delivery-csv",
        default=None,
        help="Path to processed delivery CSV (default: config.data_processed_dir/delivery_processed.csv)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before seeding",
    )
    parser.add_argument(
        "--full-import",
        action="store_true",
        help="Run full import pipeline: load raw CSV → clean → feature engineer → seed to DB",
    )
    return parser.parse_args()


def full_import(config: Optional[DabbaConfig] = None) -> None:
    """Run the full CSV→DB import: load raw → clean → feature engineer → seed.

    This is the main command for migrating from CSV to database.
    It runs the entire data processing pipeline and persists the
    results to the database, ready for API queries and model training.

    Args:
        config: Project configuration.
    """
    config = config or get_config()
    logger.info("=== Full CSV→DB Import ===")

    from dabba.data.cleaning import clean_delivery, clean_zomato
    from dabba.data.loaders import load_delivery, load_zomato
    from dabba.features.delivery_features import add_delivery_features
    from dabba.features.restaurant_features import add_restaurant_features

    # Restaurants: load → clean → feature engineer → seed
    logger.info("--- Restaurants ---")
    df_zomato = load_zomato(config)
    df_zomato = clean_zomato(df_zomato, config)
    df_zomato = add_restaurant_features(df_zomato)

    n_rest = seed_restaurants(df_zomato, config)
    logger.info("✅ %d restaurants seeded to database", n_rest)

    # Save processed CSV for reference
    processed_path = config.data_processed_dir / "restaurants_processed.csv"
    config.data_processed_dir.mkdir(parents=True, exist_ok=True)
    df_zomato.to_csv(processed_path, index=False)
    logger.info("Saved processed restaurants to %s", processed_path)

    # Delivery: load → clean → feature engineer → seed
    logger.info("--- Delivery Orders ---")
    try:
        df_delivery = load_delivery(config)
        df_delivery = clean_delivery(df_delivery, config)
        df_delivery = add_delivery_features(df_delivery, config)

        n_orders = seed_orders(df_delivery, config=config)
        logger.info("✅ %d orders seeded to database", n_orders)

        # Save processed CSV for reference
        del_path = config.data_processed_dir / "delivery_processed.csv"
        df_delivery.to_csv(del_path, index=False)
        logger.info("Saved processed delivery data to %s", del_path)
    except Exception as e:
        logger.warning("Delivery import skipped: %s", e)

    logger.info("=== Full Import Complete ===")


def main() -> None:
    """CLI entry point for database seeding."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    config = get_config()

    if args.full_import:
        full_import(config)
        return

    if args.clear:
        clear_all(config)

    rest_path = args.restaurants_csv or str(config.data_processed_dir / "restaurants_processed.csv")
    del_path = args.delivery_csv or str(config.data_processed_dir / "delivery_processed.csv")

    # Load and seed restaurants
    try:
        df_rest = pd.read_csv(rest_path)
        n = seed_restaurants(df_rest, config)
        logger.info("✅ %d restaurants seeded", n)
    except FileNotFoundError:
        logger.warning("Restaurants CSV not found at %s — skipping", rest_path)

    # Load and seed orders
    try:
        df_del = pd.read_csv(del_path)
        n = seed_orders(df_del, config=config)
        logger.info("✅ %d orders seeded", n)
    except FileNotFoundError:
        logger.warning("Delivery CSV not found at %s — skipping", del_path)


if __name__ == "__main__":
    main()
