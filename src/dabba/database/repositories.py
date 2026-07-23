"""Repository layer — read/write database access behind the API.

Provides functions to query the Dabba database (SQLite locally,
Postgres in production) for restaurants, orders, experiment results,
and drift logs. All functions accept an optional SQLAlchemy ``Session``
for dependency injection.

Usage (FastAPI):
    .. code-block:: python

        from fastapi import Depends
        from sqlalchemy.orm import Session

        from dabba.database.repositories import get_all_restaurants
        from dabba.database.session import get_db_generator

        @app.get("/v1/restaurants")
        def list_restaurants(
            db: Session = Depends(get_db_generator),
        ):
            return get_all_restaurants(db, limit=20)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from dabba.database.models import (
    DriftLog,
    ExperimentResult,
    Order,
    Prediction,
    Restaurant,
)

logger = logging.getLogger(__name__)


# ─── Restaurants ─────────────────────────────────────────────────────


def get_all_restaurants(
    db: Session,
    limit: int = 50,
    offset: int = 0,
) -> List[Restaurant]:
    """Fetch paginated list of restaurants, ordered by name.

    Args:
        db: Database session.
        limit: Max rows to return.
        offset: Row offset for pagination.

    Returns:
        List of Restaurant ORM instances.
    """
    return (
        db.query(Restaurant)
        .order_by(Restaurant.name)
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_restaurant_by_id(db: Session, restaurant_id: int) -> Optional[Restaurant]:
    """Fetch a single restaurant by primary key.

    Args:
        db: Database session.
        restaurant_id: Restaurant primary key.

    Returns:
        Restaurant instance or None.
    """
    return db.query(Restaurant).filter(Restaurant.id == restaurant_id).first()


def get_restaurant_by_name(db: Session, name: str) -> Optional[Restaurant]:
    """Fetch a single restaurant by name (case-insensitive).

    Args:
        db: Database session.
        name: Restaurant name.

    Returns:
        Restaurant instance or None.
    """
    return (
        db.query(Restaurant)
        .filter(Restaurant.name.ilike(f"%{name}%"))
        .first()
    )


def get_restaurants_by_cuisine(
    db: Session,
    cuisine: str,
    limit: int = 20,
) -> List[Restaurant]:
    """Search restaurants by cuisine keyword.

    Args:
        db: Database session.
        cuisine: Cuisine keyword to search for.
        limit: Max results.

    Returns:
        List of matching Restaurant instances.
    """
    return (
        db.query(Restaurant)
        .filter(Restaurant.cuisines.ilike(f"%{cuisine}%"))
        .order_by(desc(Restaurant.bayesian_rating))
        .limit(limit)
        .all()
    )


def count_restaurants(db: Session) -> int:
    """Return total number of restaurants in the database.

    Args:
        db: Database session.

    Returns:
        Row count.
    """
    return db.query(func.count(Restaurant.id)).scalar() or 0


# ─── Orders ──────────────────────────────────────────────────────────


def get_all_orders(
    db: Session,
    limit: int = 50,
    offset: int = 0,
) -> List[Order]:
    """Fetch paginated orders, newest first.

    Args:
        db: Database session.
        limit: Max rows.
        offset: Row offset.

    Returns:
        List of Order ORM instances.
    """
    return (
        db.query(Order)
        .order_by(desc(Order.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_orders_by_restaurant(
    db: Session,
    restaurant_id: int,
    limit: int = 20,
) -> List[Order]:
    """Fetch orders for a specific restaurant.

    Args:
        db: Database session.
        restaurant_id: Restaurant primary key.
        limit: Max results.

    Returns:
        List of Order instances.
    """
    return (
        db.query(Order)
        .filter(Order.restaurant_id == restaurant_id)
        .order_by(desc(Order.created_at))
        .limit(limit)
        .all()
    )


# ─── Experiment Results ──────────────────────────────────────────────


def get_experiment_results(
    db: Session,
    task: Optional[str] = None,
    limit: int = 20,
) -> List[ExperimentResult]:
    """Fetch experiment results, optionally filtered by task.

    Args:
        db: Database session.
        task: Filter by task ('rating' or 'eta'). If None, return all.
        limit: Max results.

    Returns:
        List of ExperimentResult instances.
    """
    q = db.query(ExperimentResult).order_by(desc(ExperimentResult.created_at))
    if task:
        q = q.filter(ExperimentResult.task == task)
    return q.limit(limit).all()


def get_winning_model(db: Session, task: str) -> Optional[ExperimentResult]:
    """Fetch the winning (best-performing) model for a task.

    Args:
        db: Database session.
        task: 'rating' or 'eta'.

    Returns:
        Winning ExperimentResult or None.
    """
    return (
        db.query(ExperimentResult)
        .filter(ExperimentResult.task == task, ExperimentResult.is_winner.is_(True))
        .order_by(desc(ExperimentResult.created_at))
        .first()
    )# ─── Predictions ──────────────────────────────────────────────────


def get_prediction_by_id(db: Session, prediction_id: int) -> Optional[Prediction]:
    """Fetch a single prediction by primary key.

    Used by the ``/v1/explain/{prediction_id}`` endpoint to retrieve
    stored model predictions with their SHAP explainability values.

    Args:
        db: Database session.
        prediction_id: Prediction primary key.

    Returns:
        Prediction instance or None.
    """
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()


# ─── Drift Logs ──────────────────────────────────────────────────────

def get_recent_drift_logs(
    db: Session,
    limit: int = 50,
    only_alerted: bool = False,
) -> List[DriftLog]:
    """Fetch recent drift detection events.

    Args:
        db: Database session.
        limit: Max results.
        only_alerted: If True, only return logs where alerted=True.

    Returns:
        List of DriftLog instances.
    """
    q = db.query(DriftLog).order_by(desc(DriftLog.detected_at))
    if only_alerted:
        q = q.filter(DriftLog.alerted.is_(True))
    return q.limit(limit).all()


def get_drift_summary(db: Session) -> Dict[str, Any]:
    """Get a summary of drift detection activity.

    Returns:
        Dict with total_drift_events, total_alerted, unique_features,
        and last_detected_at.
    """
    total = db.query(func.count(DriftLog.id)).scalar() or 0
    alerted = (
        db.query(func.count(DriftLog.id))
        .filter(DriftLog.alerted.is_(True))
        .scalar()
        or 0
    )
    features = (
        db.query(func.count(func.distinct(DriftLog.feature_name))).scalar() or 0
    )
    last = (
        db.query(DriftLog.detected_at)
        .order_by(desc(DriftLog.detected_at))
        .first()
    )
    return {
        "total_drift_events": total,
        "total_alerted": alerted,
        "unique_features_monitored": features,
        "last_detected_at": str(last[0]) if last else None,
    }
