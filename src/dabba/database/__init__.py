"""Database models, session management, and Alembic migrations.

Provides SQLAlchemy ORM models for structured storage of:
    - Restaurants (from Zomato CSV)
    - Orders (from delivery CSV)
    - Predictions (for explainability API)
    - Experiment results (mirrors MLflow for fast dashboard queries)
    - Drift logs (drift detection history with alerting)
"""

from dabba.database.models import (
    Restaurant,
    Order,
    Prediction,
    ExperimentResult,
    DriftLog,
)
from dabba.database.repositories import (
    count_restaurants,
    get_all_orders,
    get_all_restaurants,
    get_drift_summary,
    get_experiment_results,
    get_orders_by_restaurant,
    get_recent_drift_logs,
    get_restaurant_by_id,
    get_restaurant_by_name,
    get_restaurants_by_cuisine,
    get_winning_model,
)
from dabba.database.seed import seed_orders, seed_restaurants
from dabba.database.session import get_db, init_db

__all__ = [
    "Restaurant",
    "Order",
    "Prediction",
    "ExperimentResult",
    "DriftLog",
    "get_db",
    "init_db",
    "seed_restaurants",
    "seed_orders",
    "count_restaurants",
    "get_all_restaurants",
    "get_all_orders",
    "get_restaurant_by_id",
    "get_restaurant_by_name",
    "get_restaurants_by_cuisine",
    "get_orders_by_restaurant",
    "get_experiment_results",
    "get_winning_model",
    "get_recent_drift_logs",
    "get_drift_summary",
]
