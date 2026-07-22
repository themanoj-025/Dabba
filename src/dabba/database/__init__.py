"""Database models, session management, and Alembic migrations.

Provides SQLAlchemy ORM models for structured storage of:
    - Restaurants (from Zomato CSV)
    - Orders (from delivery CSV)
    - Predictions (for explainability API)
    - Experiment results (mirrors MLflow for fast dashboard queries)
    - Drift logs (drift detection history with alerting)
"""

from src.dabba.database.models import (
    Restaurant,
    Order,
    Prediction,
    ExperimentResult,
    DriftLog,
)
from src.dabba.database.session import get_db, init_db

__all__ = [
    "Restaurant",
    "Order",
    "Prediction",
    "ExperimentResult",
    "DriftLog",
    "get_db",
    "init_db",
]
