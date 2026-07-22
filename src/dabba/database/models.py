"""SQLAlchemy ORM models for the Dabba platform.

Schema design principles:
    - All tables use INTEGER auto-increment primary keys named ``id``.
    - Timestamps are stored as UTC-naive and converted on read.
    - JSON columns store dicts/lists; the driver handles serialization.
    - Foreign keys use explicit ``CASCADE`` on delete where appropriate.
    - Column types match the Pandas dtypes of the source CSV data.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all Dabba ORM models."""

    pass


# ─── Restaurants ──────────────────────────────────────────────────────


class Restaurant(Base):
    """A restaurant from the Zomato Bangalore dataset.

    Each row represents a single restaurant with its rating, location,
    cuisines, cost, and computed meta-features (sentiment, reliability).
    """

    __tablename__ = "restaurants"
    __table_args__ = {"comment": "Restaurants from Zomato Bangalore dataset"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    bayesian_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_for_two: Mapped[float | None] = mapped_column(Float, nullable=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    cuisines: Mapped[str | None] = mapped_column(Text, nullable=True)
    votes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    votes_log: Mapped[float | None] = mapped_column(Float, nullable=True)
    online_order_binary: Mapped[int | None] = mapped_column(Integer, nullable=True)
    book_table_binary: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cuisine_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)
    reliability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Restaurant id={self.id} name='{self.name}' rate={self.rate}>"


# ─── Orders ───────────────────────────────────────────────────────────


class Order(Base):
    """A delivery order with ETA prediction and actual outcome.

    Links a restaurant to a delivery event, storing both the model's
    prediction and the actual delivery time for SLA monitoring.
    """

    __tablename__ = "orders"
    __table_args__ = {"comment": "Delivery orders with ETA predictions"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    restaurant_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("restaurants.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    traffic_level: Mapped[int] = mapped_column(Integer, nullable=False)
    is_festival: Mapped[bool] = mapped_column(Boolean, default=False)
    delivery_person_age: Mapped[float | None] = mapped_column(Float, nullable=True)
    delivery_person_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    vehicle_condition: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_eta: Mapped[float] = mapped_column(Float, nullable=False)
    actual_eta: Mapped[float | None] = mapped_column(Float, nullable=True)
    sla_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    is_at_risk: Mapped[bool] = mapped_column(Boolean, default=False)
    actual_late: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationship
    restaurant: Mapped[Restaurant | None] = relationship("Restaurant", lazy="joined")

    def __repr__(self) -> str:
        return f"<Order id={self.id} restaurant_id={self.restaurant_id} eta={self.predicted_eta:.1f}>"


# ─── Predictions ──────────────────────────────────────────────────────


class Prediction(Base):
    """A model prediction logged for explainability and audit.

    Enables the ``/v1/explain/{prediction_id}`` endpoint by storing
    both the raw input and output alongside optional SHAP values.
    """

    __tablename__ = "predictions"
    __table_args__ = {"comment": "Model predictions with inputs for explainability"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    input_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True, comment="SHA-256 of serialized input"
    )
    input_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_value: Mapped[float] = mapped_column(Float, nullable=False)
    shap_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} model='{self.model_name}' "
            f"output={self.output_value:.3f}>"
        )


# ─── Experiment Results ────────────────────────────────────────────────


class ExperimentResult(Base):
    """Results from a model comparison experiment.

    Mirrors key MLflow run metrics for fast dashboard queries without
    needing to hit the MLflow Tracking Server directly.
    """

    __tablename__ = "experiment_results"
    __table_args__ = {"comment": "Model experiment results (mirrors MLflow)"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="'rating' or 'eta'"
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    r2: Mapped[float] = mapped_column(Float, nullable=False)
    train_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_winner: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<ExperimentResult id={self.id} task='{self.task}' "
            f"model='{self.model_name}' mae={self.mae:.4f}>"
        )


# ─── Drift Logs ───────────────────────────────────────────────────────


class DriftLog(Base):
    """A single drift detection event for a feature.

    Inserted when the KS-test p-value for a feature falls below the
    configured threshold. The ``alerted`` flag tracks whether a
    notification (e.g., Slack/email) has been sent.
    """

    __tablename__ = "drift_logs"
    __table_args__ = {"comment": "Drift detection events for monitored features"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    ks_statistic: Mapped[float] = mapped_column(Float, nullable=False)
    p_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    n_reference: Mapped[int] = mapped_column(Integer, nullable=False)
    n_batch: Mapped[int] = mapped_column(Integer, nullable=False)
    alerted: Mapped[bool] = mapped_column(Boolean, default=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<DriftLog feature='{self.feature_name}' "
            f"p={self.p_value:.4f} alerted={self.alerted}>"
        )
