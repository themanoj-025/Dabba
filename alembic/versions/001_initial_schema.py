"""Initial schema: restaurants, orders, predictions, experiment_results, drift_logs

Revision ID: 001
Revises: None
Create Date: 2026-07-22
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- restaurants ---
    op.create_table(
        "restaurants",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("rate", sa.Float(), nullable=True),
        sa.Column("bayesian_rating", sa.Float(), nullable=True),
        sa.Column("cost_for_two", sa.Float(), nullable=True),
        sa.Column("location", sa.String(255), nullable=True),
        sa.Column("cuisines", sa.Text(), nullable=True),
        sa.Column("votes", sa.Integer(), nullable=True),
        sa.Column("votes_log", sa.Float(), nullable=True),
        sa.Column("online_order_binary", sa.Integer(), nullable=True),
        sa.Column("book_table_binary", sa.Integer(), nullable=True),
        sa.Column("cuisine_count", sa.Integer(), nullable=True),
        sa.Column("avg_sentiment", sa.Float(), nullable=True),
        sa.Column("reliability_score", sa.Float(), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        comment="Restaurants from Zomato Bangalore dataset",
    )
    op.create_index(op.f("ix_restaurants_name"), "restaurants", ["name"])
    op.create_index(op.f("ix_restaurants_location"), "restaurants", ["location"])

    # --- orders ---
    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("restaurant_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("distance_km", sa.Float(), nullable=False),
        sa.Column("traffic_level", sa.Integer(), nullable=False),
        sa.Column("is_festival", sa.Boolean(), nullable=False),
        sa.Column("delivery_person_age", sa.Float(), nullable=True),
        sa.Column("delivery_person_rating", sa.Float(), nullable=True),
        sa.Column("vehicle_condition", sa.Integer(), nullable=True),
        sa.Column("predicted_eta", sa.Float(), nullable=False),
        sa.Column("actual_eta", sa.Float(), nullable=True),
        sa.Column("sla_threshold", sa.Float(), nullable=False),
        sa.Column("is_at_risk", sa.Boolean(), nullable=False),
        sa.Column("actual_late", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        comment="Delivery orders with ETA predictions",
    )
    op.create_foreign_key(
        "fk_orders_restaurant",
        "orders",
        "restaurants",
        ["restaurant_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # --- predictions ---
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("input_data", sa.JSON(), nullable=True),
        sa.Column("output_value", sa.Float(), nullable=False),
        sa.Column("shap_values", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        comment="Model predictions with inputs for explainability",
    )
    op.create_index(op.f("ix_predictions_model_name"), "predictions", ["model_name"])
    op.create_index(op.f("ix_predictions_input_hash"), "predictions", ["input_hash"])

    # --- experiment_results ---
    op.create_table(
        "experiment_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("task", sa.String(20), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("mae", sa.Float(), nullable=False),
        sa.Column("rmse", sa.Float(), nullable=False),
        sa.Column("r2", sa.Float(), nullable=False),
        sa.Column("train_time_s", sa.Float(), nullable=False),
        sa.Column("mlflow_run_id", sa.String(50), nullable=True),
        sa.Column("is_winner", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        comment="Model experiment results (mirrors MLflow)",
    )

    # --- drift_logs ---
    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_name", sa.String(100), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("n_reference", sa.Integer(), nullable=False),
        sa.Column("n_batch", sa.Integer(), nullable=False),
        sa.Column("alerted", sa.Boolean(), nullable=False),
        sa.Column("detected_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        comment="Drift detection events for monitored features",
    )
    op.create_index(
        op.f("ix_drift_logs_feature_name"), "drift_logs", ["feature_name"]
    )


def downgrade() -> None:
    op.drop_table("drift_logs")
    op.drop_table("experiment_results")
    op.drop_table("predictions")
    op.drop_table("orders")
    op.drop_table("restaurants")
