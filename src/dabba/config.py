"""Centralized configuration for the Dabba project.

All paths, constants, thresholds, and hyperparameters live here.
No hardcoded values should appear in any other source file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class DabbaConfig(BaseSettings):
    """Root configuration for the Dabba project.

    Attributes:
        project_root: Absolute path to the project root directory.
        data_raw_dir: Path to raw downloaded datasets.
        data_processed_dir: Path to cleaned/processed data.
        models_dir: Path to saved model artifacts (.pkl / .joblib).
        reports_dir: Path to generated reports and figures.
        random_seed: Global random seed for reproducibility.
        test_size: Fraction of data held out for testing.
        cv_folds: Number of folds for cross-validation.
        rating_metric: Primary metric for rating model selection.
        eta_metric: Primary metric for ETA model selection.
        sla_threshold_minutes: Delivery SLA threshold in minutes.
        reliability_w_rating: Weight for rating in reliability score.
        reliability_w_sentiment: Weight for sentiment in reliability score.
        reliability_w_delay: Weight for delay risk in reliability score.
        log_level: Logging level.
    """

    model_config = {"env_prefix": "DABBA_", "env_file": ".env"}

    # --- Paths ---
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    data_raw_dir: Path = Field(default=None)
    data_processed_dir: Path = Field(default=None)
    models_dir: Path = Field(default=None)
    reports_dir: Path = Field(default=None)
    reports_figures_dir: Path = Field(default=None)

    # --- Data ---
    zomato_filename: str = "zomato.csv"
    delivery_filename: str = "deliverytime.csv"
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # --- Model selection ---
    rating_metric: Literal["mae", "rmse", "r2"] = "mae"
    eta_metric: Literal["mae", "rmse", "r2"] = "mae"

    # --- Business ---
    sla_threshold_minutes: float = 40.0
    reliability_w_rating: float = 0.4
    reliability_w_sentiment: float = 0.3
    reliability_w_delay: float = 0.3

    # --- Logging ---
    log_level: str = "INFO"

    def model_post_init(self, __context: object) -> None:
        """Set default paths relative to project root if not explicitly provided."""
        root = self.project_root
        if self.data_raw_dir is None:
            self.data_raw_dir = root / "data" / "raw"
        if self.data_processed_dir is None:
            self.data_processed_dir = root / "data" / "processed"
        if self.models_dir is None:
            self.models_dir = root / "models"
        if self.reports_dir is None:
            self.reports_dir = root / "reports"
        if self.reports_figures_dir is None:
            self.reports_figures_dir = root / "reports" / "figures"

    @property
    def zomato_path(self) -> Path:
        """Full path to the raw Zomato dataset."""
        return self.data_raw_dir / self.zomato_filename

    @property
    def delivery_path(self) -> Path:
        """Full path to the raw delivery dataset."""
        return self.data_raw_dir / self.delivery_filename

    @property
    def best_rating_model_path(self) -> Path:
        """Path to the saved best rating model artifact."""
        return self.models_dir / "best_rating_model.pkl"

    @property
    def best_eta_model_path(self) -> Path:
        """Path to the saved best ETA model artifact."""
        return self.models_dir / "best_eta_model.pkl"

    @property
    def rating_comparison_path(self) -> Path:
        """Path to the rating model comparison CSV."""
        return self.reports_dir / "model_comparison_rating.csv"

    @property
    def eta_comparison_path(self) -> Path:
        """Path to the ETA model comparison CSV."""
        return self.reports_dir / "model_comparison_eta.csv"


def get_config() -> DabbaConfig:
    """Return a fresh DabbaConfig instance.

    Returns:
        DabbaConfig: The project configuration.
    """
    return DabbaConfig()
