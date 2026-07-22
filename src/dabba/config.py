"""Centralized configuration for the Dabba project v3.

All paths, constants, thresholds, hyperparameters, and API settings
live here. No hardcoded values should appear in any other source file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class DabbaConfig(BaseSettings):
    """Root configuration for the Dabba project v3.

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
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_experiment_name: MLflow experiment name.
        anthropic_api_key: Anthropic API key (optional, for LLM layer).
        llm_enabled: Whether LLM features are enabled.
        llm_model: Model string for Anthropic.
        llm_max_tokens: Max tokens for LLM responses.
        hybrid_weight_content: Weight for content-based score in hybrid recommender.
        hybrid_weight_collaborative: Weight for collaborative filtering score.
        hybrid_weight_reliability: Weight for reliability score.
    """

    model_config = {"env_prefix": "DABBA_", "env_file": ".env"}

    # --- Paths ---
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    data_raw_dir: Optional[Path] = Field(default=None)
    data_processed_dir: Optional[Path] = Field(default=None)
    models_dir: Optional[Path] = Field(default=None)
    reports_dir: Optional[Path] = Field(default=None)
    reports_figures_dir: Optional[Path] = Field(default=None)

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

    # --- Hybrid Recommender Weights ---
    hybrid_weight_content: float = 0.4
    hybrid_weight_collaborative: float = 0.3
    hybrid_weight_reliability: float = 0.3

    # --- API Auth ---
    api_key: Optional[str] = Field(default=None)

    # --- LLM Settings ---
    anthropic_api_key: Optional[str] = Field(default=None)
    llm_enabled: bool = False
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 1000

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "dabba"

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///data/dabba.db",
        description="SQLAlchemy database URL (SQLite for dev, Postgres for prod)",
    )

    # --- Cache ---
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for caching hot predictions",
    )
    cache_eta_ttl_seconds: int = Field(
        default=300,
        description="TTL for ETA prediction cache entries (seconds)",
    )
    cache_recommend_ttl_seconds: int = Field(
        default=600,
        description="TTL for recommendation cache entries (seconds)",
    )

    # --- Drift Detection ---
    drift_ks_threshold: float = 0.05  # p-value threshold for KS test
    drift_feature_sample: int = 100  # samples to use for drift detection

    # --- Optuna Hyperparameter Optimization ---
    optuna_enabled: bool = Field(
        default=True,
        description="Whether to run Optuna hyperparameter tuning before model comparison",
    )
    optuna_n_trials: int = Field(
        default=50,
        description="Number of Optuna trials per model during hyperparameter tuning",
    )
    optuna_timeout_minutes: Optional[int] = Field(
        default=None,
        description="Optional timeout per model tuning in minutes (None = no limit)",
    )
    optuna_models_to_tune: list[str] = Field(
        default=["XGBoost", "LightGBM", "CatBoost", "RandomForest", "GradientBoosting"],
        description="Which ensemble models to run Optuna tuning on",
    )

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
    def best_collaborative_model_path(self) -> Path:
        """Path to the saved collaborative filtering model."""
        return self.models_dir / "best_collaborative_model.pt"

    @property
    def rating_comparison_path(self) -> Path:
        """Path to the rating model comparison CSV."""
        return self.reports_dir / "model_comparison_rating.csv"

    @property
    def eta_comparison_path(self) -> Path:
        """Path to the ETA model comparison CSV."""
        return self.reports_dir / "model_comparison_eta.csv"

    @property
    def synthetic_interactions_path(self) -> Path:
        """Path to the synthetic user-restaurant interaction dataset."""
        return self.data_processed_dir / "synthetic_interactions.csv"

    @property
    def faiss_index_path(self) -> Path:
        """Path to the FAISS index for similar restaurant retrieval."""
        return self.models_dir / "restaurant_faiss.index"

    @property
    def restaurant_embeddings_path(self) -> Path:
        """Path to the restaurant feature embeddings for RAG."""
        return self.models_dir / "restaurant_embeddings.npy"


def get_config() -> DabbaConfig:
    """Return a fresh DabbaConfig instance.

    Returns:
        DabbaConfig: The project configuration.
    """
    return DabbaConfig()
