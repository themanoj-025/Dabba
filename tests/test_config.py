"""Tests for dabba.config — DabbaConfig defaults, paths, and properties."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestDabbaConfig:
    """DabbaConfig defaults and computed properties."""

    def test_default_config_instantiates(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.random_seed == 42
        assert cfg.test_size == 0.2
        assert cfg.cv_folds == 5

    def test_default_paths_relative_to_project_root(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        root = cfg.project_root
        assert cfg.data_raw_dir == root / "data" / "raw"
        assert cfg.data_processed_dir == root / "data" / "processed"
        assert cfg.models_dir == root / "models"
        assert cfg.reports_dir == root / "reports"
        assert cfg.reports_figures_dir == root / "reports" / "figures"

    def test_explicit_paths_override_defaults(self, tmp_path: Path) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig(
            project_root=tmp_path,
            data_raw_dir=tmp_path / "my_raw",
            models_dir=tmp_path / "my_models",
        )
        assert cfg.data_raw_dir == tmp_path / "my_raw"
        assert cfg.models_dir == tmp_path / "my_models"
        assert cfg.reports_dir == tmp_path / "reports"

    def test_zomato_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.zomato_path == cfg.data_raw_dir / "zomato.csv"

    def test_delivery_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.delivery_path == cfg.data_raw_dir / "deliverytime.csv"

    def test_best_rating_model_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.best_rating_model_path == cfg.models_dir / "best_rating_model.pkl"

    def test_best_eta_model_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.best_eta_model_path == cfg.models_dir / "best_eta_model.pkl"

    def test_best_collaborative_model_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        expected = cfg.models_dir / "best_collaborative_model.pt"
        assert cfg.best_collaborative_model_path == expected

    def test_rating_comparison_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.rating_comparison_path == cfg.reports_dir / "model_comparison_rating.csv"

    def test_eta_comparison_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.eta_comparison_path == cfg.reports_dir / "model_comparison_eta.csv"

    def test_synthetic_interactions_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.synthetic_interactions_path == cfg.data_processed_dir / "synthetic_interactions.csv"

    def test_faiss_index_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.faiss_index_path == cfg.models_dir / "restaurant_faiss.index"

    def test_restaurant_embeddings_path(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        expected = cfg.models_dir / "restaurant_embeddings.npy"
        assert cfg.restaurant_embeddings_path == expected


class TestGetConfig:
    """get_config() factory returns a fresh instance."""

    def test_returns_dabba_config(self) -> None:
        from dabba.config import DabbaConfig, get_config

        cfg = get_config()
        assert isinstance(cfg, DabbaConfig)

    def test_returns_new_instance_each_time(self) -> None:
        from dabba.config import get_config

        c1 = get_config()
        c2 = get_config()
        assert c1 is not c2


class TestLLMConfig:
    """LLM-related config values."""

    def test_llm_model_default(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.llm_model == "claude-sonnet-4-20250514"

    def test_llm_max_tokens_default(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.llm_max_tokens == 1000

    def test_llm_max_steps_default(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.llm_max_steps == 4

    def test_llm_max_steps_range(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert 1 <= cfg.llm_max_steps <= 20


class TestBusinessDefaults:
    """Business metric defaults."""

    def test_sla_threshold(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert cfg.sla_threshold_minutes == 40.0

    def test_reliability_weights_sum_to_one(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        total = cfg.reliability_w_rating + cfg.reliability_w_sentiment + cfg.reliability_w_delay
        assert abs(total - 1.0) < 1e-6

    def test_hybrid_weights_sum_to_one(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        total = cfg.hybrid_weight_content + cfg.hybrid_weight_collaborative + cfg.hybrid_weight_reliability
        assert abs(total - 1.0) < 1e-6

    def test_drift_thresholds(self) -> None:
        from dabba.config import DabbaConfig

        cfg = DabbaConfig()
        assert 0 < cfg.drift_ks_threshold < 1
        assert cfg.drift_feature_sample > 0
