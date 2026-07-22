# Changelog

All notable changes to Dabba are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Optuna hyperparameter optimization for 5 ensemble models (XGBoost, LightGBM,
  CatBoost, RandomForest, GradientBoosting) with configurable search spaces and trials
- Multi-step ReAct loop for the Food Concierge (max 4 steps, tool results fed back
  to the LLM for chained reasoning)
- Separate Dockerfiles per service (`docker/api.Dockerfile`, `docker/streamlit.Dockerfile`,
  `docker/mlflow.Dockerfile`) with healthchecks on all services
- `llm_max_steps` config field for controlling ReAct loop iterations
- Integration and end-to-end test directories
- `skorch` dependency pinned in requirements.txt

### Changed
- `_llm_concierge_response` rewritten from single-turn tool append to full ReAct loop
- docker-compose.yml replaced mono-image with per-service Dockerfiles + healthchecks
- `.dockerignore` updated to exclude `docker/` directory

### Fixed
- `skorch` dependency ghost — was referenced in `eta_model.py` but missing from `requirements.txt`

## [0.2.0] - 2026-06-01

### Added
- Rating and ETA model duplication extracted into shared `base_trainer.py`
- API key authentication on all `/v1` endpoints
- Rate limiting via `slowapi`
- API versioning under `/v1`
- HTML sanitization (`html_escape()`) for user-echoed text in chat
- Redis caching for ETA predictions and recommendations (with `fakeredis` fallback)
- Alembic migrations for database schema
- Gitignores for `kaggle.json`, model artifacts, and reports
- Input validation on `ETARequest` (gt=0, le=100, range checks)

### Changed
- Thread-safe model loading with `threading.Lock` in all API routers
- `HealthResponse` now reports actual model load status

## [0.1.0] - 2026-05-15

### Added
- Initial release with full ML pipeline
- 9-model comparison for rating prediction, 10-model for ETA
- Hybrid recommender (content + collaborative + reliability)
- LLM concierge with single-turn tool use
- Streamlit dashboard with 4 pages
- FastAPI with 5 endpoints
- MLflow experiment tracking
- SHAP explainability
- Drift detection (KS-test) in Ops Monitor
- Collaborative filtering via PyTorch matrix factorization
- Docker setup (monolithic container)
