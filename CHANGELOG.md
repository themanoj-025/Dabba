# Changelog

All notable changes to Dabba are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Comprehensive project analysis**: Full inventory of all ~60 modules across data pipeline,
  ML models, LLM layer, API, dashboard, testing, CI/CD, and documentation
- **Gaps/remaining work documentation**: Detailed 3-tier priority list (High/Medium/Low)
  with 20+ actionable items documented across all markdown files
- **Doc consistency audit**: All 8 markdown files (`README.md`, `architecture.md`, `api-map.md`,
  `database-map.md`, `dependency-graph.md`, `routes.md`, `memory.md`, `CHANGELOG.md`)
  updated to reflect v0.4.0 project state with consistent version numbers, route counts,
  test counts, and architecture descriptions
- **CSV→Postgres migration path**: `python -m dabba.database.seed --full-import` runs the
  complete pipeline (load raw CSV → clean → feature engineer → sentiment → seed to DB)
- **Database-backed data loaders**: `load_zomato_from_db()` and `load_delivery_from_db()`
  in `loaders.py` with `use_db=True` flag on existing loaders for DB-first with CSV fallback
- **REST API for restaurants from DB**: `/v1/restaurants` (list, get by ID with 404, search
  by name/cuisine) — reads from Postgres/SQLite via `get_db_generator` dependency injection
- **Restaurant API schemas**: `RestaurantItem` and `RestaurantListResponse` Pydantic models
- **Docker entrypoint**: `docker/entrypoint.sh` runs `alembic upgrade head` before starting
  uvicorn — schema always up-to-date on container start
- **Makefile DB targets**: `db-import`, `db-migrate`, `db-shell`, `db-rollback`, `db-history`,
  `db-revision` for common database operations
- **Slack drift alerting**: `_send_slack_alert()`, `detect_and_alert()` with cooldown management
  and database persistence via `DriftLog` table
- **11 new ETA features**: `is_rush_hour`, `hour_sin/cos`, `dow_sin/cos`, `city_zone`,
  `weather_encoded`, `distance_traffic_interaction`, `distance_festival_interaction`
- **Pre-existing bug fix**: `order_hour_bucket` was silently dropped from training (dead
  `startswith("order_hour_bucket_")` code in pipeline.py)
- **Integration tests**: `tests/integration/test_concierge.py` (27 tests), `tests/e2e/test_workflow.py`
  (6 tests), `tests/test_database.py` (16 tests), `tests/test_db_loaders.py` (11 tests)
- **Community files**: SECURITY.md, CHANGELOG.md, CODE_OF_CONDUCT.md, .github/ISSUE_TEMPLATE/
  (bug_report.md, feature_request.md), .github/PULL_REQUEST_TEMPLATE.md
- **Postgres + Redis services** in docker-compose.yml with healthchecks
- **Optuna HPO** for 5 ensemble models with configurable search spaces and trials
- **Multi-step ReAct loop** for the Food Concierge (max 4 steps)
- **Separate Dockerfiles** per service with healthchecks
- **`skorch` dependency** pinned in requirements.txt
- **`RESTAURANT_COL_MAP`** shared constant in models.py (eliminates duplication between
  seed.py and pipeline.py)
- **`full_import()`** with CSV existence check before clearing DB, sentiment scores,
  FileNotFoundError handling, and delivery CSV fallback

### Changed
- Module-level globals → FastAPI `app.state` + `Depends()` DI in all API routers
- `_llm_concierge_response` rewritten from single-turn tool append to full ReAct loop
- docker-compose.yml replaced mono-image with per-service Dockerfiles + healthchecks
- `get_eta_estimate()` now checks restaurant existence BEFORE `eta_model is None`
- Intent matching: `re.match` → selective `re.search` for budget/cuisine/reliability patterns
- `_INTENT_PATTERNS` reordered (budget/cuisine before search) for correct intent priority
- ETA regex handles "delivery from X take?" patterns
- `seed.py`: `clear_all()` uses single transaction (was two separate blocks)
- `loaders.py`: ORM attributes accessed inside session block (fixes DetachedInstanceError)
- `get_restaurant` endpoint raises `HTTPException(404)` instead of returning None
- SQLite `pool_size`/`max_overflow` only passed for non-SQLite dialects
- All markdown files updated with new architecture, API routes, and DB migration docs

### Fixed
- `skorch` dependency ghost — was referenced in `eta_model.py` but missing from `requirements.txt`
- Intent matching: "Find cheap restaurants" now correctly matches `budget_search` (not `search`)
- "What's the reliability score for X?" now correctly matches reliability intent
- `get_eta_estimate()` no longer returns fake ETAs for non-existent restaurants
- `order_hour_bucket` no longer silently dropped from ETA training
- `load_delivery_from_db()` handles None `actual_eta` with fallback to `predicted_eta`

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
