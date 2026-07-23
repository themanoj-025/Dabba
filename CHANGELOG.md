# Changelog

All notable changes to Dabba are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **P2 — Structured JSON logging**: New ``dabba/observability`` module with a custom
  ``JSONFormatter`` that outputs every log line as JSON with timestamp, level, logger,
  message, and request ID (via async-safe ``contextvars``). Applied globally across
  the FastAPI application (``api/main.py``).
- **P2 — Prometheus /metrics endpoint**: ``GET /metrics`` returns Prometheus text-format
  metrics including request count, latency histogram (bucketed), error rate per route,
  drift-event counter, concierge tool-call counter, and model-loaded gauges.
  ``prometheus-client`` added to ``requirements.txt``.
- **P2 — Concierge ReAct loop tracing**: Each tool execution in the concierge's ReAct loop
  is tracked with a structured log span (tool name, step, duration) and a Prometheus
  histogram (``dabba_concierge_loop_duration_seconds``).
- **P2 — Request ID middleware**: Every HTTP request receives a ``X-Request-ID`` header
  and a unique request ID that propagates into all log lines from that request context.
- **P2 — Custom drift-event and tool-call counters**: ``dabba_drift_events_total`` and
  ``dabba_concierge_tool_calls_total`` Prometheus counters added for operational insight.
- **P0 — CSV→DB migration for serving paths**: ``api/routers/model_info.py``,
  ``api/routers/recommend.py``, and ``api/routers/chat.py`` now read from the database
  via repository functions instead of ``pd.read_csv()``. All 4 Streamlit pages also
  migrated. CSV loading is now restricted to ``database/seed.py`` (the import pipeline).
- **New repository functions**: ``get_all_restaurants_as_df()`` returns all restaurants
  as a feature-engineered DataFrame (with cuisine one-hot encoding).
  ``get_all_experiment_results()`` returns experiment results sorted by MAE.
- **CSV-read prohibition tests**: Static AST checks assert no ``pd.read_csv()`` calls
  exist in API routers or Streamlit pages (``tests/test_api.py``).

### Fixed
- ``tests/integration/test_concierge.py``: ``test_get_eta_no_model`` assertion updated
  to match actual ``note`` field (``"approximate (no model loaded)"``).
- ``tests/test_narrator.py``: Fixed 6 ``TestRulesNarrate`` tests that were calling
  ``_rules_narrate()`` with 3 positional arguments — the function requires 4
  (missing ``eta_prediction=None``).

### Added (pre-v0.5.0 history, kept for attribution)

- **P1 — `/v1/explain/{prediction_id}` endpoint**: New ``api/routers/explain.py`` router that
  reads from the existing ``predictions`` table and exposes stored SHAP values through the API.
  Includes ``ExplainResponse`` schema, ``get_prediction_by_id()`` repository function, and
  router registration in ``api/main.py``. Closing the explainability loop referenced in
  ``database/models.py`` docstring.
- **P0 Bug Fix — ETA endpoint feature mismatch**: ``api/routers/eta.py`` now builds the
  full ~20-column feature vector (cyclical time encoding, rush hour, interaction terms,
  city zone, weather, age buckets) via :func:`build_eta_features_for_api` from
  ``delivery_features.py``, matching the training pipeline exactly. Previously it sent
  only 6 features to a model trained on 20+ — predictions were silently degraded.
- **Shared `ETA_FEATURE_COLS` constant**: Single source of truth in ``delivery_features.py``
  used by both ``pipeline.py`` (training) and the API serving endpoint, preventing future
  feature drift between training and serving.
- **P0 Bug Fix — Concierge ETA stub**: ``ConciergeTools.get_eta_estimate()`` now uses the
  real loaded ETA model via ``build_eta_features_for_api()`` instead of returning a hardcoded
  30-minute estimate. Falls back gracefully to a formula-based estimate on model error.
- **ETA model wired to concierge at startup**: ``api/main.py`` passes ``app.state.eta_model``
  to ``chat._load_concierge_tools()``, giving the concierge access to real predictions.
- **Helper functions for feature parity**: ``_hour_bucket()`` and ``_age_bucket()`` extracted
  from ``add_delivery_features()`` into reusable module-level functions in ``delivery_features.py``.
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
