# 🏗️ Dabba v4 — System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DABBA v4 SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────┐
 │                    DATA PIPELINE (src/dabba/)                        │
 │                                                                      │
 │  Kaggle (Zomato + Delivery CSVs) → SQLite/SQLAlchemy                │
 │         │                                                            │
 │         ▼                                                            │
 │  ┌──────────────┐                                                    │
 │  │ Data Loading  │  loaders.py (pandas)                              │
 │  └──────┬───────┘                                                    │
 │         ▼                                                            │
 │  ┌──────────────┐                                                    │
 │  │ Data Cleaning │  cleaning.py (pandas, numpy)                      │
 │  └──────┬───────┘                                                    │
 │         ▼                                                            │
 │  ┌──────────────────────┐                                            │
 │  │  Feature Engineering │  restaurant_features.py, delivery_features │
 │  │  • Cuisine encoding  │  .py, geo.py, sentiment.py                │
 │  │  • Distance (haversine)│ (now with rush hour, sin/cos encoding,  │
 │  │  • Time features     │  city zone, weather, interaction features)│
 │  └──────┬───────┬───────┘                                           │
 │         │       │                                                   │
 │         ▼       ▼                                                   │
 │  ┌──────────┐ ┌──────────┐  ┌──────────────┐  ┌──────────────┐     │
 │  │ Rating    │ │ ETA      │  │ Collaborative │  │ Geographic   │     │
 │  │ Models    │ │ Models   │  │ Filtering     │  │ Clustering   │     │
 │  │ (9+ algos) │ │ (10+ algos)│  │ (PyTorch MF)│  │ (KMeans etc)│     │
 │  │+ Optuna   │ │+ Optuna  │  │               │  │               │     │
 │  │ HPO       │ │ HPO      │  │               │  │               │     │
 │  └────┬─────┘ └────┬─────┘  └──────┬───────┘  └──────────────┘     │
 │       │            │               │                                │
 │       └────────────┼───────────────┘                                │
 │                    │                                                │
 │                    ▼                                                │
 │         ┌──────────────────────┐                                    │
 │         │  Hybrid Recommender  │  Content + CF + Reliability Score   │
 │         │  + LLM Narrator     │  + A/B weight scenarios             │
 │         └──────────┬───────────┘                                    │
 │                    │                                                │
 │                    ▼                                                │
 │         ┌──────────────────────┐                                    │
 │         │  Reliability Score   │  w1*rating + w2*sentiment          │
 │         │  + A/B Scenarios    │  - w3*delay_risk                   │
 │         └──────────┬───────────┘                                    │
 │                    │                                                │
 │         ┌──────────▼──────────┐                                     │
 │         │  Drift Detection    │  KS-test + Slack webhook            │
 │         │  + Cooldown Mgmt   │  + DB DriftLog persistence          │
 │         └─────────────────────┘                                     │
 └─────────────────────────────────────────────────────────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
     ┌──────────┐         ┌──────────┐          ┌──────────┐
     │ Streamlit│         │  FastAPI  │          │  MLflow  │
     │ Dashboard│         │  REST API │          │ Tracking │
     │ (4 pages)│         │ (8 routes)│          │ (Docker) │
     │ Custom   │         │ Models in │          │ Port 5000│
     │ Radio Nav│         │ app.state │          │          │
     │ Redis    │         │ via Depends│          │          │
     │ caching  │         │ DI        │          │          │
     └──────────┘         └──────────┘          └──────────┘
            │                     │
            ▼                     ▼
     ┌──────────────────────────────────────┐
     │        LLM Layer (Anthropic)          │
     │  ┌────────────┬──────────┬─────────┐  │
     │  │ Narrator   │ RAG      │ Chat    │  │
     │  │ Explanations│ Retrieval│ Copilot │  │
     │  │            │          │ (ReAct  │  │
     │  │            │          │ 4-step  │  │
     │  │            │          │  loop)  │  │
     │  └────────────┴──────────┴─────────┘  │
     │  Rules-based fallback (no API key)     │
     └──────────────────────────────────────┘
            │                     │
            ▼                     ▼
     ┌──────────┐         ┌──────────┐
     │ SQLite   │         │  Redis   │
     │ (dev) /  │         │  Cache   │
     │ Postgres │         │(fakeredis│
     │ (prod)   │         │ fallback)│
     │ Alembic  │         │          │
     └──────────┘         └──────────┘
```

## Key Architectural Decisions

### State Management: app.state (not module globals)
All models (ETA, recommender, concierge tools) are loaded at startup and stored in `app.state`, accessed via `Depends()` injection — no module-level globals, no thread locks.

### ML Pipeline: Optuna HPO
Ensemble models (XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting) are tuned with Optuna (TPE sampler, 50 trials default) before comparison, replacing hardcoded defaults.

### LLM Concierge: ReAct Loop
The food concierge uses a proper ReAct loop (max 4 steps, configurable via `llm_max_steps`) where tool results are fed back to the LLM for multi-step reasoning chains (e.g., search → filter → check ETA → summarize). Falls back to rules-based intent matching when no API key is configured.

### Docker: Per-Service Containers
Each service (API, Streamlit, MLflow) has its own Dockerfile with independent health checks and proper startup ordering via `depends_on: condition: service_healthy`. The API container runs `alembic upgrade head` on startup via `entrypoint.sh` to always apply the latest DB migrations.

### Authentication + Rate Limiting
All `/v1/*` endpoints require `X-API-Key` header. Dev mode bypasses auth when no key is configured. Rate limiting via `slowapi` (10-60 req/min per endpoint). Security headers (CSP, XFO, HSTS) on all responses.

### ETA Feature Engineering: Expanded Feature Set
New features added: `is_rush_hour`, `hour_sin/cos`, `dow_sin/cos`, `city_zone`, `weather_encoded`, `distance_traffic_interaction`, `distance_festival_interaction`. Previously-unused features (`order_hour`, `day_of_week`, `is_weekend`, `order_hour_bucket`) now included in training.

## Module Dependencies

```
pipeline.py (orchestrator)
 ├── config.py
 ├── data/loaders.py → config.py
 ├── data/cleaning.py → config.py
 ├── features/restaurant_features.py → config.py, geo.py
 ├── features/delivery_features.py → config.py, geo.py
 ├── features/geo.py → scikit-learn
 ├── nlp/sentiment.py → config.py, nltk
 ├── nlp/hinglish_sentiment.py → config.py, transformers (optional), nltk (fallback)
 ├── features/traffic.py → config.py, requests (optional TomTom/Mappls)
 ├── models/rating_model.py → config.py, sklearn, xgboost, lightgbm, catboost
 ├── models/eta_model.py → config.py, sklearn, xgboost, lightgbm, catboost
 ├── models/model_selection.py → config.py
 ├── models/base_trainer.py → config.py (shared CV/MLflow/HPO logic)
 ├── models/collaborative_recommender.py → config.py, torch
 ├── models/hybrid_recommender.py → config.py, recommender.py
 ├── evaluation/metrics.py → sklearn
 ├── evaluation/business_cost.py → config.py
 ├── observability/__init__.py → logging, json, contextvars, prometheus_client
 └── monitoring/retrain.py → subprocess, drift.py (DriftResult)

api/main.py (FastAPI — models stored in app.state)
 ├── config.py
 ├── api/schemas.py → pydantic
 ├── api/auth.py → config.py (API key verification)
 ├── api/limiter.py → slowapi
├── routers/recommend.py → schemas, hybrid_recommender, llm, Depends(app.state)
├── routers/eta.py → schemas, config, Depends(app.state)
├── routers/chat.py → schemas, llm/food_concierge, Depends(app.state)
├── routers/model_info.py → schemas, config
└── routers/restaurants.py → schemas, repositories, database.session

app/streamlit_app.py (Dashboard)
 ├── assets/theme.css → custom food-tech design system
 ├── pages/page_discover.py → dabba.models, dabba.llm
 ├── pages/page_ops.py → dabba.monitoring, app.components
 ├── pages/page_model_performance.py → pandas, plotly
 ├── pages/page_concierge.py → dabba.llm.food_concierge (ReAct loop)
 └── utils/sanitize.py → html_escape (XSS prevention)

Additional modules:
 ├── database/session.py → config.py, sqlalchemy (engine, session factory)
 ├── database/models.py → sqlalchemy (5 ORM tables)
 ├── database/seed.py → config.py, models (CSV→DB import)
 ├── database/repositories.py → models (12+ read functions)
 ├── cache/redis_client.py → config.py, redis/fakeredis
 └── monitoring/drift.py → config.py, scipy.stats (KS-test + Slack)

docker/
 ├── api.Dockerfile        → FastAPI + uvicorn + healthcheck + entrypoint.sh
 ├── streamlit.Dockerfile  → Streamlit dashboard + healthcheck
 ├── mlflow.Dockerfile     → MLflow tracking server + healthcheck
 └── entrypoint.sh         → alembic upgrade head + exec uvicorn
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    docker-compose.yml                    │
│                                                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  api: api.Docker│  │ streamlit:   │  │ mlflow:    │ │
│  │  file           │  │ streamlit.   │  │ mlflow.    │ │
│  │  :8000          │  │ Dockerfile   │  │ Dockerfile │ │
│  │  healthcheck:   │  │ :8501        │  │ :5000      │ │
│  │  /health        │  │ depends_on:  │  │ healthcheck│ │
│  │  depends_on:    │  │ api (healthy)│  │ /api/...   │ │
│  │  mlflow(healthy)│  │              │  │            │ │
│  └─────────────────┘  └──────────────┘  └────────────┘ │
│                                                         │
│  Healthcheck chain: mlflow → api → streamlit            │
│  All services: restart: unless-stopped                  │
└─────────────────────────────────────────────────────────┘
```

## Authentication + Security

- **All `/v1/*` endpoints**: Require `X-API-Key` header (via `api/auth.py`)
- **Dev mode**: No key configured → auth skipped for local development
- **Rate limiting**: Via `slowapi` — different limits per endpoint (10-60 req/min)
- **Health endpoint**: `/health` intentionally unauthenticated for monitoring
- **Security headers**: CSP (`default-src 'none'`), X-Content-Type-Options, X-Frame-Options (DENY), Permissions-Policy on all responses via FastAPI middleware

## Testing Infrastructure

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_api.py` | 7 | FastAPI smoke tests (health, model-info, ETA, recommend, chat, auth, CSV-read prohibition) |
| `test_features.py` | 18 | Cyclical encoding, city zone, rush hour, interaction features |
| `test_drift.py` | 13 | KS-test detection, Slack alerting, cooldown management |
| `test_database.py` | 16 | Seed functions, repository queries, in-memory SQLite |
| `test_db_loaders.py` | 11 | DB-backed loaders with CSV fallback |
| `test_optuna_tuning.py` | ~25 | HPO search spaces, trial sampling, MLflow integration |
| `test_narrator.py` | 14 | LLM + rules-based recommendation narration |
| `test_rag_similar.py` | 16 | FAISS/sklearn embedding + similarity retrieval |
| `test_redis_client.py` | 19 | Redis caching (set/get/delete/flush, fakeredis fallback) |
| `test_recommender.py` | 14 | Bayesian average, content-based recommender |
| `test_optimizer.py` | 16 | Hungarian algorithm assignment strategies |
| `test_traffic.py` | — | Traffic simulation and level estimation |
| `test_retrain.py` | — | Drift-triggered retraining hook |
| `test_metrics.py` | — | Regression metric calculations |
| `test_business_cost.py` | — | SLA analysis, reliability score, A/B scenarios |
| `test_collaborative_recommender.py` | — | Matrix factorization training |
| `test_rating_model.py` | — | Rating model pipeline |
| `test_eta_model.py` | — | ETA model pipeline |
| `test_model_selection.py` | — | Model selection and comparison |
| `test_cleaning.py` | — | Data cleaning |
| `integration/test_concierge.py` | 27 | ReAct loop, intent matching, tool execution |
| `e2e/test_workflow.py` | 6 | Full pipeline end-to-end |
