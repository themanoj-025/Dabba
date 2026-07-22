# 🏗️ Dabba v3 — System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DABBA v3 SYSTEM ARCHITECTURE                    │
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
     │ (4 pages)│         │ Models in │          │ (Docker) │
     │ Custom   │         │ app.state │          │ Port 5000│
     │ Radio Nav│         │ via DI    │          │          │
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
```

## Key Architectural Decisions

### State Management: app.state (not module globals)
All models (ETA, recommender, concierge tools) are loaded at startup and stored in `app.state`, accessed via `Depends()` injection — no module-level globals, no thread locks.

### ML Pipeline: Optuna HPO
Ensemble models (XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting) are tuned with Optuna (TPE sampler, 50 trials default) before comparison, replacing hardcoded defaults.

### LLM Concierge: ReAct Loop
The food concierge now uses a proper ReAct loop (max 4 steps) where tool results are fed back to the LLM for multi-step reasoning chains (e.g., search → filter → check ETA → summarize).

### Docker: Per-Service Containers
Each service (API, Streamlit, MLflow) has its own Dockerfile with independent health checks and proper startup ordering via `depends_on: condition: service_healthy`.

### Drift Detection: Slack Alerting + Cooldown
When drift is detected, the system can send formatted Slack webhook messages with per-feature details. Duplicate alerts are rate-limited via configurable cooldown to prevent notification fatigue.

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
 ├── models/rating_model.py → config.py, sklearn, xgboost, lightgbm, catboost
 ├── models/eta_model.py → config.py, sklearn, xgboost, lightgbm, catboost
 ├── models/model_selection.py → config.py
 ├── models/base_trainer.py → config.py (shared CV/MLflow/HPO logic)
 ├── models/collaborative_recommender.py → config.py, torch
 ├── models/hybrid_recommender.py → config.py, recommender.py
 ├── evaluation/metrics.py → sklearn
 └── evaluation/business_cost.py → config.py

api/main.py (FastAPI — models stored in app.state)
 ├── config.py
 ├── api/schemas.py → pydantic
 ├── api/auth.py → config.py (API key verification)
 ├── api/limiter.py → slowapi
 ├── routers/recommend.py → schemas, hybrid_recommender, llm, Depends(app.state)
 ├── routers/eta.py → schemas, config, Depends(app.state)
 ├── routers/chat.py → schemas, llm/food_concierge, Depends(app.state)
 └── routers/model_info.py → schemas, config

app/streamlit_app.py (Dashboard)
 ├── pages/page_discover.py → dabba.models, dabba.llm
 ├── pages/page_ops.py → dabba.monitoring, app.components
 ├── pages/page_model_performance.py → pandas, plotly
 └── pages/page_concierge.py → dabba.llm (ReAct loop concierge)

docker/
 ├── api.Dockerfile        → FastAPI + uvicorn + healthcheck
 ├── streamlit.Dockerfile  → Streamlit dashboard + healthcheck
 └── mlflow.Dockerfile     → MLflow tracking server + healthcheck
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

## Authentication

- **All `/v1/*` endpoints**: Require `X-API-Key` header (via `api/auth.py`)
- **Dev mode**: No key configured → auth skipped for local development
- **Rate limiting**: Via `slowapi` — different limits per endpoint (10-60 req/min)
- **Health endpoint**: `/health` intentionally unauthenticated for monitoring
