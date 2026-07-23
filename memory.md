# 🍛 Dabba — Restaurant Intelligence Platform (v0.4.0)

## Project Overview

**Dabba** is an India-focused restaurant ranking, recommendation, and delivery-reliability platform built to senior data-science code-review standard. It combines **deterministic ML** (rating prediction, ETA prediction, collaborative filtering) with an **LLM layer** (recommendation narration, RAG retrieval, chat copilot) — a deliberate hybrid architecture.

The project serves as a comprehensive 2026-era ML portfolio project demonstrating:
- End-to-end ML pipeline with rigorous model comparison (9 rating / 10 ETA algorithms)
- Automatic best-model selection by configurable metric (MAE, RMSE, R²)
- CatBoost + XGBoost + LightGBM + RandomForest + GradientBoosting + more
- PyTorch matrix factorization for collaborative filtering (on synthetic data)
- Hybrid recommender blending content, collaborative, and reliability signals
- SHAP explainability on winning models
- LLM layer with rules-based fallback (Anthropic Claude optional)
- MLflow experiment tracking for all training runs
- KS-test drift detection wired into the Ops Monitor UI
- Optuna HPO for 5 ensemble models
- Redis caching (with fakeredis fallback)
- PostgreSQL/SQLite via SQLAlchemy + Alembic migrations
- FastAPI + Streamlit + Docker deployment
- GitHub Actions CI/CD

---

## Business Purpose

### Problem Statement
India's food-tech landscape generates massive amounts of restaurant and delivery data, yet consumers and operators lack a unified view combining food quality, customer sentiment, and delivery reliability into a single actionable metric.

### Solution
1. Mining Zomato restaurant data for ratings, cuisine diversity, and cost signals
2. Analyzing customer sentiment from reviews using VADER NLP
3. Comparing 10 ETA algorithms to predict delivery time
4. Synthesizing everything into a proprietary **Reliability Score**
5. Adding **collaborative filtering** via PyTorch matrix factorization
6. Wrapping with an **LLM layer** (narrator, RAG retrieval, chat copilot)
7. Monitoring for **drift** via KS-test wired into the UI

### Target Users
- **Customers**: Finding reliable restaurants with good food quality
- **Operations Managers**: Monitoring delivery SLA compliance and drift
- **Data Scientists**: Understanding model performance and methodology

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11+ |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | PyTorch (matrix factorization), skorch (neural net) |
| **NLP** | NLTK (VADER sentiment) |
| **LLM** | Anthropic Claude (optional, rules-based fallback) |
| **Vector Search** | FAISS (with sklearn fallback) |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit, Plotly |
| **API** | FastAPI, Pydantic |
| **Experiment Tracking** | MLflow |
| **Monitoring** | scipy.stats.ks_2samp (drift detection) |
| **HPO** | Optuna (TPE sampler) |
| **Alerting** | Slack Incoming Webhooks |
| **Caching** | Redis (with fakeredis fallback) |
| **Database** | SQLite (dev), PostgreSQL (prod) via SQLAlchemy + Alembic |
| **Testing** | pytest, pytest-cov (100+ tests) |
| **Linting** | Ruff, Black, isort, pre-commit |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker, docker-compose (3 services) |
| **Data** | Kaggle (Zomato Bangalore, Food Delivery Time) |

---

## Repository Structure

```
dabba/
├── api/
│   ├── main.py                       # FastAPI with CORS, security, model loading
│   ├── auth.py                       # API key verification
│   ├── limiter.py                    # Rate limiting via slowapi
│   ├── schemas.py                    # Pydantic request/response models
│   └── routers/
│       ├── recommend.py              # POST /v1/recommend — hybrid recommendations
│       ├── eta.py                    # POST /v1/predict-eta — delivery time
│       ├── chat.py                   # POST /v1/chat — food concierge
│       ├── model_info.py             # GET /v1/model-info — model metadata
│       └── restaurants.py            # GET /v1/restaurants — DB-backed CRUD
├── app/
│   ├── streamlit_app.py              # Entry point, sidebar nav, theme loading
│   ├── assets/theme.css              # Warm food-tech design system
│   ├── components/
│   │   └── restaurant_card.py        # Styled card + metric card components
│   ├── pages/
│   │   ├── page_discover.py          # Customer: filters, cards, LLM explanations
│   │   ├── page_ops.py               # Ops: simulation, drift alerts, metrics
│   │   ├── page_model_performance.py # Model comparison charts, A/B scenarios
│   │   └── page_concierge.py         # Chat copilot with example prompts
│   └── utils/
│       └── sanitize.py               # HTML escaping for XSS prevention
├── data/
│   ├── raw/                          # Raw Kaggle CSVs (gitignored)
│   └── processed/                    # Cleaned + features + interactions (gitignored)
├── models/                           # Saved artifacts: .pkl, .pt (gitignored)
├── notebooks/                        # 6 EDA + prototyping notebooks
├── reports/                          # CSVs, charts, SHAP, JSON (gitignored)
├── docker/
│   ├── api.Dockerfile                # FastAPI + Alembic migration on start
│   ├── streamlit.Dockerfile          # Streamlit dashboard
│   ├── mlflow.Dockerfile             # MLflow tracking server
│   └── entrypoint.sh                 # Alembic upgrade + uvicorn
├── src/dabba/
│   ├── config.py                     # Centralized DabbaConfig (Pydantic, env vars)
│   ├── pipeline.py                   # Full training orchestrator
│   ├── data/
│   │   ├── loaders.py                # CSV + DB-backed loading with fallback
│   │   └── cleaning.py               # Zomato + delivery data cleaning
│   ├── features/
│   │   ├── geo.py                    # Haversine, clustering comparison
│   │   ├── restaurant_features.py    # Cuisine encoding, cost buckets
│   │   └── delivery_features.py      # 11+ features (cyclical, rush hour, city zone, interactions)
│   ├── models/
│   │   ├── base_trainer.py           # Shared CV/MLflow/HPO/HPO search spaces
│   │   ├── rating_model.py           # 9 models compared, MLflow tracking
│   │   ├── eta_model.py              # 10 models compared (+ neural net)
│   │   ├── model_selection.py        # Auto-select best by metric
│   │   ├── collaborative_recommender.py  # PyTorch MF on synthetic data
│   │   ├── hybrid_recommender.py     # Blends 4 signal types
│   │   ├── recommender.py            # Original content-based
│   │   └── optimizer.py              # Hungarian algorithm for partner assignment
│   ├── llm/
│   │   ├── recommendation_narrator.py    # LLM + rules-based explanations
│   │   ├── rag_similar_restaurants.py    # FAISS + sklearn retrieval
│   │   └── food_concierge.py             # ReAct loop (max 4 steps), 3 tools
│   ├── nlp/sentiment.py              # VADER sentiment analysis
│   ├── monitoring/drift.py            # KS-test drift + Slack alerting + cooldown + DB log
│   ├── cache/redis_client.py          # Redis cache (fakeredis fallback)
│   ├── database/
│   │   ├── models.py                 # 5 ORM models + RESTAURANT_COL_MAP
│   │   ├── session.py                # Engine/session factory + FastAPI Depends
│   │   ├── seed.py                   # CSV→DB import with --full-import CLI
│   │   └── repositories.py           # 12+ read functions for API
│   └── evaluation/
│       ├── metrics.py                # Regression metrics
│       └── business_cost.py          # SLA analysis, Reliability Score, A/B scenarios
├── tests/
│   ├── test_api.py                   # 7 tests — FastAPI smoke tests
│   ├── test_cleaning.py              # Data cleaning tests
│   ├── test_features.py              # 18 tests — feature engineering
│   ├── test_drift.py                 # 13 tests — drift detection + Slack
│   ├── test_database.py              # 16 tests — seed, repositories
│   ├── test_db_loaders.py            # 11 tests — DB-backed loaders
│   ├── test_optuna_tuning.py         # ~25 tests — HPO search spaces, tuning
│   ├── test_rating_model.py          # Rating model tests
│   ├── test_eta_model.py             # ETA model tests
│   ├── test_recommender.py           # Recommender tests
│   ├── test_collaborative_recommender.py  # CF tests
│   ├── test_model_selection.py       # Model selection tests
│   ├── integration/test_concierge.py # 27 tests — concierge integration
│   └── e2e/test_workflow.py          # 6 tests — end-to-end workflow
├── pyproject.toml                    # Package configuration + linting config
├── Makefile                          # 15+ targets
├── alembic.ini / alembic/            # Database migrations
├── docker-compose.yml                # 3 services with healthchecks
├── README.md
└── setup_kaggle.py                   # Kaggle authentication + download
```

---

## System Architecture

```
Kaggle (Zomato + Delivery CSVs)
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│               src/dabba/ DATA PIPELINE                               │
│                                                                      │
│  loaders.py → cleaning.py → feature engineering → sentiment (VADER)  │
│       │              │                    │              │           │
│       ▼              ▼                    ▼              ▼           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Rating    │  │ ETA      │  │ Collaborative │  │ Geographic   │     │
│  │ Model     │  │ Model    │  │ Filtering     │  │ Clustering   │     │
│  │ (9 algos) │  │ (10 algos)│  │ (PyTorch MF) │  │ (KMeans etc) │     │
│  │+ Optuna   │  │+ Optuna  │  │               │  │               │     │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └──────────────┘     │
│       │             │               │                                │
│       └─────────────┼───────────────┘                                │
│                     │              MLflow                            │
│                     ▼            ┌──────────┐                        │
│          ┌────────────────────┐  │ Rating   │ + ETA                  │
│          │  Hybrid Recommender  │  │ runs     │                        │
│          │  + LLM Narrator      │  └──────────┘                        │
│          └──────────┬──────────┘                                       │
│                     │               ┌──────────────────────┐           │
│                     ▼               │  Drift Detection     │           │
│          ┌────────────────────┐     │  (scipy KS-test)     │           │
│          │  Reliability Score  │     │  + Slack alerts      │           │
│          │  + A/B Scenarios   │     │  + Cooldown mgmt     │           │
│          └────────────────────┘     │  + DB persistence    │           │
│                     │               └──────────────────────┘           │
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
     ┌────────────────────────────────────────────┐
     │        LLM Layer (Anthropic Claude)          │
     │                                              │
     │  ┌────────────┬──────────┬─────────────────┐ │
     │  │ Narrator   │ RAG      │ Chat Copilot     │ │
     │  │ (2-3 sent) │ Retrieval│ (ReAct 4-step)   │ │
     │  │            │ (FAISS)  │                  │ │
     │  └────────────┴──────────┴─────────────────┘ │
     │  Rules-based fallback (no API key)            │
     └──────────────────────────────────────────────┘
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

---

## Routing

### Streamlit (Custom Radio Navigation — not multi-page auto-discovery)
| Page | Selector | File | Features |
|------|----------|------|----------|
| Discover | `🍽️` | `page_discover.py` | Filters, cards, LLM explanations, RAG similar |
| Ops Monitor | `🚀` | `page_ops.py` | Simulation, drift alerts, metric cards |
| Model Performance | `📊` | `page_model_performance.py` | Charts, SHAP, A/B scenarios |
| Food Concierge | `💬` | `page_concierge.py` | Chat, example prompts, tool-use |

### FastAPI (Separate routers in `api/routers/`)
| Method | Route | File | Purpose | Rate Limit |
|--------|-------|------|---------|------------|
| GET | `/health` | `api/main.py` | Health check + model status | None |
| GET | `/v1/model-info` | `routers/model_info.py` | Deployed models + metrics | 60/min |
| POST | `/v1/recommend` | `routers/recommend.py` | Hybrid recommendations (optional LLM narration) | 30/min |
| POST | `/v1/predict-eta` | `routers/eta.py` | Delivery time prediction | 30/min |
| POST | `/v1/chat` | `routers/chat.py` | Food concierge chat | 10/min |
| GET | `/v1/restaurants` | `routers/restaurants.py` | List restaurants (DB, paginated) | 60/min |
| GET | `/v1/restaurants/{id}` | `routers/restaurants.py` | Get restaurant by ID (404 if not found) | 60/min |
| GET | `/v1/restaurants/search/{q}` | `routers/restaurants.py` | Search by name or cuisine | 60/min |

---

## Model Architecture

### Rating Prediction (9 models)
**Winner**: RandomForest (MAE=0.0596, R²=0.9172) — dominant win over XGBoost (MAE=0.1373)

### ETA Prediction (10 models)
**Winner**: GradientBoosting (MAE=5.79 min) — barely edges LightGBM by 0.0008 MAE

### Collaborative Filtering
- PyTorch Matrix Factorization (embedding size=50, 20 epochs)
- 3,000 synthetic users, 52,556 interactions
- **SYNTHETIC DATA** — clearly documented as such

### Hybrid Recommender
- Blends content-based similarity + collaborative filtering + reliability score + bayesian rating
- Configurable weights (same pattern as Reliability Score)
- Supports A/B weight profiles: balanced, speed-first, quality-first
- Per-request similarity computation (avoids 13GB all-pairs matrix)

---

## Feature Inventory

| Feature | Source Files | Test Files |
|---------|-------------|------------|
| Data Cleaning | `data/cleaning.py`, `data/loaders.py` | `test_cleaning.py` |
| Feature Engineering | `features/delivery_features.py`, `features/restaurant_features.py`, `features/geo.py` | `test_features.py` (18 tests) |
| Rating Models | `models/rating_model.py`, `models/base_trainer.py` | `test_rating_model.py` |
| ETA Models | `models/eta_model.py`, `models/base_trainer.py` | `test_eta_model.py` |
| Model Selection | `models/model_selection.py` | `test_model_selection.py` |
| Content Recommender | `models/recommender.py` | `test_recommender.py` |
| Hybrid Recommender | `models/hybrid_recommender.py` | — |
| Collaborative Filtering | `models/collaborative_recommender.py` | `test_collaborative_recommender.py` |
| Optuna HPO | `models/base_trainer.py` (search spaces, tuning) | `test_optuna_tuning.py` (~25 tests) |
| LLM Narrator | `llm/recommendation_narrator.py` | — |
| RAG Retrieval | `llm/rag_similar_restaurants.py` | — |
| Food Concierge | `llm/food_concierge.py` | `integration/test_concierge.py` (27 tests) |
| Drift Detection | `monitoring/drift.py` | `test_drift.py` (13 tests) |
| SLA / Reliability | `evaluation/business_cost.py` | — |
| Redis Cache | `cache/redis_client.py` | — |
| Database ORM | `database/models.py`, `database/session.py` | `test_database.py` (16 tests) |
| DB Loaders | `database/seed.py`, `database/repositories.py` | `test_db_loaders.py` (11 tests) |
| API | `api/main.py`, `api/routers/*` | `test_api.py` (7 tests) |
| Dashboard | `app/streamlit_app.py`, `app/pages/*` | — |
| Delivery Optimizer | `models/optimizer.py` | — |
| E2E Workflow | `pipeline.py` | `e2e/test_workflow.py` (6 tests) |

---

## Key Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cv_folds` | 5 | Cross-validation folds |
| `rating_metric` | mae | Rating model selection metric |
| `eta_metric` | mae | ETA model selection metric |
| `sla_threshold_minutes` | 40 | Delivery SLA threshold |
| `reliability_w_rating` | 0.4 | Rating weight in Reliability Score |
| `reliability_w_sentiment` | 0.3 | Sentiment weight |
| `reliability_w_delay` | 0.3 | Delay risk weight |
| `hybrid_weight_content` | 0.4 | Content-based weight |
| `hybrid_weight_collaborative` | 0.3 | CF weight |
| `hybrid_weight_reliability` | 0.3 | Reliability weight |
| `llm_enabled` | False | LLM features on/off |
| `llm_model` | claude-sonnet-4-20250514 | Anthropic model |
| `llm_max_steps` | 4 | Max ReAct loop iterations |
| `optuna_n_trials` | 50 | HPO trials per model |
| `optuna_models_to_tune` | XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting | Ensembles for HPO |
| `drift_ks_threshold` | 0.05 | KS-test p-value threshold |
| `drift_alert_cooldown_hours` | 24 | Cooldown between alerts for same feature |
| `slack_webhook_url` | None | Slack Incoming Webhook for drift |
| `database_url` | sqlite:///data/dabba.db | SQLAlchemy DB URL |
| `redis_url` | redis://localhost:6379/0 | Redis URL |
| `cache_eta_ttl_seconds` | 300 | ETA prediction cache TTL |
| `cache_recommend_ttl_seconds` | 600 | Recommendation cache TTL |

---

## Development Workflow

```bash
make setup          # Install deps + pre-commit hooks
python setup_kaggle.py  # Download datasets (Kaggle API key required)
make train          # Run full pipeline (rating + ETA + CF + A/B)
make run-app        # Streamlit dashboard → :8501
make run-api        # FastAPI → :8000
make run-mlflow     # MLflow tracking → :5000
make test           # pytest with coverage (100+ tests)
make lint           # Ruff, Black --check, isort --check
make format         # Auto-format
make db-import      # Full CSV→DB import
make db-migrate     # Run Alembic migrations
```

---

## Deployment

- **Docker**: `docker-compose up --build` (API + Streamlit + MLflow with healthchecks)
- **CI**: GitHub Actions (pre-commit + pytest on push)
- **Live**: Streamlit Community Cloud + Render/Railway

---

## Remaining Work & Gaps

### 🔴 High Priority
| Gap | Details | Effort |
|-----|---------|--------|
| **ETA endpoint feature mismatch** | `POST /v1/predict-eta` sends 6 features, model trained on ~20+ | Small |
| **Concierge ETA stub** | `get_eta_estimate()` returns hardcoded 30 min | Small |
| **Real user-interaction data** | CF uses synthetic data (clearly documented) | Large |
| **Real-time traffic API** | Google Maps/OSRM for dynamic ETA | Medium |
| **Missing test coverage** | narrator, RAG, optimizer, cache, evaluation, pages | Medium |

### 🟡 Medium Priority
| Gap | Details |
|-----|---------|
| Fine-tuned small model (BART/T5) instead of Claude API at scale |
| Multi-city expansion beyond Bangalore |
| A/B testing framework for recommendation variants in production |
| Hindi/English code-switched sentiment (VADER is English-only) |
| Production hardening: dedicated test DB, secrets management, structured logging |
| Prometheus metrics + Grafana dashboard |

### 🔵 Low Priority
| Gap | Details |
|-----|---------|
| Mobile app (React Native) |
| `/v1/explain/{prediction_id}` endpoint (schema exists, no route) |
| Model auto-retraining pipeline |
| PWA support for Streamlit |
| Kubernetes manifests for zero-downtime deployment |

---

## Technical Debt & Risks

| Item | Status | Risk |
|------|--------|------|
| Collaborative filtering uses **synthetic data** | Documented, acceptable for portfolio | Low |
| **CSV dependency** — pipeline still requires CSVs to exist | DB-backed loaders added, but not enforced | Low |
| **MLflow** requires separate server | Fails gracefully when unavailable | Low |
| **VADER is English-only** | Hindi/English code-switched reviews not handled | Medium |
| **No dedicated test DB** | Tests share global SQLite engine | Medium |
| **No Prometheus metrics** | No `/metrics` endpoint on any service | Low |
| **No scheduled retraining** | Models trained manually via `make train` | Medium |
