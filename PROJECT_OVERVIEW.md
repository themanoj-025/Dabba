# Dabba — Restaurant Intelligence Platform

> **India-focused restaurant ranking, recommendation, and delivery-reliability platform with deterministic ML, collaborative filtering, and optional LLM-powered explanations.**

---

## Table of Contents

- [1. Title & Badges](#1-title--badges)
- [2. Executive Summary](#2-executive-summary)
- [3. Tech Stack & Core Technologies](#3-tech-stack--core-technologies)
- [4. High-Level Architecture](#4-high-level-architecture)
- [5. Complete Folder Structure Tree](#5-complete-folder-structure-tree)
- [6. Exhaustive File-by-File & Folder-by-Folder Breakdown](#6-exhaustive-file-by-file--folder-by-folder-breakdown)
- [7. Data Models & Schemas](#7-data-models--schemas)
- [8. API Surface](#8-api-surface)
- [9. Configuration & Environment Variables](#9-configuration--environment-variables)
- [10. Build, Run & Deployment Instructions](#10-build-run--deployment-instructions)
- [11. Data & Control Flow Walkthroughs](#11-data--control-flow-walkthroughs)
- [12. Dependency Graph Summary](#12-dependency-graph-summary)
- [13. Testing Strategy](#13-testing-strategy)
- [14. Known Issues, Technical Debt & Assumptions](#14-known-issues-technical-debt--assumptions)
- [15. Glossary](#15-glossary)
- [16. Changelog / Version History Summary](#16-changelog--version-history-summary)
- [17. Appendix](#17-appendix)
- [Security Notes](#security-notes)
- [Performance Considerations](#performance-considerations)
- [Suggested Onboarding Path](#suggested-onboarding-path)

---

## 1. Title & Badges

| | |
|---|---|
| **Project Name** | Dabba |
| **Tagline** | Restaurant Intelligence Platform |
| **Version** | 0.2.0 (pyproject.toml) / 0.5.0 (API) |
| **License** | MIT |

![CI](https://img.shields.io/badge/CI-active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-0194E2)

---

## 2. Executive Summary

**Dabba** is an India-focused restaurant intelligence platform that solves the problem of fragmented food-tech data. India's food-tech landscape generates massive amounts of restaurant and delivery data, yet consumers and operators lack a unified view combining food quality, customer sentiment, and delivery reliability.

### What it does:
1. **Restaurant Ranking:** Mines Zomato data for ratings, cuisine diversity, and cost signals
2. **Sentiment Analysis:** Analyzes customer reviews using VADER NLP (Hinglish-aware)
3. **Delivery ETA Prediction:** Predicts delivery time with rigorously selected ML models (Optuna-tuned)
4. **Reliability Score:** Proprietary composite metric: `0.4 × norm(rating) + 0.3 × norm(sentiment) - 0.3 × norm(delay_risk)`
5. **Collaborative Filtering:** PyTorch matrix factorization for personalized recommendations
6. **LLM Layer:** Anthropic Claude for natural-language explanations and chat (with fallback)

### Key Components:
- **Streamlit Dashboard:** Interactive visualization of restaurant intelligence
- **FastAPI REST API:** Production-grade API with auth, rate limiting, observability
- **MLflow Tracking:** Experiment versioning and model registry
- **Drift Detection:** KS-test based monitoring with Slack alerting

### Model Performance:
- **Best Rating Model:** RandomForest (MAE: 0.0596, R²: 0.9172)
- **Best ETA Model:** GradientBoosting (MAE: 5.789 min, R²: 0.3837)

---

## 3. Tech Stack & Core Technologies

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.11 | Core runtime |
| **ML Framework** | scikit-learn | ≥1.3,<1.9 | Traditional ML models |
| **ML Framework** | XGBoost | ≥2.0 | Gradient boosting |
| **ML Framework** | LightGBM | ≥4.0 | Fast gradient boosting |
| **ML Framework** | CatBoost | ≥1.2 | Categorical boosting |
| **Deep Learning** | PyTorch | ≥2.0 | Matrix factorization |
| **NLP** | NLTK | ≥3.8 | VADER sentiment analysis |
| **LLM** | Anthropic Claude | — | Natural language interface |
| **Vector Search** | FAISS | ≥1.7 | Similar restaurant retrieval |
| **Dashboard** | Streamlit | ≥1.28 | Interactive UI |
| **API Framework** | FastAPI | ≥0.104 | REST API |
| **Experiment Tracking** | MLflow | ≥2.0 | Model versioning |
| **HPO** | Optuna | ≥3.4 | Hyperparameter optimization |
| **Database** | SQLAlchemy | ≥2.0 | ORM |
| **Database** | PostgreSQL | — | Production storage |
| **Cache** | Redis | ≥5.0 | Rate limiting, caching |
| **Monitoring** | Prometheus | — | Metrics collection |
| **Alerting** | Slack Webhooks | — | Drift notifications |
| **Testing** | pytest | ≥7.4 | Unit + integration |
| **Linting** | Ruff | ≥0.1 | Code quality |
| **Formatting** | Black | ≥23.11 | Code formatting |

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Kaggle Datasets (Zomato + Delivery)                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Data Pipeline (Stage 1)                       │
│  load_zomato → clean_zomato → add_restaurant_features           │
│  load_delivery → clean_delivery → add_delivery_features         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│              ML Training Pipeline (Stages 2-5)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Rating Model │  │  ETA Model   │  │ Collaborative│          │
│  │ (RF/XGB/LGB) │  │ (GB/LGB/CB) │  │ Filtering    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └─────────────────┼─────────────────┘                  │
│                           ▼                                     │
│              Reliability Score + A/B Scenarios                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     Serving Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   FastAPI    │  │  Streamlit   │  │    MLflow    │          │
│  │  REST API    │  │  Dashboard   │  │   Tracking   │          │
│  └──────┬───────┘  └──────────────┘  └──────────────┘          │
│         │                                                       │
│  ┌──────▼───────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Drift Detect │  │ Slack Alert  │  │ Prometheus   │          │
│  │  (KS-test)   │  │   Webhook    │  │   Metrics    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

**Architectural Pattern:** ETL Pipeline + Serving Layer

The system separates concerns into:
1. **ETL Pipeline:** Data ingestion, cleaning, feature engineering
2. **ML Training:** Model comparison, HPO, selection
3. **Serving Layer:** API, dashboard, monitoring

---

## 5. Complete Folder Structure Tree

```
Dabba/
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── auth.py                    # API key verification
│   ├── limiter.py                 # Rate limiting
│   └── routers/
│       ├── __init__.py
│       ├── recommend.py           # /v1/recommend endpoint
│       ├── eta.py                 # /v1/predict-eta endpoint
│       ├── chat.py                # /v1/chat endpoint
│       ├── model_info.py          # /v1/model-info endpoint
│       ├── explain.py             # /v1/explain endpoint
│       └── restaurants.py         # /v1/restaurants endpoint
├── src/
│   └── dabba/
│       ├── __init__.py
│       ├── config.py              # Centralized configuration
│       ├── pipeline.py            # Full training pipeline
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py         # Dataset loading (Kaggle)
│       │   └── cleaning.py        # Data cleaning
│       ├── features/
│       │   ├── __init__.py
│       │   ├── restaurant_features.py  # Restaurant feature engineering
│       │   ├── delivery_features.py    # Delivery feature engineering
│       │   └── geo.py                  # Geographic clustering
│       ├── models/
│       │   ├── __init__.py
│       │   ├── rating_model.py    # Rating prediction models
│       │   ├── eta_model.py       # ETA prediction models
│       │   ├── collaborative_recommender.py  # PyTorch MF
│       │   └── model_selection.py # Best model selection
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── business_cost.py   # Reliability score + A/B
│       ├── nlp/
│       │   ├── __init__.py
│       │   └── sentiment.py       # VADER sentiment analysis
│       ├── database/
│       │   ├── __init__.py
│       │   ├── models.py          # SQLAlchemy models
│       │   ├── session.py         # DB session management
│       │   └── seed.py            # Database seeding
│       └── observability/
│           ├── __init__.py
│           └── metrics.py         # Prometheus metrics
├── app/
│   └── streamlit_app.py           # Streamlit dashboard
├── docker/
│   ├── api.Dockerfile             # API service image
│   ├── streamlit.Dockerfile       # Dashboard image
│   └── mlflow.Dockerfile          # MLflow server image
├── tests/
│   ├── test_api.py                # API endpoint tests
│   ├── test_models.py             # Model training tests
│   └── test_pipeline.py           # Pipeline tests
├── migrations/                    # Alembic migrations
├── data/
│   ├── raw/                       # Raw Kaggle datasets
│   └── processed/                 # Processed data
├── models/                        # Saved model artifacts
├── reports/                       # Model comparisons + figures
├── alembic.ini                    # Migration config
├── requirements.txt               # Python dependencies
├── setup_kaggle.py                # Kaggle credential setup
├── Dockerfile                     # Legacy combined Dockerfile
├── docker-compose.yml             # Service orchestration
├── Makefile                       # Build/test/deploy commands
├── pyproject.toml                 # Project metadata
├── README.md                      # Documentation
├── PROJECT_ANALYSIS.md            # Codebase audit
└── LICENSE                        # MIT License
```

---

## 6. Exhaustive File-by-File & Folder-by-Folder Breakdown

### 6.1 Root Files

#### `src/dabba/pipeline.py`
- **Type:** Python module
- **Purpose:** Full training pipeline orchestrating all stages
- **Key Stages:**
  1. **Stage 2:** Restaurant Intelligence (Zomato data)
  2. **Stage 3:** Delivery ETA Engine
  3. **Stage 4:** Reliability Score + A/B Scenarios
  4. **Stage 5:** Collaborative Filtering (Synthetic Data)
  5. **Stage 6:** Geographic Clustering
- **Key Functions:**
  - `main()` — Entry point
  - `generate_comparison_charts()` — Visual model comparison
  - `compute_shap_explanations()` — SHAP for winning model
  - `_save_restaurants_to_db()` — Persist to database

#### `src/dabba/config.py`
- **Type:** Configuration module
- **Purpose:** Centralized config with env var support
- **Key Settings:**
  - `models_dir` — Path to saved models
  - `rating_metric` — Metric for model selection (MAE)
  - `eta_metric` — Metric for ETA selection (MAE)
  - `optuna_enabled` — Enable HPO
  - `sla_threshold_minutes` — SLA threshold

### 6.2 `api/` — FastAPI Application

#### `api/main.py`
- **Purpose:** FastAPI app with middleware, CORS, rate limiting
- **Key Components:**
  - Observability middleware (request ID + Prometheus)
  - Security headers middleware
  - v1 authenticated router
  - Startup event: loads models into `app.state`
- **Endpoints:**
  - `GET /health` — Health check (no auth)
  - `GET /metrics` — Prometheus metrics (no auth)
  - `POST /v1/recommend` — Restaurant recommendations
  - `POST /v1/predict-eta` — Delivery ETA prediction
  - `POST /v1/chat` — Food concierge chat
  - `GET /v1/model-info` — Deployed model info
  - `GET /v1/restaurants` — Restaurant listing

#### `api/auth.py`
- **Purpose:** API key verification
- **Logic:** Checks `X-API-Key` header; optional if `DABBA_API_KEY` unset

#### `api/routers/recommend.py`
- **Endpoint:** `POST /v1/recommend`
- **Purpose:** Hybrid restaurant recommendations
- **Logic:** Combines collaborative filtering + content-based + reliability score

#### `api/routers/eta.py`
- **Endpoint:** `POST /v1/predict-eta`
- **Purpose:** Predict delivery ETA
- **Logic:** Uses trained ETA model with feature engineering

#### `api/routers/chat.py`
- **Endpoint:** `POST /v1/chat`
- **Purpose:** Food concierge chat (ReAct agent)
- **Logic:** Multi-step tool chain with Claude + fallback to rules

### 6.3 `src/dabba/data/` — Data Layer

#### `src/dabba/data/loaders.py`
- **Purpose:** Load datasets from Kaggle
- **Key Functions:**
  - `load_zomato()` — Load Zomato restaurant data
  - `load_delivery()` — Load delivery data
  - `describe_dataset()` — Print dataset statistics

#### `src/dabba/data/cleaning.py`
- **Purpose:** Data cleaning and preprocessing
- **Key Functions:**
  - `clean_zomato()` — Handle missing values, duplicates, formats
  - `clean_delivery()` — Clean delivery records

### 6.4 `src/dabba/features/` — Feature Engineering

#### `src/dabba/features/restaurant_features.py`
- **Purpose:** Generate restaurant features
- **Features:** `cuisine_*` (one-hot), `votes_log`, `cost_for_two`, `cuisine_count`

#### `src/dabba/features/delivery_features.py`
- **Purpose:** Generate delivery features
- **Constant:** `ETA_FEATURE_COLS` — Column names for ETA model

#### `src/dabba/features/geo.py`
- **Purpose:** Geographic clustering
- **Function:** `compare_clustering_methods()` — K-Means, DBSCAN comparison

### 6.5 `src/dabba/models/` — ML Models

#### `src/dabba/models/rating_model.py`
- **Purpose:** Train and evaluate rating prediction models
- **Models:** RandomForest, XGBoost, CatBoost, LightGBM
- **Best:** RandomForest (MAE: 0.0596, R²: 0.9172)

#### `src/dabba/models/eta_model.py`
- **Purpose:** Train and evaluate delivery ETA models
- **Models:** GradientBoosting, LightGBM, CatBoost
- **Best:** GradientBoosting (MAE: 5.789 min, R²: 0.3837)

#### `src/dabba/models/collaborative_recommender.py`
- **Purpose:** PyTorch matrix factorization
- **Key Functions:**
  - `generate_synthetic_interactions()` — Create synthetic user-restaurant data
  - `train_matrix_factorization()` — Train MF model
  - `save_collaborative_model()` — Save model artifacts

### 6.6 `src/dabba/evaluation/` — Evaluation

#### `src/dabba/evaluation/business_cost.py`
- **Purpose:** Reliability score and A/B simulation
- **Key Functions:**
  - `compute_reliability_score()` — `0.4×rating + 0.3×sentiment - 0.3×delay`
  - `compute_sla_analysis()` — SLA compliance metrics
  - `run_ab_scenario_simulation()` — A/B test scenarios

### 6.7 `src/dabba/nlp/` — NLP

#### `src/dabba/nlp/sentiment.py`
- **Purpose:** Sentiment analysis on reviews
- **Method:** VADER (Hinglish-aware)
- **Function:** `add_sentiment_scores()` — Add `avg_sentiment` column

### 6.8 `src/dabba/database/` — Database Layer

#### `src/dabba/database/models.py`
- **Purpose:** SQLAlchemy ORM models
- **Key Models:**
  - `Restaurant` — Restaurant data
  - `ExperimentResult` — Model comparison results

#### `src/dabba/database/session.py`
- **Purpose:** Database session management
- **Functions:** `init_db()`, `get_db()`

### 6.9 `app/streamlit_app.py` — Dashboard

- **Purpose:** Interactive Streamlit dashboard
- **Pages:** Restaurant explorer, Model performance, Reliability scores

---

## 7. Data Models & Schemas

### Restaurant (SQLAlchemy)

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Primary key |
| `name` | str | Restaurant name |
| `rate` | float | Rating (1-5) |
| `votes` | int | Number of votes |
| `cost_for_two` | float | Cost for two |
| `cuisine_count` | int | Number of cuisines |
| `avg_sentiment` | float | Average sentiment score |
| `reliability_score` | float | Computed reliability |
| `latitude` | float | Location latitude |
| `longitude` | float | Location longitude |

### ExperimentResult (SQLAlchemy)

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Primary key |
| `task` | str | "rating" or "eta" |
| `model_name` | str | Model identifier |
| `mae` | float | Mean Absolute Error |
| `rmse` | float | Root Mean Squared Error |
| `r2` | float | R² Score |
| `train_time_s` | float | Training time |
| `mlflow_run_id` | str | MLflow experiment ID |
| `is_winner` | bool | Best model flag |

---

## 8. API Surface

### Endpoints

| Method | Path | Purpose | Auth | Rate Limit |
|--------|------|---------|------|------------|
| `GET` | `/health` | Health check | None | 1000/min |
| `GET` | `/metrics` | Prometheus metrics | None | 1000/min |
| `POST` | `/v1/recommend` | Restaurant recommendations | API Key | 30/min |
| `POST` | `/v1/predict-eta` | Delivery ETA prediction | API Key | 30/min |
| `POST` | `/v1/chat` | Food concierge chat | API Key | 20/min |
| `GET` | `/v1/model-info` | Model metadata | API Key | 60/min |
| `GET` | `/v1/restaurants` | Restaurant listing | API Key | 60/min |

### Request/Response Examples

**POST /v1/recommend**
```json
// Request
{
    "user_id": 123,
    "cuisine_preference": "North Indian",
    "budget": 500,
    "n_recommendations": 5
}

// Response
{
    "recommendations": [
        {
            "name": "Restaurant A",
            "rating": 4.5,
            "reliability_score": 0.82,
            "reason": "High rating, good delivery reliability"
        }
    ]
}
```

---

## 9. Configuration & Environment Variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `DABBA_API_KEY` | API authentication key | — | Optional |
| `DABBA_DATABASE_URL` | PostgreSQL connection | SQLite fallback | No |
| `DABBA_MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` | No |
| `ANTHROPIC_API_KEY` | Claude API key for LLM | — | Optional |
| `REDIS_URL` | Redis connection | — | Optional |
| `SLACK_WEBHOOK_URL` | Drift alert notifications | — | Optional |
| `KAGGLE_USERNAME` | Kaggle username | — | Yes (for data) |
| `KAGGLE_KEY` | Kaggle API key | — | Yes (for data) |
| `API_PORT` | API server port | `8000` | No |
| `DASHBOARD_PORT` | Streamlit port | `8501` | No |
| `MLFLOW_PORT` | MLflow port | `5000` | No |

---

## 10. Build, Run & Deployment Instructions

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Kaggle account (for datasets)

### Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd Dabba
make setup

# 2. Setup Kaggle credentials
python setup_kaggle.py

# 3. Train all models
make train

# 4. Run dashboard
make run-app

# 5. Run API
make run-api
```

### Docker

```bash
# Start all services
docker-compose up --build

# Services:
# - Streamlit Dashboard: http://localhost:8501
# - FastAPI: http://localhost:8000
# - MLflow: http://localhost:5000
```

### Testing

```bash
make test        # pytest with coverage
make lint        # ruff, black, isort
make format      # Auto-format code
```

### Database

```bash
make db-import   # Import CSV data to DB
make db-migrate  # Run Alembic migrations
make db-shell    # Interactive DB shell
```

---

## 11. Data & Control Flow Walkthroughs

### Flow 1: Restaurant Recommendation

```
1. Client sends POST /v1/recommend with user preferences
2. api/routers/recommend.py validates input
3. Loads hybrid_recommender from app.state
4. Recommender combines:
   a. Collaborative filtering scores
   b. Content-based features
   c. Reliability scores
5. Returns top-N recommendations with explanations
```

### Flow 2: Delivery ETA Prediction

```
1. Client sends POST /v1/predict-eta with delivery details
2. api/routers/eta.py extracts features
3. add_delivery_features() generates ML features
4. ETA model predicts delivery time
5. Returns prediction + confidence interval
```

### Flow 3: Training Pipeline

```
1. python -m dabba.pipeline
2. Stage 2: Load Zomato data → clean → feature engineer
3. Stage 3: Load delivery data → clean → feature engineer
4. Stage 4: Train rating models → select best
5. Stage 5: Train ETA models → select best
6. Stage 6: Compute reliability scores
7. Stage 7: Train collaborative filtering
8. Stage 8: Geographic clustering
9. Save all artifacts to models/
10. Log to MLflow
```

---

## 12. Dependency Graph Summary

### Internal Dependencies

```
api/main.py
  ├── api/routers/* → api/auth.py, api/limiter.py
  ├── dabba.config → dabba.models.*
  └── dabba.observability

pipeline.py
  ├── dabba.data.loaders → dabba.data.cleaning
  ├── dabba.features.* → dabba.models.*
  ├── dabba.nlp.sentiment
  ├── dabba.evaluation.business_cost
  └── dabba.database.session
```

### External Package Purposes

| Package | Purpose |
|---------|---------|
| `scikit-learn` | Traditional ML (RF, GB, LR) |
| `xgboost` | Gradient boosting |
| `lightgbm` | Fast gradient boosting |
| `catboost` | Categorical boosting |
| `torch` | Matrix factorization |
| `faiss-cpu` | Vector similarity search |
| `nltk` | VADER sentiment |
| `anthropic` | LLM integration |
| `fastapi` | REST API |
| `streamlit` | Dashboard |
| `mlflow` | Experiment tracking |
| `optuna` | HPO |

---

## 13. Testing Strategy

### Test Types
- **Unit tests:** Model training, feature engineering, data cleaning
- **Integration tests:** API endpoints with mocked models
- **Pipeline tests:** End-to-end pipeline validation

### Running Tests

```bash
make test                    # All tests
make test --cov=src/dabba    # With coverage
make lint                    # Code quality
```

### Known Test Issues
- `test_api.py` fails due to `.env` file encoding (UnicodeDecodeError)
- Fix: Ensure `.env` file uses UTF-8 encoding

---

## 14. Known Issues, Technical Debt & Assumptions

### Known Issues
1. **Test failures:** UnicodeDecodeError in `.env` file parsing
2. **Synthetic data:** Collaborative filtering uses synthetic interactions (not real user data)

### Technical Debt
1. **Model serialization:** Using pickle (fragile across versions)
2. **Legacy Dockerfile:** Retained for backward compatibility
3. **Hinglish NLP:** VADER is basic; needs fine-tuned IndicBERT

### Assumptions
- Zomato dataset schema is consistent
- Delivery times are in minutes
- Geographic coordinates are valid

---

## 15. Glossary

| Term | Definition |
|------|------------|
| **Reliability Score** | Composite metric: `0.4×rating + 0.3×sentiment - 0.3×delay_risk` |
| **HPO** | Hyperparameter Optimization via Optuna |
| **Matrix Factorization** | Collaborative filtering technique |
| **KS-test** | Kolmogorov-Smirnov test for drift detection |
| **SLA** | Service Level Agreement (delivery time threshold) |

---

## 16. Changelog / Version History Summary

No explicit CHANGELOG.md found. Based on PROJECT_ANALYSIS.md:
- **Current state:** Verified & Cleaned
- **Known issues:** Test failures due to `.env` encoding
- **CI/CD:** Verified and functional

---

## 17. Appendix

### License
MIT — see LICENSE file

### Datasets
- **Zomato:** Restaurant data (ratings, cuisines, cost)
- **Delivery:** Delivery time data (distance, traffic, weather)

### Model Artifacts
- `models/best_rating_model.pkl` — Best rating model
- `models/best_eta_model.pkl` — Best ETA model
- `models/collaborative_model.pt` — PyTorch MF model
- `models/scaler.pkl` — Feature scaler

---

*This document was auto-generated from comprehensive codebase analysis.*
