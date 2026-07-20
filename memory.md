# 🍛 Dabba — Restaurant Intelligence Platform (v3)

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
- FastAPI + Streamlit + Docker deployment

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
| **Deep Learning** | PyTorch (matrix factorization) |
| **NLP** | NLTK (VADER sentiment) |
| **LLM** | Anthropic Claude (optional, rules-based fallback) |
| **Vector Search** | FAISS (with sklearn fallback) |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit, Plotly |
| **API** | FastAPI, Pydantic |
| **Experiment Tracking** | MLflow |
| **Monitoring** | scipy.stats.ks_2samp |
| **Testing** | pytest (45+ tests) |
| **Linting** | Ruff, Black, isort, pre-commit |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker, docker-compose |
| **Data** | Kaggle (Zomato Bangalore, Food Delivery Time) |

---

## Repository Structure

```
dabba/
├── api/
│   ├── main.py                       # FastAPI with CORS, security, model loading
│   ├── schemas.py                    # Pydantic request/response models
│   └── routers/
│       ├── recommend.py              # POST /recommend — hybrid recommendations
│       ├── eta.py                    # POST /predict-eta — delivery time
│       ├── chat.py                   # POST /chat — food concierge
│       └── model_info.py             # GET /model-info — model metadata
├── app/
│   ├── streamlit_app.py              # Entry point, sidebar nav, theme loading
│   ├── assets/theme.css              # Warm food-tech design system
│   ├── components/
│   │   ├── restaurant_card.py        # Styled card + metric card components
│   │   └── __init__.py
│   └── pages/
│       ├── page_discover.py          # Customer: filters, cards, LLM explanations
│       ├── page_ops.py               # Ops: simulation, drift alerts, metrics
│       ├── page_model_performance.py # Model comparison charts, A/B scenarios
│       └── page_concierge.py         # Chat copilot with example prompts
├── data/
│   ├── raw/                          # Raw Kaggle CSVs (gitignored)
│   └── processed/                    # Cleaned + features + interactions (gitignored)
├── models/                           # Saved artifacts: .pkl, .pt (gitignored)
├── notebooks/                        # 6 EDA + prototyping notebooks
├── reports/                          # CSVs, charts, SHAP, JSON (gitignored)
├── src/dabba/
│   ├── config.py                     # Centralized DabbaConfig (Pydantic)
│   ├── pipeline.py                   # Full training orchestrator
│   ├── data/
│   │   ├── loaders.py                # CSV loading with schema verification
│   │   └── cleaning.py               # Zomato + delivery data cleaning
│   ├── features/
│   │   ├── geo.py                    # Haversine, clustering comparison
│   │   ├── restaurant_features.py    # Cuisine encoding, cost buckets
│   │   └── delivery_features.py      # Distance, time, traffic features
│   ├── models/
│   │   ├── rating_model.py           # 9 models compared, MLflow tracking
│   │   ├── eta_model.py              # 10 models compared (+ neural net)
│   │   ├── model_selection.py        # Auto-select best by metric
│   │   ├── collaborative_recommender.py  # PyTorch MF on synthetic data
│   │   ├── hybrid_recommender.py     # Blends 3+ signal types
│   │   ├── recommender.py            # Original content-based
│   │   └── optimizer.py              # Hungarian algorithm for partner assignment
│   ├── llm/
│   │   ├── recommendation_narrator.py    # LLM + rules-based explanations
│   │   ├── rag_similar_restaurants.py    # FAISS + sklearn retrieval
│   │   └── food_concierge.py             # Chat copilot with tool-use
│   ├── nlp/sentiment.py              # VADER sentiment analysis
│   ├── monitoring/drift.py            # KS-test drift detection
│   └── evaluation/
│       ├── metrics.py                # Regression metrics
│       └── business_cost.py          # SLA analysis, Reliability Score, A/B scenarios
├── tests/                            # 45+ tests across all modules
├── pyproject.toml                    # Package configuration + linting config
├── .pre-commit-config.yaml
├── .gitignore                        # Data, models, reports, MLflow, caches
├── docker-compose.yml                # API + Streamlit + MLflow services
├── Dockerfile
├── Makefile                          # train, test, lint, run-app, run-api, run-mlflow
├── README.md
├── requirements.txt
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
│  │ (9 algos) │  │ (10 algos)│  │ (PyTorch MF)  │  │ (KMeans etc)│     │
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
│          │  Reliability Score  │     │  Wired into Ops UI   │           │
│          │  + A/B Scenarios   │     └──────────────────────┘           │
│          └────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────┘
           │                     │                     │
           ▼                     ▼                     ▼
    ┌──────────┐         ┌──────────┐          ┌──────────┐
    │ Streamlit│         │  FastAPI  │          │  MLflow  │
    │ Dashboard│         │  REST API │          │ Tracking │
    │ (4 pages)│         │ (5 routes)│          │ (Docker) │
    └──────────┘         └──────────┘          └──────────┘
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
| Method | Route | File | Purpose |
|--------|-------|------|---------|
| GET | `/health` | `api/main.py` | Health check + model status |
| GET | `/model-info` | `routers/model_info.py` | Deployed models + metrics |
| POST | `/recommend` | `routers/recommend.py` | Hybrid recommendations (optional LLM narration) |
| POST | `/predict-eta` | `routers/eta.py` | Delivery time prediction |
| POST | `/chat` | `routers/chat.py` | Food concierge chat |

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
- Blends content-based similarity + collaborative filtering + reliability score
- Configurable weights (same pattern as Reliability Score)
- Supports A/B weight profiles: balanced, speed-first, quality-first

---

## Feature Inventory

| Feature | Files | Tests |
|---------|-------|-------|
| Data Cleaning | `data/cleaning.py`, `data/loaders.py` | `test_cleaning.py` |
| Feature Engineering | `features/geo.py`, `features/restaurant_features.py`, `features/delivery_features.py` | `test_features.py` |
| Rating Models | `models/rating_model.py` | — |
| ETA Models | `models/eta_model.py` | — |
| Model Selection | `models/model_selection.py` | `test_model_selection.py` |
| Recommender | `models/recommender.py`, `models/hybrid_recommender.py` | — |
| Collaborative Filtering | `models/collaborative_recommender.py` | `test_collaborative_recommender.py` |
| LLM Narrator | `llm/recommendation_narrator.py` | — |
| RAG Retrieval | `llm/rag_similar_restaurants.py` | — |
| Food Concierge | `llm/food_concierge.py` | — |
| Drift Detection | `monitoring/drift.py` | `test_drift.py` |
| SLA / Reliability | `evaluation/business_cost.py` | — |
| API | `api/main.py`, `api/routers/*` | `test_api.py` |
| Dashboard | `app/streamlit_app.py`, `app/pages/*` | — |

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
| `drift_ks_threshold` | 0.05 | KS-test p-value threshold |
| `llm_enabled` | False | LLM features on/off |
| `mlflow_tracking_uri` | http://localhost:5000 | MLflow server |

---

## Development Workflow

```bash
make setup          # Install deps + pre-commit hooks
python setup_kaggle.py  # Download datasets (Kaggle API key required)
make train          # Run full pipeline (rating + ETA + CF + A/B)
make run-app        # Streamlit dashboard → :8501
make run-api        # FastAPI → :8000
make run-mlflow     # MLflow tracking → :5000
make test           # pytest with coverage (45+ tests)
make lint           # Ruff, Black --check, isort --check
make format         # Auto-format
```

## Deployment

- Docker: `docker-compose up --build` (API + Streamlit + MLflow)
- CI: GitHub Actions (pre-commit + pytest on push)
- Live: Streamlit Community Cloud + Render/Railway

## Technical Debt & Risks

- Collaborative filtering uses **synthetic data** (public dataset limitation)
- CSV-based storage — no proper database
- No authentication (development tool)
- MLflow requires separate server (fails gracefully when unavailable)
- VADER is English-only — Hindi/English code-switched reviews not handled
