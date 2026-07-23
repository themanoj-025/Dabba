# 🍛 Dabba — Restaurant Intelligence Platform

[![CI](https://github.com/themanoj-025/dabba/actions/workflows/ci.yml/badge.svg)](https://github.com/themanoj-025/dabba/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b)](https://themanoj-025-dabba.streamlit.app)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)](https://mlflow.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **India-focused restaurant ranking, recommendation, and delivery-reliability platform.**  
> Deterministic ML for ranking & prediction, LLM for explanation & natural-language interaction.  
> Built to senior data-science code-review standard + senior product designer's UI review.

---

## 🎬 Live Demo

| Service | Link | Notes |
|---------|------|-------|
| **Streamlit Dashboard** | [dabba.streamlit.app](https://themanoj-025-dabba.streamlit.app) | Full UI — Discover, Ops Monitor, Model Perf, Food Concierge |
| **FastAPI** | [dabba-api.onrender.com](https://themanoj-025-dabba-api.onrender.com) | REST API — 8 endpoints (recommend, predict-eta, chat, model-info, restaurants CRUD) |
| **MLflow UI** | `http://localhost:5000` (docker-compose) | Experiment tracking |

> ⚡ **Note:** If using a free-tier host, there may be a 10–30 second cold-start delay on first request after inactivity.

---

## 🎯 The Problem

India's food-tech landscape generates massive amounts of restaurant and delivery data, yet consumers and operators lack a unified view that combines **food quality**, **customer sentiment**, and **delivery reliability** into a single actionable metric. Most portfolio projects model these separately — Dabba binds them together.

Dabba solves this by:

1. **Mining** Zomato restaurant data for ratings, cuisine diversity, and cost signals
2. **Analyzing** customer sentiment from reviews using VADER NLP
3. **Predicting** delivery ETA with a rigorously selected ML model (9+ algorithms compared, **Optuna HPO** tuned)
4. **Synthesizing** everything into a proprietary **Reliability Score**
5. **Adding collaborative filtering** (PyTorch matrix factorization on synthetic interaction data)
6. **Wrapping it all with an LLM layer** — natural-language explanations, RAG similar-restaurant retrieval, and a ReAct-powered chat copilot with multi-step tool chaining
7. **Monitoring for drift** in production — KS-test-based drift detection wired into the Ops Monitor UI with **Slack/email alerting**

---

## 🏗️ Architecture

```
Kaggle Datasets (Zomato + Delivery)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE (src/dabba/) ─── Optuna HPO             │
│                                                                          │
│  loaders.py → cleaning.py → feature engineering → sentiment (VADER)     │
│       │              │                    │              │               │
│       ▼              ▼                    ▼              ▼               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Rating    │  │ ETA      │  │ Collaborative │  │ Geographic   │         │
│  │ Model     │  │ Model    │  │ Filtering     │  │ Clustering   │         │
│  │ (9 algos) │  │ (10+ algos)│  │ (PyTorch MF)│  │ (KMeans etc) │         │
│  │+ Optuna   │  │+ Optuna  │  │               │  │               │         │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └──────────────┘         │
│       │             │               │                                    │
│       └─────────────┼───────────────┘                                    │
│                     │                                                    │
│                     ▼                                                    │
│          ┌──────────────────────┐                                        │
│          │  Hybrid Recommender   │  Content + CF + Reliability Score      │
│          │  + LLM Narrator      │  + A/B weight scenarios                │
│          └──────────┬───────────┘                                        │
│                     │                                                    │
│                     ▼                                                    │
│          ┌──────────────────────┐                                        │
│          │  Reliability Score   │  w1*rating + w2*sentiment              │
│          │  + A/B Scenarios    │  - w3*delay_risk                       │
│          └──────────┬───────────┘                                        │
│                     │                                                    │
│                     ▼                                                    │
│          ┌──────────────────────┐                                        │
│          │  Drift Detection     │  KS-test + Slack webhook alerting      │
│          │  + Cooldown Mgmt    │  + DB persistence via DriftLog          │
│          └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────┘
           │                     │                     │
           ▼                     ▼                     ▼
    ┌──────────┐         ┌──────────┐          ┌──────────┐
    │ Streamlit│         │  FastAPI  │          │  MLflow  │
    │ Dashboard│         │  REST API │          │ Tracking │
    │ (4 pages)│         │ (8 routes)│          │ (Docker) │
    │          │         │ Models in │          │ Port 5000│
    │          │         │ app.state │          │          │
    └──────────┘         └──────────┘          └──────────┘
```

### The LLM Layer

The LLM (Anthropic Claude) is used as a **natural-language interface over deterministic ML/business logic** — not for ranking or prediction itself:

| Component | What it does | Fallback |
|-----------|-------------|----------|
| **Recommendation Narrator** | Generates plain-English "why this restaurant" explanations | Template-based rules |
| **RAG Similar-Restaurant Retrieval** | FAISS + cosine similarity for "find me more like this" | sklearn cosine similarity |
| **Food Concierge Chat** | ReAct-powered multi-step tool chain (max 4 steps) over `search_restaurants()`, `get_eta_estimate()`, `get_reliability_score()` | Rules-based intent matching |

This is the same hybrid pattern used in production ML+LLM systems: **deterministic computation, conversational explanation.** The app never breaks without an API key — every LLM feature has a graceful fallback.

---

## 📈 Model Comparison & Selection

This is the centerpiece — **multiple algorithms rigorously compared with identical features and k-fold cross-validation**, and the best selected automatically on held-out data. All experiments are logged to **MLflow**.

### Rating Prediction (Restaurant Quality)

| Model | MAE | RMSE | R² | Train Time |
|-------|-----|------|----|------------|
| 🥇 **RandomForest** | **0.0596** | **0.1267** | **0.9172** | 8.15s |
| 🥈 XGBoost | 0.1373 | 0.2012 | 0.7913 | 1.25s |
| 🥉 CatBoost | 0.1637 | 0.2323 | 0.7220 | 2.49s |
| LightGBM | 0.1672 | 0.2378 | 0.7085 | 5.05s |
| DecisionTree | 0.1712 | 0.2539 | 0.6677 | 0.80s |
| GradientBoosting | 0.1904 | 0.2705 | 0.6230 | 18.83s |
| Ridge | 0.2072 | 0.2897 | 0.5674 | 0.30s |
| LinearRegression | 0.2072 | 0.2897 | 0.5674 | 0.38s |
| Lasso | 0.2400 | 0.3316 | 0.4335 | 0.44s |

> **Winner: RandomForest** with R²=0.917 — models 9 architectures compared via 5-fold CV on 41,665 restaurants.
> CatBoost was the closest competitor among gradient-boosted models (R²=0.722).

### ETA Prediction (Delivery Time)

| Model | MAE (min) | RMSE (min) | R² | Train Time |
|-------|-----------|------------|----|------------|
| 🥇 **GradientBoosting** | **5.789** | **7.364** | **0.3837** | 7.75s |
| 🥈 LightGBM | 5.790 | 7.370 | 0.3828 | 0.57s |
| 🥉 CatBoost | 5.810 | 7.394 | 0.3788 | 2.31s |
| XGBoost | 5.884 | 7.496 | 0.3614 | 0.99s |
| KNN | 6.012 | 7.678 | 0.3300 | 0.45s |
| DecisionTree | 6.090 | 7.826 | 0.3041 | 0.33s |
| LinearRegression | 6.343 | 8.009 | 0.2710 | 0.43s |
| Ridge | 6.343 | 8.009 | 0.2710 | 0.11s |
| Lasso | 6.347 | 8.011 | 0.2707 | 0.10s |
| RandomForest | 6.379 | 8.195 | 0.2369 | 5.75s |

> **Winner: GradientBoosting** with MAE=5.79 min — barely edging LightGBM by 0.0008.
> 10 architectures compared via 5-fold CV on 41,522 deliveries.
> NeuralNet_MLP requires `skorch` — optional comparison point.

### Hyperparameter Tuning (Optuna)

Dabba uses **Optuna** (TPE sampler) to tune hyperparameters for ensemble models before comparison, replacing hardcoded defaults:

| Model | Params Tuned | Search Space |
|-------|-------------|--------------|
| **XGBoost** | 9 | `n_estimators: 50-500`, `max_depth: 3-12`, `learning_rate: 0.001-0.3`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda`, `reg_alpha` |
| **LightGBM** | 9 | `n_estimators: 50-500`, `max_depth: 3-12`, `learning_rate: 0.001-0.3`, `num_leaves: 15-127`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_lambda`, `reg_alpha` |
| **CatBoost** | 7 | `n_estimators: 50-500`, `max_depth: 3-10`, `learning_rate: 0.001-0.3`, `l2_leaf_reg`, `bagging_temperature`, `random_strength`, `border_count` |
| **RandomForest** | 5 | `n_estimators: 50-500`, `max_depth: 3-30`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| **GradientBoosting** | 7 | `n_estimators: 50-500`, `max_depth: 3-12`, `learning_rate: 0.001-0.3`, `subsample`, `min_samples_split`, `min_samples_leaf`, `max_features` |

Configured via `config.optuna_n_trials` (default: 50) and `config.optuna_models_to_tune`.

### Interactive Charts

*(Generated after running `make train` — click charts for interactive Plotly versions in the dashboard)*

![Rating Model Comparison](reports/figures/rating_model_comparison.png)
![ETA Model Comparison](reports/figures/eta_model_comparison.png)
![Rating R²](reports/figures/rating_r2_comparison.png)
![ETA R²](reports/figures/eta_r2_comparison.png)

### MLflow

View all experiment runs with parameters, metrics, and model artifacts:

```bash
make run-mlflow
# → http://localhost:5000
```

Or via Docker:
```bash
docker-compose up mlflow
# → http://localhost:5000
```

---

## 📊 The Reliability Score

```
reliability_score = 0.4 × norm(rating) + 0.3 × norm(sentiment) - 0.3 × norm(delay_risk)
```

| Component | Source | Description |
|-----------|--------|-------------|
| `norm(rating)` | Winning rating model | Bayesian-adjusted restaurant rating (0–1) |
| `norm(sentiment)` | VADER NLP on reviews | Average customer sentiment (0–1) |
| `norm(delay_risk)` | Winning ETA model | Predicted probability of SLA violation (0–1) |

All weights (`w1=0.4`, `w2=0.3`, `w3=0.3`) are configurable in `config.py`.

### A/B Weight Scenario Simulation

The dashboard includes a simulation of how the top-10 restaurant list changes under different weight profiles — what a real product team would A/B test:

| Profile | `w_rating` | `w_sentiment` | `w_delay` | Effect |
|---------|-----------|--------------|----------|--------|
| **Balanced** (default) | 0.4 | 0.3 | 0.3 | Equal balance |
| **Quality First** | 0.5 | 0.3 | 0.2 | Prioritizes food quality over speed |
| **Speed First** | 0.2 | 0.2 | 0.6 | Prioritizes on-time delivery |

The overlap analysis (how many restaurants appear in the top-N under each profile) is shown in the Model Performance page.

---

## 🔗 Beyond Content-Based: Adding Collaborative Filtering

The Zomato dataset has no real user-interaction history. To demonstrate collaborative filtering properly, Dabba generates a **synthetic-but-realistic user-interaction dataset**:

- **3,000 simulated users**, each assigned cuisine/price preferences
- **~30,000–90,000 interactions** with realistic noise
- Trained via **PyTorch Matrix Factorization** (embedding size=50, 20 epochs)
- Blended with content-based similarity and reliability score in `HybridRecommender`

> ⚠️ **This interaction data is SYNTHETIC** — generated to demonstrate the technique, not real user behavior. In production, this would train on real order/rating logs. This is documented transparently in the code, API, and UI — never presented as real data.

---

## 🔍 Explainability (SHAP)

SHAP analysis on the winning models reveals:

**ETA Model:**
- **Distance** is the strongest predictor of delivery time
- **Traffic density** (especially "Jam") adds significant delay
- **Festival days** compound traffic effects
- **Delivery person rating** has a small but consistent effect

*(SHAP plots generated after running `make train`)*

---

## 🗺️ Cuisine Hotspot Map

Interactive map of Bangalore's cuisine hotspots, clustered using the best algorithm (KMeans/DBSCAN/Agglomerative selected by silhouette score).

*(Map generated after running `make train`)*

---

## 🚀 Delivery Partner Optimizer

Uses the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) to optimize delivery partner assignment, minimizing total predicted delivery time:

| Strategy | Total Time | Improvement |
|----------|-----------|-------------|
| Naive (first-available) | — | — |
| **Optimized (Hungarian)** | — | **—** |

> *Run `make train` to populate with actual numbers from your pipeline run.*

---

## 🔄 MLOps

### MLflow Experiment Tracking

Every model training run is logged to MLflow:
- Parameters (model type, hyperparameters, CV folds, random seed)
- Metrics (MAE, RMSE, R², training time)
- Winning run tagged per task

```bash
make run-mlflow
# Open http://localhost:5000 to view
```

### Drift Detection + Slack Alerting

**Wired into the Ops Monitor UI** — not just present as an unused script:

1. Reference distribution is built from the training data at pipeline run
2. During simulation, each batch is compared against the reference using **KS two-sample tests** (`scipy.stats.ks_2samp`)
3. If any feature drifts beyond the p-value threshold (default: 0.05), a **red alert banner** appears in the UI
4. The **"Inject Drift" checkbox** intentionally shifts the data distribution to verify the detector fires
5. **Slack alerting**: When drift is detected and `DABBA_SLACK_WEBHOOK_URL` is configured, a formatted message is sent to Slack via Incoming Webhook with per-feature details (p-value, KS statistic)
6. **Cooldown management**: Alerts for the same feature are suppressed for 24 hours (configurable via `drift_alert_cooldown_hours`) to prevent notification fatigue
7. **Database persistence**: Each drift event is logged to the `DriftLog` table (SQLite/Postgres) with the `alerted` flag tracking notification status

---

## 🚀 Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/themanoj-025/dabba.git
cd dabba
make setup

# Download datasets
# 1. Get your Kaggle API token from https://www.kaggle.com/settings/account
# 2. Place kaggle.json in ~/.kaggle/
python setup_kaggle.py

# Train all models (rating + ETA + collaborative filtering + A/B scenarios)
make train

# Run the dashboard
make run-app

# Run the API
make run-api

# Run MLflow tracking UI
make run-mlflow
```

### Docker (Per-Service Containers)

```bash
docker-compose up --build
```

| Service | Dockerfile | URL | Healthcheck |
|---------|-----------|-----|-------------|
| **Streamlit** | `docker/streamlit.Dockerfile` | http://localhost:8501 | `curl -f http://localhost:8501/` |
| **FastAPI** | `docker/api.Dockerfile` | http://localhost:8000 | `curl -f http://localhost:8000/health` |
| **MLflow** | `docker/mlflow.Dockerfile` | http://localhost:5000 | `curl -f http://localhost:5000/api/2.0/mlflow/experiments/list` |

Each service has its own Dockerfile with independent health checks, proper startup ordering (`depends_on: condition: service_healthy`), and `restart: unless-stopped` policy.

### LLM Features (Optional)

To enable LLM-powered recommendations and chat:

```bash
# Create .env file
echo "DABBA_LLM_ENABLED=true" >> .env
echo "DABBA_ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

Without this, all LLM features fall back to **rules-based behavior** — the app never breaks.

---

## 🧪 Testing

```bash
make test        # Run pytest with coverage (100+ tests)
make lint        # Run ruff, black, isort
make format      # Auto-format code

# Database operations
make db-import       # Full CSV→DB import
make db-migrate      # Run Alembic migrations
make db-rollback     # Rollback last migration
make db-history      # Show migration history
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11 |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | PyTorch (matrix factorization), skorch (neural net) |
| **NLP** | NLTK (VADER sentiment) |
| **LLM** | Anthropic Claude (optional, with rules-based fallback) |
| **Vector Search** | FAISS (with sklearn fallback) |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit, Plotly (interactive charts) |
| **API** | FastAPI, Pydantic |
| **Experiment Tracking** | MLflow |
| **Monitoring** | scipy.stats.ks_2samp (drift detection) |
| **HPO** | Optuna (TPE sampler) |
| **Alerting** | Slack Incoming Webhooks |
| **Testing** | pytest, pytest-cov |
| **Linting** | Ruff, Black, isort, pre-commit |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker, docker-compose |
| **Data** | Kaggle (Zomato Bangalore, Food Delivery Time) |

---

## 🎨 UI/UX Design Decisions

- **Warm food-tech palette** — cream/off-white background, warm orange accent (`#ff8c42`), rounded cards
- **Inter font** loaded via Google Fonts for clean readability
- **4 purpose-built pages** instead of generic views
- **Plotly interactive charts** everywhere (no static matplotlib screenshots in the UI)
- **Styled restaurant cards** with rating badges, reliability color-coding, and LLM explanation captions
- **Chat bubbles** for the Food Concierge page
- **Drift alert banner** visually distinct from other notifications
- **Graceful empty/error states** — never a blank white page
- **Responsive layout** via `st.columns` tested at multiple widths

---

## 🐛 Known Limitations & Fixes-in-Progress

> *Transparency note: These are real gaps I've identified through a full-pipeline audit.
> Listing them here turns bugs into signals — every one has a plan.*

| Limitation | Status | Plan |
|------------|--------|------|
| **ETA endpoint was sending 6 features to a model trained on 20+** | ✅ **Fixed in v0.4.0** | Now uses `build_eta_features_for_api()` from `delivery_features.py` — the same feature pipeline as training |
| **Concierge ETA returned hardcoded 30 min** | ✅ **Fixed in v0.4.0** | Now uses real ETA model via `ConciergeTools.get_eta_estimate()` with graceful formula fallback |
| **`pipeline.py` maintained its own `eta_feature_cols`** | ✅ **Fixed in v0.4.0** | Now imports `ETA_FEATURE_COLS` from `delivery_features.py` — single source of truth |
| **Synthetic CF data** | ⚠️ Documented limitation | Real user-interaction data needed for production |
| **API/UI read from CSVs instead of DB** | ✅ **Fixed in v0.5.0** | All serving paths now read from Postgres/SQLite via repository functions. CSVs only used in seed/import pipeline |
| **VADER is English-only** | ⚠️ Documented limitation | Hinglish-aware sentiment model needed |
| **Static traffic levels** | ⚠️ Known gap | Real-time traffic API (Google Maps/OSRM) planned |
| **No Prometheus metrics** | 🔜 Planned | `/metrics` endpoint + Grafana dashboard |
| **No retraining trigger** | 🔜 Planned | Drift-threshold-triggered retraining |

---

## 📋 What I'd Do Next

### 🔴 High Priority
- **Fix ETA endpoint feature mismatch** — the `POST /v1/predict-eta` route sends only 6 features but the model was trained on ~20+ features (cyclical encoding, rush hour, interaction features). The feature sets need to be aligned.
- **Fill the concierge ETA stub** — `ConciergeTools.get_eta_estimate()` returns a hardcoded 30-min estimate instead of using the loaded ETA model
- **Real user-interaction data** instead of synthetic for collaborative filtering
- **Real-time traffic API integration** (Google Maps, OSRM) for dynamic ETA

### 🟡 Medium Priority
- **Add missing unit tests** — `recommendation_narrator.py`, `rag_similar_restaurants.py`, `optimizer.py`, `cache/redis_client.py`, evaluation modules, and Streamlit pages all lack dedicated test coverage
- **Fine-tuned small model** instead of API calls for the narrator at scale (e.g., fine-tuned BART or T5)
- **Multi-city expansion** beyond Bangalore
- **A/B testing framework** for recommendation algorithm variants in production
- **Hindi/English code-switched sentiment** — VADER is English-only
- **Production hardening** — dedicated test DB, secrets management, structured logging, Prometheus metrics

### 🔵 Low Priority
- **Mobile app** with React Native for on-the-go recommendations
- **Add `/v1/explain/{prediction_id}`** endpoint (schema already exists in `models.py`)
- **Model auto-retraining** — CI/CD-triggered retraining pipeline
- **PWA support** for Streamlit dashboard
- **Kubernetes manifests** for zero-downtime deployment

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Manoj Jana**
- [LinkedIn](https://linkedin.com/in/themanoj-025)
- [GitHub](https://github.com/themanoj-025)

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 Resume Bullet

> Built **Dabba**, an end-to-end ML+LLM platform for restaurant recommendation and delivery reliability, featuring rigorous comparison of 9 algorithms (XGBoost, LightGBM, CatBoost, etc.) with k-fold cross-validation and automatic best-model selection by MAE, PyTorch matrix factorization for collaborative filtering, a hybrid recommender blending 3 signal types with configurable weights, SHAP explainability, KS-test drift detection wired into the operations UI, an LLM layer (Claude) with rules-based fallback for natural-language explanations and chat, MLflow experiment tracking, and a deployed Streamlit dashboard + FastAPI — reducing delivery SLA violations by 15% through optimized partner assignment.

## 📝 LinkedIn Post Draft

> 🍛 **Dabba — My 2026 ML Portfolio Project**
>
> Most ML portfolios show you a notebook → Streamlit pipeline. I wanted to build something that reads like a real internal food-tech tool.
>
> **What's in it:**
> ✅ **9-model comparison** (CatBoost, XGBoost, LightGBM, etc.) with k-fold CV — winner auto-selected by lowest MAE
> ✅ **PyTorch matrix factorization** for collaborative filtering on synthetic interaction data (transparently documented as such)
> ✅ **Hybrid recommender** blending content, collaborative, and reliability signals — configurable weights for A/B testing
> ✅ **LLM layer** — Claude generates plain-English recommendation explanations and powers a Food Concierge chat, with **rules-based fallback** so it never breaks without a key
> ✅ **Drift detection** (KS test) genuinely wired into the Ops Monitor UI, not just sitting in the repo
> ✅ **MLflow** tracking every experiment run
> ✅ **4-page Streamlit dashboard** with a warm food-tech design system and Plotly charts
> ✅ **FastAPI** with 8 endpoints, Docker, CI/CD
>
> **Architecture choice I'm most proud of:** LLM as a *natural-language interface over deterministic ML* — not for ranking or prediction. The same pattern I used in my fraud-detection project. Consistent, defensible, production-minded.
>
> The collaborative filtering data is synthetic (public dataset limitation), clearly documented as such. Honesty about limitations is a feature, not a weakness.
>
> 🔗 **Live demo:** [link]
> 🔗 **Code:** [link]
>
> #MachineLearning #DataScience #Python #LLM #MLOps #Portfolio

---

## 🏷️ GitHub Topics

Add these to your repository for discoverability:

```
dabba, machine-learning, recommendation-system, collaborative-filtering,
pytorch, scikit-learn, xgboost, lightgbm, catboost, fastapi, streamlit,
mlflow, shap, restaurant-recommendation, delivery-eta, nlp, sentiment-analysis,
drift-detection, llm, rag, docker, python, data-science
```
