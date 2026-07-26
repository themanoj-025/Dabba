# Dabba — Restaurant Intelligence Platform

[![CI](https://github.com/themanoj-025/dabba/actions/workflows/ci.yml/badge.svg)](https://github.com/themanoj-025/dabba/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)](https://mlflow.org)

India-focused restaurant ranking, recommendation, and delivery-reliability platform. Deterministic ML for ranking and prediction, with optional LLM-powered explanations and natural-language interaction.

---

## The Problem

India's food-tech landscape generates massive amounts of restaurant and delivery data, yet consumers and operators lack a unified view that combines food quality, customer sentiment, and delivery reliability into a single actionable metric. Dabba binds them together.

**Approach:**
1. Mine Zomato restaurant data for ratings, cuisine diversity, and cost signals
2. Analyze customer sentiment from reviews using VADER NLP
3. Predict delivery ETA with rigorously selected ML models (Optuna HPO tuned)
4. Synthesize into a proprietary Reliability Score
5. Add collaborative filtering (PyTorch matrix factorization)
6. Wrap with an LLM layer for natural-language explanations and chat
7. Monitor for drift with KS-test-based detection and alerting

---

## Architecture

```
Kaggle Datasets (Zomato + Delivery)
    ↓
Data Pipeline — Optuna HPO → Rating Model → ETA Model → Collaborative Filtering
    ↓
Hybrid Recommender + Reliability Score → Drift Detection
    ↓
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Streamlit │   │  FastAPI  │   │  MLflow  │
│ Dashboard │   │ REST API  │   │ Tracking │
└──────────┘   └──────────┘   └──────────┘
```

### The LLM Layer

The LLM (Anthropic Claude) is used as a natural-language interface over deterministic ML and business logic:

| Component | What it does | Fallback |
|-----------|-------------|----------|
| Recommendation Narrator | Generates plain-English "why this restaurant" explanations | Template-based rules |
| RAG Similar-Restaurant Retrieval | FAISS + cosine similarity for "find me more like this" | sklearn cosine similarity |
| Food Concierge Chat | ReAct-powered multi-step tool chain | Rules-based intent matching |

---

## Model Performance

### Rating Prediction

| Model | MAE | RMSE | R² | Train Time |
|-------|-----|------|----|------------|
| RandomForest | 0.0596 | 0.1267 | 0.9172 | 8.15s |
| XGBoost | 0.1373 | 0.2012 | 0.7913 | 1.25s |
| CatBoost | 0.1637 | 0.2323 | 0.7220 | 2.49s |
| LightGBM | 0.1672 | 0.2378 | 0.7085 | 5.05s |

### ETA Prediction

| Model | MAE (min) | RMSE (min) | R² | Train Time |
|-------|-----------|------------|----|------------|
| GradientBoosting | 5.789 | 7.364 | 0.3837 | 7.75s |
| LightGBM | 5.790 | 7.370 | 0.3828 | 0.57s |
| CatBoost | 5.810 | 7.394 | 0.3788 | 2.31s |

### Reliability Score

```
reliability_score = 0.4 × norm(rating) + 0.3 × norm(sentiment) - 0.3 × norm(delay_risk)
```

---

## Quick Start

```bash
git clone https://github.com/themanoj-025/dabba.git
cd dabba
make setup

# Download datasets (requires Kaggle API token)
python setup_kaggle.py

# Train all models
make train

# Run the dashboard
make run-app

# Run the API
make run-api
```

### Docker

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| MLflow | http://localhost:5000 |

---

## Testing

```bash
make test        # pytest with coverage
make lint        # ruff, black, isort
make format      # Auto-format code
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11 |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | PyTorch (matrix factorization) |
| **NLP** | NLTK (VADER) |
| **LLM** | Anthropic Claude (optional, with fallback) |
| **Vector Search** | FAISS (with sklearn fallback) |
| **Dashboard** | Streamlit, Plotly |
| **API** | FastAPI, Pydantic |
| **Experiment Tracking** | MLflow |
| **Monitoring** | scipy.stats.ks_2samp (drift detection) |
| **HPO** | Optuna (TPE sampler) |
| **Alerting** | Slack Incoming Webhooks |
| **Testing** | pytest, pytest-cov |
| **CI/CD** | GitHub Actions |

---

## License

MIT
