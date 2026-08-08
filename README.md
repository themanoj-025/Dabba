<p align="center">
  <img src="https://img.shields.io/badge/Dabba-Restaurant%20Intelligence-orange?style=for-the-badge" alt="Dabba Logo" />
</p>

<h1 align="center">🍛 Dabba</h1>

<p align="center">
  <strong>India's Restaurant Intelligence Platform</strong>
</p>

<p align="center">
  <a href="https://github.com/themanoj-025/Dabba/actions"><img src="https://img.shields.io/github/actions/workflow/status/themanoj-025/Dabba/ci.yml?style=flat-square&label=CI" alt="CI Status" /></a>
  <a href="https://github.com/themanoj-025/Dabba/blob/main/LICENSE"><img src="https://img.shields.io/github/license/themanoj-025/Dabba?style=flat-square" alt="License" /></a>
  <a href="https://github.com/themanoj-025/Dabba/stargazers"><img src="https://img.shields.io/github/stars/themanoj-025/Dabba?style=social" alt="Stars" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square" alt="Python" /></a>
  <a href="#"><img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square" alt="MLflow" /></a>
</p>

---

<p align="center">
  <strong>Ranking restaurants. Predicting deliveries. Recommending meals.</strong>
  <br />
  Deterministic ML for restaurant intelligence, with optional LLM-powered explanations.
</p>

---

## 🎯 The Problem

India's food-tech landscape generates massive amounts of restaurant and delivery data, yet consumers and operators lack a unified view that combines food quality, customer sentiment, and delivery reliability into a single actionable metric.

**Dabba binds them together.**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏆 **Restaurant Ranking** | Proprietary Reliability Score combining rating, sentiment, and delivery performance |
| ⏱️ **ETA Prediction** | ML-powered delivery time estimation (MAE: 5.8 min) |
| 🎯 **Smart Recommendations** | Hybrid collaborative + content-based filtering |
| 📊 **Sentiment Analysis** | VADER NLP on customer reviews (Hinglish-aware) |
| 🤖 **AI Concierge** | Natural-language chat via Anthropic Claude |
| 📈 **Drift Detection** | KS-test monitoring with Slack alerts |
| 🔬 **Experiment Tracking** | MLflow for model versioning |
| 🎛️ **HPO** | Optuna hyperparameter optimization |

---

## 📊 Model Performance

### Rating Prediction (Best: RandomForest)

| Model | MAE | RMSE | R² | Train Time |
|-------|-----|------|-----|------------|
| **RandomForest** | **0.0596** | **0.1267** | **0.9172** | 8.15s |
| XGBoost | 0.1373 | 0.2012 | 0.7913 | 1.25s |
| CatBoost | 0.1637 | 0.2323 | 0.7220 | 2.49s |
| LightGBM | 0.1672 | 0.2378 | 0.7085 | 5.05s |

### ETA Prediction (Best: GradientBoosting)

| Model | MAE (min) | RMSE (min) | R² | Train Time |
|-------|-----------|------------|-----|------------|
| **GradientBoosting** | **5.789** | **7.364** | **0.3837** | 7.75s |
| LightGBM | 5.790 | 7.370 | 0.3828 | 0.57s |
| CatBoost | 5.810 | 7.394 | 0.3788 | 2.31s |

### Reliability Score Formula

```
reliability_score = 0.4 × norm(rating) + 0.3 × norm(sentiment) - 0.3 × norm(delay_risk)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Kaggle account (for datasets)
- Docker & Docker Compose (recommended)

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/themanoj-025/Dabba.git
cd Dabba
docker-compose up --build
```

| Service | URL |
|---------|-----|
| 🖥️ Streamlit Dashboard | http://localhost:8501 |
| 🔌 FastAPI | http://localhost:8000 |
| 📈 MLflow | http://localhost:5000 |

### Option 2: Local Development

```bash
# Setup
git clone https://github.com/themanoj-025/Dabba.git
cd Dabba
make setup

# Download datasets
python setup_kaggle.py

# Train all models
make train

# Run dashboard
make run-app

# Run API
make run-api
```

---

## 📋 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DABBA_API_KEY` | API authentication key | — | Optional |
| `DABBA_DATABASE_URL` | PostgreSQL connection | SQLite fallback | ❌ |
| `DABBA_MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` | ❌ |
| `ANTHROPIC_API_KEY` | Claude API key | — | Optional |
| `KAGGLE_USERNAME` | Kaggle username | — | ✅ |
| `KAGGLE_KEY` | Kaggle API key | — | ✅ |
| `SLACK_WEBHOOK_URL` | Drift alerts | — | Optional |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Kaggle Datasets (Zomato + Delivery)                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Data Pipeline                                 │
│  Clean → Feature Engineer → Resample → Train → Evaluate         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     ML Models                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Rating Model │  │  ETA Model   │  │ Collaborative│          │
│  │   (RF/XGB)   │  │  (GB/LGB)   │  │  Filtering   │          │
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
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Dabba/
├── api/
│   ├── main.py              # FastAPI application
│   └── routers/             # API endpoints
│       ├── recommend.py     # /v1/recommend
│       ├── eta.py           # /v1/predict-eta
│       ├── chat.py          # /v1/chat
│       └── explain.py       # /v1/explain
├── src/dabba/
│   ├── config.py            # Configuration
│   ├── pipeline.py          # Training pipeline
│   ├── data/                # Data loading & cleaning
│   ├── features/            # Feature engineering
│   ├── models/              # ML models
│   ├── evaluation/          # Metrics & scoring
│   ├── nlp/                 # Sentiment analysis
│   ├── database/            # SQLAlchemy models
│   └── observability/       # Prometheus metrics
├── app/
│   └── streamlit_app.py     # Dashboard
├── docker/                  # Per-service Dockerfiles
├── tests/                   # Test suite
├── migrations/              # Alembic migrations
├── requirements.txt
├── Makefile
└── docker-compose.yml
```

---

## 🧪 Testing

```bash
# Run tests
make test

# Run linters
make lint

# Auto-format code
make format
```

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/v1/recommend` | Restaurant recommendations |
| `POST` | `/v1/predict-eta` | Delivery ETA prediction |
| `POST` | `/v1/chat` | Food concierge chat |
| `GET` | `/v1/model-info` | Model metadata |
| `GET` | `/v1/restaurants` | Restaurant listing |

---

## 🗺️ Roadmap

- [x] Rating prediction (4 models)
- [x] ETA prediction (3 models)
- [x] Collaborative filtering (PyTorch)
- [x] Reliability score
- [x] FastAPI REST API
- [x] Streamlit dashboard
- [x] MLflow tracking
- [x] Drift detection
- [x] Docker deployment
- [ ] Real-time Zomato scraping
- [ ] User authentication
- [ ] Mobile app
- [ ] Multi-city support

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Zomato](https://www.zomato.com/) - Restaurant data
- [Kaggle](https://www.kaggle.com/) - Dataset hosting
- [scikit-learn](https://scikit-learn.org/) - ML framework
- [PyTorch](https://pytorch.org/) - Deep learning
- [FastAPI](https://fastapi.tiangolo.com/) - REST API
- [Streamlit](https://streamlit.io/) - Dashboard
- [MLflow](https://mlflow.org/) - Experiment tracking
- [Optuna](https://optuna.org/) - Hyperparameter optimization

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/themanoj-025">themanoj-025</a>
</p>

<p align="center">
  If you find this project useful, please give it a ⭐ star!
</p>
