<p align="center">
  <img src="https://img.shields.io/badge/Dabba-Restaurant%20Intelligence-orange?style=for-the-badge" alt="Dabba Logo" />
</p>

<h1 align="center">Dabba</h1>

<p align="center">
  <strong>Restaurant intelligence and delivery optimization platform combining offline ML ranking, ETA estimation, and LLM explanations.</strong>
</p>

<p align="center">
  <a href="https://github.com/themanoj-025/Dabba/actions"><img src="https://img.shields.io/github/actions/workflow/status/themanoj-025/Dabba/ci.yml?style=flat-square&label=CI" alt="CI Status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License MIT" /></a>
  <a href="https://github.com/themanoj-025/Dabba/stargazers"><img src="https://img.shields.io/github/stars/themanoj-025/Dabba?style=social" alt="GitHub Stars" /></a>
  <a href="https://github.com/themanoj-025/Dabba/issues"><img src="https://img.shields.io/github/issues/themanoj-025/Dabba?style=flat-square" alt="GitHub Issues" /></a>
  <a href="https://github.com/themanoj-025/Dabba/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome" /></a>
</p>

---

## Overview

Dabba is a machine learning platform designed for restaurant intelligence, delivery time estimation, and personalized meal recommendations. It features a deterministic offline prediction engine (for fast, reproducible scoring with zero external API dependencies) alongside an optional LLM layer for qualitative justifications and natural-language recommendations.

The platform provides dual access modes: a high-performance FastAPI backend for operational serving and an interactive multi-page Streamlit application for exploratory data analysis.

> [!NOTE]
> Core ranking, delivery ETA, and meal recommendation models run completely offline. External LLM API keys (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) are strictly optional and only leveraged for qualitative explanation prompts.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Quick Start](#installation--quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Support](#support)

---

## Key Features

- **Deterministic Restaurant Ranking**: Algorithmic scoring evaluating cuisine demand, distance decay, and operational reliability metrics.
- **Delivery ETA with Confidence Intervals**: Gradient-boosted regression modeling transit durations considering restaurant preparation backlog and geospatial variables.
- **Content-Based Meal Recommendations**: Candidate retrieval and affinity scoring across culinary catalogs.
- **MLflow Experiment Tracking**: Tracks feature transformation pipelines, hyperparameter variations, and evaluation metrics in `mlflow.db`.
- **Explainable AI Integration**: Transparent reasoning translating internal tree-split attributions into human-readable consumer summaries.

---

## Architecture

```text
               FastAPI Backend (api/main.py)
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      Deterministic Engine          LLM Service (Optional)
    (src/dabba/models/pipeline)   (OpenAI / Anthropic Provider)
               │                         │
     ┌─────────┴─────────┐               │
     ▼                   ▼               ▼
Restaurant Rank       ETA Predict  Natural Language Insights
(Gradient Boost)       (XGBoost)         │
     │                   │               │
     └─────────┬─────────┴───────────────┘
               ▼
     Streamlit Dashboard (app/streamlit_app.py)
```

---

## Project Structure

```text
Dabba/
├── api/                    # FastAPI routes, middleware, and schemas
│   ├── routers/            # Endpoint handlers (rank, ETA, explain)
│   ├── health.py           # Probe endpoints (/health, /metrics)
│   └── main.py             # Web API entry point
├── app/                    # Multi-page Streamlit application
│   ├── components/         # Reusable dashboard widgets
│   ├── pages/              # Analytics and evaluation views
│   └── streamlit_app.py    # Streamlit dashboard entry point
├── src/
│   └── dabba/              # Core installable Python package
│       ├── data/           # Ingestion and normalization pipelines
│       ├── evaluation/     # Metric scorers (NDCG, RMSE, MAE)
│       ├── features/       # Feature extraction and encoding
│       ├── llm/            # LLM explanation wrappers
│       ├── models/         # Estimators and inference pipelines
│       └── pipeline.py     # Training workflow runner
├── tests/                  # Unit and integration test suites
├── Dockerfile              # Container deployment specification
├── docker-compose.yml      # Multi-service container orchestration
├── pyproject.toml          # Build configuration and linting rules
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Prerequisites

- **Python**: `>= 3.11`
- **MLflow**: Local SQLite tracking backend (`sqlite:///mlflow.db`)
- **LLM API Key**: Optional (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)

---

## Installation & Quick Start

### 1. Clone & Setup Environment

```bash
# Clone repository
git clone https://github.com/themanoj-025/Dabba.git
cd Dabba

# Create and activate virtual environment
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
```

### 3. Run the Services

**Run FastAPI Service**:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Run Streamlit Analytics Dashboard**:
```bash
streamlit run app/streamlit_app.py
```

---

## Configuration

| Environment Variable | Default | Purpose |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Target URI for MLflow experiment tracking |
| `OPENAI_API_KEY` | `""` | Optional API key for OpenAI narrative explanations |
| `ANTHROPIC_API_KEY` | `""` | Optional API key for Claude narrative explanations |
| `DATABASE_URL` | `sqlite:///dabba.db` | Primary application relational database |

---

## API Reference

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/restaurants/rank` | Returns ordered list of restaurants scored for query parameters |
| `POST` | `/api/v1/eta` | Computes expected delivery duration with lower/upper confidence bounds |
| `GET` | `/api/v1/meals/recommended` | Recommends menu items based on catalog similarity |
| `POST` | `/api/v1/explain` | Generates narrative explanation for ranking outputs (requires LLM key) |
| `GET` | `/health` | Service health status check |

---

## Testing

```bash
# Run pytest test suite
pytest tests/ -v

# Run with code coverage reporting
pytest --cov=dabba --cov=api tests/
```

---

## Roadmap

- [x] Deterministic restaurant ranking algorithm
- [x] Delivery ETA estimation with prediction intervals
- [x] MLflow integration for model run tracking
- [x] Multi-page Streamlit analytics interface
- [x] FastAPI production API endpoints
- [ ] Real-time order simulation pipeline <!-- TODO: Track via GitHub issue -->
- [ ] Driver route optimization module <!-- TODO: Track via GitHub issue -->

---

## Contributing

Contributions are welcome! Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for code quality requirements and workflow details.

---

## Security

Please report vulnerabilities following the responsible disclosure process in [SECURITY.md](SECURITY.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Support

- 🐛 **Issue Tracker**: [GitHub Issues](https://github.com/themanoj-025/Dabba/issues)
- 💬 **Discussions & Feedback**: [GitHub Pull Requests](https://github.com/themanoj-025/Dabba/pulls)
