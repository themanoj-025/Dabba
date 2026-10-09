# 🍛 Dabba

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
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square" alt="Python" /></a>
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square" alt="MLflow" /></a>
</p>

---

## 📋 Table of Contents

- [What it does](#what-it-does)
- [📸 Screenshots](#-screenshots)
- [✨ Features](#-features)
- [📊 Model performance](#-model-performance)
- [🚀 Quick start](#-quick-start)
- [📋 Environment variables](#-environment-variables)
- [🏗️ Architecture](#️-architecture)
- [📁 Project structure](#-project-structure)
- [🧪 Testing](#-testing)
- [📡 API endpoints](#-api-endpoints)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📬 Support](#-support)
- [License](#license)

---

## What it does

Dabba is a deterministic ML platform for Indian restaurant intelligence: it ranks restaurants, predicts delivery ETA with confidence intervals, and recommends meals — with optional LLM-powered explanations layered on top. The deterministic core runs offline; the LLM layer is opt-in.

> [!NOTE] The deterministic core (ranking, ETA, recommendation) requires no API keys and is fully reproducible. The LLM explanations require `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

## Screenshots

> To add screenshots: run `make run-app`, capture your screen, save images to `docs/assets/`, and reference them below.
>
> **Suggested screenshots:**
> - Restaurant ranking dashboard
> - ETA prediction with confidence interval
> - AI concierge chat interaction

---

## ✨ Features

| Feature | Description |
| --- | --- |
| 🏪 **Restaurant ranking** | Deterministic scoring of restaurants by supply, demand, and service signals |
| 🚚 **Delivery ETA prediction** | Confidence-interval ETA given destination, traffic, and restaurant load |
| 🍛 **Meal recommendation** | Content-based recommendations over the restaurant catalog |
| 🧠 **LLM explanations (optional)** | Plain-English why-an-has-the-best-score reasoning on demand |
| 📊 **MLflow tracking** | Artifacts, parameter, and metric registry for all runs |

## 📊 Model performance

> [!IMPORTANT] Model cards, exact test metrics, and the train/validation split are maintained in `model_cards.md`/`reports/` so the README never drifts from reproduced numbers. Add a `model_cards.md` + CI check if this is still aspirational.

| Model | Task | Reported metric | Notes |
| --- | --- | --- | --- |
| Gradient Boosting | Restaurant ranking | ROC-AUC | Deterministic, offline |
| XGBoost | ETA prediction | RMSE | Confidence interval output |
| Content-based CF | Meal recommendation | NDCG@10 | Deterministic |
| LLM (optional) | Explanation generation | — | Opt-in, needs key |

### Determinism

The deterministic pipeline produces stable outputs given the same input catalog and parameters. Run `python -m dabba.deterministic.benchmark` to reproduce the benchmark exactly. The LLM path is **not** deterministic by design — it is only a post-hoc explanation layer.

## 🚀 Quick start

### Prerequisites

- Python 3.11 or newer
- MLflow (for experiment tracking)

### Install & run

```bash
# 1. Clone the repository
git clone https://github.com/themanoj-025/Dabba.git
cd Dabba

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template
cp .env.example .env
#   → Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable LLM explanations

# 5. Run the app
streamlit run app/app.py
```

### Environment variables

| Variable | Default | Required | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | — | No | Enables LLM explanations |
| `ANTHROPIC_API_KEY` | — | No | Alternate LLM provider for explanations |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | No | MLflow tracking server |

## 🏗️ Architecture

```text
Dabba/
├── dabba/
│   ├── deterministic/        # Ranking, ETA, recommendation (offline)
│   ├── llm/                  # Optional explanation generator
│   ├── mlflow/               # Tracking + model registry
│   ├── app/                  # Streamlit dashboard
│   └── scripts/              # Training + evaluation entry points
├── model_cards.md
├── reports/
├── requirements.txt
└── README.md
```

## 📁 Project structure

```
Dabba/
├── dabba/
│   ├── deterministic/        # Ranking, ETA, recommendation
│   ├── llm/                  # Optional explanations
│   ├── mlflow/               # Tracking + registry
│   ├── app/                  # Streamlit dashboard
│   └── scripts/              # Training + evaluation
├── model_cards.md
├── reports/
├── requirements.txt
└── README.md
```

## 🧪 Testing

```bash
# Run the test suite
pytest tests/ -v
```

## 📡 API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/restaurants/rank` | Rank restaurants for a given query/city |
| `POST` | `/api/v1/eta` | Predict ETA + confidence interval for a restaurant |
| `GET` | `/api/v1/meals/recommended` | Recommend meals for a user/city |
| `POST` | `/api/v1/explain` | Generate an optional LLM explanation |
| `GET` | `/health` | Health check |

## 🗺️ Roadmap

> [!CAUTION] Checked items are built and verified. Unchecked items are tracked in the issue tracker.

- [x] Restaurant ranking
- [x] Delivery ETA prediction with confidence interval
- [x] Meal recommendation
- [x] MLflow tracking
- [x] Optional LLM explanations
- [ ] Real-time dashboard (tracked public issue)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📬 Support

- 🐛 [Report a bug](https://github.com/themanoj-025/Dabba/issues)
- 💡 [Request a feature](https://github.com/themanoj-025/Dabba/issues)
- 📧 Email the maintainer via the issue tracker

## License

MIT License — see [LICENSE](LICENSE).

<!-- TODO: version/license is not yet verified in the pyproject manifest; reconcile before the next release tag. -->
