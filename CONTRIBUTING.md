# Contributing to Dabba

Thank you for your interest in contributing to Dabba! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone** the repository
2. **Create a virtual environment**: `python -m venv .venv && source .venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Install in editable mode**: `pip install -e .`
5. **Install pre-commit hooks**: `pre-commit install`
6. **Download datasets** (Kaggle): `python setup_kaggle.py` (requires Kaggle API token)
7. **Train all models**: `make train` (~15-20 minutes for full pipeline)
8. **Import to database**: `make db-import` (loads processed CSVs into SQLite/Postgres)

## Project Structure

Dabba is organized as follows:

```bash
src/dabba/          # Core ML pipeline (data, features, models, LLM, monitoring, evaluation)
  database/         # SQLAlchemy ORM, session management, seed script, repositories
  observability/    # Structured JSON logging, Prometheus metrics, request tracing
  nlp/              # VADER sentiment + Hinglish multilingual sentiment (transformers)
  features/         # Restaurant/delivery features + real-time traffic API (TomTom/Mappls)
  monitoring/       # KS-test drift detection + Slack alerts + drift-triggered retraining
app/                # Streamlit dashboard (4 pages + components + theme)
api/                # FastAPI server (10 endpoints across 8 routers: health, metrics, recommend, predict-eta, chat, model-info, explain, restaurants list/get/search)
tests/              # 220+ pytest tests (unit, integration, e2e)
notebooks/          # 6 EDA and prototyping notebooks
data/               # Raw + processed datasets (gitignored)
models/             # Saved model artifacts .pkl / .pt (gitignored)
reports/            # Comparison CSVs, charts, SHAP plots (gitignored)
docker/             # Per-service Dockerfiles (api, streamlit, mlflow) + entrypoint.sh
alembic/            # Database migrations (SQLite → Postgres)
```

## Code Style

- **Formatter:** Black (line length 88)
- **Import sorting:** isort (profile = black)
- **Linter:** Ruff (all checks passed)
- **Type hints:** Required on all function signatures
- **Docstrings:** Google style on all public functions/classes

Run before committing:
```bash
make format        # Auto-format: ruff --fix + black + isort
make lint          # Verify: ruff check + black --check + isort --check
```

## Testing

- **Run tests**: `make test` (or `pytest tests/ -v --timeout=60`)
- **220+ tests** across: cleaning, features, model selection, collaborative filtering, drift detection, API, database, narrator, RAG, redis, optimizer, traffic, retrain, integration, e2e
- **Write tests** for any new functionality using pytest
- Tests are in `tests/` and mirror the `src/dabba/` module structure
- **Slow tests** (model training, transformers download) are excluded from CI — run them locally with longer timeouts via `pytest tests/test_eta_model.py --timeout=300`

## Database Operations

```bash
make db-import     # Full CSV→DB import (load raw → clean → feature engineer → seed)
make db-migrate    # Run Alembic migrations (alembic upgrade head)
make db-rollback   # Rollback last migration (alembic downgrade -1)
make db-history    # Show migration history
make db-revision message="description"  # Create new auto-migration
make db-shell      # Open interactive DB shell
```

For manual seeding:
```bash
python -m dabba.database.seed                    # Seed from processed CSVs
python -m dabba.database.seed --full-import      # Full import pipeline
python -m dabba.database.seed --clear             # Clear all data before seeding
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commits
3. Update documentation if needed (README, memory.md, etc.)
4. Ensure all tests pass: `make test`
5. Ensure linters pass: `make lint`
6. Submit a pull request with a clear description

## LLM Features (Optional)

To enable LLM-powered recommendations and concierge chat:

```bash
echo "DABBA_LLM_ENABLED=true" >> .env
echo "DABBA_ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

Without this, all LLM features fall back to **rules-based behavior** — the app never breaks.

## Real-Time Traffic (Optional)

For dynamic ETA predictions with real traffic data, configure one of:

- **TomTom**: `DABBA_TOMTOM_API_KEY` — https://developer.tomtom.com/ (2500 req/day free)
- **Mappls**: `DABBA_MAPPLS_API_KEY` — https://www.mappls.com/ (India-specific)

If both are unset, falls back to time-of-day traffic simulation.

## Hinglish Sentiment (Optional)

For Hindi/English code-switched review sentiment, the Hinglish module uses a HuggingFace
multilingual model. Install with:

```bash
pip install transformers torch
```

Falls back to VADER (English-only) if `transformers` is not installed.

## Observability

- **Structured JSON logging**: All log lines are JSON with timestamp, level, logger, request ID
- **Prometheus `/metrics`**: Available at `http://localhost:8000/metrics` (no auth)
- **Request tracing**: Every request has a unique `X-Request-ID` header propagated through logs

## Common Commands

```bash
make setup          # Install deps + pre-commit hooks
make train          # Run full ML pipeline
make run-app        # Start Streamlit dashboard (port 8501)
make run-api        # Start FastAPI server (port 8000)
make run-mlflow     # Start MLflow tracking UI (port 5000)
make test           # Run 220+ tests with coverage
make lint           # Run ruff + black --check
make format         # Auto-format code
make clean          # Remove generated files
make db-import      # Import CSVs to database
make db-migrate     # Run database migrations
```

## Docker

```bash
docker-compose up --build   # Start all services (API, Streamlit, MLflow, Postgres, Redis)
```

The API service runs `docker/entrypoint.sh` which executes `alembic upgrade head`
before starting uvicorn, ensuring the database schema is always up-to-date.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bug reports
- Tag issues appropriately (bug, enhancement, question)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
