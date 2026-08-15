.PHONY: setup test run-app lint train format clean help run-mlflow db-import db-migrate db-shell db-test

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies and create directories
	pip install -r requirements.txt
	mkdir -p data/raw data/processed models reports/figures reports/mlruns
	pre-commit install

test:  ## Run pytest with coverage
	pytest tests/ -v --tb=short --cov=src/dabba --cov-report=term-missing -W ignore::DeprecationWarning

lint:  ## Run linters (ruff + black + isort check)
	ruff check src/ tests/ app/ api/
	black --check src/ tests/ app/ api/
	isort --check-only src/ tests/ app/ api/

format:  ## Auto-format code
	ruff check --fix src/ tests/ app/ api/
	black src/ tests/ app/ api/
	isort src/ tests/ app/ api/

train:  ## Run full v3 training pipeline
	python -m dabba.pipeline

run-app:  ## Run the Streamlit dashboard
	streamlit run app/streamlit_app.py

run-api:  ## Run the FastAPI server
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-mlflow:  ## Start MLflow tracking server
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri reports/mlruns

db-import:  ## Full CSV→DB import (load raw CSVs → clean → feature engineer → seed)
	python -m dabba.database.seed --full-import

db-migrate:  ## Run Alembic migrations
	alembic upgrade head

db-shell:  ## Open an interactive database shell
	python -c "from dabba.database.session import get_db; from dabba.database.models import *; db = next(get_db()); print('Session ready — use db.query(...)')" || true

db-rollback:  ## Rollback last Alembic migration
	alembic downgrade -1

db-history:  ## Show Alembic migration history
	alembic history

db-revision:  ## Create a new Alembic auto-migration
	alembic revision --autogenerate -m "$(message)"

clean:  ## Remove generated files
	rm -rf data/processed/* models/*.pkl models/*.pt reports/figures/*.png reports/figures/*.json reports/figures/*.html reports/*.csv reports/ab_scenarios.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
