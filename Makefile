.PHONY: setup test run-app lint train format clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies and create directories
	pip install -r requirements.txt
	mkdir -p data/raw data/processed models reports/figures
	pre-commit install

test:  ## Run pytest with coverage
	pytest tests/ -v --tb=short --cov=src/dabba --cov-report=term-missing

lint:  ## Run linters (ruff + black + isort check)
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Auto-format code
	ruff check --fix src/ tests/
	black src/ tests/
	isort src/ tests/

train:  ## Run full model training pipeline (download data, clean, train, compare)
	python -m dabba.pipeline

run-app:  ## Run the Streamlit dashboard
	streamlit run app/streamlit_app.py

run-api:  ## Run the FastAPI server
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

clean:  ## Remove generated files
	rm -rf data/processed/* models/*.pkl reports/figures/*.png reports/figures/*.html
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
