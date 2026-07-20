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

## Project Structure

Dabba v3 is organized as follows:

```bash
src/dabba/          # Core ML pipeline (data, features, models, LLM, monitoring, evaluation)
app/                # Streamlit dashboard (4 pages + components + theme)
api/                # FastAPI server (5 routes: recommend, predict-eta, chat, model-info, health)
tests/              # 45 pytest tests
notebooks/          # 6 EDA and prototyping notebooks
data/               # Raw + processed datasets (gitignored)
models/             # Saved model artifacts .pkl / .pt (gitignored)
reports/            # Comparison CSVs, charts, SHAP plots (gitignored)
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

- **Run tests**: `make test` (or `pytest tests/ -v`)
- **45 tests** across: cleaning, features, model selection, collaborative filtering, drift detection, API
- **Write tests** for any new functionality using pytest
- Tests are in `tests/` and mirror the `src/dabba/` module structure

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commits
3. Update documentation if needed (README, memory.md, etc.)
4. Ensure all tests pass: `make test`
5. Ensure linters pass: `make lint`
6. Submit a pull request with a clear description

## Common Commands

```bash
make setup          # Install deps + pre-commit hooks
make train          # Run full ML pipeline
make run-app        # Start Streamlit dashboard (port 8501)
make run-api        # Start FastAPI server (port 8000)
make run-mlflow     # Start MLflow tracking UI (port 5000)
make test           # Run 45 tests with coverage
make lint           # Run ruff + black --check
make format         # Auto-format code
make clean          # Remove generated files
```

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bug reports
- Tag issues appropriately (bug, enhancement, question)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
