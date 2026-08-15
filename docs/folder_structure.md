# Dabba — Folder Structure

```
Dabba/
├── src/dabba/                    # Core ML package (src-layout)
│   ├── __init__.py
│   ├── config.py                 # Pydantic settings
│   ├── pipeline.py               # End-to-end training pipeline
│   ├── data/                     # cleaning, loaders
│   ├── database/                 # models, repositories, session, seed
│   ├── features/                 # restaurant_features, delivery_features, geo, traffic
│   ├── models/                   # recommender, hybrid, collaborative, rating, eta,
│   │                             #   optimizer, model_selection, base_trainer
│   ├── evaluation/               # metrics, business_cost
│   ├── llm/                      # food_concierge, rag_similar_restaurants, recommendation_narrator
│   ├── nlp/                      # sentiment, hinglish_sentiment
│   ├── monitoring/               # drift, retrain
│   ├── cache/                    # redis_client
│   └── observability/
│
├── api/                          # FastAPI service layer
│   ├── main.py                   # App factory
│   ├── auth.py                   # JWT auth
│   ├── limiter.py                # Rate limiting
│   ├── routers/                  # recommend, chat, eta, explain, model_info, restaurants
│   └── schemas.py                # Pydantic DTOs
│
├── app/                          # Streamlit dashboard
│   ├── streamlit_app.py          # Entry
│   ├── pages/                    # page_concierge, page_discover, page_model_performance, page_ops
│   ├── components/               # restaurant_card
│   ├── utils/                    # sanitize
│   └── assets/                   # theme.css
│
├── tests/                        # pytest suite
│   ├── unit/ integration/ e2e/   # (conftest + test files)
│
├── alembic/                      # DB migrations
├── docker/                       # Per-service Dockerfiles + entrypoint.sh
├── notebooks/                    # EDA & prototyping notebooks
├── docs/                         # Full documentation suite
│   ├── project/                  # analysis_report.md (this pass), plans, tracker
│   ├── community/ design/ product/ reference/ technical/
├── Dockerfile                    # Root container build
├── docker-compose.yml            # Orchestration
├── pyproject.toml                # Package manifest (source of truth)
├── requirements.txt              # Dependency mirror
├── Makefile                      # Task runner
├── alembic.ini                   # Alembic config
├── .env.example                  # Env-var template
├── .pre-commit-config.yaml       # Git hooks
├── setup_kaggle.py               # Kaggle data bootstrap
├── data/                         # Runtime SQLite (untracked) + raw/processed dirs
└── README.md, LICENSE, PROJECT_ANALYSIS.md, PROJECT_OVERVIEW.md
```

## Root Hygiene

- Root contains only entry points, manifests, config, and top-level directories.
- `AGENTS_FIX.md` (AI-scaffolding duplicate) **removed** in this pass.
- `data/dabba.db*` and `mlflow.db` (runtime databases) **untracked** — regenerated at runtime, gitignored.
