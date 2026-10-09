# Dabba — Package & Module Inventory

## Installed package: `dabba` (src/dabba)

| Module | Responsibility |
|---|---|
| `__init__.py` | Package marker |
| `config.py` | Env-driven settings (DB, Redis, model paths, LLM keys) |
| `pipeline.py` | Training pipeline orchestrator (`python -m dabba.pipeline`) |
| `data/loaders.py` | Load raw CSVs + processed datasets |
| `data/cleaning.py` | Clean raw Zomato/delivery data |
| `database/session.py` | SQLAlchemy engine + session factory |
| `database/models.py` | ORM models |
| `database/repositories.py` | Data-access layer over models |
| `database/seed.py` | Seed/full-import CLI (`python -m dabba.database.seed`) |
| `features/restaurant_features.py` | Restaurant feature engineering |
| `features/delivery_features.py` | Delivery-time feature engineering |
| `features/geo.py` | Geo/spatial helpers |
| `features/traffic.py` | Traffic-window features |
| `models/base_trainer.py` | Shared trainer plumbing |
| `models/rating_model.py` | Rating regression model |
| `models/eta_model.py` | ETA regression model |
| `models/collaborative_recommender.py` | PyTorch collaborative recommender |
| `models/hybrid_recommender.py` | Hybrid rating+content recommender |
| `models/recommender.py` | FAISS-backed restaurant similarity |
| `models/model_selection.py` | Model comparison harness |
| `models/optimizer.py` | Optuna hyperparameter optimization |
| `llm/food_concierge.py` | Conversational food concierge |
| `llm/recommendation_narrator.py` | Narration of recommendations |
| `llm/rag_similar_restaurants.py` | RAG over restaurant embeddings |
| `nlp/sentiment.py` | Sentiment analysis (English) |
| `nlp/hinglish_sentiment.py` | Hinglish sentiment model |
| `monitoring/drift.py` | Data-drift detection |
| `monitoring/retrain.py` | Retraining triggers |
| `evaluation/metrics.py` | Shared metrics |
| `evaluation/business_cost.py` | Business-cost evaluation |
| `cache/redis_client.py` | Redis cache adapter |
| `observability/` | Logging/metrics hooks |

## Application packages (not installed)

| Package | Responsibility |
|---|---|
| `api/` | FastAPI: `main.py` (factory + DI), `schemas.py` (DTOs), `auth.py`, `limiter.py`, `routers/` (chat, recommend, eta, explain, restaurants, model_info) |
| `app/` | Streamlit: `streamlit_app.py` (multipage entry), `pages/` (discover, concierge, ops, model_performance), `components/` (restaurant_card), `utils/` (sanitize), `assets/theme.css` |
| `tests/` | 22 modules: unit (`test_cleaning`, `test_features`, `test_models`…), integration (`test_concierge`, `test_redis_client`), e2e (`test_workflow`), api (`test_api`) |

## Non-package trees

| Path | Purpose |
|---|---|
| `notebooks/` | 6 EDA/prototyping notebooks |
| `models/` | Trained artifacts (.pkl, .pt, .npy, FAISS .index) |
| `reports/` | Model comparisons, SHAP values, figures |
| `data/raw` + `data/processed` | Raw datasets + cleaned outputs; `data/dabba.db` runtime |
| `docker/` | Per-service Dockerfiles + entrypoint |
| `alembic/` | DB migrations |
| `docs/` | Full suite (architecture, design, technical, migration/…) |
| `setup_kaggle.py` | Kaggle dataset bootstrap |
| `mlflow.db` (untracked) | MLflow experiment store |
