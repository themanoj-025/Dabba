# 🔗 Dabba v3 — Dependency Graph

## File Dependency Map

```
src/dabba/
├── pipeline.py (orchestrator → depends on ALL modules below)
│
├── config.py  ← depended on by EVERY module
│   └── Optuna / Slack / DB settings added
│
├── data/
│   ├── loaders.py → config.py, pandas
│   └── cleaning.py → config.py, pandas, numpy
│
├── features/
│   ├── restaurant_features.py → config.py, geo.py, pandas
│   ├── delivery_features.py → config.py, geo.py, pandas
│   │   └── (rush hour, sin/cos, city_zone, weather, interactions)
│   └── geo.py → sklearn (KMeans, DBSCAN, Agglomerative, silhouette_score)
│
├── nlp/
│   └── sentiment.py → config.py, nltk (VADER), ast, pandas
│
├── models/
│   ├── base_trainer.py → config.py, sklearn, optuna, mlflow, joblib
│   │   └── (shared CV, MLflow, tune_hyperparameters, HPO search spaces)
│   ├── rating_model.py → base_trainer, sklearn, xgboost, lightgbm, catboost
│   │   └── (get_tuned_rating_models with Optuna)
│   ├── eta_model.py → base_trainer, sklearn, xgboost, lightgbm, catboost
│   │   └── (get_tuned_eta_models with Optuna)
│   ├── model_selection.py → config.py, pandas, joblib
│   ├── collaborative_recommender.py → config.py, torch, pandas
│   ├── hybrid_recommender.py → config.py, recommender.py, cf, sklearn
│   ├── recommender.py → config.py, sklearn, pandas
│   └── optimizer.py → scipy.optimize
│
├── llm/
│   ├── recommendation_narrator.py → config.py (rules-based, optional anthropic)
│   ├── rag_similar_restaurants.py → config.py, faiss, sklearn, numpy
│   └── food_concierge.py → config.py (ReAct loop, max 4 steps)
│       └── (multi-step tool chain with tool_result feedback)
│
├── monitoring/
│   └── drift.py → config.py, scipy.stats, pandas, urllib.request
│       └── (AlertResult, _send_slack_alert, cooldown, detect_and_alert)
│
├── database/
│   ├── session.py → config.py, sqlalchemy
│   └── models.py → sqlalchemy (Restaurant, Order, Prediction, DriftLog, ExperimentResult)
│
├── cache/
│   └── redis_client.py → config.py, redis/fakeredis
│
└── evaluation/
    ├── metrics.py → sklearn
    └── business_cost.py → config.py, pandas, numpy

api/
├── main.py → config.py, api/schemas.py, api/routers/*
│   └── (models stored in app.state on startup, Depends DI)
├── auth.py → config.py (API key verification)
├── limiter.py → slowapi
├── schemas.py → pydantic
├── routers/
│   ├── recommend.py → schemas, hybrid_recommender, llm, Depends(app.state)
│   ├── eta.py → schemas, joblib, Depends(app.state)
│   ├── chat.py → schemas, llm/food_concierge, Depends(app.state)
│   └── model_info.py → schemas, config.py, pandas

app/
├── streamlit_app.py → app/pages/*, dabba.config
├── components/
│   └── restaurant_card.py → streamlit
├── utils/
│   └── sanitize.py (html_escape for XSS prevention)
└── pages/
    ├── page_discover.py → hybrid_recommender, dabba.llm, pandas
    ├── page_ops.py → components, dabba.monitoring.drift, joblib, pandas
    ├── page_model_performance.py → pandas, plotly, Path
    └── page_concierge.py → dabba.llm.food_concierge (ReAct), pandas

docker/
├── api.Dockerfile       → FastAPI + uvicorn + /health curl check
├── streamlit.Dockerfile  → Streamlit + curl healthcheck
└── mlflow.Dockerfile     → MLflow server + API healthcheck

tests/
├── test_cleaning.py → src/dabba/data/cleaning.py
├── test_features.py → src/dabba/features/*.py (18 tests — cyclical, city_zone, rush hour, interactions)
├── test_model_selection.py → src/dabba/models/model_selection.py
├── test_drift.py → src/dabba/monitoring/drift.py (13 tests — Slack, cooldown, detect_and_alert)
├── test_collaborative_recommender.py → src/dabba/models/collaborative_recommender.py
├── test_api.py → api/main.py (fastapi.testclient)
├── test_rating_model.py → src/dabba/models/rating_model.py
├── test_eta_model.py → src/dabba/models/eta_model.py
├── test_recommender.py → src/dabba/models/recommender.py
├── test_optuna_tuning.py → src/dabba/models/base_trainer.py (25 tests — HPO, search spaces, MLflow)
├── integration/__init__.py → Integration test directory
└── e2e/__init__.py → E2E test directory
```

## External Package Dependencies

```
pandas          → all data processing + most modules
numpy           → features, models, evaluation, monitoring → all modules
scikit-learn    → features, models, evaluation + metrics
joblib          → model persistence (rating + ETA)

Optional:
├── xgboost     → rating model + ETA model comparison
├── lightgbm    → rating model + ETA model comparison
├── catboost    → rating model + ETA model comparison
├── optuna      → hyperparameter optimization (TPE sampler, 50 trials)
├── torch       → collaborative filtering (matrix factorization)
├── skorch      → neural network ETA model (optional)
├── mlflow      → experiment tracking (rating + ETA models)
├── anthropic   → LLM layer (narrator, RAG, ReAct concierge)
├── faiss-cpu   → RAG similar-restaurant retrieval
├── shap        → model explainability (rating + ETA)
├── plotly      → interactive charts (UI + pipeline)
├── folium      → geographic visualization
├── nltk        → VADER sentiment analysis

Infrastructure:
├── fastapi     → REST API + Depends DI + app.state
├── uvicorn     → ASGI server
├── pydantic    → schema validation
├── slowapi     → rate limiting
├── sqlalchemy  → ORM (Restaurant, Order, Prediction, DriftLog, ExperimentResult)
├── alembic     → database migrations
├── redis       → caching (ETA predictions, recommendations)
├── streamlit   → dashboard
└── docker      → 3 per-service containers with healthchecks

Testing:
├── pytest      → test framework (100+ tests)
├── pytest-cov  → coverage reporting
└── httpx       → API test client (via fastapi.testclient)
```

## Critical Files (High Impact)

1. **`src/dabba/config.py`** — Central configuration; every module depends on it
2. **`src/dabba/pipeline.py`** — Full pipeline orchestrator; coordinates all stages
3. **`src/dabba/models/rating_model.py`** — Rating training + comparison + MLflow
4. **`src/dabba/models/eta_model.py`** — ETA training + comparison + MLflow
5. **`src/dabba/data/cleaning.py`** — Data cleaning; affects all downstream results
6. **`api/main.py`** — API server with model loading

## High-Impact New v3 Files

7. **`src/dabba/models/collaborative_recommender.py`** — PyTorch MF + synthetic data gen
8. **`src/dabba/models/hybrid_recommender.py`** — Blends 3 signal types into rankings
9. **`src/dabba/llm/recommendation_narrator.py`** — LLM + rules-based explanations
10. **`src/dabba/monitoring/drift.py`** — KS-test drift detection
11. **`app/pages/page_discover.py`** — Main customer-facing UI
12. **`app/pages/page_ops.py`** — Operations UI with drift alerting
13. **`api/routers/chat.py`** — Chat endpoint with tool-use

## Files That Should Not Be Modified Lightly

| File | Reason |
|------|--------|
| `src/dabba/config.py` | All modules depend on it — changes propagate everywhere |
| `src/dabba/data/cleaning.py` | Cleaning strategy affects ALL downstream model performance |
| `src/dabba/models/rating_model.py` | MLflow logging + model comparison logic; test-critical |
| `src/dabba/models/eta_model.py` | Same as rating_model.py for ETA task |
| `api/schemas.py` | All API contracts defined here — frontend + backend depend on it |
