# 🔗 Dabba v3 — Dependency Graph

## File Dependency Map

```
src/dabba/
├── pipeline.py (orchestrator — depends on ALL modules below)
│
├── config.py  ← depended on by EVERY module
│
├── data/
│   ├── loaders.py → config.py, pandas
│   └── cleaning.py → config.py, pandas, numpy
│
├── features/
│   ├── restaurant_features.py → config.py, geo.py, pandas
│   ├── delivery_features.py → config.py, geo.py, pandas
│   └── geo.py → sklearn (KMeans, DBSCAN, Agglomerative, silhouette_score)
│
├── nlp/
│   └── sentiment.py → config.py, nltk (VADER), ast, pandas
│
├── models/
│   ├── rating_model.py → config.py, sklearn, xgboost, lightgbm, catboost, mlflow, joblib
│   ├── eta_model.py → config.py, sklearn, xgboost, lightgbm, catboost, mlflow, joblib
│   ├── model_selection.py → config.py, pandas, joblib
│   ├── collaborative_recommender.py → config.py, torch, pandas
│   ├── hybrid_recommender.py → config.py, recommender.py, collaborative_recommender.py, sklearn
│   ├── recommender.py → config.py, sklearn, pandas
│   └── optimizer.py → scipy.optimize
│
├── llm/
│   ├── recommendation_narrator.py → config.py (rules-based, optional anthropic)
│   ├── rag_similar_restaurants.py → config.py, faiss, sklearn (fallback), numpy
│   └── food_concierge.py → config.py (rules-based, optional anthropic)
│
├── monitoring/
│   └── drift.py → config.py, scipy.stats, pandas
│
└── evaluation/
    ├── metrics.py → sklearn
    └── business_cost.py → config.py, pandas, numpy

api/
├── main.py → config.py, api/schemas.py, api/routers/*, joblib, pandas
├── schemas.py → pydantic
├── routers/
│   ├── recommend.py → api/schemas.py, dabba.models.hybrid_recommender, dabba.llm, joblib
│   ├── eta.py → api/schemas.py, joblib, pandas
│   ├── chat.py → api/schemas.py, dabba.llm.food_concierge
│   └── model_info.py → api/schemas.py, config.py, pandas

app/
├── streamlit_app.py → app/pages/*, dabba.config
├── components/
│   └── restaurant_card.py → streamlit
└── pages/
    ├── page_discover.py → dabba.models.hybrid_recommender, dabba.llm, pandas
    ├── page_ops.py → app.components, dabba.monitoring.drift, joblib, pandas
    ├── page_model_performance.py → pandas, plotly, Path
    └── page_concierge.py → dabba.llm.food_concierge, pandas, Path

tests/
├── test_cleaning.py → src/dabba/data/cleaning.py
├── test_features.py → src/dabba/features/*.py
├── test_model_selection.py → src/dabba/models/model_selection.py
├── test_drift.py → src/dabba/monitoring/drift.py
├── test_collaborative_recommender.py → src/dabba/models/collaborative_recommender.py
├── test_api.py → api/main.py (fastapi.testclient)
├── test_rating_model.py → src/dabba/models/rating_model.py
├── test_eta_model.py → src/dabba/models/eta_model.py
└── test_recommender.py → src/dabba/models/recommender.py
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
├── torch       → collaborative filtering (matrix factorization)
├── mlflow      → experiment tracking (rating + ETA models)
├── anthropic   → LLM layer (narrator, RAG, concierge)
├── faiss-cpu   → RAG similar-restaurant retrieval
├── shap        → model explainability (rating + ETA)
├── plotly      → interactive charts (UI + pipeline)
├── folium      → geographic visualization
├── nltk        → VADER sentiment analysis
└── skorch      → neural network ETA model (optional)

Testing:
├── pytest      → test framework
├── pytest-cov  → coverage reporting
├── httpx       → API test client (via fastapi.testclient)

Infrastructure:
├── streamlit   → dashboard
├── fastapi     → REST API
├── uvicorn     → ASGI server
├── pydantic    → schema validation
└── docker      → containerization
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
