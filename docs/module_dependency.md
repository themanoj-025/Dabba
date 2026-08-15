# Dabba — Module Dependency Map

## Core package (src/dabba) internal dependencies

```
dabba.config                ← imported by every domain module (paths, model names, env)
dabba.data.loaders          ← used by features.*, database.seed, pipeline
dabba.data.cleaning         ← used by features.delivery_features, pipeline
dabba.database.session      ← used by database.repositories, api routers, app
dabba.database.models       ← ORM models; used by repositories, seed
dabba.database.repositories ← used by api routers (via schemas), database.seed
dabba.features.*            ← depend on data.* only (geo, traffic are leaf helpers)
dabba.models.base_trainer   ← shared trainer; used by all model modules
dabba.models.rating_model / eta_model / collaborative_recommender / hybrid_recommender
                            ← depend on features.* + base_trainer
dabba.models.model_selection ← depends on models.* + evaluation.metrics
dabba.models.optimizer      ← Optuna tuning; used by model_selection
dabba.evaluation.metrics    ← depends on models.* (used by model_selection)
dabba.evaluation.business_cost ← used by pipeline / reports
dabba.llm.food_concierge    ← depends on llm.recommendation_narrator, llm.rag_similar_restaurants
dabba.llm.rag_similar_restaurants ← depends on models.recommender (FAISS index)
dabba.nlp.sentiment / hinglish_sentiment ← depend on config only (leaf)
dabba.monitoring.drift / retrain ← depend on models, features, database
dabba.observability         ← leaf (logging/tracing hooks)
dabba.cache.redis_client    ← leaf adapter (Redis)
dabba.pipeline              ← **orchestrator**: depends on data, features, models, evaluation
```

## Interface layer → core

```
api/main.py          → dabba.config, dabba.database.session/repositories,
                       dabba.llm.food_concierge, dabba.models.*, dabba.cache
api/routers/*        → api.main (DI via Depends), api.schemas, dabba.* facades
api/auth.py / limiter.py → dabba.config (leaf middleware)
app/streamlit_app.py → dabba.llm.food_concierge, dabba.database.repositories,
                       dabba.monitoring, dabba.models (metadata)
app/pages/*          → dabba.* facades via streamlit_app
app/utils/sanitize.py → leaf
```

## Dependency rules (why)

- **`dabba.pipeline` is the only sanctioned training orchestrator** — the API,
  UI, and notebooks never re-implement the pipeline.
- **`data.*` and `features.*` never import models or the LLM layer** — feature
  engineering stays upstream of modeling.
- **`llm.rag_similar_restaurants` is the single bridge to
  `models.recommender`** — the LLM layer reads the FAISS index but never
  trains models.
- **No circular imports** — domain modules depend only downward or on shared
  infra (`config`, `database.session`); interface modules (api/app) depend on
  core, never the reverse.
- **Alembic env.py** imports `dabba.database.models` to autogenerate — kept in
  `alembic/env.py` per the v5.0 fix.

## External dependencies

FastAPI + uvicorn · Streamlit · SQLite/Postgres (SQLAlchemy + Alembic) ·
Redis (cache) · PyTorch (collaborative recommender, `models/best_collaborative_model.pt`)
· scikit-learn + joblib (rating/ETA models) · FAISS (restaurant index) ·
Optuna (hyperparameter tuning) · MLflow (experiment tracking) · LLM provider
(concierge narration)
