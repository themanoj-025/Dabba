# Dabba — Architecture

> Textual architecture of the Dabba restaurant recommendation platform (as-is; no behavior changes).

## System Overview

Dabba is a layered ML platform with three primary surfaces:

1. **CLI / Pipeline** — `src/dabba/pipeline.py` runs the full training flow (data → features → models → evaluate → register to MLflow).
2. **API (FastAPI)** — REST endpoints for recommendations, ratings, ETA prediction, explanation, restaurant search, and the food concierge chat. Protected by JWT auth and rate limiting.
3. **Dashboard (Streamlit)** — concierge, discover, model-performance, and ops pages.

```mermaid
graph TD
    subgraph CLI
        PIP[pipeline.py]
    end

    subgraph API[FastAPI api/]
        API_MAIN[main.py]
        ROUTERS[routers: recommend, rating/eta, explain,
                 chat, restaurants, model_info]
        AUTH[auth.py]
        LIM[limiter.py]
    end

    subgraph DASH[Streamlit app/]
        SA[streamlit_app.py]
        PAGES[pages: concierge, discover,
               model_performance, ops]
        COMP[components: restaurant_card]
    end

    subgraph CORE[src/dabba]
        DATA[data: cleaning, loaders]
        FEAT[features: restaurant, delivery, geo, traffic]
        MODELS[models: recommender, hybrid, rating, eta,
               collaborative, optimizer, model_selection]
        EVAL[evaluation: metrics, business_cost]
        LLM[llm: food_concierge, rag_similar_restaurants,
             recommendation_narrator]
        NLP[nlp: sentiment, hinglish_sentiment]
        MON[monitoring: drift, retrain]
        DB_LAYER[database: models, repositories, session, seed]
        CACHE[cache: redis_client]
    end

    subgraph INFRA
        DB[(SQLite/Postgres via Alembic)]
        REDIS[(Redis)]
        MLF[(MLflow)]
    end

    PIP --> CORE
    API_MAIN --> ROUTERS
    ROUTERS --> AUTH
    ROUTERS --> LIM
    ROUTERS --> CORE
    SA --> PAGES
    PAGES --> COMP
    PAGES --> CORE
    CORE --> DB_LAYER --> DB
    CORE --> CACHE --> REDIS
    MODELS --> MLF
    MON --> PIP
    LLM --> MODELS
    NLP --> LLM
```

## Layering Rules (as observed)

- **API layer** (`api/`) depends on `src/dabba/`; never the reverse.
- **Database** access is isolated in `database/repositories.py` (function-based repository layer).
- **Streamlit pages** call the core package directly (no separate API client — unlike FraudLens).
- **LLM features** (concierge, narration, RAG) are wrapped modules with graceful fallbacks.
- **Config** centralized in `src/dabba/config.py` (Pydantic settings).

## Data Flow (recommendation path)

```
GET /v1/recommend ──► auth ──► limiter ──► router ──► repositories ──► features.engineering
                                                              │
                                                              ▼
                                             hybrid recommender (collaborative + content)
                                                              │
                                                              ▼
                                     LLM recommendation narrator (optional)
                                                              │
                                                              ▼
                                                  JSON response
```

## Deployment

- Multi-service containers: `docker/api.Dockerfile`, `docker/streamlit.Dockerfile`, `docker/mlflow.Dockerfile` + `docker-compose.yml`.
- Alembic manages schema; Redis for caching; MLflow for experiment tracking.
- Drift detection can trigger automatic retraining via `monitoring/retrain.py` (subprocess, rate-limited).
