# 🏗️ Dabba v3 — System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DABBA v3 SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────┐
 │                        DATA PIPELINE (src/dabba/)                    │
 │                                                                      │
 │  Kaggle (Zomato + Delivery CSVs)                                     │
 │         │                                                            │
 │         ▼                                                            │
 │  ┌──────────────┐                                                    │
 │  │ Data Loading  │  loaders.py (pandas)                              │
 │  └──────┬───────┘                                                    │
 │         ▼                                                            │
 │  ┌──────────────┐                                                    │
 │  │ Data Cleaning │  cleaning.py (pandas, numpy)                      │
 │  └──────┬───────┘                                                    │
 │         ▼                                                            │
 │  ┌──────────────────────┐                                            │
 │  │  Feature Engineering │  restaurant_features.py, delivery_features  │
 │  │  • Cuisine encoding  │  .py, geo.py, sentiment.py                │
 │  │  • Distance (haversine)│                                         │
 │  │  • Time features     │                                           │
 │  └──────┬───────┬───────┘                                           │
 │         │       │                                                   │
 │         ▼       ▼                                                   │
 │  ┌──────────┐ ┌──────────┐  ┌──────────────┐  ┌──────────────┐     │
 │  │ Rating    │ │ ETA      │  │ Collaborative │  │ Geographic   │     │
 │  │ Models    │ │ Models   │  │ Filtering     │  │ Clustering   │     │
 │  │ (9 algos) │ │ (10 algos)│  │ (PyTorch MF)  │  │ (KMeans etc)│     │
 │  └────┬─────┘ └────┬─────┘  └──────┬───────┘  └──────────────┘     │
 │       │            │               │                                │
 │       └────────────┼───────────────┘                                │
 │                    │                                                │
 │                    ▼                                                │
 │         ┌──────────────────────┐                                    │
 │         │  Hybrid Recommender  │  Content + CF + Reliability Score   │
 │         │  + LLM Narrator     │  + A/B weight scenarios             │
 │         └──────────┬───────────┘                                    │
 │                    │                                                │
 │                    ▼                                                │
 │         ┌──────────────────────┐                                    │
 │         │  Reliability Score   │  w1*rating + w2*sentiment          │
 │         │  + A/B Scenarios    │  - w3*delay_risk                   │
 │         └──────────┬───────────┘                                    │
 │                    │                                                │
 │         ┌──────────▼──────────┐                                     │
 │         │  Drift Detection    │  KS-test, wired into Ops UI         │
 │         └─────────────────────┘                                     │
 └─────────────────────────────────────────────────────────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
     ┌──────────┐         ┌──────────┐          ┌──────────┐
     │ Streamlit│         │  FastAPI  │          │  MLflow  │
     │ Dashboard│         │  REST API │          │ Tracking │
     │ (4 pages)│         │ (5 routes)│          │ (Docker) │
     │ Custom   │         │ Routers   │          │ Port 5000│
     │ Radio Nav│         │ Pattern   │          │          │
     └──────────┘         └──────────┘          └──────────┘
            │                     │
            ▼                     ▼
     ┌──────────────────────────────────────┐
     │        LLM Layer (Anthropic)          │
     │  ┌────────────┬──────────┬─────────┐  │
     │  │ Narrator   │ RAG      │ Chat    │  │
     │  │ Explanations│ Retrieval│ Copilot │  │
     │  └────────────┴──────────┴─────────┘  │
     │  Rules-based fallback (no API key)     │
     └──────────────────────────────────────┘
```

## Module Dependencies

```
pipeline.py (orchestrator)
 ├── config.py
 ├── data/loaders.py → config.py
 ├── data/cleaning.py → config.py
 ├── features/restaurant_features.py → config.py, geo.py
 ├── features/delivery_features.py → config.py, geo.py
 ├── features/geo.py → scikit-learn
 ├── nlp/sentiment.py → config.py, nltk
 ├── models/rating_model.py → config.py, sklearn, xgboost, lightgbm, catboost
 ├── models/eta_model.py → config.py, sklearn, xgboost, lightgbm, catboost
 ├── models/model_selection.py → config.py
 ├── models/collaborative_recommender.py → config.py, torch
 ├── models/hybrid_recommender.py → config.py, recommender.py
 ├── evaluation/metrics.py → sklearn
 └── evaluation/business_cost.py → config.py

api/main.py (FastAPI)
 ├── config.py
 ├── api/schemas.py → pydantic
 ├── routers/recommend.py → schemas, hybrid_recommender, llm
 ├── routers/eta.py → schemas, config
 ├── routers/chat.py → schemas, llm/food_concierge
 └── routers/model_info.py → schemas, config

app/streamlit_app.py (Dashboard)
 ├── pages/page_discover.py → dabba.models, dabba.llm
 ├── pages/page_ops.py → dabba.monitoring, app.components
 ├── pages/page_model_performance.py → pandas, plotly
 └── pages/page_concierge.py → dabba.llm
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                  docker-compose.yml                   │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ Streamlit    │  │ FastAPI     │  │ MLflow       │  │
│  │ :8501        │  │ :8000       │  │ :5000        │  │
│  │ app/         │  │ api/        │  │ reports/     │  │
│  │              │  │             │  │ mlruns/      │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Authentication

No authentication — development/demo tool. Security headers (CSP, X-Frame-Options, XSS) only.
