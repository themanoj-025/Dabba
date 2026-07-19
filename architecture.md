# 🏗️ Dabba — System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DABBA SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────┐
 │                        DATA PIPELINE                                 │
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
 │  │   • Dedup    │  • Parse ratings/costs                             │
 │  │   • Validate │  • Fill missing                                   │
 │  └──────┬───────┘                                                    │
 │         ▼                                                            │
 │  ┌──────────────────────┐                                            │
 │  │  Feature Engineering │  restaurant_features.py, delivery_features │
 │  │  • Cuisine encoding  │  .py, geo.py, sentiment.py                │
 │  │  • Distance (haversine)│  • Traffic ordinal                     │
 │  │  • Time features     │  • VADER sentiment                        │
 │  └──────┬───────┬───────┘                                            │
 │         │       │                                                    │
 │         ▼       ▼                                                    │
 │  ┌────────────┐ ┌────────────┐                                       │
 │  │ Rating Model│ │ ETA Model  │  rating_model.py, eta_model.py       │
 │  │ Comparison  │ │ Comparison │  • 5-fold CV on identical features   │
 │  │ 8 models    │ │ 9 models   │  • Lowest MAE wins                  │
 │  └──────┬─────┘ └──────┬─────┘                                       │
 │         │              │                                              │
 │         ▼              ▼                                              │
 │  ┌────────────┐ ┌────────────┐                                       │
 │  │ Best Model  │ │ Best Model │  model_selection.py, joblib          │
 │  │ (full data) │ │ (full data)│  → Saved to models/                 │
 │  └────────────┘ └────────────┘                                       │
 │         │              │                                              │
 │         └──────┬───────┘                                              │
 │                ▼                                                     │
 │  ┌──────────────────────┐                                            │
 │  │   Reliability Score  │  business_cost.py                          │
 │  │   0.4×rating +       │                                            │
 │  │   0.3×sentiment -    │                                            │
 │  │   0.3×delay_risk     │                                            │
 │  └──────────────────────┘                                            │
 └─────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────┐
 │                       APPLICATION LAYER                              │
 │                                                                      │
 │  ┌─────────────────┐    ┌──────────────────┐                         │
 │  │   Streamlit App  │    │    FastAPI        │                       │
 │  │   (Port 8501)    │    │   (Port 8000)     │                       │
 │  │                  │    │                   │                       │
 │  │  ├─ Customer View│    │  GET  /health     │                       │
 │  │  ├─ Ops View     │    │  GET  /model-info │                       │
 │  │  └─ Model Info   │    │  POST /recommend  │                       │
 │  │                  │    │  POST /predict-eta│                       │
 │  └────────┬─────────┘    └────────┬──────────┘                       │
 │           │                      │                                   │
 │           └──────────────────────┘                                   │
 │                    │   (CORS-enabled)                                │
 │                    ▼                                                 │
 │           ┌──────────────┐                                           │
 │           │  Saved Models │  models/best_rating_model.pkl           │
 │           │  + CSVs      │  models/best_eta_model.pkl               │
 │           │              │  reports/model_comparison_*.csv          │
 │           └──────────────┘                                           │
 └─────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────┐
 │                       INFRASTRUCTURE                                 │
 │                                                                      │
 │  ┌──────────────────────────────────────────────────────────────┐    │
 │  │                    Docker Compose                             │    │
 │  │  ┌──────────────┐     ┌──────────────────┐                   │    │
 │  │  │ api service   │     │ streamlit service│                  │    │
 │  │  │ uvicorn:8000  │     │ streamlit:8501   │                  │    │
 │  │  │ Volumes:      │     │ Volumes:          │                  │    │
 │  │  │  ./models     │     │  ./models         │                  │    │
 │  │  │  ./data       │     │  ./data           │                  │    │
 │  │  └──────────────┘     └──────────────────┘                   │    │
 │  └──────────────────────────────────────────────────────────────┘    │
 │                                                                      │
 │  ┌──────────────────────────────────────────────────────────────┐    │
 │  │                    CI/CD (GitHub Actions)                     │    │
 │  │  push/PR → Python 3.11 → pip install → pre-commit → pytest  │    │
 │  │  → coverage report → Codecov                                 │    │
 │  └──────────────────────────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────────────────────────┘
```

## Component Interaction Diagram

```
┌─────────┐     HTTP/JSON      ┌──────────┐     joblib.load()    ┌──────────┐
│ Browser │ ◄────────────────► │ FastAPI  │ ◄──────────────────► │ models/  │
│         │                    │ (api/)   │                      │ *.pkl    │
└─────────┘                    └────┬─────┘                      └──────────┘
                                    │
                                    │ pandas.read_csv()
                                    ▼
                            ┌───────────────┐
                            │ data/processed/│
                            │ *.csv         │
                            └───────────────┘

┌─────────┐                   ┌──────────────┐    pandas.read_csv() ┌──────────┐
│ Browser │ ◄───────────────► │ Streamlit    │ ◄──────────────────► │ data/    │
│         │                   │ (app/)       │                      │ *.csv    │
└─────────┘                   └──────────────┘                      └──────────┘

┌─────────┐                   ┌──────────────┐    joblib.load()    ┌──────────┐
│ Browser │ ◄───────────────► │ Streamlit    │ ◄──────────────────► │ models/  │
│         │                   │ Model Info   │                     │ *.pkl    │
└─────────┘                   └──────────────┘                     └──────────┘
```

## Module Dependencies

```
                    ┌─────────────────────────┐
                    │      dabba.config        │ (Read by ALL modules)
                    └─────────────────────────┘
                              ▲
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────────┐
   │   data/  │        │ features/│        │   models/    │
   │ loading +│◄───────│ + nlp/   │◄───────│ training +   │
   │ cleaning │        │          │        │ selection    │
   └──────────┘        └──────────┘        └──────┬───────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ evaluation/      │
                    │ metrics + SLA    │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   pipeline.py    │ (Orchestrator)
                    └──────────────────┘
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Host                                 │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │   API Container       │    │   Dashboard Container        │   │
│  │   uvicorn api.main    │    │   streamlit run app/...      │   │
│  │   Port 8000           │    │   Port 8501                  │   │
│  │                       │    │                              │   │
│  │   Volumes:            │    │   Volumes:                   │   │
│  │   - ./models:/app/models│   │   - ./models:/app/models    │   │
│  │   - ./data:/app/data  │    │   - ./data:/app/data         │   │
│  └──────┬───────────────┘    └──────────┬───────────────────┘   │
│         │                               │                       │
│         └───────────────────────────────┘                       │
│                         │                                       │
│                         ▼                                       │
│              ┌──────────────────────┐                           │
│              │   Docker Network      │                           │
│              └──────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Data Volume & Storage

| Artifact | Format | Location | Size (Est.) | Persistence |
|----------|--------|----------|-------------|-------------|
| Raw Zomato data | CSV | `data/raw/zomato.csv` | ~5-10 MB | Gitignored |
| Raw delivery data | CSV | `data/raw/deliverytime.csv` | ~5-10 MB | Gitignored |
| Processed restaurants | CSV | `data/processed/restaurants_processed.csv` | ~5 MB | Gitignored |
| Best rating model | Pickle | `models/best_rating_model.pkl` | ~1-100 MB | Gitignored |
| Best ETA model | Pickle | `models/best_eta_model.pkl` | ~1-100 MB | Gitignored |
| Comparison CSVs | CSV | `reports/` | < 1 MB | Gitignored |
| Charts/images | PNG | `reports/figures/` | < 10 MB | Gitignored |
