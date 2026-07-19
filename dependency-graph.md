# 🔗 Dabba — Dependency Graph

## Overview

This document maps all module dependencies in the Dabba project. Understanding these dependencies is critical before making any modifications.

---

## Import Dependency Map

### Legend
- `A → B` means module A imports from module B
- **Bold** = Core system file (high impact)
- *Italic* = External library

---

### Source Package: `src/dabba/`

```
dabba/
├── __init__.py
│   └── No imports (package definition)
│
├── config.py
│   ├── pydantic (BaseSettings)
│   ├── pydantic_settings (BaseSettings)
│   ├── os (environment variables)
│   └── pathlib.Path
│
├── pipeline.py
│   ├── **dabba.config**
│   ├── **dabba.data.cleaning**
│   ├── **dabba.data.loaders**
│   ├── **dabba.evaluation.business_cost**
│   ├── **dabba.features.delivery_features**
│   ├── **dabba.features.geo**
│   ├── **dabba.features.restaurant_features**
│   ├── **dabba.models.eta_model**
│   ├── **dabba.models.model_selection**
│   ├── **dabba.models.rating_model**
│   ├── **dabba.nlp.sentiment**
│   ├── matplotlib (Agg backend + pyplot)
│   ├── numpy
│   ├── pandas
│   └── sklearn.preprocessing.StandardScaler
│
├── data/
│   ├── __init__.py (no imports)
│   │
│   ├── loaders.py
│   │   ├── **dabba.config**
│   │   └── pandas
│   │
│   └── cleaning.py
│       ├── **dabba.config**
│       ├── numpy
│       ├── pandas
│       └── re (regular expressions)
│
├── features/
│   ├── __init__.py (no imports)
│   │
│   ├── geo.py
│   │   ├── numpy
│   │   ├── pandas
│   │   ├── sklearn.cluster (KMeans, DBSCAN, Agglomerative)
│   │   └── sklearn.metrics (silhouette_score)
│   │
│   ├── restaurant_features.py
│   │   ├── **dabba.config**
│   │   ├── numpy
│   │   └── pandas
│   │
│   └── delivery_features.py
│       ├── **dabba.config**
│       ├── **dabba.features.geo** (haversine_distance)
│       ├── numpy
│       └── pandas
│
├── models/
│   ├── __init__.py (no imports)
│   │
│   ├── rating_model.py
│   │   ├── **dabba.config**
│   │   ├── joblib
│   │   ├── numpy
│   │   ├── pandas
│   │   ├── sklearn.* (pipeline, preprocessing, model_selection,
│   │   │          ensemble, linear_model, tree, metrics, neighbors)
│   │   ├── xgboost (optional import)
│   │   └── lightgbm (optional import)
│   │
│   ├── eta_model.py
│   │   ├── **dabba.config**
│   │   ├── joblib
│   │   ├── numpy
│   │   ├── pandas
│   │   ├── sklearn.* (same as rating_model + KNeighborsRegressor)
│   │   ├── xgboost (optional import)
│   │   └── lightgbm (optional import)
│   │
│   ├── model_selection.py
│   │   ├── **dabba.config**
│   │   ├── joblib
│   │   └── pandas
│   │
│   ├── optimizer.py
│   │   ├── **dabba.config**
│   │   ├── numpy
│   │   ├── pandas
│   │   └── scipy.optimize (linear_sum_assignment)
│   │
│   └── recommender.py
│       ├── **dabba.config**
│       ├── joblib
│       ├── numpy
│       ├── pandas
│       └── sklearn.metrics.pairwise (cosine_similarity)
│
├── nlp/
│   ├── __init__.py (no imports)
│   │
│   └── sentiment.py
│       ├── **dabba.config**
│       ├── nltk.sentiment.vader (VADER SentimentIntensityAnalyzer)
│       ├── numpy
│       └── pandas
│
└── evaluation/
    ├── __init__.py (no imports)
    │
    ├── metrics.py
    │   ├── numpy
    │   └── sklearn.metrics (mean_absolute_error, mean_squared_error, r2_score)
    │
    └── business_cost.py
        ├── **dabba.config**
        ├── numpy
        └── pandas
```

---

### Application Layer

```
api/
├── main.py
│   ├── **dabba.config**
│   ├── api.schemas
│   ├── fastapi (FastAPI, HTTPException, CORSMiddleware)
│   ├── joblib
│   ├── pandas
│   └── pathlib.Path
│
└── schemas.py
    ├── pydantic (BaseModel, Field)
    └── typing (Optional, List, Dict, Any)
```

```
app/
├── streamlit_app.py
│   └── streamlit
│
└── pages/
    ├── 1_customer_view.py
    │   ├── streamlit
    │   ├── pandas
    │   └── pathlib.Path
    │
    ├── 2_ops_view.py
    │   ├── streamlit
    │   ├── pandas
    │   ├── numpy
    │   ├── time
    │   └── random
    │
    └── 3_model_info.py
        ├── streamlit
        ├── pandas
        └── pathlib.Path
```

---

### Test Package

```
tests/
├── __init__.py
│
├── test_api.py
│   ├── pytest
│   ├── fastapi.testclient (TestClient)
│   └── api.main
│
├── test_cleaning.py
│   ├── numpy
│   ├── pandas
│   ├── pytest
│   └── dabba.data.cleaning
│
├── test_eta_model.py
│   ├── numpy
│   ├── pandas
│   ├── pytest
│   └── dabba.models.eta_model
│
├── test_features.py
│   ├── numpy
│   ├── pandas
│   ├── pytest
│   ├── dabba.features.geo
│   ├── dabba.features.restaurant_features
│   └── dabba.features.delivery_features
│
├── test_model_selection.py
│   ├── numpy
│   ├── pandas
│   ├── pytest
│   └── dabba.models.model_selection
│
├── test_rating_model.py
│   ├── numpy
│   ├── pandas
│   ├── pytest
│   ├── dabba.models.rating_model
│   └── joblib
│
└── test_recommender.py
    ├── numpy
    ├── pandas
    ├── pytest
    └── dabba.models.recommender
```

---

## Critical Dependency Chain

The following chain represents the **highest-impact path** in the codebase. Changes to any node in this chain affect everything downstream.

```
config.py
    │ (ALL modules depend on this)
    ▼
cleaning.py → loaders.py
    │              │
    ▼              ▼
restaurant_features.py ← geo.py ← delivery_features.py
    │                                │
    ▼                                ▼
sentiment.py (NLP)          eta_model.py
    │                                │
    ▼                                ▼
rating_model.py             eta_model.py (training)
    │                                │
    ▼                                ▼
model_selection.py ← ← ← ← ← ← ← ← ┘
    │
    ▼
pipeline.py (orchestrates all)
    │
    ├──► evaluation/business_cost.py (Reliability Score)
    ├──► models/best_rating_model.pkl → api/main.py, app/pages/
    └──► models/best_eta_model.pkl → api/main.py, app/pages/
```

---

## External Dependency Graph

```
PYTHON PACKAGE DEPENDENCIES
├── pandas ≥2.0
├── numpy ≥1.24
├── scipy ≥1.10
├── scikit-learn ≥1.3
├── joblib ≥1.3
├── xgboost ≥2.0 (optional)
├── lightgbm ≥4.0 (optional)
├── nltk ≥3.8 (VADER lexicon)
├── fastapi ≥0.104
├── uvicorn ≥0.24
├── pydantic ≥2.5
├── pydantic-settings ≥2.1
├── streamlit ≥1.28
├── plotly ≥5.18
├── matplotlib ≥3.7
├── folium ≥0.15
├── seaborn ≥0.13
├── shap ≥0.43
├── pytest ≥7.4
├── pytest-cov ≥4.1
├── ruff ≥0.1
├── black ≥23.11
├── isort ≥5.12
├── kaggle ≥1.5
└── python-dotenv ≥1.0
```

---

## File Impact Analysis

### Tier 1 — Core System Files (Highest Impact)
*Changes here affect ALL downstream modules*

| File | Impact Radius | Reason |
|------|--------------|--------|
| `src/dabba/config.py` | **Entire project** | All modules import configuration |
| `src/dabba/data/cleaning.py` | All features, models, pipeline | Raw data quality affects everything |
| `src/dabba/models/rating_model.py` | Model selection, pipeline, API, dashboard | Rating model training & comparison |
| `src/dabba/models/eta_model.py` | Model selection, pipeline, API, dashboard | ETA model training & comparison |
| `src/dabba/models/model_selection.py` | Pipeline, reports, charts | Best-model selection logic |

### Tier 2 — High Impact Files
*Changes affect multiple modules*

| File | Impact Radius | Reason |
|------|--------------|--------|
| `src/dabba/pipeline.py` | All generated artifacts | Orchestrates entire pipeline |
| `src/dabba/features/restaurant_features.py` | Rating model, recommender | Feature set for rating ML |
| `src/dabba/features/delivery_features.py` | ETA model, optimizer | Feature set for ETA ML |
| `src/dabba/features/geo.py` | Delivery features, clustering | Haversine distance + clustering |
| `src/dabba/nlp/sentiment.py` | Restaurant features, reliability score | Sentiment scores |
| `src/dabba/evaluation/business_cost.py` | Reliability score, SLA analysis | Business metrics |
| `api/main.py` | API consumers, dashboard | All API endpoints |

### Tier 3 — Medium Impact Files
*Changes affect specific features*

| File | Impact Radius | Reason |
|------|--------------|--------|
| `src/dabba/models/recommender.py` | Customer View (future) | Recommender logic |
| `src/dabba/models/optimizer.py` | Operations View (potential) | Partner assignment |
| `src/dabba/data/loaders.py` | Pipeline, cleaning | CSV loading |
| `src/dabba/evaluation/metrics.py` | Model evaluation | Regression metrics |

### Tier 4 — Application/Surface Files
*Changes affect user-facing functionality*

| File | Impact Radius | Reason |
|------|--------------|--------|
| `app/streamlit_app.py` | All dashboard pages | Entry point + sidebar |
| `app/pages/1_customer_view.py` | Customer-facing UI | Restaurant recommendations |
| `app/pages/2_ops_view.py` | Operations UI | Delivery simulation |
| `app/pages/3_model_info.py` | Model documentation UI | Model comparison display |
| `api/schemas.py` | All API endpoints | Request/response models |

### Tier 5 — Configuration & Infrastructure
*Changes affect project setup*

| File | Impact Radius | Reason |
|------|--------------|--------|
| `requirements.txt` | All developers | Python dependencies |
| `Makefile` | All developers | Build commands |
| `Dockerfile` | Deployment | Container image |
| `docker-compose.yml` | Deployment | Service orchestration |
| `.github/workflows/ci.yml` | CI/CD | Continuous integration |

---

## Dead/Dormant Code

| File/Function | Status | Notes |
|---------------|--------|-------|
| `src/dabba/models/optimizer.py` | **Not used anywhere** | `compare_assignment_strategies()` and related functions are defined but never imported or called |
| `api/main.py:recommend()` | **Stub** | Returns empty recommendations; doesn't use `RestaurantRecommender` |
| `api/main.py:add_distance_columns()` in `geo.py` | **Defined but not imported** | No module imports this function |
| `api/main.py:describe_dataset()` in `loaders.py` | Only used in `pipeline.py` | Not used in API or dashboard |
| `src/dabba/models/recommender.py:load_rating_model()` | **Not called** | Method exists but never invoked |

---

## Test Coverage Map

| Test File | Tests What | Coverage Focus |
|-----------|-----------|----------------|
| `test_cleaning.py` | `cleaning.py` | Rating parsing, cost parsing, duplicate removal, lat/long validation |
| `test_features.py` | `features/*.py` | Haversine distance, geocoding, clustering, cuisine encoding, delivery features |
| `test_rating_model.py` | `models/rating_model.py` | ModelResult dataclass, model dictionary, training pipeline, fitting/saving |
| `test_eta_model.py` | `models/eta_model.py` | ETAModelResult dataclass, training pipeline, prediction shapes |
| `test_model_selection.py` | `models/model_selection.py` | DataFrame conversion, sorting, best-model selection by all metrics |
| `test_recommender.py` | `models/recommender.py` | Bayesian average, recommender filtering, empty results |
| `test_api.py` | `api/main.py` | Health endpoint, model-info, ETA endpoint (smoke tests) |

### Untested Modules
- `evaluation/business_cost.py` — No direct tests (SLA analysis, reliability score)
- `evaluation/metrics.py` — No direct tests (tested indirectly via model tests)
- `nlp/sentiment.py` — No direct tests
- `pipeline.py` — No direct tests (integration-level)
- `optimizer.py` — No direct tests (dead code)
- Streamlit pages — No tests (UI-level)
