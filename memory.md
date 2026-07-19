# 🍛 Dabba — Restaurant Intelligence Platform

## Project Overview

**Dabba** is an India-focused restaurant ranking, recommendation, and delivery-reliability platform built to senior data-science code-review standard. It combines restaurant quality data (from Zomato), customer sentiment (from reviews), and delivery ETA predictions (from delivery time data) into a unified, actionable **Reliability Score**.

The project serves as a comprehensive ML portfolio project demonstrating:
- End-to-end ML pipeline development
- Rigorous model comparison with k-fold cross-validation
- Automatic best-model selection
- SHAP explainability
- Production-ready deployment (FastAPI + Streamlit + Docker + CI/CD)
- Hybrid recommender systems

---

## Business Purpose

### Problem Statement
India's food-tech landscape generates massive amounts of restaurant and delivery data, yet consumers and operators lack a unified view that combines food quality, customer sentiment, and delivery reliability into a single actionable metric.

### Solution
Dabba solves this by:
1. Mining Zomato restaurant data for ratings, cuisine diversity, and cost signals
2. Analyzing customer sentiment from reviews using VADER NLP
3. Predicting delivery ETA with a rigorously selected ML model (8+ algorithms compared)
4. Synthesizing everything into a proprietary **Reliability Score**

### Target Users
- **Customers**: Finding reliable restaurants with good food quality
- **Operations Managers**: Monitoring delivery SLA compliance and optimizing partner assignment
- **Data Scientists**: Understanding model performance and methodology

---

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| **Language** | Python | 3.11 |
| **ML Framework** | scikit-learn, XGBoost, LightGBM | 1.3+, 2.0+, 4.0+ |
| **NLP** | NLTK (VADER) | 3.8+ |
| **Dashboard** | Streamlit | 1.28+ |
| **API Framework** | FastAPI | 0.104+ |
| **Data Validation** | Pydantic / Pydantic-Settings | 2.5+ |
| **Testing** | pytest, pytest-cov | 7.4+, 4.1+ |
| **Linting** | Ruff, Black, isort | 0.1+, 23.11+, 5.12+ |
| **CI/CD** | GitHub Actions | — |
| **Containerization** | Docker, docker-compose | — |
| **Explainability** | SHAP | 0.43+ |
| **Visualization** | plotly, matplotlib, folium, seaborn | Latest |
| **Data Sources** | Kaggle (Zomato Bangalore, Food Delivery Time) | — |
| **Version Control** | pre-commit hooks | 4.5+ |

---

## Repository Structure

```
dabba/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI pipeline
├── api/
│   ├── main.py                       # FastAPI application (4 endpoints)
│   └── schemas.py                    # Pydantic request/response models
├── app/
│   ├── streamlit_app.py              # Streamlit dashboard entry point
│   └── pages/
│       ├── 1_customer_view.py        # Restaurant recommendations UI
│       ├── 2_ops_view.py             # Delivery SLA monitoring & simulation
│       └── 3_model_info.py           # Model comparison results & charts
├── data/
│   ├── raw/                          # Raw Kaggle datasets (gitignored)
│   └── processed/                    # Cleaned/processed data (gitignored)
├── models/                           # Saved model artifacts (.pkl) (gitignored)
├── notebooks/
│   ├── 01_eda_restaurants.ipynb      # EDA on Zomato data
│   ├── 02_eda_delivery.ipynb         # EDA on delivery data
│   ├── 03_sentiment_analysis.ipynb   # Sentiment analysis exploration
│   ├── 04_recommender_prototyping.ipynb  # Recommender prototyping
│   ├── 05_eta_modeling.ipynb         # ETA modeling exploration
│   └── 06_explainability.ipynb       # SHAP explainability analysis
├── reports/
│   ├── figures/                      # Generated charts & images (gitignored)
│   └── model_comparison_*.csv        # Model comparison CSVs (gitignored)
├── src/
│   └── dabba/
│       ├── __init__.py               # Package init, version 0.1.0
│       ├── config.py                 # Centralized DabbaConfig (Pydantic BaseSettings)
│       ├── pipeline.py               # Full training pipeline orchestrator
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py            # CSV loading with schema verification
│       │   └── cleaning.py           # Zomato & delivery data cleaning
│       ├── features/
│       │   ├── __init__.py
│       │   ├── geo.py                # Haversine distance, geocoding, clustering
│       │   ├── restaurant_features.py # Cuisine encoding, cost buckets, flags
│       │   └── delivery_features.py  # Distance, time features, traffic encoding
│       ├── models/
│       │   ├── __init__.py
│       │   ├── rating_model.py       # Rating prediction (8 models compared)
│       │   ├── eta_model.py          # ETA prediction (9 models compared)
│       │   ├── model_selection.py    # Best model selection by metric
│       │   ├── optimizer.py          # Hungarian algorithm for partner assignment
│       │   └── recommender.py        # Hybrid recommender (content + popularity)
│       ├── nlp/
│       │   ├── __init__.py
│       │   └── sentiment.py          # VADER sentiment analysis
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py            # Regression metrics (MAE, RMSE, R²)
│           └── business_cost.py      # SLA analysis & Reliability Score
├── tests/
│   ├── __init__.py
│   ├── test_api.py                   # FastAPI endpoint smoke tests
│   ├── test_cleaning.py              # Data cleaning tests
│   ├── test_eta_model.py             # ETA model training tests
│   ├── test_features.py              # Feature engineering tests
│   ├── test_model_selection.py       # Model selection logic tests
│   ├── test_rating_model.py          # Rating model training tests
│   └── test_recommender.py           # Recommender tests
├── .pre-commit-config.yaml           # Pre-commit hooks (black, isort, ruff)
├── .gitignore
├── CONTRIBUTING.md                   # Contribution guidelines
├── docker-compose.yml                # Docker services (api + streamlit)
├── Dockerfile                        # Single Docker image for both services
├── LICENSE                           # MIT License
├── Makefile                          # Build commands (setup, test, train, etc.)
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
└── setup_kaggle.py                   # Kaggle setup & download script
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES (Kaggle)                        │
│  ┌─────────────────────┐          ┌──────────────────────────┐     │
│  │ Zomato Bangalore     │          │ Food Delivery Time       │     │
│  │ (restaurants, ratings,│         │ (delivery logs, ETA)     │     │
│  │  cuisines, reviews)  │          │                          │     │
│  └─────────┬───────────┘          └──────────┬───────────────┘     │
│            ↓                                 ↓                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              DATA CLEANING (cleaning.py)                      │  │
│  │  • Remove duplicates  • Parse ratings/costs/time_taken       │  │
│  │  • Validate lat/long  • Remove outliers  • Fill missing       │  │
│  │  • Normalize column names to snake_case                       │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             ↓                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              FEATURE ENGINEERING                              │  │
│  │  • Restaurant: cuisine encoding, cost buckets, binary flags,  │  │
│  │    log-transformed votes, sentiment scores (VADER)            │  │
│  │  • Delivery: haversine distance, time features, traffic       │  │
│  │    ordinal encoding, festival flags, age buckets              │  │
│  │  • Geographic: neighborhood centroids, clustering comparison  │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             ↓                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              MODEL COMPARISON (k-fold CV)                     │  │
│  │  ┌─────────────────────┐    ┌──────────────────────────┐     │  │
│  │  │ Rating Prediction   │    │ ETA Prediction            │     │  │
│  │  │ 8 models compared   │    │ 9 models compared         │     │  │
│  │  │ Same features/CV    │    │ Same features/CV          │     │  │
│  │  └──────────┬──────────┘    └─────────────┬────────────┘     │  │
│  │             ↓                              ↓                  │  │
│  │  ┌──────────────────────────────────────────────────────┐    │  │
│  │  │        BEST MODEL SELECTION (lowest MAE)              │    │  │
│  │  │  • Comparison CSV saved to reports/                   │    │  │
│  │  │  • Charts + Residual plots generated                  │    │  │
│  │  │  • Winner retrained on full data & saved to models/   │    │  │
│  │  └─────────────────────┬────────────────────────────────┘    │  │
│  └────────────────────────┼────────────────────────────────────┘  │
│                           ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              RELIABILITY SCORE                               │  │
│  │  score = 0.4×norm(rating) + 0.3×norm(sentiment)              │  │
│  │          - 0.3×norm(delay_risk)                              │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             ↓                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         DEPLOYMENT                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │  Streamlit App  │  │  FastAPI       │  │  Container (Docker)  │  │
│  │  (app/)         │  │  (api/)        │  │  docker-compose      │  │
│  │  Port 8501      │  │  Port 8000     │  │  CI/CD: GH Actions   │  │
│  └────────────────┘  └────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Frontend Architecture (Streamlit)

### Entry Point
- **File**: `app/streamlit_app.py`
- **Configuration**: `st.set_page_config(layout="wide")`
- **Navigation**: Multi-page app via `app/pages/` directory (no explicit router needed)

### Pages

#### 1. Customer View (`1_customer_view.py`)
- **Purpose**: Restaurant discovery and recommendations
- **State**: Session state filters (cuisine multi-select, budget slider, area select)
- **Data Source**: `data/processed/restaurants_processed.csv` (loaded via `st.cache_data`)
- **Features**:
  - Sidebar filters: cuisine type, budget (₹100-5000), area, result count
  - Filtered restaurant listing with rating and cost metrics
  - Loading state when data is absent

#### 2. Operations View (`2_ops_view.py`)
- **Purpose**: Delivery SLA monitoring and simulation
- **State**: Streamlit session state for simulation
- **Features**:
  - SLA threshold configuration slider
  - Real-time delivery simulation with progress bar
  - Live metrics (total orders, on-time rate, at-risk count)
  - Results table with conditional formatting (red for late)
  - Confusion matrix display for prediction accuracy

#### 3. Model Info (`3_model_info.py`)
- **Purpose**: Transparent model comparison display
- **Data Source**: `reports/model_comparison_rating.csv` and `reports/model_comparison_eta.csv`
- **Features**:
  - Winner callout (success alert)
  - Full comparison table with highlighting (green for best)
  - Bar charts (MAE/RMSE, R²) loaded from PNG files
  - Residual plots for top 3 models
  - Methodology explanation

### Styling
- Uses Streamlit's built-in components and theming
- Emoji icons for visual hierarchy
- Conditional formatting via `st.dataframe.style`

---

## Backend Architecture (FastAPI)

### Application Setup (`api/main.py`)
- CORS middleware configured for localhost (ports 8000, 8501)
- Security headers middleware (CSP, X-Frame-Options, XSS, etc.)
- Lazy-loaded models on startup
- Graceful handling of missing models (returns 503)

### Endpoints

| Method | Route | Purpose | Auth |
|--------|-------|---------|------|
| GET | `/health` | Health check + model load status | No |
| GET | `/model-info` | Deployed model names & metrics | No |
| POST | `/recommend` | Restaurant recommendations (stub) | No |
| POST | `/predict-eta` | Delivery ETA prediction | No |

### Request/Response Schemas (`api/schemas.py`)
- `HealthResponse`: status, rating_model_loaded, eta_model_loaded
- `ModelInfoResponse`: rating_model dict, eta_model dict
- `RecommendRequest`: cuisine, budget, area, top_n
- `Recommendation`: name, rating, bayesian_rating, cost, location, etc.
- `RecommendResponse`: list of Recommendation, message
- `ETARequest`: distance_km, traffic_level, is_festival, delivery person fields
- `ETAResponse`: predicted_minutes, is_at_risk, sla_threshold

---

## ML Model Architecture

### Candidate Models

**Rating Prediction (8 models):**
- LinearRegression, Ridge, Lasso, DecisionTree
- RandomForest, GradientBoosting, XGBoost, LightGBM

**ETA Prediction (9 models):**
- LinearRegression, Ridge, Lasso, KNN, DecisionTree
- RandomForest, GradientBoosting, XGBoost, LightGBM

### Training Process
1. **Same features** — all models train on identical feature sets
2. **Same split** — 5-fold cross-validation with random seed 42
3. **Same preprocessing** — StandardScaler for numerics, OneHotEncoder for categoricals
4. **Selection rule** — Lowest MAE (configurable via config)
5. **Winner only** — wired into dashboard, API, and SLA logic

### Preprocessing
- `ColumnTransformer` with:
  - `StandardScaler` for numeric columns
  - `OneHotEncoder(handle_unknown="ignore")` for categorical columns

### Model Persistence
- Winning models saved via `joblib.dump()` to `models/`
- Comparison CSVs saved to `reports/`

---

## Database Architecture

**No database is used in this project.** All data is loaded from CSV files (Kaggle datasets) into pandas DataFrames. Data is processed in-memory and saved back to CSV/parquet files.

### Data Sources

#### Zomato Bangalore Restaurants Dataset
- **Source**: Kaggle (`himanshupoddar/zomato-bangalore-restaurants`)
- **File**: `data/raw/zomato.csv`
- **Key Columns**: name, rate, approx_cost(for two people), cuisines, location, votes, online_order, book_table, reviews_list, restaurant_lat, restaurant_lon
- **After Processing**: `data/processed/restaurants_processed.csv`

#### Food Delivery Time Dataset
- **Source**: Kaggle (`rajatkumar30/food-delivery-time`)
- **File**: `data/raw/deliverytime.csv`
- **Key Columns**: Time_taken(min), Restaurant_latitude/longitude, Delivery_location_latitude/longitude, Road_traffic_density, Festival, Delivery_person_Age, Delivery_person_Ratings, Vehicle_condition, Order_Date
- **After Processing**: Used in-memory for ETA model

---

## Authentication & Authorization

**No authentication is implemented.** The API and dashboard are designed for local/development use. Security headers are added to the FastAPI responses as a basic security measure.

---

## API Inventory

| # | Method | Route | Input Schema | Output Schema | Purpose | Status |
|---|--------|-------|-------------|---------------|---------|--------|
| 1 | GET | `/health` | — | `HealthResponse` | Health check + model load status | ✅ Implemented |
| 2 | GET | `/model-info` | — | `ModelInfoResponse` | Deployed model info & metrics | ✅ Implemented |
| 3 | POST | `/recommend` | `RecommendRequest` | `RecommendResponse` | Restaurant recommendations | ⚠️ Stub (needs processed data) |
| 4 | POST | `/predict-eta` | `ETARequest` | `ETAResponse` | Delivery ETA prediction | ✅ Implemented (requires model) |

---

## Data Flow Diagrams

### Restaurant Data Flow
```
Kaggle (Zomato CSV)
    ↓
load_zomato() → DataFrame
    ↓
clean_zomato():
    - Remove duplicates
    - Parse ratings (4.1/5 → 4.1)
    - Parse costs (₹1,200 → 1200.0)
    - Drop rows with missing rate
    - Fill missing values (mode/median)
    - Normalize column names (snake_case)
    ↓
add_restaurant_features():
    - Multi-hot encode cuisines
    - Cost buckets (budget→luxury)
    - Binary flags (online_order, book_table)
    - Log-transformed votes (log1p)
    ↓
add_sentiment_scores():
    - Parse reviews_list column (ast.literal_eval / regex / fallback)
    - Compute VADER compound sentiment per review
    - Average across all reviews per restaurant
    ↓
train_and_evaluate_rating_models():
    - 8 models with 5-fold CV
    - Select best by lowest MAE
    ↓
fit_best_rating_model() → models/best_rating_model.pkl
```

### Delivery Data Flow
```
Kaggle (Delivery CSV)
    ↓
load_delivery() → DataFrame
    ↓
clean_delivery():
    - Remove duplicates
    - Strip whitespace from strings
    - Parse Time_taken(min) → time_taken_min
    - Validate lat/long (India bounds)
    - Remove time outliers (>120 min)
    - Fill missing values
    - Normalize column names
    ↓
add_delivery_features():
    - Haversine distance (restaurant → delivery)
    - Time features (hour, day_of_week, weekend, hour buckets)
    - Festival flag (binary)
    - Traffic ordinal encoding (Low=0 → Jam=3)
    - Delivery person age bucket
    - Speed (km/h) for outlier detection
    ↓
train_and_evaluate_eta_models():
    - 9 models with 5-fold CV
    - Select best by lowest MAE
    ↓
fit_best_eta_model() → models/best_eta_model.pkl
```

### Recommendation Flow (API)
```
POST /recommend
    ↓
RecommendRequest { cuisine, budget, area, top_n }
    ↓
Recommender.apply_filters() → candidate restaurants
    ↓
cosine_similarity(query_features, restaurant_features)
    ↓
combined_score = 0.5 * similarity + 0.5 * bayesian_rating_norm
    ↓
sort descending, take top_n
    ↓
RecommendResponse { recommendations, message }
```

### ETA Prediction Flow (API)
```
POST /predict-eta
    ↓
ETARequest { distance, traffic, festival, delivery_person, vehicle }
    ↓
Build feature DataFrame
    ↓
_eta_model.predict(features) → predicted_minutes
    ↓
Compare with SLA threshold → is_at_risk
    ↓
ETAResponse { predicted_minutes, is_at_risk, sla_threshold }
```

---

## Environment Variables & Configuration

All configuration is centralized in `src/dabba/config.py` via Pydantic BaseSettings.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DABBA_LOG_LEVEL` | `INFO` | Logging level |
| — | `.env` file supported | Additional overrides |

**Environment variables from `docker-compose.yml`:**
- `DABBA_LOG_LEVEL=INFO`

### Internal Configuration Paths (config.py)
| Property | Path |
|----------|------|
| `project_root` | Auto-detected (3 levels up from config.py) |
| `data_raw_dir` | `<root>/data/raw` |
| `data_processed_dir` | `<root>/data/processed` |
| `models_dir` | `<root>/models` |
| `reports_dir` | `<root>/reports` |
| `best_rating_model_path` | `<root>/models/best_rating_model.pkl` |
| `best_eta_model_path` | `<root>/models/best_eta_model.pkl` |
| `rating_comparison_path` | `<root>/reports/model_comparison_rating.csv` |
| `eta_comparison_path` | `<root>/reports/model_comparison_eta.csv` |

### Key Configuration Constants
| Parameter | Default | Description |
|-----------|---------|-------------|
| `random_seed` | 42 | Global random seed |
| `test_size` | 0.2 | Hold-out fraction |
| `cv_folds` | 5 | Cross-validation folds |
| `rating_metric` | `mae` | Rating model selection metric |
| `eta_metric` | `mae` | ETA model selection metric |
| `sla_threshold_minutes` | 40 | Delivery SLA threshold |
| `reliability_w_rating` | 0.4 | Rating weight in reliability score |
| `reliability_w_sentiment` | 0.3 | Sentiment weight in reliability score |
| `reliability_w_delay` | 0.3 | Delay risk weight in reliability score |

---

## Feature Inventory

### 1. Data Loading & Cleaning
- **Files**: `data/loaders.py`, `data/cleaning.py`
- **Tests**: `test_cleaning.py`
- **Dependencies**: pandas, numpy
- **Description**: Loads CSV files with validation, cleans Zomato ratings/costs, cleans delivery data with lat/long validation

### 2. Feature Engineering
- **Files**: `features/restaurant_features.py`, `features/delivery_features.py`, `features/geo.py`
- **Tests**: `test_features.py`
- **Dependencies**: pandas, numpy, scikit-learn
- **Description**: Creates model-ready features from cleaned data

### 3. Rating Prediction Models
- **Files**: `models/rating_model.py`
- **Tests**: `test_rating_model.py`
- **Dependencies**: scikit-learn, xgboost, lightgbm, joblib
- **Description**: 8-model comparison with k-fold CV, best model selection and persistence

### 4. ETA Prediction Models
- **Files**: `models/eta_model.py`
- **Tests**: `test_eta_model.py`
- **Dependencies**: scikit-learn, xgboost, lightgbm, joblib
- **Description**: 9-model comparison with k-fold CV, best model selection and persistence

### 5. Model Selection
- **Files**: `models/model_selection.py`
- **Tests**: `test_model_selection.py`
- **Dependencies**: pandas, joblib
- **Description**: Comparison CSV generation and best-model selection by metric

### 6. Hybrid Recommender
- **Files**: `models/recommender.py`
- **Tests**: `test_recommender.py`
- **Dependencies**: scikit-learn (cosine_similarity), joblib, numpy, pandas
- **Description**: Content-based + popularity-based hybrid recommender with Bayesian-adjusted ratings

### 7. Delivery Partner Optimizer
- **Files**: `models/optimizer.py`
- **Tests**: None
- **Dependencies**: scipy (linear_sum_assignment)
- **Description**: Hungarian algorithm for optimal delivery partner assignment

### 8. NLP Sentiment Analysis
- **Files**: `nlp/sentiment.py`
- **Tests**: None
- **Dependencies**: NLTK (VADER)
- **Description**: Restaurant review sentiment scoring with robust parsing

### 9. Evaluation Metrics
- **Files**: `evaluation/metrics.py`, `evaluation/business_cost.py`
- **Tests**: None (metrics tested indirectly via model tests)
- **Dependencies**: scikit-learn, numpy
- **Description**: Regression metrics, SLA analysis, Reliability Score computation

### 10. Full Training Pipeline
- **Files**: `pipeline.py`
- **Tests**: None
- **Dependencies**: All modules above
- **Description**: Orchestrates full data-to-model pipeline with chart generation

### 11. Streamlit Dashboard
- **Files**: `app/streamlit_app.py`, `app/pages/*.py`
- **Dependencies**: streamlit, pandas
- **Description**: 3-page interactive dashboard

### 12. FastAPI Backend
- **Files**: `api/main.py`, `api/schemas.py`
- **Tests**: `test_api.py`
- **Dependencies**: FastAPI, Pydantic, joblib
- **Description**: REST API with 4 endpoints

### 13. Kaggle Setup
- **Files**: `setup_kaggle.py`
- **Description**: Helper script for Kaggle authentication and dataset download

---

## Dependency Graph

```
pipeline.py (main orchestrator)
├── config.py (all modules depend on this)
├── data/loaders.py
│   └── config.py
├── data/cleaning.py
│   └── config.py
├── features/restaurant_features.py
│   └── config.py
├── features/delivery_features.py
│   └── features/geo.py
├── features/geo.py
│   └── scikit-learn (KMeans, DBSCAN, Agglomerative, silhouette_score)
├── models/rating_model.py
│   ├── config.py
│   ├── scikit-learn (8+ model families)
│   ├── xgboost (optional)
│   └── lightgbm (optional)
├── models/eta_model.py
│   ├── config.py
│   ├── scikit-learn (9+ model families)
│   ├── xgboost (optional)
│   └── lightgbm (optional)
├── models/model_selection.py
│   └── config.py
├── nlp/sentiment.py
│   ├── config.py
│   └── nltk (VADER)
├── evaluation/metrics.py
│   └── scikit-learn (MAE, RMSE, R²)
└── evaluation/business_cost.py
    └── config.py

api/main.py (FastAPI server)
├── config.py
├── api/schemas.py (Pydantic)
├── joblib (model loading)
└── pandas

app/ (Streamlit dashboard)
├── pages/1_customer_view.py
│   └── pandas (data loading)
├── pages/2_ops_view.py
│   └── pandas, numpy, random
└── pages/3_model_info.py
    └── pandas (CSV loading)

tests/
├── test_api.py → api/main.py
├── test_cleaning.py → data/cleaning.py
├── test_eta_model.py → models/eta_model.py
├── test_features.py → features/*
├── test_model_selection.py → models/model_selection.py
├── test_rating_model.py → models/rating_model.py
└── test_recommender.py → models/recommender.py
```

---

## Important Files

### Core System Files (High Impact — Modify with Caution)
1. **`src/dabba/config.py`** — Central configuration; all modules depend on it
2. **`src/dabba/pipeline.py`** — Full pipeline orchestrator; coordinates all stages
3. **`src/dabba/models/rating_model.py`** — Rating model training & comparison
4. **`src/dabba/models/eta_model.py`** — ETA model training & comparison
5. **`src/dabba/models/model_selection.py`** — Best-model selection logic
6. **`src/dabba/data/cleaning.py`** — Data cleaning; affects all downstream results
7. **`api/main.py`** — API server with model loading

### High Impact — Frequently Modified
8. **`app/pages/1_customer_view.py`** — Main user-facing interface
9. **`app/pages/2_ops_view.py`** — Operations simulation interface
10. **`app/pages/3_model_info.py`** — Model comparison display
11. **`src/dabba/nlp/sentiment.py`** — Sentiment analysis logic

### Configuration
12. **`requirements.txt`** — Python dependencies
13. **`Makefile`** — Build commands
14. **`Dockerfile`** / **`docker-compose.yml`** — Container configuration
15. **`.github/workflows/ci.yml`** — CI pipeline

---

## Performance Notes & Technical Debt

### Performance Considerations
1. **Model files are large** — XGBoost/LightGBM models with 100 estimators can be large; consider reducing n_estimators or using smaller models for development
2. **CSV data loading** — Entire datasets loaded into memory; for production, consider chunking or a database
3. **VADER sentiment** — Processes all reviews sequentially; can be slow with large datasets
4. **Cross-validation** — 5-fold CV with 8-9 models can be compute-intensive; consider reducing CV folds during development
5. **Streamlit caching** — Uses `@st.cache_data` to avoid re-reading CSVs on each interaction

### Technical Debt
1. **`POST /recommend` is a stub** — Returns empty recommendations; needs processed data loading
2. **No authentication** — API is fully open; should add JWT or API key auth
3. **No database** — CSV-based storage is fragile; consider PostgreSQL or SQLite
4. **No error tracking** — Missing Sentry or similar monitoring
5. **No logging aggregation** — Logs go to stdout only
6. **Recommender not wired into API** — `/recommend` endpoint doesn't use `RestaurantRecommender` class
7. **No data versioning** — Changes to CSV data are not tracked
8. **Simulation in Operations View** — Uses random data instead of actual model predictions
9. **Hardcoded Bangalore centroids** — Not scalable to other cities
10. **No model monitoring** — No drift detection or performance tracking in production

### Known Risks
1. **Kaggle API changes** — Dataset download script could break
2. **NLTK VADER download** — Requires internet on first run
3. **XGBoost/LightGBM optional** — Models silently skipped if not installed
4. **File structure assumptions** — Path-based config assumes specific directory structure
5. **Lat/Long validation** — India-specific bounds (6-37 lat, 68-98 lon) won't work for other regions

---

## Development Workflow

```
1. Fork and clone repository
2. python -m venv .venv && source .venv/bin/activate
3. make setup          # Install deps + pre-commit hooks
4. python setup_kaggle.py  # Download datasets
5. make train          # Run full pipeline
6. make run-app        # Start Streamlit dashboard
7. make run-api        # Start FastAPI server
8. make test           # Run tests with coverage
```

### Available Make Commands
| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies, create dirs, install pre-commit |
| `make test` | Run pytest with coverage |
| `make lint` | Run ruff, black --check, isort --check |
| `make format` | Auto-format with ruff, black, isort |
| `make train` | Run full training pipeline |
| `make run-app` | Start Streamlit (port 8501) |
| `make run-api` | Start FastAPI (port 8000) |
| `make clean` | Remove generated files |

### Docker Workflow
```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## Deployment Process

### Production Readiness
- **Containerization**: Docker + docker-compose (single image, two services)
- **CI/CD**: GitHub Actions (lint → test → coverage)
- **Hosting**: Designed for Streamlit Community Cloud / any Docker host
- **Live Demo**: Deployed at `https://yourname-dabba.streamlit.app`

### CI Pipeline (`.github/workflows/ci.yml`)
1. Checkout code
2. Set up Python 3.11
3. Install dependencies
4. Run pre-commit checks (black, isort, ruff, trailing-whitespace, etc.)
5. Run pytest with coverage
6. Upload coverage to Codecov

---

## Future Recommendations

1. **Fine-tune an Indic NLP model** (multilingual BERT) for Hindi/English code-switched reviews
2. **Real-time data pipeline** with Kafka + PostgreSQL for live restaurant/delivery data
3. **A/B testing framework** for recommendation algorithm variants
4. **Mobile app** with React Native for on-the-go recommendations
5. **Multi-city expansion** beyond Bangalore
6. **Add authentication** (JWT or OAuth2)
7. **Wire recommender into API endpoint**
8. **Add model monitoring and drift detection**
9. **Replace CSV storage with proper database**
10. **Add CI/CD for deployment to cloud (AWS/GCP/Azure)**
