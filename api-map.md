# 🌐 Dabba — API Map

## Overview

Dabba exposes a **FastAPI** application with 4 endpoints. The API is designed for internal/local use with no authentication.

**Base URL**: `http://localhost:8000` (configurable via docker-compose)

---

## Endpoint Inventory

### 1. Health Check

```
GET /health
```

**Purpose**: Verify the API is running and models are loaded.

**Response** (`HealthResponse`):
```json
{
  "status": "ok",
  "rating_model_loaded": false,
  "eta_model_loaded": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always "ok" when server is running |
| `rating_model_loaded` | bool | Whether the rating model artifact was loaded |
| `eta_model_loaded` | bool | Whether the ETA model artifact was loaded |

**Status Codes**: 200 OK

**Used By**: Monitoring, health checks, Docker health probes

---

### 2. Model Info

```
GET /model-info
```

**Purpose**: Return information about deployed ML models and their performance metrics.

**Response** (`ModelInfoResponse`):
```json
{
  "rating_model": {
    "model": "XGBoost",
    "mae": 0.2856,
    "rmse": 0.3872,
    "r2": 0.8247
  },
  "eta_model": {
    "model": "GradientBoosting",
    "mae": 7.5432,
    "rmse": 10.2345,
    "r2": 0.6142
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `rating_model` | object or null | Winning rating model info |
| `rating_model.model` | string | Model name |
| `rating_model.mae` | float | Mean Absolute Error |
| `rating_model.rmse` | float | Root Mean Squared Error |
| `rating_model.r2` | float | R² Score |
| `eta_model` | object or null | Winning ETA model info (same sub-fields) |

**Data Source**: `reports/model_comparison_rating.csv` and `reports/model_comparison_eta.csv` (first row = winner).

**Status Codes**: 200 OK

**Used By**: Streamlit Model Info page

---

### 3. Restaurant Recommendation

```
POST /recommend
```

**Purpose**: Get personalized restaurant recommendations based on user preferences.

**⚠️ NOTE**: This endpoint is currently a **stub**. It returns an empty recommendation list with a message instructing the user to run `make train`. The full implementation would use `RestaurantRecommender` from `src/dabba/models/recommender.py`.

**Request Body** (`RecommendRequest`):
```json
{
  "cuisine": "North Indian",
  "budget": 1000,
  "area": "Koramangala",
  "top_n": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `cuisine` | string | No | null | Preferred cuisine for filtering |
| `budget` | float | No | null | Maximum cost for two (INR) |
| `area` | string | No | null | Area/neighborhood filter |
| `top_n` | int | No | 5 | Number of recommendations |

**Response** (`RecommendResponse`):
```json
{
  "recommendations": [],
  "message": "Recommender requires processed data. Run `make train` first."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `recommendations` | array of objects | List of `Recommendation` objects |
| `message` | string or null | Status/info message |

**Recommendation Object**:
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Restaurant name |
| `rating` | float or null | Raw rating (/5) |
| `bayesian_rating` | float or null | Bayesian-adjusted rating |
| `cost_for_two` | float or null | Cost for two (INR) |
| `location` | string or null | Area/neighborhood |
| `cuisines` | string or null | Comma-separated cuisines |
| `similarity_score` | float or null | Content similarity to query |
| `combined_score` | float or null | Hybrid score (similarity + popularity) |
| `explanation` | string or null | Human-readable explanation |

**Status Codes**: 200 OK

**Used By**: Streamlit Customer View page

---

### 4. Predict ETA

```
POST /predict-eta
```

**Purpose**: Predict delivery time for an order and determine SLA compliance.

**Request Body** (`ETARequest`):
```json
{
  "distance_km": 5.2,
  "traffic_level": 2,
  "is_festival": false,
  "delivery_person_age": 28,
  "delivery_person_rating": 4.5,
  "vehicle_condition": 2
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `distance_km` | float | **Yes** | — | Haversine distance in km |
| `traffic_level` | int | No | 1 | Traffic density (0=Low, 1=Medium, 2=High, 3=Jam) |
| `is_festival` | bool | No | false | Whether it's a festival day |
| `delivery_person_age` | float | No | null | Delivery person age |
| `delivery_person_rating` | float | No | null | Delivery person rating |
| `vehicle_condition` | int | No | null | Vehicle condition score |

**Response** (`ETAResponse`):
```json
{
  "predicted_minutes": 32.5,
  "is_at_risk": false,
  "sla_threshold": 40.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted_minutes` | float | Predicted delivery time in minutes |
| `is_at_risk` | bool | Whether predicted time exceeds SLA threshold |
| `sla_threshold` | float | Current SLA threshold in minutes |

**Error States**:
- **503 Service Unavailable**: ETA model not loaded (run `make train`)
- **422 Unprocessable Entity**: Validation failure (e.g., non-numeric distance)
- **500 Internal Server Error**: Model prediction failure

**Status Codes**: 200 OK, 422, 503, 500

**Used By**: Operations simulation, external integrations

---

## Error Handling

| Status Code | Meaning | Example |
|-------------|---------|---------|
| 200 | Success | Request processed |
| 422 | Validation Error | Invalid input types |
| 503 | Service Unavailable | Model not loaded |
| 500 | Internal Server Error | Model prediction failed |

**Validation Error Format (422)**:
```json
{
  "detail": [
    {
      "loc": ["body", "distance_km"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

---

## CORS Configuration

Allowed origins:
- `http://localhost:8000`
- `http://127.0.0.1:8000`
- `http://localhost:8501`
- `http://127.0.0.1:8501`

Allowed methods: `*`
Allowed headers: `*`

---

## Integration Points

| Endpoint | Integrates With | Data Flow |
|----------|----------------|-----------|
| `/health` | Docker, monitoring | Direct response |
| `/model-info` | Streamlit Model Info page | Reads comparison CSVs |
| `/recommend` | Streamlit Customer View (future) | Requires processed data + recommender |
| `/predict-eta` | Operations View (simulation), external systems | Uses loaded ETA model |
