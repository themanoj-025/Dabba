# 📡 Dabba v4 — API Inventory

**Base URL**: `http://localhost:8000` (configurable via docker-compose)

**Authentication**: All `/v1/*` endpoints require `X-API-Key` header.
See `routes.md` for dev-mode bypass behavior.

**Rate Limiting**: All `/v1/*` endpoints are rate-limited per IP.
429 responses include `Retry-After` header.

---

## Endpoints

---

### 1. GET `/health`

**Auth**: None

**Purpose**: Health check + model load status. Intentionally unauthenticated
for load balancers, Docker health checks, and monitoring.

**Response** (`HealthResponse`):
```json
{
    "status": "ok",
    "rating_model_loaded": true,
    "eta_model_loaded": true
}
```

**Used by**: Streamlit sidebar status indicator, Docker health checks.

---

### 2. GET `/v1/model-info`

**Auth**: `X-API-Key` header

**Rate Limit**: 60/minute

**Purpose**: Deployed model names and metrics — reads from comparison CSVs.

**Response**:
```json
{
    "rating_model": {
        "model": "RandomForest",
        "mae": 0.0596,
        "rmse": 0.1267,
        "r2": 0.9172
    },
    "eta_model": {
        "model": "GradientBoosting",
        "mae": 5.7893,
        "rmse": 7.3641,
        "r2": 0.3837
    }
}
```

**Dependencies**: `reports/model_comparison_rating.csv`, `reports/model_comparison_eta.csv`.

---

### 3. POST `/v1/recommend`

**Auth**: `X-API-Key` header

**Rate Limit**: 30/minute

**Purpose**: Generate hybrid restaurant recommendations with optional LLM narration.

**Request** (`RecommendRequest`):
```json
{
    "cuisine": "North Indian",
    "budget": 500,
    "area": "Koramangala",
    "top_n": 5,
    "prioritize": "balanced",
    "use_llm_narration": false
}
```

**Response** (`RecommendResponse`):
```json
{
    "recommendations": [
        {
            "name": "Meghana Foods",
            "rating": 4.8,
            "cost_for_two": 400,
            "location": "Koramangala",
            "cuisines": "North Indian, Chinese",
            "combined_score": 0.92,
            "explanation": "Highly rated with excellent reliability in your area"
        }
    ],
    "message": null
}
```

**Dependencies**: `models/best_rating_model.pkl`, processed restaurant CSV.

---

### 4. POST `/v1/predict-eta`

**Auth**: `X-API-Key` header

**Rate Limit**: 30/minute

**Purpose**: Predict delivery time using the winning ETA model.

**Request** (`ETARequest`):
```json
{
    "distance_km": 5.2,
    "traffic_level": 1,
    "is_festival": false,
    "delivery_person_age": 28,
    "delivery_person_rating": 4.5,
    "vehicle_condition": 2
}
```

**Response** (`ETAResponse`):
```json
{
    "predicted_minutes": 28.4,
    "is_at_risk": false,
    "sla_threshold": 40
}
```

**Dependencies**: `models/best_eta_model.pkl`.

---

### 5. POST `/v1/chat`

**Auth**: `X-API-Key` header

**Rate Limit**: 10/minute (LLM calls are expensive)

**Purpose**: Food concierge chat with **ReAct tool loop** (max 4 steps) and rules-based fallback.

**Request** (`ChatRequest`):
```json
{
    "message": "Find me something spicy under 400 rupees near Koramangala, check the ETA and reliability of the top pick",
    "history": []
}
```

**Response** (`ChatResponse`):
```json
{
    "reply": "I found Meghana Foods (₹400, rating 4.8) in Koramangala. Delivery ETA is ~28 min and reliability score is 0.88/1.0 — highly recommended!"
}
```

**ReAct Loop**: The concierge can chain multiple tool calls per user message — search → check ETA → check reliability → summarize. Tool results are fed back to the LLM for multi-step reasoning.

**Fallback**: Rules-based intent matching when no API key is configured.

**Dependencies**: `dabba.llm.food_concierge` (ReAct loop with LLM, rules-based fallback without API key). Models loaded via `app.state` DI.

---

### 6. GET `/v1/restaurants`

**Auth**: `X-API-Key` header

**Rate Limit**: 60/minute

**Purpose**: List restaurants from the database (proves CSV→DB migration).

**Query Parameters**:
- `limit` (int, 1-200, default 50) — Max results per page
- `offset` (int, default 0) — Pagination offset

**Response** (`RestaurantListResponse`):
```json
{
    "restaurants": [
        {
            "id": 1,
            "name": "Meghana Foods",
            "rate": 4.8,
            "bayesian_rating": 4.75,
            "cost_for_two": 400,
            "location": "Koramangala",
            "cuisines": "North Indian, Chinese",
            "votes": 500,
            "reliability_score": 0.88
        }
    ],
    "total": 4268,
    "limit": 50,
    "offset": 0
}
```

**Dependencies**: Database (SQLite/Postgres) via `get_db_generator` DI.

---

### 7. GET `/v1/restaurants/{restaurant_id}`

**Auth**: `X-API-Key` header

**Rate Limit**: 60/minute

**Purpose**: Get a single restaurant by ID.

**Path Parameters**:
- `restaurant_id` (int) — Restaurant primary key

**Response** (`RestaurantItem`):
```json
{
    "id": 1,
    "name": "Meghana Foods",
    "rate": 4.8,
    "bayesian_rating": 4.75,
    "cost_for_two": 400,
    "location": "Koramangala",
    "cuisines": "North Indian, Chinese",
    "votes": 500,
    "reliability_score": 0.88
}
```

**Error Responses**:
- `404` — Restaurant not found
- `503` — Database not available

---

### 8. GET `/v1/restaurants/search/{query}`

**Auth**: `X-API-Key` header

**Rate Limit**: 60/minute

**Purpose**: Search restaurants by name or cuisine.

**Path Parameters**:
- `query` (str) — Search term (matches name or cuisine)

**Query Parameters**:
- `limit` (int, 1-100, default 20) — Max results

**Response** (`RestaurantListResponse`):
```json
{
    "restaurants": [...],
    "total": 3,
    "limit": 20,
    "offset": 0
}
```

**Error Responses**:
- `404` — No restaurants found matching query
- `503` — Database not available

---

## Schema Summary

| Schema | Endpoint | Fields |
|--------|----------|--------|
| `HealthResponse` | GET /health | status, rating_model_loaded, eta_model_loaded |
| `RecommendRequest` | POST /v1/recommend | cuisine, budget, area, top_n, prioritize, use_llm_narration |
| `Recommendation` | POST /v1/recommend | name, rating, cost_for_two, location, cuisines, combined_score, explanation |
| `RecommendResponse` | POST /v1/recommend | recommendations (list), message |
| `ETARequest` | POST /v1/predict-eta | distance_km, traffic_level, is_festival, delivery_person_age, delivery_person_rating, vehicle_condition |
| `ETAResponse` | POST /v1/predict-eta | predicted_minutes, is_at_risk, sla_threshold |
| `ChatMessage` | POST /v1/chat | role, content |
| `ChatRequest` | POST /v1/chat | message, history (list of ChatMessage) |
| `ChatResponse` | POST /v1/chat | reply |
| `RestaurantItem` | GET /v1/restaurants/* | id, name, rate, bayesian_rating, cost_for_two, location, cuisines, votes, reliability_score |
| `RestaurantListResponse` | GET /v1/restaurants | restaurants (list), total, limit, offset |

---

## Error Responses

| Status | Meaning | Example |
|--------|---------|---------|
| 400 | Bad request (validation) | `{"detail":[{"loc":["body","distance_km"],"msg":"..."}]}` |
| 401 | Missing/invalid API key | `{"detail":"Missing X-API-Key header"}` |
| 404 | Resource not found | `{"detail":"Restaurant not found"}` |
| 429 | Rate limit exceeded | `{"detail":"Rate limit exceeded: 10/minute"}` |
| 503 | Model/tools not loaded | `{"detail":"ETA model not loaded. Run `make train` first."}` |
