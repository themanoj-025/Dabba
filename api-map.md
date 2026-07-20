# 📡 Dabba v3 — API Inventory

**Base URL**: `http://localhost:8000` (configurable via docker-compose)

---

## Endpoints

### 1. GET `/health`

**Purpose**: Health check + model load status.

**Response** (`HealthResponse`):
```json
{
    "status": "ok",
    "rating_model_loaded": true,
    "eta_model_loaded": true,
    "version": "3.0.0"
}
```

**Used by**: Streamlit sidebar status indicator, Docker health checks.

---

### 2. GET `/model-info`

**Purpose**: Deployed model names and metrics — reads from comparison CSVs.

**Response** (`ModelInfoResponse`):
```json
{
    "rating_model": {
        "name": "RandomForest",
        "mae": 0.0596,
        "rmse": 0.1267,
        "r2": 0.9172
    },
    "eta_model": {
        "name": "GradientBoosting",
        "mae": 5.7893,
        "rmse": 7.3641,
        "r2": 0.3837
    }
}
```

**Dependencies**: `reports/model_comparison_rating.csv`, `reports/model_comparison_eta.csv`.

---

### 3. POST `/recommend`

**Purpose**: Generate hybrid restaurant recommendations.

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
            "reliability_score": 0.85,
            "combined_score": 0.92,
            "explanation": "Highly rated with excellent reliability in your area"
        }
    ],
    "message": "Found 5 restaurants matching your preferences"
}
```

**Query Params**: `?use_llm_narration=true` optionally invokes the LLM narrator.

**Dependencies**: `models/best_rating_model.pkl`, `models/best_collaborative_model.pt`, processed restaurant CSV.

---

### 4. POST `/predict-eta`

**Purpose**: Predict delivery time using the winning ETA model.

**Request** (`ETARequest`):
```json
{
    "distance_km": 5.2,
    "traffic_level": "Medium",
    "is_festival": false,
    "delivery_person_age": 28,
    "delivery_person_ratings": 4.5,
    "vehicle_condition": 2
}
```

**Response** (`ETAResponse`):
```json
{
    "predicted_minutes": 28.4,
    "is_at_risk": false,
    "sla_threshold": 40,
    "confidence_interval": [22.0, 34.8]
}
```

**Dependencies**: `models/best_eta_model.pkl`.

---

### 5. POST `/chat`

**Purpose**: Food concierge chat with tool-use.

**Request** (`ChatRequest`):
```json
{
    "message": "Find me something spicy under 400 rupees near Koramangala",
    "history": []
}
```

**Response** (`ChatResponse`):
```json
{
    "reply": "I found several options! Meghana Foods (₹400, rating 4.8) tops the list with excellent reliability. Also try Anupam's Coast to Coast (₹350, rating 4.5) for spicy Andhra-style dishes."
}
```

**Fallback**: Rules-based intent matching when no API key is configured.

**Dependencies**: `dabba.llm.food_concierge` (rules-based without API key, LLM-powered with Anthropic key).

---

## Schema Summary

| Schema | Endpoint | Fields |
|--------|----------|--------|
| `HealthResponse` | GET /health | status, rating_model_loaded, eta_model_loaded, version |
| `ModelInfoResponse` | GET /model-info | rating_model (dict), eta_model (dict) |
| `RecommendRequest` | POST /recommend | cuisine, budget, area, top_n, prioritize, use_llm_narration |
| `Recommendation` | POST /recommend | name, rate, bayesian_rating, cost_for_two, location, cuisines, reliability_score, cf_score, combined_score, explanation |
| `RecommendResponse` | POST /recommend | recommendations (list), message |
| `ETARequest` | POST /predict-eta | distance_km, traffic_level, is_festival, delivery_person_age, delivery_person_ratings, vehicle_condition |
| `ETAResponse` | POST /predict-eta | predicted_minutes, is_at_risk, sla_threshold |
| `ChatMessage` | POST /chat | role, content |
| `ChatRequest` | POST /chat | message, history (list of ChatMessage) |
| `ChatResponse` | POST /chat | reply |
