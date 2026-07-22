# 🗺️ Dabba v4 — Routing Map

## Streamlit Routing

Dabba v4 uses **custom radio navigation** (not Streamlit multi-page auto-discovery).

**How it works:**
1. `app/streamlit_app.py` renders a `st.radio` in the sidebar with 4 page options
2. Based on the selected radio value, the appropriate page module is called

### Streamlit Pages

| Radio Label | Module | File | Description |
|-------------|--------|------|-------------|
| `🍽️ Discover` | `page_discover` | `app/pages/page_discover.py` | Restaurant discovery with filters, styled cards, LLM explanations, RAG similar-retrieval |
| `🚀 Ops Monitor` | `page_ops` | `app/pages/page_ops.py` | Delivery SLA monitoring, simulation, drift alerts, metric cards |
| `📊 Model Performance` | `page_model_performance` | `app/pages/page_model_performance.py` | Interactive comparison charts, SHAP plots, A/B scenario display |
| `💬 Food Concierge` | `page_concierge` | `app/pages/page_concierge.py` | Chat copilot with example prompts, tool-use integration |

### Navigation Flow

```
streamlit_app.py (entry point)
    │
    ├── Load theme CSS (assets/theme.css)
    │
    ├── Sidebar: logo, description, radio navigation
    │
    ├── Page routing:
    │   ├── "🍽️"  → discover.show()
    │   ├── "🚀"  → ops.show()
    │   ├── "📊"  → model_perf.show()
    │   └── "💬"  → concierge.show()
    │
    └── Default: welcome screen with Reliability Score explanation
```

---

## FastAPI Routing

Dabba v4 uses **versioned API routing** under `/v1` with API key authentication and rate limiting.

### Router Architecture

```
app (no auth)
  └── GET /health  [reads from app.state via request parameter]
  └── v1_router (auth + rate limit, models via Depends DI)
       ├── POST /v1/recommend       (30/min)  [Depends(get_recommender)]
       ├── POST /v1/predict-eta     (30/min)  [Depends(get_eta_model)]
       ├── POST /v1/chat            (10/min)  [Depends(get_tools)]
       └── GET  /v1/model-info      (60/min)
```

### Endpoint Table

| # | Method | Route | Module | Auth | Rate Limit | Purpose |
|---|--------|-------|--------|------|------------|---------|
| 1 | GET | `/health` | `api/main.py` (inline) | No | No | Health check + model load status |
| 2 | GET | `/v1/model-info` | `routers/model_info.py` | `X-API-Key` | 60/min | Deployed model names & metrics |
| 3 | POST | `/v1/recommend` | `routers/recommend.py` | `X-API-Key` | 30/min | Hybrid recommendations, optional LLM narration |
| 4 | POST | `/v1/predict-eta` | `routers/eta.py` | `X-API-Key` | 30/min | Delivery ETA using winning model |
| 5 | POST | `/v1/chat` | `routers/chat.py` | `X-API-Key` | 10/min | Food concierge with tool-use |

### Authentication

- All `/v1/*` endpoints require an `X-API-Key` header
- Configure via `DABBA_API_KEY` environment variable (or `.env` file)
- **Dev mode:** If `DABBA_API_KEY` is not set, authentication is skipped — the API works without a key for local development
- The `/health` endpoint is intentionally unauthenticated for monitoring/load balancer access

### Rate Limiting

- Uses `slowapi` with IP-based key function (`get_remote_address`)
- Limits applied per endpoint type:
  - `POST /v1/chat`: **10/minute** — LLM calls are expensive
  - `POST /v1/recommend`: **30/minute** — moderate usage
  - `POST /v1/predict-eta`: **30/minute** — moderate usage
  - `GET /v1/model-info`: **60/minute** — lightweight reads
- Rate-limited requests return `429 Too Many Requests`

### CORS Configuration

```python
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
```

Allows the Streamlit dashboard (port 8501) to call the API (port 8000).

### Security Headers

All responses include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `X-XSS-Protection: 0`
- `Permissions-Policy: camera=(), microphone=(), geolocation=(), interest-cohort=()`
- `Content-Security-Policy: default-src 'none'; frame-ancestors 'none';`

---

## Data Flow (Feature → API → UI)

### Recommendation Flow
```
User selects filters in Discover page
    ↓
page_discover.py reads session state filters
    ↓
HybridRecommender.recommend() blends 3 signals
    ↓
narrate_recommendation() generates explanation (LLM or rules)
    ↓
render_restaurant_card() displays styled card with explanation
```

### ETA Flow
```
User enters delivery details
    ↓
POST /v1/predict-eta → ETARequest schema (X-API-Key required)
    ↓
joblib.load('best_eta_model.pkl') → predict()
    ↓
Compare with SLA threshold → is_at_risk flag
    ↓
ETAResponse schema returned
```

### Chat Flow (ReAct Loop)
```
User types message in Food Concierge
    ↓
POST /v1/chat → ChatRequest schema (X-API-Key required, 10/min)
    ↓
ConciergeTools loaded via app.state (Depends DI, no module globals)
    ↓
LLM: ReAct loop (max 4 steps, config.llm_max_steps):
    ├── Step 1: LLM returns tool_use call(s) + optional text
    ├── Step 2: Execute tool(s) → tool_result fed back to conversation
    ├── Step 3: LLM analyzes results, decides: more tools or final answer
    └── Step 4: Final summarization (or max_steps reached)
    ↓
Tool calls available: search_restaurants() / get_eta_estimate() / get_reliability_score()
    ↓
Fallback: rules-based intent matching when no API key configured
    ↓
Formatted response with styled chat bubbles
```

### Model Loading (app.state)
```
api/main.py startup:
    app.state.eta_model = eta._load_eta_model()
    app.state.hybrid_recommender = recommend._load_hybrid_recommender()
    app.state.concierge_tools = chat._load_concierge_tools()

Router endpoints access via Depends:
    model = Depends(get_eta_model)     # → request.app.state.eta_model
    rec   = Depends(get_recommender)   # → request.app.state.hybrid_recommender
    tools = Depends(get_tools)         # → request.app.state.concierge_tools
```
