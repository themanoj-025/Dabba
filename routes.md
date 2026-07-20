# 🗺️ Dabba v3 — Routing Map

## Streamlit Routing

Dabba v3 uses **custom radio navigation** (not Streamlit multi-page auto-discovery).

**How it works:**
1. `app/streamlit_app.py` renders a `st.radio` in the sidebar with 4 page options
2. Based on the selected radio value, the appropriate page module is called
3. Old `1_customer_view.py`, `2_ops_view.py`, `3_model_info.py` files are deleted

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

Dabba v3 uses separate router modules under `api/routers/` for clean separation.

### Router Table

| # | Method | Route | Module | Purpose | Auth |
|---|--------|-------|--------|---------|------|
| 1 | GET | `/health` | `api/main.py` (inline) | Health check + model load status | No |
| 2 | GET | `/model-info` | `routers/model_info.py` | Deployed model names & metrics from CSVs | No |
| 3 | POST | `/recommend` | `routers/recommend.py` | Hybrid recommendations, optional LLM narration | No |
| 4 | POST | `/predict-eta` | `routers/eta.py` | Delivery ETA using winning model | No |
| 5 | POST | `/chat` | `routers/chat.py` | Food concierge with tool-use | No |

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

### API Setup

```python
app = FastAPI(title="Dabba API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=origins, ...)
app.include_router(recommend_router)
app.include_router(eta_router)
app.include_router(chat_router)
app.include_router(model_info_router)
```

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
POST /predict-eta → ETARequest schema
    ↓
joblib.load('best_eta_model.pkl') → predict()
    ↓
Compare with SLA threshold → is_at_risk flag
    ↓
ETAResponse schema returned
```

### Chat Flow
```
User types message in Food Concierge
    ↓
get_concierge_response() with ConciergeTools
    ↓
LLM or rules-based intent matching → tool calls
    ↓
search_restaurants() / get_eta_estimate() / get_reliability_score()
    ↓
Formatted response with styled chat bubbles
```
