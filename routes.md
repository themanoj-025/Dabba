# 🧭 Dabba — Route Map

## Overview

Dabba uses **Streamlit's built-in multi-page app routing**, which automatically detects pages placed in the `app/pages/` directory. No explicit router configuration is needed.

---

## Route Table

| Route | File | Title | Purpose | Auth Required |
|-------|------|-------|---------|---------------|
| `/` (root) | `app/streamlit_app.py` | Dabba — Restaurant Intelligence Platform | Home/landing page with nav links | No |
| `/customer_view` | `app/pages/1_customer_view.py` | Customer View — Dabba | Restaurant recommendations with filters | No |
| `/ops_view` | `app/pages/2_ops_view.py` | Operations View — Dabba | Delivery SLA monitoring & simulation | No |
| `/model_info` | `app/pages/3_model_info.py` | Model Info — Dabba | Model comparison results & charts | No |

**Streamlit Multi-Page Behavior:**
- Streamlit reads the `app/pages/` directory alphabetically
- Files prefixed with numbers (e.g., `1_customer_view.py`) appear ordered in the sidebar
- The page filename suffix after the number becomes the label (e.g., `1_customer_view.py` → "Customer View")
- Each page runs independently with its own `st.set_page_config()`

---

## API Routes

| Route | Method | File | Purpose | Auth Required |
|-------|--------|------|---------|---------------|
| `/health` | GET | `api/main.py` | Health check + model load status | No |
| `/model-info` | GET | `api/main.py` | Deployed model names & metrics | No |
| `/recommend` | POST | `api/main.py` | Restaurant recommendations | No |
| `/predict-eta` | POST | `api/main.py` | Delivery ETA prediction | No |

---

## Route Details

### Home Page (`/`)
- **File**: `app/streamlit_app.py`
- **Sidebar**: Logo, project description, author links
- **Content**: Welcome message, feature descriptions, Reliability Score formula
- **Navigation**: Sidebar links to all three sub-pages + info hint

### Customer View (`/customer_view`)
- **File**: `app/pages/1_customer_view.py`
- **Sidebar Filters**:
  - Cuisine (multi-select, 15 options)
  - Budget (slider, ₹100-5000)
  - Area (dropdown, 12 Bangalore areas)
  - Result count (slider, 3-20)
- **Main Content**:
  - Filtered restaurant listing with name, location, cuisines, rating, cost
  - Loading state when data missing
- **Data Source**: `data/processed/restaurants_processed.csv`

### Operations View (`/ops_view`)
- **File**: `app/pages/2_ops_view.py`
- **Controls**:
  - SLA threshold slider (20-60 min)
  - Number of simulated orders (10-500)
  - "Run Simulation" button
- **Display**:
  - Real-time progress bar
  - Live metrics (total orders, on-time rate, at-risk count)
  - Results table with conditional formatting
  - Confusion matrix table (TP, FP, FN, TN)

### Model Info (`/model_info`)
- **File**: `app/pages/3_model_info.py`
- **Sections**:
  - Rating Prediction Comparison
  - ETA Prediction Comparison
  - Methodology explanation
- **Dynamic Content**:
  - Winner callout (success alert)
  - Comparison tables with green highlighting
  - Bar charts (PNG)
  - R² comparison charts (PNG)
  - Residual plots (PNG)
  - Interpretation text
- **Fallback**: Warning messages when data/charts not found

---

## Middleware

### FastAPI Middleware Chain
1. **CORS Middleware** — Allows origins from localhost:8000, 8501
2. **Security Headers Middleware** — Adds:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `Referrer-Policy: strict-origin-when-cross-origin`
   - `X-XSS-Protection: 0`
   - `Permissions-Policy`: restricts camera, microphone, geolocation
   - `Content-Security-Policy`: `default-src 'none'; frame-ancestors 'none';`

### Streamlit
- No middleware; pages are rendered server-side
- `st.cache_data` decorator caches data loading results

---

## Navigation Flow

```
User opens Streamlit (port 8501)
        │
        ▼
    Home Page (/)
    ┌─────────────────────────────────┐
    │ Welcome + Reliability Score    │
    │ Sidebar: 3 page links          │
    └─────────────────────────────────┘
        │
        ├───▶ Customer View (/customer_view)
        │     ┌──────────────────────────┐
        │     │ Filters → Recommendations│
        │     └──────────────────────────┘
        │
        ├───▶ Operations View (/ops_view)
        │     ┌──────────────────────────┐
        │     │ Simulation → Metrics     │
        │     └──────────────────────────┘
        │
        └───▶ Model Info (/model_info)
              ┌──────────────────────────┐
              │ Charts → Winner Info     │
              └──────────────────────────┘
```
