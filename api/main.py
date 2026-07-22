"""FastAPI application v4 — Dabba ML endpoints.

Routing structure:
    Root (no auth, no rate limit):
        GET  /health               — health check

    v1 (auth + rate limited):
        POST /v1/recommend         — hybrid restaurant recommendations
        POST /v1/predict-eta       — delivery ETA prediction
        POST /v1/chat              — food concierge chat
        GET  /v1/model-info        — deployed model info
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from dabba.config import get_config
from api.auth import verify_api_key
from api.limiter import limiter
from api.routers import recommend, eta, chat, model_info
from api.routers.eta import get_eta_model
from api.routers.recommend import get_recommender
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)

# ─── App instance ────────────────────────────────────────────────────
app = FastAPI(
    title="Dabba API",
    description="India-focused restaurant recommendation and delivery ETA API",
    version="0.4.0",
)

# ─── Rate limiter ─────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Security Headers ────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), interest-cohort=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; frame-ancestors 'none';"
    )
    return response


config = get_config()

# ─── v1 authenticated router ─────────────────────────────────────────
# All v1 endpoints require X-API-Key (unless DABBA_API_KEY is unset)
v1_router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(verify_api_key)],
)

v1_router.include_router(recommend.router)
v1_router.include_router(eta.router)
v1_router.include_router(chat.router)
v1_router.include_router(model_info.router)

app.include_router(v1_router)


# ─── Startup ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup() -> None:
    """Load models and tools on startup."""
    config.models_dir.mkdir(parents=True, exist_ok=True)

    eta.load_eta_model()
    recommend.load_recommender()
    chat.load_concierge_tools()

    logger.info("Dabba API v0.4.0 started")


# ─── Health check (no auth) ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint — reports model load status.

    This endpoint is intentionally unauthenticated so that
    load balancers, Docker health checks, and monitoring
    tools can always reach it.
    """
    return HealthResponse(
        status="ok",
        rating_model_loaded=get_recommender() is not None,
        eta_model_loaded=get_eta_model() is not None,
    )
