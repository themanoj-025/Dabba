"""FastAPI application v5 — Dabba ML endpoints with observability.

Routing structure:
    Root (no auth, no rate limit):
        GET  /health               — health check
        GET  /metrics              — Prometheus metrics

    v1 (auth + rate limited):
        POST /v1/recommend         — hybrid restaurant recommendations
        POST /v1/predict-eta       — delivery ETA prediction
        POST /v1/chat              — food concierge chat
        GET  /v1/model-info        — deployed model info

State management:
    All models/tools are loaded at startup and stored in ``app.state``
    rather than module-level globals. Router endpoints access them
    via FastAPI ``Depends()`` instead of ``get_*()`` accessors.

Observability:
    - JSON-formatted structured logging (via :mod:`dabba.observability`)
    - Prometheus metrics at ``/metrics`` (counter, latency histogram)
    - Request ID in every log line via middleware
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from dabba.config import get_config
from dabba.observability import (
    generate_request_id,
    http_request_duration_seconds,
    http_requests_total,
    metrics_endpoint,
    models_loaded,
    set_request_id,
    setup_logging,
)
from api.auth import verify_api_key
from api.limiter import limiter
from api.routers import recommend, eta, chat, explain, model_info, restaurants
from api.schemas import HealthResponse
from dabba.database.session import init_db as init_database

# ─── Configure JSON logging ──────────────────────────────────────────
config = get_config()
setup_logging(config.log_level)

logger = logging.getLogger(__name__)


# ─── App instance ────────────────────────────────────────────────────
app = FastAPI(
    title="Dabba API",
    description="India-focused restaurant recommendation and delivery ETA API",
    version="0.5.0",
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


# ─── Observability middleware (request ID + Prometheus) ───────────────
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = generate_request_id()
    set_request_id(request_id)
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    endpoint = request.url.path
    method = request.method
    status = str(response.status_code)

    # Record Prometheus metrics
    http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

    # Add request_id to response headers
    response.headers["X-Request-ID"] = request_id

    return response


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
v1_router.include_router(explain.router)
v1_router.include_router(restaurants.router)

app.include_router(v1_router)


# ─── Startup ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup() -> None:
    """Load models and tools into ``app.state`` on startup.

    All models are stored in ``app.state`` and accessed via
    FastAPI ``Depends()`` in the router endpoints — no module-level
    globals, no threading locks needed.
    """
    config.models_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database (creates tables if they don't exist)
    try:
        init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database init failed (non-fatal): %s", e)

    # Load each model and store in app.state
    app.state.eta_model = eta._load_eta_model()
    app.state.hybrid_recommender = recommend._load_hybrid_recommender()
    # Pass the ETA model to the concierge so get_eta_estimate() uses real predictions
    app.state.concierge_tools = chat._load_concierge_tools(
        eta_model=app.state.eta_model,
    )

    # Set Prometheus model-loaded gauges
    models_loaded.labels(model="eta").set(
        1 if app.state.eta_model is not None else 0
    )
    models_loaded.labels(model="recommender").set(
        1 if app.state.hybrid_recommender is not None else 0
    )
    models_loaded.labels(model="concierge").set(
        1 if app.state.concierge_tools is not None else 0
    )

    logger.info(
        "Dabba API started",
        extra={
            "eta_model": "loaded" if app.state.eta_model is not None else "missing",
            "recommender": (
                "loaded" if app.state.hybrid_recommender is not None else "missing"
            ),
            "concierge": (
                "loaded" if app.state.concierge_tools is not None else "empty"
            ),
            "version": "0.5.0",
        },
    )


# ─── Metrics (no auth) ───────────────────────────────────────────────
@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint — no auth required.

    Returns metrics in Prometheus text format for scraping by
    Prometheus or Grafana.
    """
    return Response(
        content=metrics_endpoint(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ─── Health check (no auth) ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint — reports model load status from ``app.state``.

    This endpoint is intentionally unauthenticated so that
    load balancers, Docker health checks, and monitoring
    tools can always reach it.
    """
    return HealthResponse(
        status="ok",
        rating_model_loaded=(
            getattr(request.app.state, "hybrid_recommender", None) is not None
        ),
        eta_model_loaded=(
            getattr(request.app.state, "eta_model", None) is not None
        ),
    )
