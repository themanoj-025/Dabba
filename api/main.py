"""FastAPI application v3 — Dabba ML endpoints.

Routers:
    POST /recommend       — hybrid restaurant recommendations
    POST /predict-eta      — delivery ETA prediction
    POST /chat              — food concierge chat
    GET  /model-info        — deployed model info
    GET  /health            — health check
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dabba.config import get_config
from api.routers import recommend, eta, chat, model_info
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dabba API",
    description="India-focused restaurant recommendation and delivery ETA API",
    version="0.2.0",
)

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

# ─── Include routers ──────────────────────────────────────────────────
app.include_router(recommend.router)
app.include_router(eta.router)
app.include_router(chat.router)
app.include_router(model_info.router)


@app.on_event("startup")
async def startup() -> None:
    """Load models and tools on startup."""
    # Check that required directories exist
    config.models_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    eta.load_eta_model()
    recommend.load_recommender()
    chat.load_concierge_tools()

    logger.info("Dabba API v0.2.0 started")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")
