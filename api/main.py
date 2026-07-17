"""FastAPI application exposing Dabba ML endpoints.

Endpoints:
    POST /recommend — restaurant recommendations
    POST /predict-eta — delivery ETA prediction
    GET /model-info — deployed model info and metrics
    GET /health — health check
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dabba.config import get_config
from api.schemas import (
    ETARequest,
    ETAResponse,
    HealthResponse,
    ModelInfoResponse,
    RecommendRequest,
    RecommendResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dabba API",
    description="India-focused restaurant recommendation and delivery ETA API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = get_config()

# Lazy-loaded models
_rating_model = None
_eta_model = None
_rating_model_info = None
_eta_model_info = None


def _load_models() -> None:
    """Load the winning models from disk."""
    global _rating_model, _eta_model
    global _rating_model_info, _eta_model_info

    try:
        _rating_model = joblib.load(config.best_rating_model_path)
        logger.info("Loaded rating model from %s", config.best_rating_model_path)
    except FileNotFoundError:
        logger.warning("Rating model not found at %s", config.best_rating_model_path)

    try:
        _eta_model = joblib.load(config.best_eta_model_path)
        logger.info("Loaded ETA model from %s", config.best_eta_model_path)
    except FileNotFoundError:
        logger.warning("ETA model not found at %s", config.best_eta_model_path)

    # Load comparison results for info endpoint
    try:
        _rating_model_info = pd.read_csv(config.rating_comparison_path)
    except FileNotFoundError:
        pass
    try:
        _eta_model_info = pd.read_csv(config.eta_comparison_path)
    except FileNotFoundError:
        pass


@app.on_event("startup")
async def startup() -> None:
    """Load models on application startup."""
    _load_models()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        rating_model_loaded=_rating_model is not None,
        eta_model_loaded=_eta_model is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return which models are deployed and their metrics."""
    rating_winner = None
    eta_winner = None

    if _rating_model_info is not None and not _rating_model_info.empty:
        best_row = _rating_model_info.iloc[0]
        rating_winner = {
            "model": best_row.get("model", "unknown"),
            "mae": float(best_row.get("mae", 0)),
            "rmse": float(best_row.get("rmse", 0)),
            "r2": float(best_row.get("r2", 0)),
        }

    if _eta_model_info is not None and not _eta_model_info.empty:
        best_row = _eta_model_info.iloc[0]
        eta_winner = {
            "model": best_row.get("model", "unknown"),
            "mae": float(best_row.get("mae", 0)),
            "rmse": float(best_row.get("rmse", 0)),
            "r2": float(best_row.get("r2", 0)),
        }

    return ModelInfoResponse(
        rating_model=rating_winner,
        eta_model=eta_winner,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Get restaurant recommendations based on preferences."""
    # This endpoint requires processed data to be loaded
    # For now, return a placeholder that documents the API contract
    return RecommendResponse(
        recommendations=[],
        message="Recommender requires processed data. Run `make train` first.",
    )


@app.post("/predict-eta", response_model=ETAResponse)
async def predict_eta(request: ETARequest) -> ETAResponse:
    """Predict delivery time for an order."""
    if _eta_model is None:
        raise HTTPException(
            status_code=503,
            detail="ETA model not loaded. Run `make train` first.",
        )

    # Build feature DataFrame from request
    features = pd.DataFrame([{
        "haversine_distance_km": request.distance_km,
        "traffic_ordinal": request.traffic_level,
        "is_festival": int(request.is_festival),
        "delivery_person_age": request.delivery_person_age,
        "delivery_person_ratings": request.delivery_person_rating,
        "vehicle_condition": request.vehicle_condition,
    }])

    try:
        prediction = _eta_model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    is_at_risk = prediction > config.sla_threshold_minutes

    return ETAResponse(
        predicted_minutes=round(float(prediction), 1),
        is_at_risk=is_at_risk,
        sla_threshold=config.sla_threshold_minutes,
    )
