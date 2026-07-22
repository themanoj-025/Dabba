"""ETA Prediction router — delivery time estimation using winning model."""

from __future__ import annotations

import logging
import threading

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.limiter import limiter
from dabba.config import get_config
from api.schemas import ETARequest, ETAResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict-eta", tags=["eta"])

config = get_config()
_eta_model = None
_eta_model_lock = threading.Lock()


def load_eta_model() -> None:
    """Load the winning ETA model from disk (thread-safe)."""
    global _eta_model
    with _eta_model_lock:
        if _eta_model is not None:
            return
        try:
            _eta_model = joblib.load(config.best_eta_model_path)
            logger.info("Loaded ETA model from %s", config.best_eta_model_path)
        except FileNotFoundError:
            logger.warning("ETA model not found at %s", config.best_eta_model_path)


def get_eta_model():
    """Thread-safe accessor for the loaded ETA model."""
    global _eta_model
    with _eta_model_lock:
        return _eta_model


@router.post("", response_model=ETAResponse)
@limiter.limit("30/minute")
async def predict_eta(request: Request, body: ETARequest) -> ETAResponse:
    """Predict delivery time for an order.

    Args:
        request: Incoming HTTP request (required by rate limiter).
        body: ETARequest with distance, traffic, festival, etc.

    Returns:
        ETAResponse with predicted minutes and SLA risk.
    """
    model = get_eta_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ETA model not loaded. Run `make train` first.",
        )

    features = pd.DataFrame(
        [
            {
                "haversine_distance_km": body.distance_km,
                "traffic_ordinal": body.traffic_level,
                "is_festival": int(body.is_festival),
                "delivery_person_age": body.delivery_person_age,
                "delivery_person_ratings": body.delivery_person_rating,
                "vehicle_condition": body.vehicle_condition,
            }
        ]
    )

    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    is_at_risk = prediction > config.sla_threshold_minutes

    return ETAResponse(
        predicted_minutes=round(float(prediction), 1),
        is_at_risk=is_at_risk,
        sla_threshold=config.sla_threshold_minutes,
    )
