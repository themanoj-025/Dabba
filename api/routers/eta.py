"""ETA Prediction router — delivery time estimation using winning model.
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from dabba.config import get_config
from api.schemas import ETARequest, ETAResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict-eta", tags=["eta"])

config = get_config()
_eta_model = None


def load_eta_model() -> None:
    """Load the winning ETA model from disk."""
    global _eta_model
    try:
        _eta_model = joblib.load(config.best_eta_model_path)
        logger.info("Loaded ETA model from %s", config.best_eta_model_path)
    except FileNotFoundError:
        logger.warning("ETA model not found at %s", config.best_eta_model_path)


@router.post("", response_model=ETAResponse)
async def predict_eta(request: ETARequest) -> ETAResponse:
    """Predict delivery time for an order.

    Args:
        request: ETARequest with distance, traffic, festival, etc.

    Returns:
        ETAResponse with predicted minutes and SLA risk.
    """
    if _eta_model is None:
        raise HTTPException(
            status_code=503,
            detail="ETA model not loaded. Run `make train` first.",
        )

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
