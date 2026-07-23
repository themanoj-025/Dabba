"""ETA Prediction router — delivery time estimation using winning model.

Models are loaded at app startup and stored in ``app.state``,
then injected via FastAPI ``Depends()`` — no module-level globals.
"""

from __future__ import annotations

import logging
from typing import Optional

import joblib
from fastapi import APIRouter, Depends, HTTPException, Request

from api.limiter import limiter
from dabba.cache.redis_client import get_cache
from dabba.config import DabbaConfig, get_config
from dabba.features.delivery_features import build_eta_features_for_api
from api.schemas import ETARequest, ETAResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict-eta", tags=["eta"])

config = get_config()

# Feature columns are built using build_eta_features_for_api() from
# delivery_features.py, which is the single source of truth matching the
# training pipeline (pipeline.py imports ETA_FEATURE_COLS from there).


def _load_eta_model() -> Optional[object]:
    """Load the winning ETA model from disk.

    Called once at app startup by ``api.main``. Returns the model
    or ``None`` if the artifact hasn't been trained yet.

    Returns:
        The fitted Pipeline (joblib-loaded), or None.
    """
    try:
        model = joblib.load(config.best_eta_model_path)
        logger.info("Loaded ETA model from %s", config.best_eta_model_path)
        return model
    except FileNotFoundError:
        logger.warning("ETA model not found at %s", config.best_eta_model_path)
        return None


def get_eta_model(request: Request) -> Optional[object]:
    """FastAPI dependency: return the ETA model from ``app.state``.

    Usage:
        .. code-block:: python

            @router.post(...)
            async def predict(body: ETARequest, model = Depends(get_eta_model)):
                ...

    Returns:
        The loaded ETA model Pipeline, or ``None`` if not loaded.
    """
    return getattr(request.app.state, "eta_model", None)


@router.post("", response_model=ETAResponse)
@limiter.limit("30/minute")
async def predict_eta(
    request: Request,
    body: ETARequest,
    model: Optional[object] = Depends(get_eta_model),
) -> ETAResponse:
    """Predict delivery time for an order.

    Args:
        request: Incoming HTTP request (required by rate limiter).
        body: ETARequest with distance, traffic, festival, etc.
        model: The loaded ETA model (injected via ``Depends``).

    Returns:
        ETAResponse with predicted minutes and SLA risk.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ETA model not loaded. Run `make train` first.",
        )

    # Check cache first
    cache = get_cache(config)
    cache_key = cache.make_eta_key(body.model_dump())
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("ETA cache hit for key=%s", cache_key)
        return ETAResponse(**cached)

    # Build the full feature vector matching the training pipeline.
    # Previously this only had 6 features — the model was trained on ~20.
    # Reusing build_eta_features_for_api() from delivery_features.py
    # ensures the serving endpoint stays in sync with the training pipeline.
    features = build_eta_features_for_api(
        distance_km=body.distance_km,
        traffic_level=body.traffic_level,
        is_festival=body.is_festival,
        delivery_person_age=body.delivery_person_age or 30.0,
        delivery_person_rating=body.delivery_person_rating or 4.0,
        vehicle_condition=body.vehicle_condition or 1,
    )

    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    is_at_risk = prediction > config.sla_threshold_minutes

    response = ETAResponse(
        predicted_minutes=round(float(prediction), 1),
        is_at_risk=is_at_risk,
        sla_threshold=config.sla_threshold_minutes,
    )

    # Cache the result
    cache.set(cache_key, response.model_dump(), ttl_seconds=config.cache_eta_ttl_seconds)

    return response
