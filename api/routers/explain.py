"""Explainability router — retrieve stored predictions with SHAP values.

Provides the ``/v1/explain/{prediction_id}`` endpoint that closes the
explainability loop by exposing ``predictions.shap_values`` (which
already exists in the database schema) through the API.

This is purely a read operation — SHAP values are computed during the
training pipeline (:func:`dabba.pipeline.compute_shap_explanations`)
and stored in the ``Predictions`` table.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.limiter import limiter
from api.schemas import ExplainResponse
from dabba.database.repositories import get_prediction_by_id
from dabba.database.session import get_db_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])


@router.get("/{prediction_id}", response_model=ExplainResponse)
@limiter.limit("60/minute")
async def explain_prediction(
    request: Request,
    prediction_id: int,
    db: Session = Depends(get_db_generator),
) -> ExplainResponse:
    """Retrieve a stored model prediction with SHAP explanations.

    Args:
        request: Incoming HTTP request (required by rate limiter).
        prediction_id: Primary key of the Prediction record.
        db: Database session (injected).

    Returns:
        ExplainResponse with prediction details, input data, output value,
        and SHAP feature importance values.

    Raises:
        HTTPException 404: Prediction not found.
    """
    prediction = get_prediction_by_id(db, prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction {prediction_id} not found.",
        )

    return ExplainResponse(
        id=prediction.id,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        input_data=prediction.input_data,
        output_value=prediction.output_value,
        shap_values=prediction.shap_values,
        created_at=prediction.created_at.isoformat(),
    )
