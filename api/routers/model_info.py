"""Model Info router — deployed model names and metrics.

Reads from the ``ExperimentResult`` database table (seeded by the training
pipeline) instead of comparison CSVs, completing the CSV→DB migration.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from api.limiter import limiter
from dabba.database.repositories import get_winning_model
from dabba.database.session import get_db_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model-info", tags=["model-info"])


@router.get("")
@limiter.limit("60/minute")
async def model_info(
    request: Request,
    db: Session = Depends(get_db_generator),
) -> dict:
    """Return which models are deployed and their metrics.

    Reads from the ExperimentResult table which stores data from the
    last training run, seeded by the pipeline.

    Args:
        request: Incoming HTTP request (required by rate limiter).
        db: Database session (injected).

    Returns:
        Dict with rating_model and eta_model info.
    """
    rating_winner: Optional[dict] = None
    eta_winner: Optional[dict] = None

    rating_result = get_winning_model(db, "rating")
    if rating_result is not None:
        rating_winner = {
            "model": rating_result.model_name,
            "mae": rating_result.mae,
            "rmse": rating_result.rmse,
            "r2": rating_result.r2,
        }

    eta_result = get_winning_model(db, "eta")
    if eta_result is not None:
        eta_winner = {
            "model": eta_result.model_name,
            "mae": eta_result.mae,
            "rmse": eta_result.rmse,
            "r2": eta_result.r2,
        }

    return {"rating_model": rating_winner, "eta_model": eta_winner}
