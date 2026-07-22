"""Model Info router — deployed model names and metrics."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Request

from api.limiter import limiter
from dabba.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model-info", tags=["model-info"])

config = get_config()


@router.get("")
@limiter.limit("60/minute")
async def model_info(request: Request):
    """Return which models are deployed and their metrics.

    Args:
        request: Incoming HTTP request (required by rate limiter).

    Returns:
        Dict with rating_model and eta_model info.
    """
    rating_winner = None
    eta_winner = None

    try:
        rating_df = pd.read_csv(config.rating_comparison_path)
        if not rating_df.empty:
            best = rating_df.iloc[0]
            rating_winner = {
                "model": str(best.get("model", "unknown")),
                "mae": float(best.get("mae", 0)),
                "rmse": float(best.get("rmse", 0)),
                "r2": float(best.get("r2", 0)),
            }
    except FileNotFoundError:
        pass

    try:
        eta_df = pd.read_csv(config.eta_comparison_path)
        if not eta_df.empty:
            best = eta_df.iloc[0]
            eta_winner = {
                "model": str(best.get("model", "unknown")),
                "mae": float(best.get("mae", 0)),
                "rmse": float(best.get("rmse", 0)),
                "r2": float(best.get("r2", 0)),
            }
    except FileNotFoundError:
        pass

    return {"rating_model": rating_winner, "eta_model": eta_winner}
