"""Recommendation router — hybrid recommendations with optional LLM narration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from dabba.config import get_config
from dabba.llm.recommendation_narrator import narrate_recommendation
from dabba.models.hybrid_recommender import HybridRecommender
from api.schemas import RecommendRequest, RecommendResponse, Recommendation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["recommend"])

config = get_config()

_hybrid_recommender: Optional[HybridRecommender] = None


def load_recommender() -> None:
    """Load or reload the hybrid recommender."""
    global _hybrid_recommender
    data_path = config.data_processed_dir / "restaurants_processed.csv"
    if not data_path.exists():
        logger.warning("Processed data not found at %s", data_path)
        return

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c.startswith("cuisine_")]
    feature_cols += [c for c in [
        "votes_log", "cost_for_two", "online_order_binary",
        "book_table_binary", "cuisine_count", "avg_sentiment",
    ] if c in df.columns]

    _hybrid_recommender = HybridRecommender(df, feature_cols, config=config)
    logger.info("Hybrid recommender loaded with %d restaurants", len(df))


@router.post("", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Get hybrid restaurant recommendations with optional LLM narration.

    Args:
        request: RecommendRequest with cuisine, budget, area, top_n, etc.

    Returns:
        RecommendResponse with ranked recommendations.
    """
    if _hybrid_recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not loaded. Run `make train` first.",
        )

    results = _hybrid_recommender.recommend(
        cuisine=request.cuisine,
        budget=request.budget,
        area=request.area,
        prioritize=request.prioritize or "balanced",
        top_n=request.top_n,
    )

    if results.empty:
        return RecommendResponse(
            recommendations=[],
            message="No restaurants match your filters.",
        )

    recommendations: List[Recommendation] = []
    for _, row in results.iterrows():
        rest_dict = row.to_dict()

        # Generate narration if requested
        explanation = None
        if request.use_llm_narration and config.llm_enabled:
            rs = rest_dict.get("reliability_score_display", 0.5)
            explanation = narrate_recommendation(
                rest_dict, rs, config=config,
            )

        rec = Recommendation(
            name=str(row.get("name", "Unknown")),
            rating=float(row["rate"]) if "rate" in row and pd.notna(row["rate"]) else None,
            bayesian_rating=float(row["bayesian_rating"]) if "bayesian_rating" in row else None,
            cost_for_two=float(row["cost_for_two"]) if "cost_for_two" in row and pd.notna(row["cost_for_two"]) else None,
            location=str(row.get("location", "")) or None,
            cuisines=str(row.get("cuisines", "")) or None,
            combined_score=float(row["combined_score"]) if "combined_score" in row else None,
            explanation=explanation,
        )
        recommendations.append(rec)

    return RecommendResponse(recommendations=recommendations)
