"""Recommendation router — hybrid recommendations with optional LLM narration."""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.limiter import limiter
from dabba.cache.redis_client import get_cache
from dabba.config import get_config
from dabba.llm.recommendation_narrator import narrate_recommendation
from dabba.models.hybrid_recommender import HybridRecommender
from api.schemas import RecommendRequest, RecommendResponse, Recommendation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["recommend"])

config = get_config()

_hybrid_recommender: Optional[HybridRecommender] = None
_hybrid_recommender_lock = threading.Lock()


def load_recommender() -> None:
    """Load or reload the hybrid recommender (thread-safe)."""
    global _hybrid_recommender
    with _hybrid_recommender_lock:
        if _hybrid_recommender is not None:
            return
        data_path = config.data_processed_dir / "restaurants_processed.csv"
        if not data_path.exists():
            logger.warning("Processed data not found at %s", data_path)
            return

        df = pd.read_csv(data_path)
        feature_cols = [c for c in df.columns if c.startswith("cuisine_")]
        feature_cols += [
            c
            for c in [
                "votes_log",
                "cost_for_two",
                "online_order_binary",
                "book_table_binary",
                "cuisine_count",
                "avg_sentiment",
            ]
            if c in df.columns
        ]

        _hybrid_recommender = HybridRecommender(df, feature_cols, config=config)
        logger.info("Hybrid recommender loaded with %d restaurants", len(df))


def get_recommender() -> Optional[HybridRecommender]:
    """Thread-safe accessor for the hybrid recommender."""
    global _hybrid_recommender
    with _hybrid_recommender_lock:
        return _hybrid_recommender


@router.post("", response_model=RecommendResponse)
@limiter.limit("30/minute")
async def recommend(request: Request, body: RecommendRequest) -> RecommendResponse:
    """Get hybrid restaurant recommendations with optional LLM narration.

    Args:
        request: Incoming HTTP request (required by rate limiter).
        body: RecommendRequest with cuisine, budget, area, top_n, etc.

    Returns:
        RecommendResponse with ranked recommendations.
    """
    # Check cache first (skip for LLM-narrated requests since explanations vary)
    cache = get_cache(config)
    cache_key = cache.make_recommend_key(
        {**body.model_dump(), "use_llm_narration": False}
    )
    if not body.use_llm_narration:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Recommend cache hit for key=%s", cache_key)
            return RecommendResponse(**cached)

    model = get_recommender()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not loaded. Run `make train` first.",
        )

    results = model.recommend(
        cuisine=body.cuisine,
        budget=body.budget,
        area=body.area,
        prioritize=body.prioritize or "balanced",
        top_n=body.top_n,
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
        if body.use_llm_narration and config.llm_enabled:
            rs = rest_dict.get("reliability_score_display", 0.5)
            sentiment = rest_dict.get("avg_sentiment", 0.0)
            explanation = narrate_recommendation(
                rest_dict,
                rs,
                sentiment_avg=sentiment,
                config=config,
            )

        rec = Recommendation(
            name=str(row.get("name", "Unknown")),
            rating=(
                float(row["rate"]) if "rate" in row and pd.notna(row["rate"]) else None
            ),
            bayesian_rating=(
                float(row["bayesian_rating"]) if "bayesian_rating" in row else None
            ),
            cost_for_two=(
                float(row["cost_for_two"])
                if "cost_for_two" in row and pd.notna(row["cost_for_two"])
                else None
            ),
            location=str(row.get("location", "")) or None,
            cuisines=str(row.get("cuisines", "")) or None,
            combined_score=(
                float(row["combined_score"]) if "combined_score" in row else None
            ),
            explanation=explanation,
        )
        recommendations.append(rec)

    response = RecommendResponse(recommendations=recommendations)

    # Cache the result (only non-LLM versions)
    if not body.use_llm_narration:
        cache.set(
            cache_key,
            response.model_dump(),
            ttl_seconds=config.cache_recommend_ttl_seconds,
        )

    return response
