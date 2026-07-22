"""Recommendation router — hybrid recommendations with optional LLM narration.

Models are loaded at app startup and stored in ``app.state``,
then injected via FastAPI ``Depends()`` — no module-level globals.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.limiter import limiter
from dabba.cache.redis_client import get_cache
from dabba.config import get_config
from dabba.llm.recommendation_narrator import narrate_recommendation
from dabba.models.hybrid_recommender import HybridRecommender
from api.schemas import RecommendRequest, RecommendResponse, Recommendation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["recommend"])

config = get_config()


def _load_hybrid_recommender() -> Optional[HybridRecommender]:
    """Build and return the hybrid recommender from processed data.

    Called once at app startup by ``api.main``. Returns ``None`` if
    the processed restaurant data hasn't been generated yet.

    Returns:
        HybridRecommender instance, or None.
    """
    data_path = config.data_processed_dir / "restaurants_processed.csv"
    if not data_path.exists():
        logger.warning("Processed data not found at %s", data_path)
        return None

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

    recommender = HybridRecommender(df, feature_cols, config=config)
    logger.info("Hybrid recommender loaded with %d restaurants", len(df))
    return recommender


def get_recommender(request: Request) -> Optional[HybridRecommender]:
    """FastAPI dependency: return the hybrid recommender from ``app.state``.

    Usage:
        .. code-block:: python

            @router.post(...)
            async def recommend(body: Request, recommender = Depends(get_recommender)):
                ...

    Returns:
        The HybridRecommender instance, or ``None`` if not loaded.
    """
    return getattr(request.app.state, "hybrid_recommender", None)


@router.post("", response_model=RecommendResponse)
@limiter.limit("30/minute")
async def recommend(
    request: Request,
    body: RecommendRequest,
    recommender: Optional[HybridRecommender] = Depends(get_recommender),
) -> RecommendResponse:
    """Get hybrid restaurant recommendations with optional LLM narration.

    Args:
        request: Incoming HTTP request (required by rate limiter).
        body: RecommendRequest with cuisine, budget, area, top_n, etc.
        recommender: The hybrid recommender (injected via ``Depends``).

    Returns:
        RecommendResponse with ranked recommendations.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not loaded. Run `make train` first.",
        )

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

    results = recommender.recommend(
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
