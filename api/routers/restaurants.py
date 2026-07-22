"""Restaurants router — read from the database (proves CSV→DB migration).

Provides endpoints for listing, searching, and inspecting restaurants
that have been seeded into the database. These endpoints demonstrate
that the CSV→DB migration is complete by reading from Postgres/SQLite
instead of the raw CSV files.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from api.limiter import limiter
from api.schemas import RestaurantItem, RestaurantListResponse
from dabba.database.repositories import (
    count_restaurants,
    get_all_restaurants,
    get_restaurant_by_id,
    get_restaurant_by_name,
    get_restaurants_by_cuisine,
)
from dabba.database.session import get_db_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/restaurants", tags=["restaurants"])


@router.get("", response_model=RestaurantListResponse)
@limiter.limit("60/minute")
async def list_restaurants(
    request: Request,
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db_generator),
) -> RestaurantListResponse:
    """List all restaurants from the database (paginated).

    This endpoint proves the CSV→DB migration by reading from
    the database instead of CSV files.

    Args:
        request: Incoming HTTP request (rate limiter).
        limit: Max results per page.
        offset: Pagination offset.
        db: Database session (injected).

    Returns:
        Paginated list of restaurants.
    """
    restaurants = get_all_restaurants(db, limit=limit, offset=offset)
    total = count_restaurants(db)

    return RestaurantListResponse(
        restaurants=[
            RestaurantItem(
                id=r.id,
                name=r.name,
                rate=r.rate,
                bayesian_rating=r.bayesian_rating,
                cost_for_two=r.cost_for_two,
                location=r.location,
                cuisines=r.cuisines,
                votes=r.votes,
                reliability_score=r.reliability_score,
            )
            for r in restaurants
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{restaurant_id}", response_model=Optional[RestaurantItem])
@limiter.limit("60/minute")
async def get_restaurant(
    request: Request,
    restaurant_id: int,
    db: Session = Depends(get_db_generator),
) -> RestaurantItem:
    """Get a single restaurant by ID.

    Args:
        request: Incoming HTTP request.
        restaurant_id: Restaurant primary key.
        db: Database session.

    Returns:
        RestaurantItem for the requested restaurant.

    Raises:
        HTTPException 404: Restaurant not found.
    """
    r = get_restaurant_by_id(db, restaurant_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Restaurant not found")

    return RestaurantItem(
        id=r.id,
        name=r.name,
        rate=r.rate,
        bayesian_rating=r.bayesian_rating,
        cost_for_two=r.cost_for_two,
        location=r.location,
        cuisines=r.cuisines,
        votes=r.votes,
        reliability_score=r.reliability_score,
    )


@router.get("/search/{query}", response_model=RestaurantListResponse)
@limiter.limit("60/minute")
async def search_restaurants(
    request: Request,
    query: str,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db_generator),
) -> RestaurantListResponse:
    """Search restaurants by name or cuisine.

    Args:
        request: Incoming HTTP request.
        query: Search term (matches name or cuisine).
        limit: Max results.
        db: Database session.

    Returns:
        Matching restaurants.
    """
    # Try name match first, then cuisine
    by_name = get_restaurant_by_name(db, query)
    if by_name:
        return RestaurantListResponse(
            restaurants=[
                RestaurantItem(
                    id=by_name.id,
                    name=by_name.name,
                    rate=by_name.rate,
                    bayesian_rating=by_name.bayesian_rating,
                    cost_for_two=by_name.cost_for_two,
                    location=by_name.location,
                    cuisines=by_name.cuisines,
                    votes=by_name.votes,
                    reliability_score=by_name.reliability_score,
                )
            ],
            total=1,
            limit=limit,
            offset=0,
        )

    by_cuisine = get_restaurants_by_cuisine(db, query, limit=limit)
    if not by_cuisine:
        raise HTTPException(status_code=404, detail=f"No restaurants found matching '{query}'")

    return RestaurantListResponse(
        restaurants=[
            RestaurantItem(
                id=r.id,
                name=r.name,
                rate=r.rate,
                bayesian_rating=r.bayesian_rating,
                cost_for_two=r.cost_for_two,
                location=r.location,
                cuisines=r.cuisines,
                votes=r.votes,
                reliability_score=r.reliability_score,
            )
            for r in by_cuisine
        ],
        total=len(by_cuisine),
        limit=limit,
        offset=0,
    )
