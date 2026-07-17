"""Pydantic schemas for the Dabba FastAPI application.

Defines request and response models for all API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    rating_model_loaded: bool = False
    eta_model_loaded: bool = False


class ModelInfoResponse(BaseModel):
    """Deployed model information response."""

    rating_model: Optional[Dict[str, Any]] = None
    eta_model: Optional[Dict[str, Any]] = None


class RecommendRequest(BaseModel):
    """Restaurant recommendation request."""

    cuisine: Optional[str] = Field(None, description="Preferred cuisine")
    budget: Optional[float] = Field(None, description="Max cost for two (INR)")
    area: Optional[str] = Field(None, description="Area/neighborhood")
    top_n: int = Field(5, description="Number of recommendations")


class Recommendation(BaseModel):
    """A single restaurant recommendation."""

    name: str
    rating: Optional[float] = None
    bayesian_rating: Optional[float] = None
    cost_for_two: Optional[float] = None
    location: Optional[str] = None
    cuisines: Optional[str] = None
    similarity_score: Optional[float] = None
    combined_score: Optional[float] = None
    explanation: Optional[str] = None


class RecommendResponse(BaseModel):
    """Restaurant recommendation response."""

    recommendations: List[Recommendation] = []
    message: Optional[str] = None


class ETARequest(BaseModel):
    """Delivery ETA prediction request."""

    distance_km: float = Field(..., description="Haversine distance in km")
    traffic_level: int = Field(1, description="Traffic density (0=Low, 3=Jam)")
    is_festival: bool = Field(False, description="Whether it's a festival day")
    delivery_person_age: Optional[float] = Field(None, description="Delivery person age")
    delivery_person_rating: Optional[float] = Field(None, description="Delivery person rating")
    vehicle_condition: Optional[int] = Field(None, description="Vehicle condition score")


class ETAResponse(BaseModel):
    """Delivery ETA prediction response."""

    predicted_minutes: float
    is_at_risk: bool
    sla_threshold: float
