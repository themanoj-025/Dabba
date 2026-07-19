"""Pydantic schemas for the Dabba FastAPI application v3.

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


class RecommendRequest(BaseModel):
    """Restaurant recommendation request."""

    cuisine: Optional[str] = Field(None, description="Preferred cuisine")
    budget: Optional[float] = Field(None, description="Max cost for two (INR)")
    area: Optional[str] = Field(None, description="Area/neighborhood")
    top_n: int = Field(5, description="Number of recommendations")
    prioritize: Optional[str] = Field(
        "balanced", description="'balanced', 'speed', or 'quality'"
    )
    use_llm_narration: bool = Field(
        False, description="Generate LLM-powered explanation"
    )


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
    delivery_person_age: Optional[float] = Field(
        None, description="Delivery person age"
    )
    delivery_person_rating: Optional[float] = Field(
        None, description="Delivery person rating"
    )
    vehicle_condition: Optional[int] = Field(
        None, description="Vehicle condition score"
    )


class ETAResponse(BaseModel):
    """Delivery ETA prediction response."""

    predicted_minutes: float
    is_at_risk: bool
    sla_threshold: float


class ChatMessage(BaseModel):
    """A message in the chat history."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Food Concierge chat request."""

    message: str = Field(..., description="User's message")
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list, description="Conversation history"
    )


class ChatResponse(BaseModel):
    """Food Concierge chat response."""

    reply: str = Field(..., description="Concierge's reply")
