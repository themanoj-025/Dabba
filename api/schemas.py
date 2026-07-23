"""Pydantic schemas for the Dabba FastAPI application v3.

Defines request and response models for all API endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    rating_model_loaded: bool = False
    eta_model_loaded: bool = False


class RecommendRequest(BaseModel):
    """Restaurant recommendation request."""

    cuisine: Optional[str] = Field(None, description="Preferred cuisine")
    budget: Optional[float] = Field(None, ge=0, description="Max cost for two (INR)")
    area: Optional[str] = Field(None, description="Area/neighborhood")
    top_n: int = Field(5, ge=1, le=50, description="Number of recommendations")
    prioritize: Optional[str] = Field(
        "balanced", description="'balanced', 'speed', or 'quality'"
    )
    use_llm_narration: bool = Field(
        False, description="Generate LLM-powered explanation"
    )

    @field_validator("prioritize")
    @classmethod
    def validate_prioritize(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("balanced", "speed", "quality"):
            raise ValueError("prioritize must be 'balanced', 'speed', or 'quality'")
        return v


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

    distance_km: float = Field(
        ..., gt=0, le=100, description="Haversine distance in km (0-100)"
    )
    traffic_level: int = Field(
        1, ge=0, le=3, description="Traffic density (0=Low, 1=Medium, 2=High, 3=Jam)"
    )
    is_festival: bool = Field(False, description="Whether it's a festival day")
    delivery_person_age: Optional[float] = Field(
        None, ge=18, le=70, description="Delivery person age (18-70)"
    )
    delivery_person_rating: Optional[float] = Field(
        None, ge=1.0, le=5.0, description="Delivery person rating (1.0-5.0)"
    )
    vehicle_condition: Optional[int] = Field(
        None, ge=0, le=3, description="Vehicle condition score (0-3)"
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

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    """Food Concierge chat request."""

    message: str = Field(
        ..., min_length=1, max_length=2000, description="User's message (1-2000 chars)"
    )
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list, description="Conversation history"
    )


class RestaurantItem(BaseModel):
    """A single restaurant returned from the database."""

    id: int
    name: str
    rate: Optional[float] = None
    bayesian_rating: Optional[float] = None
    cost_for_two: Optional[float] = None
    location: Optional[str] = None
    cuisines: Optional[str] = None
    votes: Optional[int] = None
    reliability_score: Optional[float] = None


class RestaurantListResponse(BaseModel):
    """Paginated list of restaurants from the database."""

    restaurants: List[RestaurantItem] = []
    total: int = 0
    limit: int = 50
    offset: int = 0


class ChatResponse(BaseModel):
    """Food Concierge chat response."""

    reply: str = Field(..., description="Concierge's reply")


class ExplainResponse(BaseModel):
    """Model prediction explanation response.

    Returns the stored SHAP values alongside the prediction details
    for a single inference request, enabling the ``/v1/explain``
    endpoint to close the explainability loop.
    """

    id: int
    model_name: str
    model_version: Optional[str] = None
    input_data: Optional[dict] = None
    output_value: float
    shap_values: Optional[dict] = None
    created_at: str
