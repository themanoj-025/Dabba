"""Pydantic schemas for the Dabba FastAPI application v3.

Defines request and response models for all API endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    rating_model_loaded: bool = False
    eta_model_loaded: bool = False


class RecommendRequest(BaseModel):
    """Restaurant recommendation request."""

    cuisine: str | None = Field(None, description="Preferred cuisine")
    budget: float | None = Field(None, ge=0, description="Max cost for two (INR)")
    area: str | None = Field(None, description="Area/neighborhood")
    top_n: int = Field(5, ge=1, le=50, description="Number of recommendations")
    prioritize: str | None = Field(
        "balanced", description="'balanced', 'speed', or 'quality'"
    )
    use_llm_narration: bool = Field(
        False, description="Generate LLM-powered explanation"
    )

    @field_validator("prioritize")
    @classmethod
    def validate_prioritize(cls, v: str | None) -> str | None:
        if v is not None and v not in ("balanced", "speed", "quality"):
            raise ValueError("prioritize must be 'balanced', 'speed', or 'quality'")
        return v


class Recommendation(BaseModel):
    """A single restaurant recommendation."""

    name: str
    rating: float | None = None
    bayesian_rating: float | None = None
    cost_for_two: float | None = None
    location: str | None = None
    cuisines: str | None = None
    similarity_score: float | None = None
    combined_score: float | None = None
    explanation: str | None = None


class RecommendResponse(BaseModel):
    """Restaurant recommendation response."""

    recommendations: list[Recommendation] = []
    message: str | None = None


class ETARequest(BaseModel):
    """Delivery ETA prediction request."""

    distance_km: float = Field(
        ..., gt=0, le=100, description="Haversine distance in km (0-100)"
    )
    traffic_level: int = Field(
        1, ge=0, le=3, description="Traffic density (0=Low, 1=Medium, 2=High, 3=Jam)"
    )
    is_festival: bool = Field(False, description="Whether it's a festival day")
    delivery_person_age: float | None = Field(
        None, ge=18, le=70, description="Delivery person age (18-70)"
    )
    delivery_person_rating: float | None = Field(
        None, ge=1.0, le=5.0, description="Delivery person rating (1.0-5.0)"
    )
    vehicle_condition: int | None = Field(
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
    history: list[ChatMessage] | None = Field(
        default_factory=list, description="Conversation history"
    )


class RestaurantItem(BaseModel):
    """A single restaurant returned from the database."""

    id: int
    name: str
    rate: float | None = None
    bayesian_rating: float | None = None
    cost_for_two: float | None = None
    location: str | None = None
    cuisines: str | None = None
    votes: int | None = None
    reliability_score: float | None = None


class RestaurantListResponse(BaseModel):
    """Paginated list of restaurants from the database."""

    restaurants: list[RestaurantItem] = []
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
    model_version: str | None = None
    input_data: dict | None = None
    output_value: float
    shap_values: dict | None = None
    created_at: str
