"""Recommendation Narrator — generates plain-English explanations for
ranked restaurant recommendations.

PRIMARY MODE: Uses Anthropic Claude API to generate natural-language
  explanations from restaurant features, reliability score components,
  and sentiment summary.

FALLBACK MODE: Template-based rules when LLM is unavailable or
  not configured — the app never breaks without a key.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)

# ─── Anthropic client (lazy-loaded) ─────────────────────────────────────

_anthropic_client = None


def _get_anthropic_client(config: DabbaConfig):
    """Initialize Anthropic client if API key is configured."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    if not config.anthropic_api_key:
        return None
    try:
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        logger.info("Anthropic client initialized")
        return _anthropic_client
    except Exception as e:
        logger.warning("Failed to initialize Anthropic client: %s", e)
        return None


# ─── LLM-powered narration ──────────────────────────────────────────────


def _llm_narrate(
    restaurant: Dict[str, Any],
    reliability_score: float,
    sentiment_avg: float,
    eta_prediction: Optional[float],
    config: DabbaConfig,
) -> Optional[str]:
    """Generate a natural-language recommendation explanation via Claude."""
    client = _get_anthropic_client(config)
    if client is None:
        return None

    prompt = (
        f"You are a friendly food recommendation assistant. "
        f"Explain in 2-3 sentences why this restaurant is being recommended. "
        f"Be specific, natural, and helpful — mention cuisine, price, "
        f"quality rating, delivery reliability, and customer sentiment.\n\n"
        f"Restaurant: {restaurant.get('name', 'Unknown')}\n"
        f"Cuisine: {restaurant.get('cuisines', 'Various')}\n"
        f"Rating: {restaurant.get('rate', 'N/A')}/5\n"
        f"Cost for two: ₹{restaurant.get('cost_for_two', 'N/A')}\n"
        f"Area: {restaurant.get('location', 'N/A')}\n"
        f"Customer Sentiment: {sentiment_avg:.2f}/1.0\n"
        f"Reliability Score: {reliability_score:.2f}/1.0\n"
    )
    if eta_prediction is not None:
        prompt += f"Predicted delivery time: {eta_prediction:.0f} min\n"
    prompt += "\nWrite a short recommendation explanation:"

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        logger.info("LLM narration generated for %s", restaurant.get("name"))
        return text
    except Exception as e:
        logger.warning("LLM narration failed for %s: %s", restaurant.get("name"), e)
        return None


# ─── Rules-based fallback narration ─────────────────────────────────────


def _rules_narrate(
    restaurant: Dict[str, Any],
    reliability_score: float,
    sentiment_avg: float,
    eta_prediction: Optional[float],
) -> str:
    """Generate a template-based recommendation explanation."""
    name = restaurant.get("name", "This restaurant")
    cuisine = restaurant.get("cuisines", "a variety of dishes")
    rating = restaurant.get("rate", "N/A")
    cost = restaurant.get("cost_for_two", "N/A")
    location = restaurant.get("location", "your area")

    parts = [f"**{name}** — {cuisine} in **{location}**"]

    try:
        rating_f = float(rating)
        if rating_f >= 4.0:
            parts.append(f"Highly rated ({rating_f}/5)")
        elif rating_f >= 3.0:
            parts.append(f"Decent rating ({rating_f}/5)")
        else:
            parts.append(f"Average rating ({rating_f}/5)")
    except (ValueError, TypeError):
        pass

    try:
        cost_f = float(cost)
        if cost_f <= 300:
            parts.append("budget-friendly")
        elif cost_f <= 800:
            parts.append("moderately priced")
        else:
            parts.append("premium")
    except (ValueError, TypeError):
        pass

    if sentiment_avg > 0.3:
        parts.append("positive customer sentiment")
    elif sentiment_avg < -0.2:
        parts.append("mixed customer reviews")
    else:
        parts.append("neutral customer sentiment")

    if reliability_score >= 0.7:
        parts.append("high overall reliability")
    elif reliability_score >= 0.4:
        parts.append("moderate reliability")
    else:
        parts.append("lower reliability — consider alternatives")

    if eta_prediction is not None:
        parts.append(f"~{eta_prediction:.0f} min delivery estimate")

    return " · ".join(parts) + "."


# ─── Public API ─────────────────────────────────────────────────────────


def narrate_recommendation(
    restaurant: Dict[str, Any],
    reliability_score: float,
    sentiment_avg: float = 0.0,
    eta_prediction: Optional[float] = None,
    config: Optional[DabbaConfig] = None,
) -> str:
    """Generate a natural-language recommendation explanation.

    Tries LLM first (if configured), falls back to template-based rules.

    Args:
        restaurant: Restaurant data dict (name, cuisines, rate, cost, location).
        reliability_score: Reliability score in [0, 1].
        sentiment_avg: Average customer sentiment score.
        eta_prediction: Predicted delivery time in minutes.
        config: Project configuration.

    Returns:
        str: Human-readable explanation for the recommendation.
    """
    config = config or get_config()

    # Try LLM
    if config.llm_enabled:
        llm_text = _llm_narrate(
            restaurant, reliability_score, sentiment_avg, eta_prediction, config
        )
        if llm_text:
            return llm_text

    # Fallback to rules
    return _rules_narrate(restaurant, reliability_score, sentiment_avg, eta_prediction)
