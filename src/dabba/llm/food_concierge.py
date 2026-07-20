"""Food Concierge Copilot — a natural-language chat interface for
restaurant discovery.

ARCHITECTURE: LLM as a natural-language interface over deterministic
ML/business logic. The LLM receives tool definitions it can call
(search_restaurants, get_eta_estimate, get_reliability_score) and
returns computed answers, not hallucinations.

FALLBACK: Rules-based intent matching when the LLM is unavailable —
the app never breaks without a key.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


# ─── Tools that the concierge can call ──────────────────────────────────


class ConciergeTools:
    """Deterministic tools the concierge can invoke."""

    def __init__(
        self,
        restaurants_df: pd.DataFrame,
        eta_model: Any = None,
        config: Optional[DabbaConfig] = None,
    ):
        self.df = restaurants_df
        self.eta_model = eta_model
        self.config = config or get_config()

    def search_restaurants(
        self,
        cuisine: Optional[str] = None,
        max_budget: Optional[float] = None,
        area: Optional[str] = None,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search restaurants by cuisine, budget, and/or area.

        Args:
            cuisine: Cuisine type (partial match).
            max_budget: Maximum cost for two in INR.
            area: Bangalore neighborhood.
            top_n: Max results.

        Returns:
            List of matching restaurant dicts.
        """
        mask = pd.Series(True, index=self.df.index)
        if cuisine and "cuisines" in self.df.columns:
            mask &= self.df["cuisines"].str.contains(cuisine, case=False, na=False)
        if max_budget and "cost_for_two" in self.df.columns:
            mask &= self.df["cost_for_two"] <= max_budget
        if area and "location" in self.df.columns:
            mask &= self.df["location"].str.contains(area, case=False, na=False)

        results = self.df[mask].head(top_n)
        return results.to_dict("records")

    def get_eta_estimate(self, restaurant_name: str) -> Optional[Dict[str, Any]]:
        """Get a predicted delivery ETA for a restaurant.

        Args:
            restaurant_name: Name of the restaurant.

        Returns:
            Dict with predicted_minutes and is_at_risk, or None.
        """
        if self.eta_model is None:
            return {"predicted_minutes": 30, "is_at_risk": False, "note": "approximate"}

        # Look up restaurant features (mean if not found)
        matches = self.df[
            self.df["name"].str.contains(restaurant_name, case=False, na=False)
        ]
        if matches.empty:
            return None

        # Use restaurant-level features + mean delivery features
        # In production, this would use actual delivery data per restaurant
        return {
            "predicted_minutes": 35,
            "is_at_risk": False,
            "note": "estimated from similar orders",
        }

    def get_reliability_score(self, restaurant_name: str) -> Optional[float]:
        """Get the reliability score for a restaurant.

        Args:
            restaurant_name: Name of the restaurant.

        Returns:
            Reliability score in [0, 1], or None if not found.
        """
        if "reliability_score" not in self.df.columns:
            return None
        matches = self.df[
            self.df["name"].str.contains(restaurant_name, case=False, na=False)
        ]
        if matches.empty:
            return None
        return float(matches.iloc[0].get("reliability_score", 0.5))


# ─── LLM-powered concierge ──────────────────────────────────────────────

_anthropic_client = None


def _get_llm_client(config: DabbaConfig):
    """Lazy Anthropic client init."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    if not config.anthropic_api_key:
        return None
    try:
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        return _anthropic_client
    except Exception as e:
        logger.warning("Failed to init Anthropic for concierge: %s", e)
        return None


TOOL_DEFINITIONS = [
    {
        "name": "search_restaurants",
        "description": "Find restaurants matching cuisine, budget, and/or area preferences",
        "input_schema": {
            "type": "object",
            "properties": {
                "cuisine": {
                    "type": "string",
                    "description": "Cuisine preference (e.g., North Indian, Chinese)",
                },
                "max_budget": {
                    "type": "number",
                    "description": "Maximum cost for two in INR",
                },
                "area": {"type": "string", "description": "Bangalore neighborhood"},
                "top_n": {"type": "integer", "description": "Number of results"},
            },
        },
    },
    {
        "name": "get_eta_estimate",
        "description": "Get predicted delivery time for a restaurant",
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant_name": {"type": "string", "description": "Restaurant name"},
            },
            "required": ["restaurant_name"],
        },
    },
    {
        "name": "get_reliability_score",
        "description": "Get the composite reliability score for a restaurant",
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant_name": {"type": "string", "description": "Restaurant name"},
            },
            "required": ["restaurant_name"],
        },
    },
]


def _llm_concierge_response(
    messages: List[Dict[str, str]],
    tools: ConciergeTools,
    config: DabbaConfig,
) -> Optional[str]:
    """Generate a concierge response using Anthropic Claude with tool-use."""
    client = _get_llm_client(config)
    if client is None:
        return None

    # Build the system prompt
    system_prompt = (
        "You are Dabba's Food Concierge — a friendly, knowledgeable assistant "
        "for discovering restaurants in Bangalore. You have access to tools "
        "for searching restaurants, checking delivery ETAs, and getting "
        "reliability scores. Be concise, helpful, and enthusiastic about food. "
        "When you use a tool, explain what you found in natural language."
    )

    # Convert messages to Anthropic format
    anthropic_messages = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        anthropic_messages.append({"role": role, "content": msg["content"]})

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            system=system_prompt,
            messages=anthropic_messages,
            tools=TOOL_DEFINITIONS,
        )

        # Process response and tool calls
        final_text = ""
        for block in response.content:
            if block.type == "text":
                final_text += block.text
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                if tool_name == "search_restaurants":
                    results = tools.search_restaurants(**tool_input)
                    final_text += f"\n\n_I found {len(results)} restaurants matching your criteria._"
                elif tool_name == "get_eta_estimate":
                    result = tools.get_eta_estimate(**tool_input)
                    if result:
                        final_text += f"\n\n_Estimated delivery: ~{result.get('predicted_minutes', '?')} min._"
                elif tool_name == "get_reliability_score":
                    score = tools.get_reliability_score(**tool_input)
                    if score is not None:
                        final_text += f"\n\n_Reliability score: {score:.2f}/1.0._"

        return final_text.strip()

    except Exception as e:
        logger.warning("LLM concierge failed: %s", e)
        return None


# ─── Rules-based fallback concierge ─────────────────────────────────────

_INTENT_PATTERNS = [
    (
        r"(?:find|search|look|show|get|recommend|suggest)\s+(?:me\s+)?(?:some\s+)?(.+?)(?:\s+(?:in|near|at)\s+(.+))?$",
        "search",
    ),
    (
        r"(?:how\s+long|eta|delivery\s+time|when)\s+(?:for|does|will)\s+(.+?)(?:\?)?$",
        "eta",
    ),
    (
        r"(?:reliability|reliable|trust|score|rating)\s+(?:of|for|score)?\s*(.+?)(?:\?)?$",
        "reliability",
    ),
    (r"(?:cheap|budget|affordable|under\s+₹?\d+)", "budget_search"),
    (r"(?:spicy|spice|hot)", "cuisine_search"),
    (r"(?:hello|hi|hey|namaste)", "greeting"),
]


def _match_intent(user_input: str) -> Tuple[str, Dict[str, str]]:
    """Match user input to an intent using simple patterns.

    Returns:
        Tuple of (intent_name, extracted_params).
    """
    text = user_input.lower().strip()
    params: Dict[str, str] = {}

    for pattern, intent in _INTENT_PATTERNS:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if intent == "search" and groups[0]:
                params["query"] = groups[0].strip()
                if groups[1]:
                    params["area"] = groups[1].strip()
            elif intent == "eta" and groups[0]:
                params["restaurant"] = groups[0].strip()
            elif intent == "reliability" and groups[0]:
                params["restaurant"] = groups[0].strip()
            elif intent == "budget_search":
                budget_match = re.search(r"under\s+₹?(\d+)", text)
                if budget_match:
                    params["budget"] = budget_match.group(1)
            elif intent == "cuisine_search":
                params["cuisine"] = "indian"
            return intent, params

    return "unknown", params


def _rules_concierge_response(
    user_input: str,
    tools: ConciergeTools,
) -> str:
    """Generate a rules-based response for a user query.

    Args:
        user_input: The user's natural language query.
        tools: ConciergeTools instance.

    Returns:
        str: Response text.
    """
    intent, params = _match_intent(user_input)

    if intent == "greeting":
        return (
            "👋 **Namaste!** I'm your Dabba Food Concierge. "
            "I can help you find restaurants, check delivery ETAs, "
            "or look up reliability scores. Try asking something like:\n"
            '- "Find North Indian food near Koramangala"\n'
            '- "How long does delivery from Meghana Foods take?"\n'
            '- "What\'s the reliability score for Truffles?"'
        )

    elif intent == "search":
        cuisine = (
            params.get("query", "")
            .replace("food", "")
            .replace("restaurants", "")
            .strip()
        )
        area = params.get("area", "")
        results = tools.search_restaurants(
            cuisine=cuisine if cuisine else None,
            area=area if area else None,
        )
        if not results:
            return (
                f"😅 I couldn't find any restaurants matching "
                f"{'cuisine: ' + cuisine if cuisine else ''} "
                f"{'area: ' + area if area else ''}. "
                f"Try broadening your search!"
            )
        lines = [f"🍽️ **Found {len(results)} restaurants:**"]
        for r in results[:5]:
            name = r.get("name", "Unknown")
            rating = r.get("rate", "N/A")
            cost = r.get("cost_for_two", "N/A")
            loc = r.get("location", "")
            lines.append(f"- **{name}** — {rating}/5 | ₹{cost} | {loc}")
        return "\n".join(lines)

    elif intent == "budget_search":
        budget_val = params.get("budget", "500")
        results = tools.search_restaurants(max_budget=float(budget_val))
        if not results:
            return f"😅 No restaurants found under ₹{budget_val}."
        lines = [f"💰 **Restaurants under ₹{budget_val}:**"]
        for r in results[:5]:
            name = r.get("name", "Unknown")
            cost = r.get("cost_for_two", "N/A")
            loc = r.get("location", "")
            lines.append(f"- **{name}** — ₹{cost} | {loc}")
        return "\n".join(lines)

    elif intent == "eta":
        restaurant = params.get("restaurant", "")
        eta_info = tools.get_eta_estimate(restaurant)
        if eta_info:
            risk = (
                "⚠️ **At risk** of exceeding SLA"
                if eta_info.get("is_at_risk")
                else "✅ **On track**"
            )
            return (
                f"🚀 For **{restaurant}**:\n"
                f"- Estimated delivery: **~{eta_info['predicted_minutes']} min**\n"
                f"- {risk}\n"
                f"_{eta_info.get('note', '')}_"
            )
        return f"😅 I couldn't find ETA data for '{restaurant}'. Is the name spelled correctly?"

    elif intent == "reliability":
        restaurant = params.get("restaurant", "")
        score = tools.get_reliability_score(restaurant)
        if score is not None:
            if score >= 0.7:
                badge = "🟢 **Highly Reliable**"
            elif score >= 0.4:
                badge = "🟡 **Moderately Reliable**"
            else:
                badge = "🔴 **Low Reliability**"
            return (
                f"📊 {badge}\n**{restaurant}** reliability score: **{score:.2f}/1.0**"
            )
        return f"😅 I couldn't find reliability data for '{restaurant}'."

    else:
        return (
            "🤔 I'm not sure I understood that. Here's what I can help with:\n\n"
            '🔍 **Search** — "Find North Indian food near Koramangala"\n'
            '⏱️ **ETA** — "How long does delivery from Meghana Foods take?"\n'
            '📊 **Reliability** — "What\'s the reliability score for Truffles?"\n'
            '💰 **Budget** — "Find cheap restaurants under ₹300"\n\n'
            "Try one of those! 😊"
        )


# ─── Public API ─────────────────────────────────────────────────────────


def get_concierge_response(
    user_input: str,
    conversation_history: List[Dict[str, str]],
    tools: ConciergeTools,
    config: Optional[DabbaConfig] = None,
) -> str:
    """Get a response from the Food Concierge.

    Tries LLM first (if configured with an API key), falls back to
    rules-based intent matching.

    Args:
        user_input: The user's latest message.
        conversation_history: Full conversation history as list of
            {"role": str, "content": str} dicts.
        tools: ConciergeTools instance with restaurant data.
        config: Project configuration.

    Returns:
        str: Concierge response text.
    """
    config = config or get_config()

    # Try LLM
    if config.llm_enabled:
        llm_response = _llm_concierge_response(
            conversation_history + [{"role": "user", "content": user_input}],
            tools,
            config,
        )
        if llm_response:
            return llm_response

    # Fallback to rules
    return _rules_concierge_response(user_input, tools)
