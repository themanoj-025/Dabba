"""Food Concierge Copilot — a natural-language chat interface for
restaurant discovery with a ReAct tool-use loop.

ARCHITECTURE:
    Multi-step ReAct agent (max 4 steps, configurable via
    ``DabbaConfig.llm_max_steps``). The LLM receives tool definitions
    it can call (search_restaurants, get_eta_estimate,
    get_reliability_score), and tool results are fed back into the
    conversation so the LLM can reason over them and decide whether
    to call more tools or give a final answer.

FLOW (per user message):
    1. Send conversation + user input (+ tool results from previous
       step) to Claude with tool definitions
    2. Claude returns text blocks (accumulated into final answer)
       and/or tool_use blocks
    3. If tool_use → execute tool → add tool_result to conversation
       → loop back to step 1 (max N steps)
    4. If no tool_use → break, return accumulated text

FALLBACK:
    Rules-based intent matching when the LLM is unavailable —
    the app never breaks without a key.

NOTE ON ETA: ``get_eta_estimate()`` now uses the real loaded ETA model
(via ``build_eta_features_for_api``) instead of a hardcoded 30-min stub.
The model is passed to ``ConciergeTools`` at construction time.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dabba.config import DabbaConfig, get_config
from dabba.features.delivery_features import build_eta_features_for_api
from dabba.features.geo import haversine_distance

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
        """Get a predicted delivery ETA for a restaurant using the real ETA model.

        Builds a full feature vector (matching the training pipeline's 20+ features)
        from the restaurant's data and current temporal context, then calls the
        loaded ETA model. Falls back to a reasonable estimate if no model is loaded.

        Args:
            restaurant_name: Name of the restaurant.

        Returns:
            Dict with predicted_minutes and is_at_risk, or None if restaurant not found.
        """
        # Check restaurant existence FIRST
        matches = self.df[
            self.df["name"].str.contains(restaurant_name, case=False, na=False)
        ]
        if matches.empty:
            return None

        restaurant = matches.iloc[0]

        # Estimate distance from the restaurant's location to a central delivery point
        # (Bangalore centroid ≈ 12.97, 77.59). Falls back to 5km if no coords available.
        lat = restaurant.get("latitude", restaurant.get("restaurant_latitude", None))
        lon = restaurant.get("longitude", restaurant.get("restaurant_longitude", None))

        if pd.notna(lat) and pd.notna(lon):
            distance_km = float(haversine_distance(lat, lon, 12.97, 77.59))
        else:
            distance_km = 5.0

        if self.eta_model is None:
            return {
                "predicted_minutes": round(max(10, distance_km * 3 + 15)),
                "is_at_risk": False,
                "note": "approximate (no model loaded)",
            }

        try:
            features = build_eta_features_for_api(
                distance_km=distance_km,
                traffic_level=1,  # default: Medium
                is_festival=False,
                delivery_person_age=30.0,
                delivery_person_rating=4.0,
                vehicle_condition=2,
            )

            prediction = float(self.eta_model.predict(features)[0])
            sla_threshold = self.config.sla_threshold_minutes

            return {
                "predicted_minutes": round(prediction, 1),
                "is_at_risk": prediction > sla_threshold,
                "note": "estimated from restaurant data",
            }
        except Exception as e:
            logger.warning(
                "ETA model prediction failed for '%s': %s — falling back to formula",
                restaurant_name,
                e,
            )
            return {
                "predicted_minutes": round(max(10, distance_km * 3 + 15)),
                "is_at_risk": False,
                "note": "fallback (model error)",
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


# ─── Tool execution helpers ──────────────────────────────────────────


def _execute_tool(tool_name: str, tool_input: Dict[str, Any], tools: ConciergeTools) -> str:
    """Execute a concierge tool and format the result as structured text for the LLM.

    The returned string is passed back to the LLM as a ``tool_result``
    content block, so it should be structured for machine reading
    (the LLM will rephrase it naturally).

    Args:
        tool_name: Name of the tool to execute.
        tool_input: Arguments dict for the tool.
        tools: ConciergeTools instance with data and models.

    Returns:
        Structured text result for the LLM to consume.
    """
    if tool_name == "search_restaurants":
        results = tools.search_restaurants(**tool_input)
        if not results:
            return "No restaurants found matching the criteria."
        lines = [f"Found {len(results)} restaurants:"]
        for r in results[:10]:
            name = r.get("name", "Unknown")
            rating = r.get("rate", "N/A")
            cost = r.get("cost_for_two", "N/A")
            cuisines = r.get("cuisines", "")
            location = r.get("location", "")
            lines.append(
                f"- {name} | Rating: {rating}/5 | ₹{cost} | {cuisines} | {location}"
            )
        return "\n".join(lines)

    elif tool_name == "get_eta_estimate":
        result = tools.get_eta_estimate(**tool_input)
        if result is None:
            return (
                f"Restaurant '{tool_input.get('restaurant_name', '')}' not found."
            )
        risk = "at risk of exceeding SLA" if result.get("is_at_risk") else "on track"
        note = result.get("note", "")
        eta = result.get("predicted_minutes", "?")
        return (
            f"ETA for {tool_input.get('restaurant_name', '')}: "
            f"~{eta} min ({risk}). {note}"
        )

    elif tool_name == "get_reliability_score":
        score = tools.get_reliability_score(**tool_input)
        if score is None:
            return (
                f"Reliability data not found for "
                f"'{tool_input.get('restaurant_name', '')}'."
            )
        return (
            f"Reliability score for {tool_input.get('restaurant_name', '')}: "
            f"{score:.2f}/1.0"
        )

    logger.warning("Unknown tool called: %s", tool_name)
    return f"Error: unknown tool '{tool_name}'."


# ─── LLM-powered concierge (ReAct loop) ─────────────────────────────

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
    """Generate a concierge response using Anthropic Claude with a ReAct tool loop.

    The ReAct loop works as follows:

        1. Send the accumulated conversation (including any tool results
           from previous steps) to Claude with tool definitions.
        2. Claude returns ``text`` blocks (accumulated into final answer)
           and/or ``tool_use`` blocks.
        3. If any ``tool_use`` → execute each tool → add a ``tool_result``
           content block to the conversation → loop back to step 1.
        4. If no ``tool_use`` → break, return accumulated text.

    The loop runs at most ``config.llm_max_steps`` iterations.

    Args:
        messages: Conversation history as list of
            ``{"role": str, "content": str}`` dicts.
        tools: ConciergeTools instance with restaurant data.
        config: Project configuration.

    Returns:
        The final response text, or ``None`` if the LLM call completely
        failed (triggers fallback in caller).
    """
    client = _get_llm_client(config)
    if client is None:
        return None

    system_prompt = (
        "You are Dabba's Food Concierge — a friendly, knowledgeable assistant "
        "for discovering restaurants in Bangalore. You have access to tools "
        "for searching restaurants, checking delivery ETAs, and getting "
        "reliability scores. Be concise, helpful, and enthusiastic about food.\n\n"
        "You can use MULTIPLE tools in sequence to answer complex questions. "
        "For example: first search for restaurants, then check the ETA and "
        "reliability of the top result. After you receive tool results, "
        "summarise them naturally for the user.\n\n"
        "When you use a tool, explain what you found. If the user's request "
        "needs multiple pieces of information, use the tools one at a time."
    )

    # Build initial Anthropic messages from conversation history
    anthropic_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        anthropic_messages.append({"role": role, "content": msg["content"]})

    max_steps = config.llm_max_steps
    final_text_parts: List[str] = []

    for step in range(1, max_steps + 1):
        logger.debug("Concierge ReAct step %d/%d", step, max_steps)

        try:
            response = client.messages.create(
                model=config.llm_model,
                max_tokens=config.llm_max_tokens,
                system=system_prompt,
                messages=anthropic_messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            logger.warning("LLM call failed at ReAct step %d: %s", step, e)
            break

        # Collect assistant content blocks (text + tool_use)
        assistant_content: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_tool_use = False

        for block in response.content:
            if block.type == "text":
                final_text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                has_tool_use = True
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                tool_calls.append(block)

        # Add the assistant's response (text + tool_use blocks) to conversation
        if assistant_content:
            anthropic_messages.append(
                {"role": "assistant", "content": assistant_content}
            )

        # If no tool was used, this is the final answer — break
        if not has_tool_use:
            break

        # Execute each tool and add tool_result content blocks
        tool_results: List[Dict[str, Any]] = []
        for block in tool_calls:
            result_text = _execute_tool(block.name, block.input, tools)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )

        # Tool results are sent as a user message with tool_result blocks
        anthropic_messages.append({"role": "user", "content": tool_results})

        if step >= max_steps:
            logger.info(
                "Reached max ReAct steps (%d) for concierge", max_steps
            )

    if not final_text_parts:
        return None

    return "\n".join(final_text_parts).strip()


# ─── Rules-based fallback concierge ─────────────────────────────────────

_INTENT_PATTERNS = [
    # Budget/cuisine keywords checked BEFORE search so "find cheap..."
    # matches budget_search, not search.
    (r"(?:cheap|budget|affordable|under\s+₹?\d+)", "budget_search"),
    (r"(?:spicy|spice|hot)", "cuisine_search"),
    (
        r"(?:find|search|look|show|get|recommend|suggest)\s+(?:me\s+)?(?:some\s+)?(.+?)(?:\s+(?:in|near|at)\s+(.+))?$",
        "search",
    ),
    (
        # Handles: "How long does [delivery from] X take?" / "ETA for X" / etc.
        r"(?:how\s+long|eta|delivery\s+time|when)\s+(?:for|does|will)\s+(?:delivery\s+)?"
        r"(?:from\s+)?(.+?)(?:\s+take)?(?:\?)?$",
        "eta",
    ),
    (
        r"(?:reliability|reliable|trust|score|rating)\s+(?:of|for|score)?\s*(.+?)(?:\?)?$",
        "reliability",
    ),
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
        # Patterns that should match anywhere in the text use re.search;
        # patterns that require a keyword at the start use re.match.
        if intent in ("budget_search", "cuisine_search", "reliability"):
            match = re.search(pattern, text, re.IGNORECASE)
        else:
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
