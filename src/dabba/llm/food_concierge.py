"""LLM-powered and rules-based food concierge."""

from __future__ import annotations

import logging
from typing import Any

from dabba.config import DabbaConfig
from dabba.llm.concierge_tools import ConciergeTools, _execute_tool

logger = logging.getLogger(__name__)

# ─── LLM-powered concierge (ReAct loop) ─────────────────────────────

_anthropic_client = None


def _get_llm_client(config: DabbaConfig) -> Any:
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
    except (ImportError, OSError) as e:
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
    messages: list[dict[str, str]],
    tools: ConciergeTools,
    config: DabbaConfig,
) -> str | None:
    """Generate a concierge response using Anthropic Claude with a ReAct tool loop.

    Uses circuit breaker to prevent cascading LLM failures.

    The ReAct loop works as follows:

        1. Send the accumulated conversation (including any tool results
           from previous steps) to Claude with tool definitions.
        2. Claude returns ``text`` blocks (accumulated into final answer)
           and/or ``tool_use`` blocks.
        3. If any ``tool_use`` → execute each tool → add a ``tool_result``
           content block to the conversation → loop back to step 1.
        4. If no ``tool_use`` → break, return accumulated text.

    The loop runs at most ``config.llm_max_steps`` iterations.
    Each iteration is traced via Prometheus metrics (tool call counter,
    loop duration histogram).

    Args:
        messages: Conversation history as list of
            ``{"role": str, "content": str}`` dicts.
        tools: ConciergeTools instance with restaurant data.
        config: Project configuration.

    Returns:
        The final response text, or ``None`` if the LLM call completely
        failed (triggers fallback in caller).
    """
    from dabba.llm.circuit_breaker import llm_breaker

    client = _get_llm_client(config)
    if client is None:
        return None

    # Check circuit breaker before making LLM call
    if llm_breaker.is_open():
        logger.warning("LLM circuit breaker open — skipping concierge response")
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
    anthropic_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        anthropic_messages.append({"role": role, "content": msg["content"]})

    max_steps = config.llm_max_steps
    final_text_parts: list[str] = []

    for step in range(1, max_steps + 1):
        logger.debug("Concierge ReAct step %d/%d", step, max_steps)
        step_start_time = time.monotonic()

        try:
            response = client.messages.create(
                model=config.llm_model,
                max_tokens=config.llm_max_tokens,
                system=system_prompt,
                messages=anthropic_messages,
                tools=TOOL_DEFINITIONS,
            )
        except (OSError, ValueError) as e:
            llm_breaker.record_failure()
            logger.warning("LLM call failed at ReAct step %d: %s", step, e)
            break

        # Collect assistant content blocks (text + tool_use)
        assistant_content: list[dict[str, Any]] = []
        tool_calls: list[Any] = []
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
        tool_results: list[dict[str, Any]] = []
        for block in tool_calls:
            tool_start = time.monotonic()

            # Record tool call metric
            concierge_tool_calls_total.labels(tool=block.name).inc()

            result_text = _execute_tool(block.name, block.input, tools)

            tool_duration = time.monotonic() - tool_start

            # Log trace span for this tool execution as a structured log line
            logger.info(
                "Concierge tool executed",
                extra={
                    "span_type": "tool_execution",
                    "tool": block.name,
                    "step": step,
                    "duration_s": round(tool_duration, 3),
                    "result_length": len(result_text),
                },
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )

        # Record loop iteration duration
        loop_duration = time.monotonic() - step_start_time
        concierge_loop_duration_seconds.labels(step=str(step)).observe(loop_duration)

        # Tool results are sent as a user message with tool_result blocks
        anthropic_messages.append({"role": "user", "content": tool_results})

        if step >= max_steps:
            logger.info("Reached max ReAct steps (%d) for concierge", max_steps)

    if not final_text_parts:
        return None

    llm_breaker.record_success()
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


def _match_intent(user_input: str) -> tuple[str, dict[str, str]]:
    """Match user input to an intent using simple patterns.

    Returns:
        Tuple of (intent_name, extracted_params).
    """
    text = user_input.lower().strip()
    params: dict[str, str] = {}

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
            elif (intent == "eta" and groups[0]) or (intent == "reliability" and groups[0]):
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
    conversation_history: list[dict[str, str]],
    tools: ConciergeTools,
    config: DabbaConfig | None = None,
) -> str -> None:
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
