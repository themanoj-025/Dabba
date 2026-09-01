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
import time
from typing import Any

import pandas as pd

from dabba.config import DabbaConfig, get_config
from dabba.features.delivery_features import build_eta_features_for_api
from dabba.features.geo import haversine_distance
from dabba.observability import (
    concierge_loop_duration_seconds,
    concierge_tool_calls_total,
)

logger = logging.getLogger(__name__)


# ─── Tools that the concierge can call ──────────────────────────────────


class ConciergeTools:
    """Deterministic tools the concierge can invoke."""

    def __init__(
        self,
        restaurants_df: pd.DataFrame,
        eta_model: Any = None,
        config: DabbaConfig | None = None,
    ) -> Any:
        self.df = restaurants_df
        self.eta_model = eta_model
        self.config = config or get_config()

    def search_restaurants(
        self,
        cuisine: str | None = None,
        max_budget: float | None = None,
        area: str | None = None,
        top_n: int = 5,
    ) -> list[dict[str, Any]] -> None:
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

    def get_eta_estimate(self, restaurant_name: str) -> dict[str, Any] | None:
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
        except (ValueError, OSError) as e:
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

    def get_reliability_score(self, restaurant_name: str) -> float | None:
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


def _execute_tool(
    tool_name: str, tool_input: dict[str, Any], tools: ConciergeTools
) -> str:
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
            return f"Restaurant '{tool_input.get('restaurant_name', '')}' not found."
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


