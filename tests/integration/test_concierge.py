"""Integration tests for the Food Concierge agent.

Tests the full non-LLM pipeline:
    - ConciergeTools with actual DataFrames
    - _execute_tool formatting for all 3 tools
    - _match_intent pattern matching
    - _rules_concierge_response for all 7 intents
    - get_concierge_response fallback behavior
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from dabba.config import DabbaConfig, get_config
from dabba.llm.food_concierge import (
    ConciergeTools,
    _execute_tool,
    _match_intent,
    _rules_concierge_response,
    get_concierge_response,
)


@pytest.fixture
def sample_restaurants() -> pd.DataFrame:
    """Create a small restaurant DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": [
                "Meghana Foods",
                "Truffles",
                "KFC",
                "Byg Brewski",
                "A2B Veg",
            ],
            "rate": [4.8, 4.5, 3.8, 4.6, 4.2],
            "cost_for_two": [400, 600, 350, 1200, 300],
            "location": [
                "Koramangala",
                "Indiranagar",
                "MG Road",
                "Koramangala",
                "BTM Layout",
            ],
            "cuisines": [
                "North Indian, Chinese",
                "American, Italian",
                "American, Fast Food",
                "North Indian, Continental",
                "South Indian, Jain",
            ],
            "reliability_score": [0.88, 0.75, 0.60, 0.92, 0.80],
        }
    )


@pytest.fixture
def tools(sample_restaurants) -> ConciergeTools:
    """Create ConciergeTools with sample data."""
    return ConciergeTools(sample_restaurants)


@pytest.fixture
def config_no_llm() -> DabbaConfig:
    """Config with LLM disabled (to test rules fallback)."""
    cfg = get_config()
    cfg.llm_enabled = False
    return cfg


class TestConciergeTools:
    """Tests for ConciergeTools with actual data."""

    def test_search_by_cuisine(self, tools):
        """Should filter restaurants by cuisine."""
        results = tools.search_restaurants(cuisine="North Indian")
        assert len(results) >= 2  # Meghana Foods + Byg Brewski
        assert all("North Indian" in r["cuisines"] for r in results)

    def test_search_by_budget(self, tools):
        """Should filter restaurants by max budget."""
        results = tools.search_restaurants(max_budget=400)
        assert len(results) >= 3
        assert all(r["cost_for_two"] <= 400 for r in results)

    def test_search_by_area(self, tools):
        """Should filter restaurants by area."""
        results = tools.search_restaurants(area="Koramangala")
        assert len(results) >= 2
        assert all("Koramangala" in r["location"] for r in results)

    def test_search_combined(self, tools):
        """Should combine multiple filters."""
        results = tools.search_restaurants(
            cuisine="North Indian", max_budget=500, area="Koramangala"
        )
        assert len(results) >= 1
        # Meghana Foods: North Indian, ₹400, Koramangala
        assert results[0]["name"] == "Meghana Foods"

    def test_search_empty_result(self, tools):
        """Should return empty list for no match."""
        results = tools.search_restaurants(cuisine="Sushi")
        assert len(results) == 0

    def test_get_eta_no_model(self, tools):
        """Should return approximate ETA when no model loaded."""
        result = tools.get_eta_estimate("Meghana Foods")
        assert result is not None
        assert result["predicted_minutes"] == 30
        assert result["note"] == "approximate"

    def test_get_eta_unknown_restaurant(self, tools):
        """Should return None for unknown restaurant."""
        result = tools.get_eta_estimate("Nonexistent Restaurant")
        assert result is None

    def test_get_reliability_score(self, tools):
        """Should return correct reliability score."""
        score = tools.get_reliability_score("Meghana Foods")
        assert score is not None
        assert score == pytest.approx(0.88, abs=0.01)

    def test_get_reliability_score_unknown(self, tools):
        """Should return None for unknown restaurant."""
        score = tools.get_reliability_score("Nonexistent")
        assert score is None

    def test_top_n_limit(self, tools):
        """Should respect top_n parameter."""
        results = tools.search_restaurants(top_n=2)
        assert len(results) == 2


class TestExecuteTool:
    """Tests for the _execute_tool LLM-side formatting."""

    def test_search_restaurants_formatted(self, tools):
        """Should format search results as structured text."""
        result = _execute_tool("search_restaurants", {"cuisine": "South Indian"}, tools)
        assert "Found" in result
        assert "A2B Veg" in result
        assert "Rating:" in result
        assert "₹" in result

    def test_search_no_results(self, tools):
        """Should return clean message for no results."""
        result = _execute_tool("search_restaurants", {"cuisine": "Sushi"}, tools)
        assert "No restaurants found" in result

    def test_get_eta_formatted(self, tools):
        """Should format ETA result."""
        result = _execute_tool("get_eta_estimate", {"restaurant_name": "Truffles"}, tools)
        assert "ETA for Truffles" in result
        assert "on track" in result
        assert "min" in result

    def test_get_eta_not_found(self, tools):
        """Should format not-found message."""
        result = _execute_tool(
            "get_eta_estimate", {"restaurant_name": "Fake"}, tools
        )
        assert "not found" in result

    def test_reliability_formatted(self, tools):
        """Should format reliability score."""
        result = _execute_tool(
            "get_reliability_score", {"restaurant_name": "A2B Veg"}, tools
        )
        assert "Reliability score for A2B Veg" in result
        assert "/1.0" in result

    def test_reliability_not_found(self, tools):
        """Should format not-found message."""
        result = _execute_tool(
            "get_reliability_score", {"restaurant_name": "Fake"}, tools
        )
        assert "not found" in result

    def test_unknown_tool(self, tools):
        """Should handle unknown tool gracefully."""
        result = _execute_tool("unknown_tool", {}, tools)
        assert "Error" in result
        assert "unknown tool" in result


class TestMatchIntent:
    """Tests for rules-based intent matching."""

    def test_search_intent(self):
        """Should match search queries."""
        intent, params = _match_intent("Find North Indian food near Koramangala")
        assert intent == "search"
        # Input is lowercased before matching, so params use lowercase
        assert "north indian" in params.get("query", "")

    def test_eta_intent(self):
        """Should match ETA queries."""
        intent, params = _match_intent("How long does delivery from Meghana Foods take?")
        assert intent == "eta"
        # Input is lowercased; restaurant name extracted from pattern
        assert "meghana" in params.get("restaurant", "").lower()

    def test_reliability_intent(self):
        """Should match reliability queries."""
        intent, params = _match_intent("What's the reliability score for Truffles?")
        assert intent == "reliability"
        assert "truffles" in params.get("restaurant", "").lower()

    def test_budget_intent(self):
        """Should match budget queries."""
        intent, params = _match_intent("Find cheap restaurants under 300")
        assert intent == "budget_search"
        assert params.get("budget") == "300"

    def test_greeting_intent(self):
        """Should match greetings."""
        intent, params = _match_intent("Hello!")
        assert intent == "greeting"

    def test_unknown_intent(self):
        """Should return unknown for nonsense."""
        intent, params = _match_intent("asdfghjkl")
        assert intent == "unknown"


class TestRulesConciergeResponse:
    """Tests for the rules-based fallback concierge."""

    def test_greeting_response(self, tools):
        """Should return greeting message."""
        response = _rules_concierge_response("Hi there!", tools)
        assert "Namaste" in response
        assert "Food Concierge" in response

    def test_search_response(self, tools):
        """Should return formatted search results."""
        response = _rules_concierge_response(
            "Find North Indian food near Koramangala", tools
        )
        assert "Found" in response
        assert "Meghana Foods" in response

    def test_eta_response(self, tools):
        """Should return formatted ETA estimate."""
        response = _rules_concierge_response(
            "How long does delivery from Meghana Foods take?", tools
        )
        assert "Meghana Foods" in response or "meghana" in response.lower()
        assert "min" in response or "30" in response

    def test_reliability_response(self, tools):
        """Should return formatted reliability score."""
        response = _rules_concierge_response(
            "What's the reliability score for Byg Brewski?", tools
        )
        assert "Byg Brewski" in response or "byg brewski" in response.lower()
        assert "0.92" in response or "reliability" in response.lower()

    def test_budget_response(self, tools):
        """Should return budget search results."""
        response = _rules_concierge_response(
            "Find cheap restaurants under 400", tools
        )
        assert "under" in response

    def test_unknown_response(self, tools):
        """Should return help message for unknown intent."""
        response = _rules_concierge_response("asdfghjkl", tools)
        assert "not sure" in response
        assert "Search" in response
        assert "ETA" in response


class TestGetConciergeResponse:
    """Tests for the public concierge API."""

    def test_rules_fallback_without_llm(self, tools, config_no_llm):
        """Should use rules-based response when LLM is disabled."""
        response = get_concierge_response(
            "Find North Indian food",
            [],
            tools,
            config=config_no_llm,
        )
        assert len(response) > 0
        assert "Meghana Foods" in response or "Found" in response

    def test_rules_with_history(self, tools, config_no_llm):
        """Should work with conversation history."""
        history = [
            {"role": "user", "content": "Find me some food"},
            {"role": "assistant", "content": "Here are some options!"},
        ]
        response = get_concierge_response(
            "Show me cheap options",
            history,
            tools,
            config=config_no_llm,
        )
        assert len(response) > 0

    def test_llm_fallback_on_error(self, tools):
        """Should fall back to rules when LLM key is missing."""
        config = get_config()
        config.llm_enabled = True
        config.anthropic_api_key = None  # No key = client returns None

        response = get_concierge_response(
            "Find South Indian food",
            [],
            tools,
            config=config,
        )
        assert len(response) > 0
        # Falls back to rules, so should still return valid response
