"""Tests for the recommendation narrator — LLM-powered with rules fallback."""

import pytest

from dabba.config import DabbaConfig, get_config
from dabba.llm.recommendation_narrator import (
    _rules_narrate,
    narrate_recommendation,
)

SAMPLE_RESTAURANT = {
    "name": "Meghana Foods",
    "cuisines": "North Indian, Chinese",
    "rate": "4.5",
    "cost_for_two": "400",
    "location": "Koramangala",
}


class TestRulesNarrate:
    """Tests for the _rules_narrate fallback function."""

    def test_high_rating_high_reliability(self):
        """High rating + high reliability should mention both positively."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.85, 0.6, eta_prediction=25.0)
        assert "Meghana Foods" in result
        assert "Highly rated" in result
        assert "high overall reliability" in result
        assert "positive customer sentiment" in result
        assert "25 min" in result

    def test_low_reliability_and_negative_sentiment(self):
        """Low reliability + negative sentiment should be called out."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.25, -0.5, eta_prediction=None)
        assert "lower reliability" in result
        assert "mixed customer reviews" in result

    def test_neutral_sentiment(self):
        """Sentiment in [-0.2, 0.3] should mention 'neutral customer sentiment'."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.5, 0.0, eta_prediction=None)
        assert "neutral customer sentiment" in result

    def test_moderate_reliability(self):
        """Reliability between 0.4 and 0.7 should say moderate."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.55, 0.5, eta_prediction=None)
        assert "moderate reliability" in result

    def test_budget_restaurant(self):
        """Cost <= 300 should be flagged as budget-friendly."""
        rest = {**SAMPLE_RESTAURANT, "cost_for_two": "250"}
        result = _rules_narrate(rest, 0.8, 0.5, eta_prediction=None)
        assert "budget-friendly" in result

    def test_premium_restaurant(self):
        """Cost > 800 should be flagged as premium."""
        rest = {**SAMPLE_RESTAURANT, "cost_for_two": "1500"}
        result = _rules_narrate(rest, 0.8, 0.5, eta_prediction=None)
        assert "premium" in result

    def test_empty_restaurant_dict(self):
        """An empty restaurant dict should not crash."""
        result = _rules_narrate({}, 0.5, 0.0, eta_prediction=None)
        assert "This restaurant" in result
        assert "a variety of dishes" in result

    def test_missing_optional_fields(self):
        """Missing rate/cost should be handled gracefully (no crash)."""
        rest = {"name": "Test Place"}
        result = _rules_narrate(rest, 0.5, 0.0, eta_prediction=None)
        assert "Test Place" in result

    def test_no_eta_no_crash(self):
        """Eta_prediction=None should omit delivery time from output."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.5, eta_prediction=None)
        assert "min delivery" not in result


class TestNarrateRecommendation:
    """Tests for the public narrate_recommendation() API."""

    def test_rules_fallback_when_llm_disabled(self):
        """With LLM disabled, should return rules-based narration."""
        config = get_config()
        config.llm_enabled = False
        result = narrate_recommendation(
            SAMPLE_RESTAURANT, 0.8, sentiment_avg=0.5, config=config
        )
        assert "Meghana Foods" in result
        assert "positive customer sentiment" in result

    def test_rules_fallback_when_llm_enabled_no_key(self):
        """With LLM enabled but no API key, should fall back to rules."""
        config = get_config()
        config.llm_enabled = True
        config.anthropic_api_key = None
        result = narrate_recommendation(
            SAMPLE_RESTAURANT, 0.8, sentiment_avg=0.5, config=config
        )
        assert len(result) > 0
        assert "Meghana Foods" in result

    def test_with_eta_prediction(self):
        """Eta_prediction should be included in the output."""
        result = narrate_recommendation(
            SAMPLE_RESTAURANT, 0.8, sentiment_avg=0.5, eta_prediction=30.0, config=get_config()
        )
        assert "30 min" in result or "30" in result

    def test_low_confidence_restaurant(self):
        """Very low reliability scores should be reflected."""
        result = narrate_recommendation(
            SAMPLE_RESTAURANT, 0.15, sentiment_avg=-0.3, config=get_config()
        )
        assert "lower reliability" in result or "reliability" in result.lower()

    def test_uses_default_config_when_none(self):
        """Should use default config when none is provided."""
        result = narrate_recommendation(SAMPLE_RESTAURANT, 0.7, sentiment_avg=0.4)
        assert len(result) > 0
