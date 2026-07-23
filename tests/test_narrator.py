"""Tests for the Recommendation Narrator — LLM and rules-based explanations."""

import pytest

from dabba.llm.recommendation_narrator import _rules_narrate, narrate_recommendation


# ─── Sample restaurant data ────────────────────────────────────────────

SAMPLE_RESTAURANT = {
    "name": "Test Kitchen",
    "cuisines": "North Indian, Chinese",
    "rate": "4.5",
    "cost_for_two": "600",
    "location": "Koramangala",
}


class TestRulesNarrate:
    """Tests for the rules-based narration fallback."""

    def test_returns_string(self):
        """Should return a string."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.5, 25.0)
        assert isinstance(result, str)

    def test_includes_restaurant_name(self):
        """Result should include the restaurant name."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.5, 25.0)
        assert "Test Kitchen" in result

    def test_includes_cuisine(self):
        """Result should include cuisine information."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.5, 25.0)
        assert "North Indian" in result

    def test_includes_location(self):
        """Result should include location."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.5, 25.0)
        assert "Koramangala" in result

    def test_high_rating_badge(self):
        """Rating >= 4.0 should mention 'Highly rated'."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.5, 25.0)
        assert "Highly rated" in result

    def test_average_rating_badge(self):
        """Rating in [3.0, 4.0) should mention 'Decent rating'."""
        restaurant = {**SAMPLE_RESTAURANT, "rate": "3.5"}
        result = _rules_narrate(restaurant, 0.8, 0.5, 25.0)
        assert "Decent rating" in result

    def test_low_rating_badge(self):
        """Rating < 3.0 should mention 'Average rating'."""
        restaurant = {**SAMPLE_RESTAURANT, "rate": "2.5"}
        result = _rules_narrate(restaurant, 0.8, 0.5, 25.0)
        assert "Average rating" in result

    def test_budget_friendly(self):
        """Cost <= 300 should mention 'budget-friendly'."""
        restaurant = {**SAMPLE_RESTAURANT, "cost_for_two": "250"}
        result = _rules_narrate(restaurant, 0.8, 0.5, 25.0)
        assert "budget-friendly" in result

    def test_moderately_priced(self):
        """Cost in (300, 800] should mention 'moderately priced'."""
        restaurant = {**SAMPLE_RESTAURANT, "cost_for_two": "500"}
        result = _rules_narrate(restaurant, 0.8, 0.5, 25.0)
        assert "moderately priced" in result

    def test_premium(self):
        """Cost > 800 should mention 'premium'."""
        restaurant = {**SAMPLE_RESTAURANT, "cost_for_two": "1200"}
        result = _rules_narrate(restaurant, 0.8, 0.5, 25.0)
        assert "premium" in result

    def test_high_reliability(self):
        """Reliability >= 0.7 should mention 'high overall reliability'."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.85, 0.5, 25.0)
        assert "high overall reliability" in result

    def test_moderate_reliability(self):
        """Reliability in [0.4, 0.7) should mention 'moderate reliability'."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.55, 0.5, 25.0)
        assert "moderate reliability" in result

    def test_low_reliability(self):
        """Reliability < 0.4 should mention 'lower reliability'."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.25, 0.5, 25.0)
        assert "lower reliability" in result

    def test_positive_sentiment(self):
        """Sentiment > 0.3 should mention 'positive customer sentiment'."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.6, 25.0)
        assert "positive customer sentiment" in result

    def test_negative_sentiment(self):
        """Sentiment < -0.2 should mention 'mixed customer reviews'."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, -0.5, 25.0)
        assert "mixed customer reviews" in result

    def test_neutral_sentiment(self):
        """Sentiment in [-0.2, 0.3] should mention 'neutral customer sentiment'."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.0, 25.0)
        assert "neutral customer sentiment" in result

    def test_includes_eta_estimate(self):
        """Should include delivery time estimate when provided."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.5, 25.0)
        assert "min" in result

    def test_no_eta_estimate(self):
        """Should not mention delivery time when None."""
        result = _rules_narrate(SAMPLE_RESTAURANT, 0.8, 0.5, None)
        assert "delivery" not in result.lower() or "min" not in result

    def test_minimal_restaurant_data(self):
        """Should handle minimal restaurant data gracefully."""
        result = _rules_narrate({}, 0.5, 0.0, None)
        assert isinstance(result, str)
        assert len(result) > 0


class TestNarrateRecommendation:
    """Tests for the narrate_recommendation entry point."""

    def test_returns_string(self):
        """Should return a string (falls back to rules since no LLM configured)."""
        result = narrate_recommendation(SAMPLE_RESTAURANT, 0.8)
        assert isinstance(result, str)

    def test_falls_back_to_rules_without_llm(self):
        """Without LLM configured, should use rules-based narration."""
        result = narrate_recommendation(SAMPLE_RESTAURANT, 0.8, sentiment_avg=0.5)
        assert "Test Kitchen" in result
        assert "Highly rated" in result

    def test_custom_sentiment(self):
        """Sentiment value should influence the narration."""
        result = narrate_recommendation(
            SAMPLE_RESTAURANT, 0.8, sentiment_avg=0.9, eta_prediction=30.0
        )
        assert isinstance(result, str)

    def test_custom_eta(self):
        """ETA prediction should be included in the narration."""
        result = narrate_recommendation(
            SAMPLE_RESTAURANT, 0.8, sentiment_avg=0.5, eta_prediction=45.0
        )
        assert "45" in result or "min" in result

    def test_low_reliability_narration(self):
        """Low reliability should mention alternatives."""
        result = narrate_recommendation(SAMPLE_RESTAURANT, 0.2, sentiment_avg=0.0)
        assert "consider alternatives" in result or "lower reliability" in result
