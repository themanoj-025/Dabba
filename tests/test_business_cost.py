"""Tests for business cost analysis: SLA, Reliability Score, A/B simulations."""

import numpy as np
import pandas as pd
import pytest

from dabba.evaluation.business_cost import (
    WEIGHT_PROFILES,
    compute_reliability_score,
    compute_sla_analysis,
    run_ab_scenario_simulation,
)


class TestComputeSlaAnalysis:
    """Tests for compute_sla_analysis()."""

    def test_returns_expected_keys(self):
        """Should return dict with all expected metric keys."""
        y_true = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
        y_pred = np.array([22.0, 24.0, 32.0, 33.0, 38.0])
        result = compute_sla_analysis(y_true, y_pred, sla_threshold=30.0)
        assert isinstance(result, dict)
        assert "sla_threshold_min" in result
        assert "actual_on_time_rate" in result
        assert "precision" in result
        assert "recall" in result
        assert "true_positives" in result
        assert "false_positives" in result
        assert "false_negatives" in result
        assert "true_negatives" in result

    def test_perfect_predictions(self):
        """Perfect predictions should yield 100% precision and recall."""
        y_true = np.array([20.0, 25.0, 35.0, 40.0])
        y_pred = np.array([20.0, 25.0, 35.0, 40.0])
        result = compute_sla_analysis(y_true, y_pred, sla_threshold=30.0)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_all_late(self):
        """When all deliveries exceed SLA, on-time rate should be 0."""
        y_true = np.array([35.0, 40.0, 45.0])
        y_pred = np.array([33.0, 38.0, 42.0])
        result = compute_sla_analysis(y_true, y_pred, sla_threshold=30.0)
        assert result["actual_on_time_rate"] == 0.0

    def test_all_on_time(self):
        """When all deliveries are within SLA, on-time rate should be 1."""
        y_true = np.array([15.0, 20.0, 25.0])
        y_pred = np.array([14.0, 21.0, 24.0])
        result = compute_sla_analysis(y_true, y_pred, sla_threshold=30.0)
        assert result["actual_on_time_rate"] == 1.0

    def test_empty_arrays(self):
        """Empty arrays should not crash — precision/recall default to 0."""
        y_true = np.array([])
        y_pred = np.array([])
        result = compute_sla_analysis(y_true, y_pred, sla_threshold=30.0)
        assert result["total_orders"] == 0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0


class TestReliabilityScore:
    """Tests for compute_reliability_score()."""

    def test_returns_float_for_scalars(self):
        """Scalar inputs should return a float score in [0, 1]."""
        score = compute_reliability_score(
            rating=4.5, sentiment=0.8, delay_risk=0.2
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_returns_array_for_arrays(self):
        """Array inputs should return an array of scores in [0, 1]."""
        ratings = np.array([4.5, 3.0, 5.0])
        sentiments = np.array([0.8, 0.2, -0.1])
        delays = np.array([0.2, 0.5, 0.8])
        scores = compute_reliability_score(ratings, sentiments, delays)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_higher_rating_increases_score(self):
        """Higher rating should yield a higher reliability score (all else equal)."""
        # Use arrays so min-max normalization produces distinct values
        score_low = compute_reliability_score(
            rating=np.array([2.0, 3.0]), sentiment=np.array([0.5, 0.5]),
            delay_risk=np.array([0.3, 0.3])
        )
        score_high = compute_reliability_score(
            rating=np.array([4.0, 5.0]), sentiment=np.array([0.5, 0.5]),
            delay_risk=np.array([0.3, 0.3])
        )
        # Compare high-rated array's max vs low-rated array's max
        assert np.max(score_high) > np.max(score_low)

    def test_higher_delay_decreases_score(self):
        """Higher delay risk should decrease reliability score."""
        score_low_delay = compute_reliability_score(
            rating=np.array([4.0, 4.0]), sentiment=np.array([0.5, 0.5]),
            delay_risk=np.array([0.1, 0.5])
        )
        score_high_delay = compute_reliability_score(
            rating=np.array([4.0, 4.0]), sentiment=np.array([0.5, 0.5]),
            delay_risk=np.array([0.5, 0.9])
        )
        assert np.max(score_low_delay) > np.max(score_high_delay)

    def test_clips_to_zero_one(self):
        """Score should be clipped to [0, 1] even with extreme inputs."""
        # Large negative component
        score = compute_reliability_score(
            rating=0.0, sentiment=-1.0, delay_risk=1.0,
            weights={"w_rating": 1.0, "w_sentiment": 0.0, "w_delay": 1.0},
        )
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        """Custom weight dict (rating=1.0) should produce score = norm(rating)."""
        # With w_rating=1.0 and single scalar input, min-max norm returns 0.5
        score_custom = compute_reliability_score(
            rating=4.5, sentiment=0.8, delay_risk=0.2,
            weights={"w_rating": 1.0, "w_sentiment": 0.0, "w_delay": 0.0},
        )
        assert score_custom == pytest.approx(0.5)

    def test_identical_inputs_identical_scores(self):
        """Same inputs should produce the same score."""
        s1 = compute_reliability_score(rating=4.0, sentiment=0.5, delay_risk=0.3)
        s2 = compute_reliability_score(rating=4.0, sentiment=0.5, delay_risk=0.3)
        assert s1 == pytest.approx(s2)


class TestWeightProfiles:
    """Tests for the WEIGHT_PROFILES constant."""

    def test_has_expected_profiles(self):
        """Should have balanced, quality_first, and speed_first profiles."""
        assert "balanced" in WEIGHT_PROFILES
        assert "quality_first" in WEIGHT_PROFILES
        assert "speed_first" in WEIGHT_PROFILES

    def test_balanced_profile_description(self):
        """Balanced profile should have a description."""
        assert "description" in WEIGHT_PROFILES["balanced"]

    def test_weights_sum_to_one(self):
        """Each profile's weights should sum to approximately 1.0."""
        for name, profile in WEIGHT_PROFILES.items():
            total = profile["w_rating"] + profile["w_sentiment"] + profile["w_delay"]
            assert total == pytest.approx(1.0), f"{name} weights sum to {total}"


class TestAbScenarioSimulation:
    """Tests for run_ab_scenario_simulation()."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample restaurant DataFrame for A/B testing."""
        rng = np.random.RandomState(42)
        n = 10
        return pd.DataFrame({
            "name": [f"Rest_{i}" for i in range(n)],
            "rate": rng.uniform(3.0, 5.0, n),
            "avg_sentiment": rng.uniform(-0.5, 1.0, n),
            "delay_risk": rng.uniform(0.1, 0.9, n),
            "cost_for_two": rng.randint(200, 1500, n),
            "location": rng.choice(["Koramangala", "Indiranagar"], n),
            "cuisines": rng.choice(["North Indian", "Chinese"], n),
        })

    def test_returns_all_profiles(self, sample_df):
        """Should return results for all weight profiles plus _meta."""
        results = run_ab_scenario_simulation(sample_df, top_n=5)
        assert "balanced" in results
        assert "quality_first" in results
        assert "speed_first" in results
        assert "_meta" in results

    def test_profile_has_top_restaurants(self, sample_df):
        """Each profile result should contain top_restaurants."""
        results = run_ab_scenario_simulation(sample_df, top_n=5)
        for profile in ["balanced", "quality_first", "speed_first"]:
            assert "top_restaurants" in results[profile]
            assert len(results[profile]["top_restaurants"]) <= 5

    def test_top_restaurant_has_name_and_score(self, sample_df):
        """Top restaurant entries should have name and score fields."""
        results = run_ab_scenario_simulation(sample_df, top_n=3)
        top = results["balanced"]["top_restaurants"][0]
        assert "name" in top
        assert "score" in top

    def test_meta_overlap_keys(self, sample_df):
        """_meta should contain overlap counts between profiles."""
        results = run_ab_scenario_simulation(sample_df, top_n=5)
        meta = results["_meta"]
        assert "balanced_vs_quality_overlap" in meta
        assert "balanced_vs_speed_overlap" in meta
        assert "quality_vs_speed_overlap" in meta

    def test_missing_columns_do_not_crash(self):
        """Missing columns should be handled gracefully with warnings."""
        df = pd.DataFrame({"name": ["Test"], "rate": [4.0]})
        # Should not crash despite missing sentiment and delay columns
        results = run_ab_scenario_simulation(df, top_n=5)
        assert "balanced" in results

    def test_mean_score_in_range(self, sample_df):
        """Mean scores should be in [0, 1] range."""
        results = run_ab_scenario_simulation(sample_df, top_n=5)
        for profile in ["balanced", "quality_first", "speed_first"]:
            ms = results[profile]["mean_score"]
            assert 0.0 <= ms <= 1.0, f"{profile} mean_score={ms} out of range"

    def test_custom_column_names(self, sample_df):
        """Custom column names should be accepted."""
        df_renamed = sample_df.rename(
            columns={
                "rate": "rating",
                "avg_sentiment": "sentiment",
                "delay_risk": "delay",
            }
        )
        results = run_ab_scenario_simulation(
            df_renamed,
            rating_col="rating",
            sentiment_col="sentiment",
            delay_col="delay",
            top_n=3,
        )
        assert "balanced" in results
