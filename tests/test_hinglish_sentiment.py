"""Tests for the Hinglish sentiment module (P3)."""

import pytest

from dabba.nlp.hinglish_sentiment import score_sentiment, add_hinglish_sentiment_scores


class TestScoreSentiment:
    """Tests for score_sentiment() — falls back to VADER when transformers unavailable."""

    def test_empty_text(self):
        """Empty text should return 0.0."""
        assert score_sentiment("") == 0.0
        assert score_sentiment(None) == 0.0

    def test_english_text(self):
        """English text should return a non-zero score via VADER."""
        score = score_sentiment("This food was absolutely amazing and delicious!")
        assert score != 0.0

    def test_negative_text(self):
        """Negative text should return a negative score."""
        score = score_sentiment("Terrible food, worst experience ever, disgusting.")
        assert score < 0

    def test_positive_text(self):
        """Positive text should return a positive score."""
        score = score_sentiment("Amazing food, wonderful service, highly recommend!")
        assert score > 0

    def test_hinglish_falls_back_to_vader(self):
        """Hinglish text should work (via VADER fallback since transformers may not be available)."""
        score = score_sentiment("Ye bahut achha hai, amazing food!")
        # Should not crash — returns whatever VADER gives for the English parts
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


class TestAddHinglishSentimentScores:
    """Tests for add_hinglish_sentiment_scores()."""

    def test_adds_column(self):
        """Should add avg_sentiment column."""
        import pandas as pd

        df = pd.DataFrame({
            "name": ["Test Restaurant"],
            "reviews_list": [["Great food!"]],
        })
        result = add_hinglish_sentiment_scores(df)
        assert "avg_sentiment" in result.columns

    def test_empty_reviews(self):
        """Empty reviews should result in 0.0 sentiment."""
        import pandas as pd

        df = pd.DataFrame({
            "name": ["Test"],
            "reviews_list": [None],
        })
        result = add_hinglish_sentiment_scores(df)
        assert result["avg_sentiment"].iloc[0] == 0.0

    def test_missing_column(self):
        """Missing review column should not crash."""
        import pandas as pd

        df = pd.DataFrame({"name": ["Test"]})
        result = add_hinglish_sentiment_scores(df, review_col="nonexistent")
        assert "avg_sentiment" in result.columns
