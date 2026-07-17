"""NLP sentiment analysis for restaurant reviews.

Uses VADER for fast sentiment scoring. Notes in README that a fine-tuned
Indic model (e.g., on Hindi/English code-switched text) would be the
production choice for Indian restaurant reviews.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


def _get_vader():
    """Safely import and return VADER SentimentIntensityAnalyzer."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("NLTK VADER not installed — attempting download")
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()


def score_sentiment(text: str, analyzer=None) -> float:
    """Score a single text for sentiment polarity.

    Args:
        text: Input text string.
        analyzer: Pre-initialized VADER analyzer (created if None).

    Returns:
        Float sentiment score in [-1, 1] range (compound).
    """
    if analyzer is None:
        analyzer = _get_vader()
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return 0.0
    scores = analyzer.polarity_scores(text)
    return scores["compound"]


def add_sentiment_scores(
    df: pd.DataFrame,
    review_col: str = "reviews_list",
    config: Optional[DabbaConfig] = None,
) -> pd.DataFrame:
    """Add per-restaurant average sentiment scores to the DataFrame.

    Processes the reviews_list column (which contains a stringified list of
    review tuples) and computes an average sentiment per restaurant.

    Args:
        df: DataFrame with review text column.
        review_col: Name of the column containing reviews.
        config: Project configuration.

    Returns:
        pd.DataFrame: DataFrame with added 'avg_sentiment' column.
    """
    config = config or get_config()
    df = df.copy()
    logger.info("Computing sentiment scores for %d restaurants", len(df))

    analyzer = _get_vader()

    def _extract_reviews(raw) -> list:
        """Parse the reviews_list column robustly.

        The Zomato reviews_list column can contain:
        - A stringified Python list of tuples: "[('5.0', 'Great food'), ...]"
        - An actual Python list (already parsed by pandas)
        - A plain string
        - NaN / empty

        This function handles all cases with multiple fallback strategies.
        """
        import re

        if pd.isna(raw):
            return []
        if isinstance(raw, list):
            return [str(r) for r in raw]
        s = str(raw).strip()
        if not s or s == "[]":
            return []

        # Strategy 1: Try ast.literal_eval (handles well-formed stringified lists)
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                texts = []
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        texts.append(str(item[1]))
                    elif isinstance(item, str):
                        texts.append(item)
                if texts:
                    return texts
        except (ValueError, SyntaxError, RecursionError):
            pass

        # Strategy 2: Regex extraction — pull quoted strings from the raw text
        # Matches single or double quoted strings that look like review text
        quoted = re.findall(r'["\']([^"\']{10,})["\']', s)
        if quoted:
            return quoted

        # Strategy 3: Split on common delimiters and take substantial chunks
        for delim in ["\n", "; ", "], "]:
            parts = s.split(delim)
            texts = [p.strip().strip("'\"()[]") for p in parts if len(p.strip()) > 10]
            if texts:
                return texts

        # Fallback: treat the whole string as one review (if long enough)
        return [s] if len(s) > 5 else []

    def _avg_sentiment(raw) -> float:
        reviews = _extract_reviews(raw)
        if not reviews:
            return 0.0
        sentiments = [score_sentiment(r, analyzer) for r in reviews]
        return float(np.mean(sentiments))

    if review_col in df.columns:
        df["avg_sentiment"] = df[review_col].apply(_avg_sentiment)
        logger.info(
            "Sentiment scores computed — mean=%.3f, std=%.3f",
            df["avg_sentiment"].mean(),
            df["avg_sentiment"].std(),
        )
    else:
        logger.warning("Column '%s' not found — defaulting sentiment to 0", review_col)
        df["avg_sentiment"] = 0.0

    return df
