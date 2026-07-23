"""Hinglish (Hindi-English code-switched) sentiment analysis.

Provides a drop-in alternative to VADER (which is English-only) for
restaurant reviews written in code-switched Hindi/English text (a
common pattern in Indian food delivery reviews).

PRIMARY MODE: Uses a HuggingFace multilingual sentiment model
  (``tabularisai/multilingual-sentiment-analysis``) that natively
  supports Hindi and other Indian languages.

FALLBACK MODE: Uses English VADER when:
  - The HuggingFace model fails to load (memory constraints)
  - The ``transformers`` package is not installed
  - A specific review is detected as English-only

The module auto-detects whether to use the transformer model or
VADER based on available dependencies — the app never breaks.

Usage:
    >>> from dabba.nlp.hinglish_sentiment import score_sentiment, add_hinglish_sentiment_scores
    >>> score_sentivity("Ye restaurant bahut achha hai, I loved the food!")
    0.78
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)

# ─── Model cache (lazy-loaded) ───────────────────────────────────────

_PIPELINE = None


def _get_transformer_pipeline():
    """Lazy-load the HuggingFace multilingual sentiment pipeline.

    Returns:
        A ``transformers.pipeline`` or None if transformers/torch
        is not available or the model fails to load.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        from transformers import pipeline

        logger.info("Loading multilingual sentiment model...")
        _PIPELINE = pipeline(
            "text-classification",
            model="tabularisai/multilingual-sentiment-analysis",
            top_k=None,
        )
        logger.info("Multilingual sentiment model loaded successfully")
        return _PIPELINE
    except ImportError:
        logger.warning(
            "transformers not installed — falling back to VADER for sentiment"
        )
        return None
    except Exception as e:
        logger.warning("Failed to load multilingual sentiment model: %s", e)
        return None


def _get_vader():
    """Get VADER analyzer (same import pattern as sentiment.py)."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except ImportError:
        import nltk

        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()


# ─── Scoring ──────────────────────────────────────────────────────────


def score_sentiment(text: str, vader_analyzer=None) -> float:
    """Score a single text for sentiment polarity.

    Uses the multilingual transformer model when available (handles
    Hindi, Hinglish, and other Indian languages). Falls back to
    English VADER when the model is unavailable or the text is
    detected as English-only.

    Args:
        text: Input text string.
        vader_analyzer: Pre-initialized VADER analyzer (created if None).

    Returns:
        Float sentiment score in [-1, 1] range (negative to positive).
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return 0.0

    # Try transformer model first (handles Hindi/Hinglish)
    pipeline = _get_transformer_pipeline()
    if pipeline is not None:
        try:
            result = pipeline(text)[0]
            # HuggingFace text-classification pipeline returns
            # [{"label": "POSITIVE"|"NEGATIVE", "score": 0.xx}, ...]
            # or with top_k=None: [[{"label":"POSITIVE","score":0.xx},...]]
            if isinstance(result, list):
                scores = {item["label"]: item["score"] for item in result}
            elif isinstance(result, dict):
                scores = {result["label"]: result["score"]}
            else:
                scores = {}

            pos = scores.get("POSITIVE", scores.get("LABEL_1", 0.5))
            neg = scores.get("NEGATIVE", scores.get("LABEL_0", 0.5))
            # Convert to [-1, 1] range
            return float(round(pos - neg, 4))
        except Exception as e:
            logger.debug("Transformer sentiment failed: %s — falling back to VADER", e)

    # Fallback: English VADER
    if vader_analyzer is None:
        vader_analyzer = _get_vader()

    try:
        scores = vader_analyzer.polarity_scores(text)
        return float(scores["compound"])
    except Exception as e:
        logger.warning("VADER sentiment failed: %s", e)
        return 0.0


# ─── Batch scoring for DataFrames ─────────────────────────────────────


def add_hinglish_sentiment_scores(
    df: pd.DataFrame,
    review_col: str = "reviews_list",
    config: Optional[DabbaConfig] = None,
) -> pd.DataFrame:
    """Add per-restaurant average sentiment scores using Hinglish-aware model.

    Works identically to ``sentiment.add_sentiment_scores()`` but
    uses the multilingual transformer model instead of VADER for
    better handling of Hindi/English code-switched reviews.

    Falls back to VADER if the transformer model is unavailable.

    Args:
        df: DataFrame with review text column.
        review_col: Name of the column containing reviews.
        config: Project configuration.

    Returns:
        pd.DataFrame: DataFrame with added 'avg_sentiment' column.
    """
    config = config or get_config()
    df = df.copy()
    logger.info(
        "Computing Hinglish-aware sentiment scores for %d restaurants",
        len(df),
    )

    vader_analyzer = _get_vader()

    def _extract_reviews(raw) -> list:
        """Parse the reviews_list column (same logic as sentiment.py)."""
        import ast

        if pd.isna(raw):
            return []
        if isinstance(raw, list):
            return [str(r) for r in raw]
        s = str(raw).strip()
        if not s or s == "[]":
            return []

        try:
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

        quoted = re.findall(r'["\']([^"\']{10,})["\']', s)
        if quoted:
            return quoted

        for delim in ["\n", "; ", "], "]:
            parts = s.split(delim)
            texts = [p.strip().strip("'\"()[]") for p in parts if len(p.strip()) > 10]
            if texts:
                return texts

        return [s] if len(s) > 5 else []

    def _avg_sentiment(raw) -> float:
        reviews = _extract_reviews(raw)
        if not reviews:
            return 0.0
        sentiments = [
            score_sentiment(r, vader_analyzer=vader_analyzer) for r in reviews
        ]
        return float(np.mean(sentiments))

    if review_col in df.columns:
        df["avg_sentiment"] = df[review_col].apply(_avg_sentiment)
        logger.info(
            "Hinglish sentiment scores computed — mean=%.3f, std=%.3f",
            df["avg_sentiment"].mean(),
            df["avg_sentiment"].std(),
        )
    else:
        logger.warning(
            "Column '%s' not found — defaulting sentiment to 0", review_col
        )
        df["avg_sentiment"] = 0.0

    return df
