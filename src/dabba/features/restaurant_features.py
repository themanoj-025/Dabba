"""Feature engineering for restaurant intelligence module.

Transforms cleaned Zomato data into model-ready features.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)

# Canonical cuisine list — top 30 most frequent in the dataset
TOP_CUISINES: List[str] = [
    "North Indian", "Chinese", "South Indian", "Mughlai", "Cafe",
    "Bakery", "Italian", "Fast Food", "Continental", "Desserts",
    "Biryani", "Street Food", "Ice Cream", "Andhra", "Thai",
    "Kerala", "Seafood", "Bengali", "Rajasthani", "Goan",
    "Japanese", "Korean", "Mexican", "Mediterranean", "Lebanese",
    "American", "French", "German", "Vietnamese", "Middle Eastern",
]


def encode_cuisines(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Multi-hot encode the cuisines column.

    Splits comma-separated cuisine strings, keeps only the top N most
    frequent cuisines, and creates binary columns for each.

    Args:
        df: DataFrame with a 'cuisines' column (comma-separated strings).
        top_n: Number of top cuisines to encode.

    Returns:
        pd.DataFrame: Original df with added cuisine binary columns.
    """
    df = df.copy()
    if "cuisines" not in df.columns:
        logger.warning("No 'cuisines' column found — skipping cuisine encoding")
        return df

    # Explode cuisines to find top N
    all_cuisines = (
        df["cuisines"]
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    for cuisine in all_cuisines:
        col_name = f"cuisine_{cuisine.lower().replace(' ', '_')}"
        df[col_name] = df["cuisines"].str.contains(cuisine, case=False, na=False).astype(int)

    logger.info("Encoded %d cuisine binary columns", len(all_cuisines))
    return df


def add_restaurant_features(df: pd.DataFrame, config: Optional[DabbaConfig] = None) -> pd.DataFrame:
    """Engineer features for the rating prediction model.

    Features created:
        - cost_for_two_bucket: ordinal bucket of restaurant cost
        - cuisine_count: number of cuisines offered
        - online_order_binary: 1/0 for Yes/No
        - book_table_binary: 1/0 for Yes/No
        - votes_log: log-transformed vote count
        - cuisine_* : multi-hot encoded top cuisines

    Args:
        df: Cleaned Zomato DataFrame.
        config: Project configuration.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    config = config or get_config()
    df = df.copy()
    logger.info("Engineering restaurant features — input shape: %s", df.shape)

    # Cost buckets
    if "cost_for_two" in df.columns:
        bins = [0, 200, 500, 1000, 2000, float("inf")]
        labels = ["budget", "affordable", "moderate", "premium", "luxury"]
        df["cost_for_two_bucket"] = pd.cut(
            df["cost_for_two"], bins=bins, labels=labels
        )

    # Cuisine count
    if "cuisines" in df.columns:
        df["cuisine_count"] = df["cuisines"].str.split(",").str.len().fillna(1).astype(int)

    # Binary flags
    if "online_order" in df.columns:
        df["online_order_binary"] = df["online_order"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    if "book_table" in df.columns:
        df["book_table_binary"] = df["book_table"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Log-transformed votes
    if "votes" in df.columns:
        df["votes_log"] = np.log1p(df["votes"].fillna(0))

    # Multi-hot cuisine encoding
    df = encode_cuisines(df)

    logger.info("Restaurant features complete — output shape: %s", df.shape)
    return df
