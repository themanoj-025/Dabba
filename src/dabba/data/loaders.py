"""Data loading utilities for the Dabba project.

Provides functions to load raw CSV files with schema verification.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


def load_zomato(config: Optional[DabbaConfig] = None) -> pd.DataFrame:
    """Load the raw Zomato Bangalore Restaurants dataset.

    Args:
        config: Project configuration. Uses default if None.

    Returns:
        pd.DataFrame: Raw Zomato data.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    config = config or get_config()
    path = config.zomato_path
    logger.info("Loading Zomato dataset from %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Zomato dataset not found at {path}. "
            "Please download it first — see README for instructions."
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded Zomato dataset: shape=%s, columns=%s", df.shape, list(df.columns))
    return df


def load_delivery(config: Optional[DabbaConfig] = None) -> pd.DataFrame:
    """Load the raw food delivery time dataset.

    Args:
        config: Project configuration. Uses default if None.

    Returns:
        pd.DataFrame: Raw delivery data.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    config = config or get_config()
    path = config.delivery_path
    logger.info("Loading delivery dataset from %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Delivery dataset not found at {path}. "
            "Please download it first — see README for instructions."
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded delivery dataset: shape=%s, columns=%s", df.shape, list(df.columns))
    return df


def describe_dataset(df: pd.DataFrame, name: str) -> None:
    """Print schema summary for a loaded DataFrame.

    Args:
        df: The DataFrame to describe.
        name: Human-readable name for logging.
    """
    logger.info("=== %s Schema Summary ===", name)
    logger.info("Shape: %s", df.shape)
    logger.info("Columns:\n%s", df.dtypes.to_string())
    logger.info("First 5 rows:\n%s", df.head().to_string())
    logger.info("Null counts:\n%s", df.isnull().sum().to_string())
    logger.info("============================")
