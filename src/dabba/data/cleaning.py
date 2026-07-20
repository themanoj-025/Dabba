"""Data cleaning utilities for Zomato and delivery datasets.

All cleaning strategies are documented and justified — no silent dropna.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

# Suppress FutureWarning from pandas fillna downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

from dabba.config import DabbaConfig, get_config  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zomato cleaning
# ---------------------------------------------------------------------------

def clean_zomato_rating(series: pd.Series) -> pd.Series:
    """Parse the messy Zomato rating column (e.g. '4.1/5', 'NEW', '-').
    
    Strategy:
        - Extract numeric portion before '/5'
        - Replace non-numeric sentinels ('NEW', '-', etc.) with NaN
        - Cast to float.

    Args:
        series: Raw 'rate' column.

    Returns:
        pd.Series: Cleaned numeric rating (float, 0-5 scale).
    """
    def _extract(value: object) -> Optional[float]:
        if pd.isna(value):
            return np.nan
        s = str(value).strip()
        match = re.match(r"([\d.]+)\s*/\s*5", s)
        if match:
            return float(match.group(1))
        return np.nan

    cleaned = series.map(_extract)
    n_nulls = cleaned.isna().sum()
    logger.info("Rating cleaned: %d nulls out of %d", n_nulls, len(cleaned))
    return cleaned


def clean_zomato_cost(series: pd.Series) -> pd.Series:
    """Parse the cost-for-two column (e.g. '1,200', '₹300').
    
    Strategy:
        - Remove non-numeric characters (comma, rupee sign, spaces)
        - Cast to float; unparseable values become NaN.

    Args:
        series: Raw 'approx_cost(for two people)' column.

    Returns:
        pd.Series: Cleaned cost as float (INR).
    """
    def _parse(value: object) -> Optional[float]:
        if pd.isna(value):
            return np.nan
        s = str(value).replace(",", "").replace("₹", "").replace(" ", "")
        try:
            return float(s)
        except ValueError:
            return np.nan

    cleaned = series.map(_parse)
    n_nulls = cleaned.isna().sum()
    logger.info("Cost cleaned: %d nulls out of %d", n_nulls, len(cleaned))
    return cleaned


def clean_zomato(df: pd.DataFrame, config: Optional[DabbaConfig] = None) -> pd.DataFrame:
    """Apply full cleaning pipeline to the raw Zomato dataframe.

    Steps:
        1. Remove exact duplicate rows.
        2. Clean 'rate' and 'approx_cost(for two people)' columns.
        3. Drop rows where the TARGET (rate) is missing — justified because
           we cannot impute a restaurant's rating without external data.
        4. For other columns: fill categoricals with mode, numerics with median.
        5. Normalize column names to snake_case.

    Args:
        df: Raw Zomato DataFrame.
        config: Project configuration.

    Returns:
        pd.DataFrame: Cleaned Zomato data.
    """
    config = config or get_config()
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    logger.info("Starting Zomato cleaning — input shape: %s", df.shape)

    # 1. Deduplicate
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows", n_before - len(df))

    # 2. Clean specific columns
    if "rate" in df.columns:
        df.loc[:, "rate"] = clean_zomato_rating(df["rate"])

    cost_col = "approx_cost(for two people)"
    if cost_col in df.columns:
        df.loc[:, cost_col] = clean_zomato_cost(df[cost_col])
        # Rename for consistency
        df = df.rename(columns={cost_col: "cost_for_two"})

    # 3. Drop rows with missing target (rate)
    n_before = len(df)
    df = df.dropna(subset=["rate"])
    logger.info("Dropped %d rows with missing rate", n_before - len(df))

    # 4. Fill remaining missing values
    for col in df.select_dtypes(include="object").columns:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val.iloc[0]).infer_objects(copy=False)
        else:
            df[col] = df[col].fillna("Unknown").infer_objects(copy=False)

    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median()).infer_objects(copy=False)

    # 5. Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    logger.info("Zomato cleaning complete — output shape: %s", df.shape)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Delivery cleaning
# ---------------------------------------------------------------------------

def clean_delivery(df: pd.DataFrame, config: Optional[DabbaConfig] = None) -> pd.DataFrame:
    """Apply full cleaning pipeline to the raw delivery dataframe.

    Steps:
        1. Remove exact duplicate rows.
        2. Strip whitespace from string columns.
        3. Parse 'Time_taken(min)' to float, removing the 'min' suffix.
        4. Validate lat/long ranges (India: lat 6–37, long 68–98).
        5. Remove physically impossible outliers (speed > 200 km/h implies error).
        6. Fill missing values with documented strategies.
        7. Normalize column names to snake_case.

    Args:
        df: Raw delivery DataFrame.
        config: Project configuration.

    Returns:
        pd.DataFrame: Cleaned delivery data.
    """
    config = config or get_config()
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    logger.info("Starting delivery cleaning — input shape: %s", df.shape)

    # 1. Deduplicate
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows", n_before - len(df))

    # 2. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 3. Parse target: Time_taken(min)
    target_col = "Time_taken(min)"
    if target_col in df.columns:
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
        df = df.rename(columns={target_col: "time_taken_min"})

    # 4. Validate lat/long
    lat_cols = [c for c in df.columns if "latitude" in c.lower()]
    lon_cols = [c for c in df.columns if "longitude" in c.lower()]
    for col in lat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = (df[col] < 6) | (df[col] > 37)
        n_bad = mask.sum()
        if n_bad > 0:
            logger.warning("Removed %d rows with invalid %s", n_bad, col)
            df = df[~mask]
    for col in lon_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = (df[col] < 68) | (df[col] > 98)
        n_bad = mask.sum()
        if n_bad > 0:
            logger.warning("Removed %d rows with invalid %s", n_bad, col)
            df = df[~mask]

    # 5. Remove physically impossible speed outliers (distance/time > 200 km/h)
    # Approximate: if haversine distance / time implies speed > 200 km/h
    # We'll do a rough filter after haversine is computed; for now just drop
    # rows where time_taken is extremely high (> 120 min) or missing.
    if "time_taken_min" in df.columns:
        n_before = len(df)
        df = df[df["time_taken_min"].notna() & (df["time_taken_min"] > 0)]
        df = df[df["time_taken_min"] <= 120]
        logger.info("Removed %d rows with invalid time_taken", n_before - len(df))

    # 6. Fill missing values
    for col in df.select_dtypes(include="object").columns:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val.iloc[0]).infer_objects(copy=False)
        else:
            df[col] = df[col].fillna("Unknown").infer_objects(copy=False)

    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median()).infer_objects(copy=False)

    # 7. Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    logger.info("Delivery cleaning complete — output shape: %s", df.shape)
    return df.reset_index(drop=True)
