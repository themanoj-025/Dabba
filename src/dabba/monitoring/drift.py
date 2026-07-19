"""Drift detection for ML model monitoring.

Uses Kolmogorov-Smirnov two-sample tests (scipy.stats.ks_2samp) to
compare live inference batches against the training distribution for
each feature. Reports which features have drifted and at what
significance level.

WIRED INTO: Ops Monitor page's simulation loop — fires a drift alert
banner when drift is detected in a simulated batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection on a batch of inference data.

    Attributes:
        drifted_features: Dict mapping feature name to (p-value, statistic).
        total_features: Number of features checked.
        drifted_count: Number of features with significant drift.
        has_drift: True if any feature drifted beyond threshold.
        threshold: p-value threshold used.
        message: Human-readable summary.
    """

    drifted_features: Dict[str, tuple] = field(default_factory=dict)
    total_features: int = 0
    drifted_count: int = 0
    has_drift: bool = False
    threshold: float = 0.05
    message: str = ""


class DriftDetector:
    """Compares live inference batches against a reference distribution.

    Args:
        reference_data: DataFrame of training/inference reference data.
        config: Project configuration.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        config: Optional[DabbaConfig] = None,
    ):
        self.config = config or get_config()
        self.reference_stats: Dict[str, Dict] = {}
        self._fit(reference_data)

    def _fit(self, df: pd.DataFrame) -> None:
        """Compute reference statistics from training data.

        Args:
            df: Reference (training) DataFrame.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            values = df[col].dropna().values
            if len(values) > 0:
                self.reference_stats[col] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }
        logger.info(
            "DriftDetector fit on %d numeric features from %d samples",
            len(self.reference_stats), len(df),
        )

    def detect(self, batch: pd.DataFrame) -> DriftResult:
        """Run drift detection on a batch of inference data.

        Compares each numeric feature's distribution in the batch
        against the reference distribution using a two-sample KS test.

        Args:
            batch: DataFrame of new inference data.

        Returns:
            DriftResult with per-feature results and summary.
        """
        threshold = self.config.drift_ks_threshold
        drifted: Dict[str, tuple] = {}
        total = 0

        for col, ref_info in self.reference_stats.items():
            if col not in batch.columns:
                continue
            batch_values = batch[col].dropna().values
            if len(batch_values) < 5:
                continue

            total += 1
            ref_values = ref_info["values"]

            # Sample if too large (for speed)
            if len(ref_values) > self.config.drift_feature_sample:
                rng = np.random.RandomState(self.config.random_seed)
                ref_values = rng.choice(ref_values, self.config.drift_feature_sample, replace=False)
            if len(batch_values) > self.config.drift_feature_sample:
                rng = np.random.RandomState(self.config.random_seed)
                batch_values = rng.choice(batch_values, self.config.drift_feature_sample, replace=False)

            statistic, p_value = ks_2samp(ref_values, batch_values)

            if p_value < threshold:
                drifted[col] = (float(p_value), float(statistic))

        has_drift = len(drifted) > 0
        msg = (
            f"🚨 **Drift Detected!** {len(drifted)}/{total} features "
            f"have drifted (p<{threshold}). "
            f"Affected: {', '.join(list(drifted.keys())[:5])}"
            if has_drift else
            f"✅ No significant drift detected across {total} features."
        )

        return DriftResult(
            drifted_features=drifted,
            total_features=total,
            drifted_count=len(drifted),
            has_drift=has_drift,
            threshold=threshold,
            message=msg,
        )

    def generate_drift_batch(
        self,
        n_samples: int = 100,
        shift_scale: float = 2.0,
    ) -> pd.DataFrame:
        """Generate a synthetic batch with intentional drift for testing.

        Args:
            n_samples: Number of samples to generate.
            shift_scale: How much to shift the distribution (multiples of std).

        Returns:
            DataFrame with shifted distributions for testing.
        """
        rng = np.random.RandomState(self.config.random_seed)
        data: Dict[str, np.ndarray] = {}

        for col, info in self.reference_stats.items():
            mean = info["mean"]
            std = info["std"] if info["std"] > 0 else 1.0
            # Shift the mean to simulate drift
            data[col] = rng.normal(
                loc=mean + shift_scale * std,
                scale=std * 1.2,
                size=n_samples,
            )

        return pd.DataFrame(data)
