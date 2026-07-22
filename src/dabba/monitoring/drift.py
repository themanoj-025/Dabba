"""Drift detection for ML model monitoring with Slack alerting.

Drift detection uses Kolmogorov-Smirnov two-sample tests
(``scipy.stats.ks_2samp``) to compare live inference batches against
the training distribution for each feature.

Alerting sends a formatted Slack message via Incoming Webhook when
drift is detected. Alerts are rate-limited per-feature based on a
configurable cooldown window to prevent notification fatigue.

WIRED INTO:
    - Ops Monitor page's simulation loop (UI alert banner)
    - :meth:`DriftDetector.detect_and_alert` (Slack + UI + DB log)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


# ─── Shared result types ──────────────────────────────────────────────


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


@dataclass
class AlertResult:
    """Result of sending a drift alert to Slack.

    Attributes:
        channel: Slack channel the alert was addressed to.
        message: The alert text that was (or would have been) sent.
        sent: Whether the alert was successfully delivered.
        reason: Human-readable explanation of the outcome.
    """

    channel: str = ""
    message: str = ""
    sent: bool = False
    reason: str = ""


# ─── Slack webhook sender ───────────────────────────────────────────


def _send_slack_alert(
    webhook_url: str,
    message: str,
    channel: Optional[str] = None,
) -> AlertResult:
    """Send a formatted message to Slack via Incoming Webhook.

    Uses ``urllib.request`` (stdlib) — no extra dependencies.
    The message is wrapped in a Slack Block Kit payload with the
    optional channel override.

    Args:
        webhook_url: Slack Incoming Webhook URL.
        message: Plain-text message to send (Mrkdwn is supported).
        channel: Optional Slack channel override (e.g. "#alerts").

    Returns:
        AlertResult indicating success/failure.
    """
    payload: Dict[str, object] = {
        "text": message,
        "mrkdwn": True,
    }
    if channel:
        payload["channel"] = channel

    try:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            if 200 <= status < 300:
                logger.info("Slack alert sent successfully to %s", channel or "default")
                return AlertResult(
                    channel=channel or "default",
                    message=message,
                    sent=True,
                    reason=f"HTTP {status}",
                )
            else:
                logger.warning("Slack webhook returned HTTP %d", status)
                return AlertResult(
                    channel=channel or "default",
                    message=message,
                    sent=False,
                    reason=f"HTTP {status}",
                )
    except Exception as e:
        logger.warning("Failed to send Slack alert: %s", e)
        return AlertResult(
            channel=channel or "default",
            message=message,
            sent=False,
            reason=str(e),
        )


def _format_drift_message(result: DriftResult) -> str:
    """Format a DriftResult into a Slack-friendly alert message.

    Args:
        result: The drift detection result.

    Returns:
        Formatted message string with Mrkdwn formatting.
    """
    if not result.has_drift:
        return f"✅ No drift detected across {result.total_features} features."

    lines = [
        f"🚨 *Drift Detected!*",
        f"",
        f"• {result.drifted_count}/{result.total_features} features drifted "
        f"(p < {result.threshold})",
        f"",
        f"*Affected features:*",
    ]
    for feature_name, (p_value, statistic) in result.drifted_features.items():
        lines.append(f"  • `{feature_name}` — p={p_value:.4f}, KS={statistic:.3f}")

    lines.extend(
        [
            "",
            "_Recommended action: Review feature distributions and consider "
            "triggering a model retrain._",
        ]
    )
    return "\n".join(lines)


def _save_drift_log(
    result: DriftResult,
    alerted: bool = False,
    config: Optional[DabbaConfig] = None,
) -> None:
    """Persist drift detection results to the database (best-effort).

    Writes one ``DriftLog`` row per drifted feature. Logs a warning
    if the database is not available — never crashes the caller.

    Args:
        result: Drift detection result to persist.
        alerted: Whether a Slack alert was sent for this batch.
        config: Project configuration.
    """
    config = config or get_config()
    if not result.has_drift:
        return

    try:
        from dabba.database.session import get_db, init_db
        from dabba.database.models import DriftLog

        init_db(config)
        with get_db() as db:
            now = datetime.now(timezone.utc)
            for feature_name, (p_value, statistic) in result.drifted_features.items():
                log_entry = DriftLog(
                    feature_name=feature_name,
                    ks_statistic=statistic,
                    p_value=p_value,
                    threshold=result.threshold,
                    n_reference=0,  # Not tracked per-feature in current impl
                    n_batch=0,  # Not tracked per-feature in current impl
                    alerted=alerted,
                    detected_at=now,
                )
                db.add(log_entry)
            logger.info(
                "Logged %d drift events to database (alerted=%s)",
                result.drifted_count,
                alerted,
            )
    except Exception as e:
        logger.warning("Failed to persist drift log to database: %s", e)


# ─── DriftDetector ────────────────────────────────────────────────────


class DriftDetector:
    """Compares live inference batches against a reference distribution.

    Supports both in-memory drift detection (:meth:`detect`) and
    full alerting pipeline (:meth:`detect_and_alert`) that includes
    Slack notification + database logging + cooldown management.

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
        # Per-feature cooldown tracking: {feature_name: last_alert_timestamp}
        self._alert_cooldowns: Dict[str, float] = {}
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
            len(self.reference_stats),
            len(df),
        )

    def _is_alert_on_cooldown(self, feature_name: str) -> bool:
        """Check whether an alert for a feature is on cooldown.

        Compares the last alert timestamp against the configured
        cooldown window. Returns ``True`` if we should suppress
        the alert.

        Args:
            feature_name: Name of the drifted feature.

        Returns:
            ``True`` if the feature is in cooldown period.
        """
        now = time.time()
        cooldown_seconds = self.config.drift_alert_cooldown_hours * 3600
        last_alert = self._alert_cooldowns.get(feature_name)
        if last_alert is not None and (now - last_alert) < cooldown_seconds:
            remaining_hours = (cooldown_seconds - (now - last_alert)) / 3600
            logger.debug(
                "Alert for '%s' suppressed — %.1f hours remaining in cooldown",
                feature_name,
                remaining_hours,
            )
            return True
        return False

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
                ref_values = rng.choice(
                    ref_values, self.config.drift_feature_sample, replace=False
                )
            if len(batch_values) > self.config.drift_feature_sample:
                rng = np.random.RandomState(self.config.random_seed)
                batch_values = rng.choice(
                    batch_values, self.config.drift_feature_sample, replace=False
                )

            statistic, p_value = ks_2samp(ref_values, batch_values)

            if p_value < threshold:
                drifted[col] = (float(p_value), float(statistic))

        has_drift = len(drifted) > 0
        msg = (
            f"🚨 **Drift Detected!** {len(drifted)}/{total} features "
            f"have drifted (p<{threshold}). "
            f"Affected: {', '.join(list(drifted.keys())[:5])}"
            if has_drift
            else f"✅ No significant drift detected across {total} features."
        )

        return DriftResult(
            drifted_features=drifted,
            total_features=total,
            drifted_count=len(drifted),
            has_drift=has_drift,
            threshold=threshold,
            message=msg,
        )

    def detect_and_alert(
        self,
        batch: pd.DataFrame,
    ) -> DriftResult:
        """Drift detection + Slack notification + DB logging.

      Combines :meth:`detect` with the full alerting pipeline:

        1. Run drift detection on the batch
        2. If drift detected, check per-feature cooldown
        3. For features not in cooldown, send Slack alert
           (if ``slack_webhook_url`` is configured)
        4. Persist drift events to the database

        Args:
            batch: DataFrame of new inference data.

        Returns:
            The DriftResult from detection (unchanged by alerting).
        """
        result = self.detect(batch)

        if not result.has_drift:
            return result

        # Determine which features are not in cooldown
        features_to_alert = [
            f
            for f in result.drifted_features
            if not self._is_alert_on_cooldown(f)
        ]

        # Build a sub-result for features that should trigger alerts
        if features_to_alert:
            alert_features = {
                f: result.drifted_features[f] for f in features_to_alert
            }
            alert_result = DriftResult(
                drifted_features=alert_features,
                total_features=result.total_features,
                drifted_count=len(alert_features),
                has_drift=True,
                threshold=result.threshold,
                message="",
            )
            alert_message = _format_drift_message(alert_result)

            # Send Slack alert
            slack_url = self.config.slack_webhook_url
            slack_sent = False
            if slack_url:
                alert_outcome = _send_slack_alert(
                    slack_url,
                    alert_message,
                    channel=self.config.slack_alert_channel,
                )
                slack_sent = alert_outcome.sent
                if alert_outcome.sent:
                    # Update cooldown timestamps for alerted features
                    now = time.time()
                    for f in features_to_alert:
                        self._alert_cooldowns[f] = now
                    logger.info(
                        "Drift alerts sent for %d features: %s",
                        len(features_to_alert),
                        features_to_alert,
                    )
            else:
                logger.info(
                    "Drift detected but no slack_webhook_url configured — "
                    "alert suppressed. Affected features: %s",
                    features_to_alert,
                )

            # Persist to database
            _save_drift_log(alert_result, alerted=slack_sent, config=self.config)

        return result

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
