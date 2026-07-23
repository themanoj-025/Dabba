"""Drift-triggered retraining hook for Dabba.

When :class:`~dabba.monitoring.drift.DriftDetector` detects drift
above a configurable severity threshold, this module can spawn the
training pipeline as a subprocess.

Usage:
    >>> from dabba.monitoring.retrain import maybe_trigger_retraining
    >>> result = drift_detector.detect_and_alert(batch)
    >>> maybe_trigger_retraining(result, drift_threshold=0.3)

Architecture notes:

- **Subprocess-based**: Runs ``python -m dabba.pipeline`` in a
  detached subprocess.  No job queue, no Kubernetes Job — this is
  intentionally simple for a portfolio project.

- **Rate-limited**: Respects a configurable cooldown (default 6 hours)
  to avoid retraining storms on flapping drift signals.

- **Graceful fallback**: If the pipeline binary or model directory is
  not found, logs a warning and returns — never crashes the caller.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cooldown tracking
_last_retrain_time: float = 0.0


def maybe_trigger_retraining(
    drift_result: DriftResult,
    retrain_cooldown_hours: float = 6.0,
    drift_threshold: float = 0.3,
    project_root: Optional[Path] = None,
    dry_run: bool = False,
) -> bool:
    from dabba.monitoring.drift import DriftResult  # lazy import avoids circular dependency

    """Trigger model retraining if drift is severe enough.

    Compares the fraction of drifted features against
    ``drift_threshold``.  If exceeded and not in cooldown, spawns
    ``python -m dabba.pipeline --force`` as a subprocess.

    Args:
        drift_result: Output of :meth:`DriftDetector.detect`.
        retrain_cooldown_hours: Minimum hours between retrains.
        drift_threshold: Fraction of features that must drift to
            trigger retraining (e.g. 0.3 = 30%).
        project_root: Explicit project root.  Falls back to parent of
            ``src/dabba`` if not given.
        dry_run: If ``True``, log what would happen but don't spawn
            the subprocess.

    Returns:
        ``True`` if retraining was triggered, ``False`` otherwise.
    """
    global _last_retrain_time

    if not drift_result.has_drift or drift_result.total_features == 0:
        return False

    # Check severity: fraction of features that drifted
    drift_fraction = drift_result.drifted_count / drift_result.total_features
    if drift_fraction < drift_threshold:
        logger.info(
            "Drift fraction %.2f below threshold %.2f — no retrain needed",
            drift_fraction,
            drift_threshold,
        )
        return False

    # Check cooldown
    now = time.time()
    cooldown_seconds = retrain_cooldown_hours * 3600
    if _last_retrain_time > 0 and (now - _last_retrain_time) < cooldown_seconds:
        remaining_hours = (cooldown_seconds - (now - _last_retrain_time)) / 3600
        logger.info(
            "Retrain on cooldown — %.1f hours remaining (drift fraction=%.2f)",
            remaining_hours,
            drift_fraction,
        )
        return False

    # Determine project root
    if project_root is None:
        # Walk up from this file's location to find project root
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "src" / "dabba").is_dir() and (parent / "pyproject.toml").is_file():
                project_root = parent
                break
        if project_root is None:
            logger.warning("Could not determine project root — skipping retrain")
            return False

    # Build the command
    cmd = [
        sys.executable or "python",
        "-m",
        "dabba.pipeline",
        # Use --force to re-run even if models exist
        "--force",
    ]

    drifted_features = list(drift_result.drifted_features.keys())
    logger.warning(
        "⚠️  Drift threshold exceeded: %.0f%% of features drifted. "
        "Triggering retrain...  Features: %s",
        drift_fraction * 100,
        drifted_features[:8],
    )

    if dry_run:
        logger.info("[DRY RUN] Would run: %s", " ".join(cmd))
        return True

    try:
        # Use DABBA_RETRAIN=1 env var so the pipeline can detect
        # it was triggered by drift (for custom logging).
        env = os.environ.copy()
        env["DABBA_RETRAIN"] = "1"

        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        _last_retrain_time = now
        logger.info(
            "Retrain subprocess spawned (PID=%d) — drift fraction=%.2f, "
            "features=%s",
            process.pid,
            drift_fraction,
            drifted_features[:8],
        )
        return True

    except Exception as e:
        logger.error("Failed to spawn retrain subprocess: %s", e)
        return False
