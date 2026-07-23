"""Observability module for Dabba — structured JSON logging, Prometheus metrics,
and tracing spans.

Usage:
    >>> from dabba.observability import setup_logging, get_json_logger
    >>> setup_logging()
    >>> logger = get_json_logger(__name__)
    >>> logger.info("Model loaded", extra={"model": "eta", "mae": 5.79})
"""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

# ─── Request ID (async-safe via contextvars) ─────────────────────────
# DO NOT use threading.local() — FastAPI is async and multiple
# coroutines on the same thread would leak request IDs.

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_dabba_request_id", default=""
)


def generate_request_id() -> str:
    """Generate a unique request ID (short UUID hex)."""
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str) -> None:
    """Store the current request ID in the async-safe context variable.

    Called by the FastAPI middleware at the start of each request.
    The value is automatically scoped to the current async context.

    Args:
        request_id: The request ID string.
    """
    _request_id_var.set(request_id)


def get_request_id() -> str:
    """Return the current request ID, or empty string if not set."""
    return _request_id_var.get()


# ─── JSON Formatter ──────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON lines.

    Every log line includes:
        - timestamp (ISO 8601 with timezone)
        - level
        - logger (name)
        - message
        - request_id (if set via :func:`set_request_id`)
        - Extra fields passed via the ``extra`` dict.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include request_id if available
        rid = get_request_id()
        if rid:
            log_entry["request_id"] = rid

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include any extra fields passed via the ``extra`` dict
        # (skip standard LogRecord attributes)
        standard_attrs = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "levelname", "levelno", "lineno",
            "message", "module", "msecs", "msg", "name", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "thread", "threadName", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                if isinstance(value, str) and len(value) > 500:
                    value = value[:500] + "..."
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


# ─── JSON Logger helper ──────────────────────────────────────────────


def get_json_logger(name: str) -> logging.Logger:
    """Get a logger that outputs JSON when the root handler is configured.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a JSON formatter.

    Removes any existing handlers and adds a single stream handler
    that outputs JSON lines to stderr.

    Safe to call multiple times — only the first call configures.

    Args:
        level: Log level string (e.g., 'INFO', 'DEBUG').
    """
    root_logger = logging.getLogger()

    # Skip if already configured with a JSON handler
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and isinstance(
            handler.formatter, JSONFormatter
        ):
            return

    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)


# ─── Prometheus metrics ──────────────────────────────────────────────

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Stub classes for when prometheus_client is not installed
    class Counter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> Counter:
            return self

        def inc(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Histogram:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> Histogram:
            return self

        def observe(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Gauge:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> Gauge:
            return self

        def set(self, *args: Any, **kwargs: Any) -> None:
            pass

    def generate_latest(*args: Any, **kwargs: Any) -> bytes:
        return b""


# ─── Metric definitions ──────────────────────────────────────────────

http_requests_total = Counter(
    "dabba_http_requests_total",
    "Total HTTP requests by method, endpoint, and status",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "dabba_http_request_duration_seconds",
    "HTTP request duration in seconds by endpoint",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

drift_events_total = Counter(
    "dabba_drift_events_total",
    "Total drift detection events by feature",
    ["feature"],
)

concierge_tool_calls_total = Counter(
    "dabba_concierge_tool_calls_total",
    "Total concierge tool calls by tool name",
    ["tool"],
)

concierge_loop_duration_seconds = Histogram(
    "dabba_concierge_loop_duration_seconds",
    "Concierge ReAct loop iteration duration by step",
    ["step"],
)

models_loaded = Gauge(
    "dabba_models_loaded",
    "Whether each model type is loaded (1=loaded, 0=missing)",
    ["model"],
)


def metrics_endpoint() -> str:
    """Return the Prometheus metrics text.

    Returns:
        Prometheus text-format metrics as a string.
    """
    if PROMETHEUS_AVAILABLE:
        return generate_latest().decode("utf-8")
    return "# Prometheus client not available"
