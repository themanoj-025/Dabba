"""Circuit breaker for Dabba LLM API calls.

Prevents cascading failures by temporarily disabling LLM calls
after a configurable number of consecutive failures.

States:
    CLOSED: Normal operation — LLM calls proceed
    OPEN: LLM calls are blocked (fail fast)
    HALF_OPEN: Trial request to check if LLM recovered
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class LLMCircuitBreaker:
    """Circuit breaker for LLM API calls."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        cooldown_multiplier: float = 2.0,
    ) -> Any:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.cooldown_multiplier = cooldown_multiplier

        self._failure_count = 0
        self._state = "CLOSED"
        self._last_open_time = 0.0
        self._current_timeout = recovery_timeout

    def is_open(self) -> bool:
        """Check if the circuit breaker is open (LLM calls blocked)."""
        if self._state == "CLOSED":
            return False

        if self._state == "OPEN":
            if time.monotonic() - self._last_open_time >= self._current_timeout:
                self._state = "HALF_OPEN"
                logger.info("Dabba LLM circuit breaker half-open — allowing trial request")
                return False
            return True

        return False  # HALF_OPEN — allow trial request

    def record_success(self) -> None:
        """Record a successful LLM call. Resets the circuit breaker."""
        self._failure_count = 0
        self._state = "CLOSED"
        self._current_timeout = self.recovery_timeout
        logger.info("Dabba LLM circuit breaker closed — API available")

    def record_failure(self) -> None:
        """Record a failed LLM call. May open the circuit breaker."""
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            self._last_open_time = time.monotonic()
            self._current_timeout *= self.cooldown_multiplier
            logger.warning(
                "Dabba LLM circuit breaker OPEN after %d failures (timeout=%.1fs)",
                self._failure_count,
                self._current_timeout,
            )

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._failure_count = 0
        self._state = "CLOSED"
        self._current_timeout = self.recovery_timeout
        logger.info("Dabba LLM circuit breaker manually reset")


# Module-level singleton
llm_breaker = LLMCircuitBreaker()
