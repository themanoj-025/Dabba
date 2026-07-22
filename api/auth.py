"""API Key authentication for Dabba FastAPI application.

Provides a FastAPI dependency that verifies the X-API-Key header
against the configured DABBA_API_KEY environment variable.

If DABBA_API_KEY is not set, authentication is skipped (dev-mode)
so the API remains usable without configuration during development.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


def verify_api_key(
    request: Request,
    config: DabbaConfig = Depends(get_config),
) -> None:
    """Verify the X-API-Key header matches the configured API key.

    If config.api_key is None (not configured), authentication is
    skipped so the API works in dev/demo mode without a key.

    Raises:
        HTTPException(401): If the key is missing or invalid.
    """
    # Skip auth when no key is configured (dev mode)
    if config.api_key is None:
        return

    api_key: Optional[str] = request.headers.get("X-API-Key")

    if api_key is None:
        logger.warning("Request missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header. Provide your API key to access this endpoint.",
            headers={"WWW-Authenticate": "API-Key"},
        )

    if api_key != config.api_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "API-Key"},
        )
