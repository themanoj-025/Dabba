"""Rate limiter instance for the Dabba FastAPI application.

Uses slowapi with IP-based rate limiting. The limiter is configured
in api/main.py and shared with routers via this module.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
