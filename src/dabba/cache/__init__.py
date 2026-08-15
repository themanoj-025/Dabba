"""Prediction caching layer for the Dabba API.

Provides a Redis-backed cache with a ``fakeredis`` fallback when
Redis is unavailable — allowing the application to work in dev
mode without a Redis server running.
"""

from src.dabba.cache.redis_client import CacheClient, get_cache

__all__ = ["CacheClient", "get_cache"]
