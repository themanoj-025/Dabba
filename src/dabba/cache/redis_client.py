"""Redis-backed cache service for hot predictions.

Provides a ``CacheClient`` that wraps a Redis-compatible client.
If Redis is unavailable, it falls back to ``fakeredis`` (in-memory)
so the application works in dev mode without a Redis server.

Usage:
    >>> from src.dabba.cache.redis_client import CacheClient
    >>> cache = CacheClient()
    >>> cache.set("key", {"prediction": 28.4}, ttl_seconds=300)
    >>> result = cache.get("key")
    >>> result
    {"prediction": 28.4}
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from dabba.config import DabbaConfig, get_config

logger = logging.getLogger(__name__)


class CacheClient:
    """Simple Redis-compatible cache for model predictions.

    Uses the actual Redis client when available, falling back to
    ``fakeredis`` for development/testing.

    Args:
        config: Project configuration (for Redis URL).
    """

    def __init__(self, config: Optional[DabbaConfig] = None):
        self.config = config or get_config()
        self._client: Any = None
        self._fakeredis: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Redis or fakeredis client."""
        url = self.config.redis_url
        try:
            import redis as redis_lib

            self._client = redis_lib.from_url(
                url, decode_responses=True, socket_connect_timeout=2
            )
            self._client.ping()
            logger.info("Redis cache connected: %s", url)
        except Exception:
            logger.warning(
                "Redis unavailable at %s — falling back to fakeredis (in-memory)", url
            )
            self._client = None
            try:
                import fakeredis

                self._fakeredis = fakeredis.FakeStrictRedis(decode_responses=True)
                logger.info("Using fakeredis in-memory cache")
            except ImportError:
                logger.warning(
                    "fakeredis not installed — cache disabled. "
                    "Install with: pip install fakeredis"
                )
                self._fakeredis = None

    def _get_redis(self) -> Any:
        """Return the active Redis client or fakeredis instance."""
        if self._client is not None:
            return self._client
        return self._fakeredis

    @property
    def available(self) -> bool:
        """Whether the cache is available (Redis or fakeredis connected)."""
        return self._get_redis() is not None

    @staticmethod
    def _make_key(prefix: str, data: dict) -> str:
        """Create a deterministic cache key from input data.

        Args:
            prefix: Key prefix (e.g., 'eta', 'recommend').
            data: Input data dict.

        Returns:
            Cache key string: ``dabba:{prefix}:{sha256_hex}``.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        hash_digest = hashlib.sha256(serialized.encode()).hexdigest()[:16]
        return f"dabba:{prefix}:{hash_digest}"

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set a cache entry.

        Args:
            key: Cache key.
            value: Value to cache (will be JSON-serialized).
            ttl_seconds: Time-to-live in seconds.
        """
        r = self._get_redis()
        if r is None:
            return
        try:
            serialized = json.dumps(value, default=str)
            r.setex(key, ttl_seconds, serialized)
        except Exception as e:
            logger.warning("Cache set failed: %s", e)

    def get(self, key: str) -> Optional[Any]:
        """Get a cache entry.

        Args:
            key: Cache key.

        Returns:
            Deserialized value, or None if not found.
        """
        r = self._get_redis()
        if r is None:
            return None
        try:
            raw = r.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("Cache get failed: %s", e)
            return None

    def delete(self, key: str) -> None:
        """Delete a cache entry.

        Args:
            key: Cache key.
        """
        r = self._get_redis()
        if r is None:
            return
        try:
            r.delete(key)
        except Exception as e:
            logger.warning("Cache delete failed: %s", e)

    def flush(self) -> None:
        """Flush all cache entries (use with care)."""
        r = self._get_redis()
        if r is None:
            return
        try:
            r.flushdb()
            logger.info("Cache flushed")
        except Exception as e:
            logger.warning("Cache flush failed: %s", e)

    def make_eta_key(self, eta_request: dict) -> str:
        """Create a cache key for an ETA prediction request.

        Args:
            eta_request: The ETARequest fields as a dict.

        Returns:
            Cache key string.
        """
        return self._make_key("eta", eta_request)

    def make_recommend_key(self, recommend_request: dict) -> str:
        """Create a cache key for a recommendation request.

        Args:
            recommend_request: The RecommendRequest fields as a dict.

        Returns:
            Cache key string.
        """
        return self._make_key("recommend", recommend_request)


# Module-level singleton for reuse across routers
_cache_instance: Optional[CacheClient] = None


def get_cache(config: Optional[DabbaConfig] = None) -> CacheClient:
    """Return a singleton CacheClient instance.

    Args:
        config: Project configuration.

    Returns:
        CacheClient instance.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheClient(config)
    return _cache_instance
