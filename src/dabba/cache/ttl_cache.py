"""TTL + single-flight cache for heavy module-level artifacts.

Fixes two failure modes of the classic ``_remote_model = None`` global:

1. **Thundering herd** — N concurrent async requests all see an empty
   cache and each kick off the expensive load. A single
   :class:`asyncio.Lock` guarantees exactly one load runs; everyone
   else awaits the same result (single-flight).
2. **Stale cache** — a module-level global lives forever, so a model
   retrained on disk is never picked up. Every entry carries a TTL;
   after expiry the next caller reloads.

Usage::

    _cache = TTLAsyncCache(ttl_seconds=300)
    loader = _Loader()          # any callable, sync or async

    async def get_model():
        return await _cache.get_or_load("eta", loader.load)

Sync loaders run via ``asyncio.to_thread`` so a slow ``joblib.load``
never blocks the event loop; async loaders are awaited directly.

``get_or_load_sync`` is provided for Streamlit pages (no running loop)
and uses a :class:`threading.Lock` for the same single-flight guarantee.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 300.0


class TTLAsyncCache:
    """Per-key TTL cache with async single-flight loading."""

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._values: dict[str, Any] = {}
        self._expires_at: dict[str, float] = {}
        self._async_locks: dict[str, asyncio.Lock] = {}
        self._sync_locks: dict[str, threading.Lock] = {}
        self._lock_creation_guard = threading.Lock()

    def _is_fresh(self, key: str) -> bool:
        if key not in self._values:
            return False
        return time.monotonic() < self._expires_at.get(key, 0.0)

    def _get_async_lock(self, key: str) -> asyncio.Lock:
        # asyncio.Lock creation is not coroutine-safe across the first
        # gather tick in every case; serialize creation with a threading
        # guard (held for microseconds).
        with self._lock_creation_guard:
            if key not in self._async_locks:
                self._async_locks[key] = asyncio.Lock()
            return self._async_locks[key]

    def _get_sync_lock(self, key: str) -> threading.Lock:
        with self._lock_creation_guard:
            if key not in self._sync_locks:
                self._sync_locks[key] = threading.Lock()
            return self._sync_locks[key]

    def invalidate(self, key: str | None = None) -> None:
        """Drop one key or the whole cache (e.g. after a retrain)."""
        if key is None:
            self._values.clear()
            self._expires_at.clear()
        else:
            self._values.pop(key, None)
            self._expires_at.pop(key, None)

    async def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any] | Callable[[], Awaitable[Any]],
    ) -> Any:
        """Return the cached value for ``key``, loading it once if stale.

        Args:
            key: Cache key (e.g. ``"eta_model"``, ``"faiss_index"``).
            loader: Zero-arg callable returning the artifact. May be sync
                (executed in a worker thread) or async.

        Returns:
            The cached or freshly loaded artifact (``None`` if the loader
            failed — callers keep their existing fallback behaviour).
        """
        if self._is_fresh(key):
            return self._values[key]

        lock = self._get_async_lock(key)
        async with lock:
            # Double-check inside the lock: another coroutine may have
            # finished the load while we awaited the lock acquisition.
            if self._is_fresh(key):
                return self._values[key]

            try:
                if inspect.iscoroutinefunction(loader):
                    loader_callable = cast(
                        "Callable[[], Awaitable[Any]]", loader
                    )
                    value = await loader_callable()
                else:
                    value = await asyncio.to_thread(loader)
            except Exception as e:
                logger.warning("Cache load failed for %s: %s", key, e)
                return None

            self._values[key] = value
            self._expires_at[key] = time.monotonic() + self._ttl
            return value

    def get_or_load_sync(
        self,
        key: str,
        loader: Callable[[], Any],
    ) -> Any:
        """Sync variant for Streamlit pages (no running event loop).

        Same semantics: single-flight via a per-key ``threading.Lock``
        and TTL-based expiry. On load failure returns ``None``.
        """
        if self._is_fresh(key):
            return self._values[key]

        lock = self._get_sync_lock(key)

        with lock:
            if self._is_fresh(key):
                return self._values[key]

            try:
                value = loader()
            except Exception as e:
                logger.warning("Cache load failed for %s: %s", key, e)
                return None

            self._values[key] = value
            self._expires_at[key] = time.monotonic() + self._ttl
            return value


__all__ = ["TTLAsyncCache"]
