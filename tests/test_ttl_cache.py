"""Tests for the TTL + single-flight async cache (dabba.cache.ttl_cache).

Covers the two failure modes the cache exists to prevent:
thundering-herd duplicate loads and stale-cache-forever globals.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from dabba.cache.ttl_cache import TTLAsyncCache

pytestmark = pytest.mark.unit


class TestTTLAsyncCacheAsync:
    """Async single-flight + TTL behaviour."""

    def test_single_flight_one_load_for_concurrent_callers(self) -> None:
        """10 concurrent callers with a cold cache trigger exactly 1 load."""
        cache = TTLAsyncCache(ttl_seconds=60.0)
        calls: list[int] = []

        async def loader() -> str:
            calls.append(1)
            await asyncio.sleep(0.05)  # simulate slow joblib/torch load
            return "model"

        async def scenario() -> list[str]:
            return list(
                await asyncio.gather(
                    *(cache.get_or_load("m", loader) for _ in range(10))
                )
            )

        results = asyncio.run(scenario())
        assert results == ["model"] * 10
        assert len(calls) == 1, "thundering herd: loader ran more than once"

    def test_fresh_entry_does_not_reload(self) -> None:
        """A second call within the TTL is served from cache."""
        cache = TTLAsyncCache(ttl_seconds=60.0)
        calls: list[int] = []

        def loader() -> str:
            calls.append(1)
            return "model"

        asyncio.run(cache.get_or_load("m", loader))
        asyncio.run(cache.get_or_load("m", loader))
        assert len(calls) == 1

    def test_ttl_expiry_triggers_reload(self) -> None:
        """After the TTL lapses the next call reloads (stale cache fixed)."""
        cache = TTLAsyncCache(ttl_seconds=0.1)
        calls: list[int] = []

        def loader() -> str:
            calls.append(1)
            return f"v{len(calls)}"

        first = asyncio.run(cache.get_or_load("m", loader))
        time.sleep(0.15)
        second = asyncio.run(cache.get_or_load("m", loader))

        assert first == "v1"
        assert second == "v2"
        assert len(calls) == 2

    def test_async_loader_is_awaited(self) -> None:
        """Coroutine loaders are awaited directly (not sent to a thread)."""

        async def loader() -> str:
            return "async-value"

        cache = TTLAsyncCache(ttl_seconds=60.0)
        assert asyncio.run(cache.get_or_load("m", loader)) == "async-value"

    def test_loader_failure_returns_none_and_does_not_poison(self) -> None:
        """A failed load degrades to None; the next call retries."""

        async def bad() -> str:
            raise RuntimeError("boom")

        cache = TTLAsyncCache(ttl_seconds=60.0)
        assert asyncio.run(cache.get_or_load("m", bad)) is None

        async def good() -> str:
            return "ok"

        assert asyncio.run(cache.get_or_load("m", good)) == "ok"

    def test_invalidate_forces_reload(self) -> None:
        """invalidate() drops the entry so the next call reloads."""
        cache = TTLAsyncCache(ttl_seconds=60.0)
        state = {"v": 1}

        def loader() -> int:
            state["v"] += 1
            return state["v"]

        assert asyncio.run(cache.get_or_load("m", loader)) == 2
        cache.invalidate("m")
        assert asyncio.run(cache.get_or_load("m", loader)) == 3
        cache.invalidate()  # whole-cache
        assert asyncio.run(cache.get_or_load("m", loader)) == 4


class TestTTLAsyncCacheSync:
    """Sync variant used by Streamlit pages (no running event loop)."""

    def test_single_flight_across_threads(self) -> None:
        """5 threads with a cold cache trigger exactly 1 load."""
        cache = TTLAsyncCache(ttl_seconds=60.0)
        calls: list[int] = []

        def loader() -> str:
            calls.append(1)
            time.sleep(0.05)
            return "model"

        def worker() -> None:
            assert cache.get_or_load_sync("m", loader) == "model"

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(calls) == 1, "thundering herd in sync path"

    def test_ttl_expiry_triggers_reload(self) -> None:
        cache = TTLAsyncCache(ttl_seconds=0.1)
        calls: list[int] = []

        def loader() -> int:
            calls.append(1)
            return len(calls)

        assert cache.get_or_load_sync("m", loader) == 1
        time.sleep(0.15)
        assert cache.get_or_load_sync("m", loader) == 2

    def test_loader_failure_returns_none(self) -> None:
        def bad() -> str:
            raise RuntimeError("boom")

        cache = TTLAsyncCache(ttl_seconds=60.0)
        assert cache.get_or_load_sync("m", bad) is None
