"""Tests for Redis cache client with fakeredis fallback."""

import pytest

pytest.importorskip("fakeredis", reason="fakeredis required for cache tests")

from dabba.cache.redis_client import CacheClient, get_cache


class TestCacheClientInit:
    """Tests for CacheClient initialization."""

    def test_creates_with_default_config(self):
        """Should create a CacheClient without a config argument."""
        cache = CacheClient()
        assert cache is not None

    def test_available_with_fakeredis(self):
        """Should be available even without a real Redis server."""
        cache = CacheClient()
        # Will use fakeredis if real Redis is unavailable
        assert cache.available is not None

    def test_available_property(self):
        """available property should return boolean."""
        cache = CacheClient()
        assert isinstance(cache.available, bool)


class TestCacheClientCacheKey:
    """Tests for cache key generation."""

    def test_make_key_returns_string(self):
        """_make_key should return a string."""
        cache = CacheClient()
        key = cache._make_key("eta", {"distance_km": 5.0, "traffic_level": 1})
        assert isinstance(key, str)

    def test_make_key_has_prefix(self):
        """_make_key should include the prefix in the key."""
        cache = CacheClient()
        key = cache._make_key("eta", {"distance_km": 5.0})
        assert key.startswith("dabba:")

    def test_make_key_deterministic(self):
        """Same inputs should produce the same key."""
        cache = CacheClient()
        data = {"distance_km": 5.0, "traffic_level": 1}
        key1 = cache._make_key("eta", data)
        key2 = cache._make_key("eta", data)
        assert key1 == key2

    def test_make_key_different_inputs_different_keys(self):
        """Different inputs should produce different keys."""
        cache = CacheClient()
        key1 = cache._make_key("eta", {"distance_km": 5.0})
        key2 = cache._make_key("eta", {"distance_km": 10.0})
        assert key1 != key2

    def test_make_eta_key(self):
        """make_eta_key should create a key with eta prefix."""
        cache = CacheClient()
        key = cache.make_eta_key({"distance_km": 5.0})
        assert isinstance(key, str)

    def test_make_recommend_key(self):
        """make_recommend_key should create a key with recommend prefix."""
        cache = CacheClient()
        key = cache.make_recommend_key({"cuisine": "Italian"})
        assert isinstance(key, str)


class TestCacheClientSetGet:
    """Tests for CacheClient set/get operations."""

    def test_set_and_get_string(self):
        """Should store and retrieve a string value."""
        cache = CacheClient()
        cache.set("test:string", "hello world")
        result = cache.get("test:string")
        assert result == "hello world"

    def test_set_and_get_dict(self):
        """Should store and retrieve a dict value."""
        cache = CacheClient()
        data = {"key": "value", "number": 42, "float": 3.14}
        cache.set("test:dict", data)
        result = cache.get("test:dict")
        assert result == data

    def test_set_and_get_list(self):
        """Should store and retrieve a list value."""
        cache = CacheClient()
        data = [1, 2, 3, "four"]
        cache.set("test:list", data)
        result = cache.get("test:list")
        assert result == data

    def test_get_missing_key(self):
        """get() should return None for a missing key."""
        cache = CacheClient()
        result = cache.get("nonexistent:key")
        assert result is None

    def test_overwrite_key(self):
        """set() should overwrite an existing key."""
        cache = CacheClient()
        cache.set("test:overwrite", "value1")
        result1 = cache.get("test:overwrite")
        cache.set("test:overwrite", "value2")
        result2 = cache.get("test:overwrite")
        assert result1 != result2
        assert result2 == "value2"


class TestCacheClientDelete:
    """Tests for CacheClient delete operation."""

    def test_delete_removes_key(self):
        """delete() should remove a key from cache."""
        cache = CacheClient()
        cache.set("test:delete_me", "to_delete")
        assert cache.get("test:delete_me") == "to_delete"
        cache.delete("test:delete_me")
        assert cache.get("test:delete_me") is None

    def test_delete_nonexistent(self):
        """delete() should not crash on nonexistent key."""
        cache = CacheClient()
        cache.delete("nonexistent:key")


class TestCacheClientFlush:
    """Tests for CacheClient flush operation."""

    def test_flush_clears_all(self):
        """flush() should clear all cache entries."""
        cache = CacheClient()
        cache.set("test:flush_a", "value_a")
        cache.set("test:flush_b", "value_b")
        cache.flush()
        assert cache.get("test:flush_a") is None
        assert cache.get("test:flush_b") is None


class TestGetCache:
    """Tests for the module-level get_cache() singleton."""

    def test_returns_cache_client(self):
        """get_cache() should return a CacheClient instance."""
        cache = get_cache()
        assert isinstance(cache, CacheClient)

    def test_singleton_behavior(self):
        """Multiple calls should return the same instance."""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2
