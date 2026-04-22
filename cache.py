"""
cache.py — Deterministic in-memory cache with TTL for EvoDoc Drug Safety Engine.

Cache key: SHA-256 of sorted(proposed_medicines) + sorted(current_medications)
TTL: 1 hour (configurable via CACHE_TTL_SECONDS env var)
Thread-safe for async FastAPI usage.
"""

import hashlib
import json
import time
import asyncio
from typing import Optional, Any
from functools import lru_cache
import os

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default


class CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: int = CACHE_TTL_SECONDS):
        self.value = value
        self.expires_at = time.monotonic() + ttl

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class InMemoryCache:
    """
    Thread-safe async-compatible in-memory cache.
    Uses asyncio.Lock for safe concurrent writes.
    Includes passive TTL eviction on each read, and periodic cleanup.
    """

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._hit_count = 0
        self._miss_count = 0

    @staticmethod
    def build_key(proposed_medicines: list[str], current_medications: list[str]) -> str:
        """
        Deterministic cache key: order-independent hash of medicines + current meds.
        Sorted lowercase ensures [Aspirin, Warfarin] == [Warfarin, Aspirin].
        """
        normalised = {
            "medicines": sorted(m.lower().strip() for m in proposed_medicines),
            "current": sorted(m.lower().strip() for m in current_medications),
        }
        raw = json.dumps(normalised, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.is_expired():
                if entry is not None:
                    del self._store[key]  # passive eviction
                self._miss_count += 1
                return None
            self._hit_count += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
        async with self._lock:
            self._store[key] = CacheEntry(value, ttl)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def evict_expired(self) -> int:
        """Remove all expired entries. Returns count of evicted entries."""
        async with self._lock:
            expired = [k for k, v in self._store.items() if v.is_expired()]
            for k in expired:
                del self._store[k]
            return len(expired)

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": (
                self._hit_count / max(1, self._hit_count + self._miss_count)
            ),
        }


# Singleton cache instance shared across the app
drug_safety_cache = InMemoryCache()
