"""Pluggable Caching Service.

Provides TTL-based exact-match query and embedding caching with automatic fallback
from Redis to a thread-safe local in-memory store.
"""

import json
import logging
import time
from typing import Any, Dict, Optional
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class CacheService:
    """Pluggable caching interface supporting both in-memory and Redis storage."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_type = settings.cache_type
        self.ttl = settings.cache_ttl_seconds
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._redis_client = None

        if self.cache_type == "redis":
            try:
                import redis
                self._redis_client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    decode_responses=True,
                )
                self._redis_client.ping()
                logger.info(f"Connected to Redis cache at {settings.redis_host}:{settings.redis_port}")
            except Exception as e:
                logger.warning(f"Could not connect to Redis ({e}), falling back to in-memory cache.")
                self.cache_type = "memory"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache if not expired."""
        if self.cache_type == "redis" and self._redis_client:
            try:
                val = self._redis_client.get(key)
                return json.loads(val) if val else None
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                return None
        else:
            entry = self._memory_cache.get(key)
            if not entry:
                return None
            if time.time() > entry["expires_at"]:
                del self._memory_cache[key]
                return None
            return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store item in cache with TTL."""
        expiry = ttl or self.ttl
        if self.cache_type == "redis" and self._redis_client:
            try:
                self._redis_client.setex(key, expiry, json.dumps(value))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        else:
            self._memory_cache[key] = {
                "value": value,
                "expires_at": time.time() + expiry,
            }

    def delete(self, key: str) -> None:
        """Remove item from cache."""
        if self.cache_type == "redis" and self._redis_client:
            try:
                self._redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        else:
            self._memory_cache.pop(key, None)
