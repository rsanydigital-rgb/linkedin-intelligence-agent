"""
cache.py
--------
Redis-backed query caching with 24-hour TTL.

Phase 4 feature. Gracefully degrades when Redis is unavailable (no crash,
just skips cache and falls through to the live pipeline).

Usage:
  from app.cache import get_cached_result, set_cached_result, make_cache_key

  key = make_cache_key(topic)
  cached = get_cached_result(key)
  if cached:
      return cached          # fast path
  result = run_pipeline(topic)
  set_cached_result(key, result)
"""

import os
import json
import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# TTL in seconds — default 24 hours
_CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 60 * 60)))
_REDIS_URL = os.getenv("REDIS_URL", "")

# Lazy singleton — only connect when first used
_redis_client = None


def _get_client():
    """Return a Redis client, or None if Redis is not configured / unavailable."""
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    if not _REDIS_URL:
        logger.debug("REDIS_URL not set — caching disabled")
        return None

    try:
        import redis  # type: ignore

        client = redis.from_url(_REDIS_URL, decode_responses=True, socket_connect_timeout=3)
        client.ping()
        _redis_client = client
        logger.info("Redis cache connected at %s (TTL=%ds)", _REDIS_URL, _CACHE_TTL)
        return _redis_client
    except Exception as exc:
        logger.warning("Redis unavailable — caching disabled: %s", exc)
        return None


def make_cache_key(topic: str) -> str:
    """
    Produce a deterministic cache key for a topic string.

    The key is a SHA-256 fingerprint of the lower-cased, whitespace-normalised
    topic so that 'Robotic Technologies' and 'robotic technologies' share a key.
    """
    normalised = " ".join(topic.lower().split())
    digest = hashlib.sha256(normalised.encode()).hexdigest()[:24]
    return f"lia:v1:{digest}"


def get_cached_result(key: str) -> Optional[Dict[str, Any]]:
    """
    Return the cached result dict for *key*, or None if not found / expired.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        raw = client.get(key)
        if raw is None:
            logger.debug("Cache MISS for key %s", key)
            return None
        logger.info("Cache HIT for key %s", key)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Cache read error for key %s: %s", key, exc)
        return None


def set_cached_result(key: str, result: Dict[str, Any]) -> bool:
    """
    Store *result* under *key* with the configured TTL.

    Returns True on success, False on any failure (caller continues normally).
    """
    client = _get_client()
    if client is None:
        return False

    try:
        client.setex(key, _CACHE_TTL, json.dumps(result, default=str))
        logger.info("Cached result under key %s (TTL=%ds)", key, _CACHE_TTL)
        return True
    except Exception as exc:
        logger.warning("Cache write error for key %s: %s", key, exc)
        return False


def invalidate(key: str) -> bool:
    """Delete a cached entry explicitly (e.g. for forced refresh)."""
    client = _get_client()
    if client is None:
        return False

    try:
        deleted = client.delete(key)
        logger.info("Invalidated cache key %s (deleted=%d)", key, deleted)
        return bool(deleted)
    except Exception as exc:
        logger.warning("Cache invalidation error for key %s: %s", key, exc)
        return False


def cache_stats() -> Dict[str, Any]:
    """Return basic cache statistics for the /metrics endpoint."""
    client = _get_client()
    if client is None:
        return {"status": "disabled", "redis_url": None}

    try:
        info = client.info("stats")
        keys = client.dbsize()
        return {
            "status": "connected",
            "redis_url": _REDIS_URL,
            "total_keys": keys,
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "ttl_seconds": _CACHE_TTL,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}