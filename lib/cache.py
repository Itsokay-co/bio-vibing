"""Simple file-based cache for API responses."""

import hashlib
import json
import os
import time


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "cache.json")
DEFAULT_TTL = 3600  # 1 hour


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache(cache: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _make_key(provider: str, start_date: str, end_date: str, category: str) -> str:
    raw = f"{provider}:{start_date}:{end_date}:{category}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached(provider: str, start_date: str, end_date: str, category: str, ttl: int = DEFAULT_TTL):
    """Return cached data if fresh, else None."""
    cache = _load_cache()
    key = _make_key(provider, start_date, end_date, category)
    entry = cache.get(key)
    if entry and (time.time() - entry.get("ts", 0)) < ttl:
        return entry.get("data")
    return None


def set_cached(provider: str, start_date: str, end_date: str, category: str, data):
    """Store data in cache."""
    cache = _load_cache()
    key = _make_key(provider, start_date, end_date, category)
    cache[key] = {"ts": time.time(), "data": data}

    # Prune expired entries
    now = time.time()
    cache = {k: v for k, v in cache.items() if (now - v.get("ts", 0)) < DEFAULT_TTL * 24}

    _save_cache(cache)


def clear_cache():
    """Remove all cached data."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
