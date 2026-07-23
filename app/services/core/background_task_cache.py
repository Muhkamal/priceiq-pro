"""
PriceIQ Pro — Background Task Cache v1.0

Prevents FastAPI timeout on heavy computation endpoints.

Problem:
    GET /api/v5/montecarlo  → runs 1,000 MC paths on request thread
    GET /api/v5/sensitivity → runs 50 backtests on request thread
    Both can take 5-30 seconds → FastAPI 30s timeout → 503 error

Solution:
    Background worker runs heavy computations on a schedule.
    Results cached with TTL.
    API endpoints return cached results instantly.
    If cache is stale: return last known result + "stale" flag.
    If cache is empty: return 202 Accepted + trigger computation.

Pattern:
    POST /api/v5/montecarlo/run   → triggers background computation
    GET  /api/v5/montecarlo       → returns cached result (instant)
    GET  /api/v5/montecarlo/status → shows if computation is running

Usage:
    cache = BackgroundTaskCache()

    # In lifespan startup:
    asyncio.create_task(cache.background_worker())

    # In API endpoint:
    result = cache.get("montecarlo")
    if result is None:
        cache.trigger("montecarlo")
        return {"status": "computing", "retry_after": 30}
    return result
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, Optional

logger = logging.getLogger(__name__)

# Default TTLs per task type
DEFAULT_TTLS = {
    "montecarlo":   3600,   # 1 hour
    "sensitivity":  7200,   # 2 hours
    "attribution":  1800,   # 30 minutes
    "full_status":  60,     # 1 minute
    "var_report":   300,    # 5 minutes
}


@dataclass
class CachedResult:
    key:        str
    value:      Any
    computed_at: float    # unix timestamp
    ttl:        int       # seconds
    stale:      bool = False

    def is_fresh(self) -> bool:
        return (time.monotonic() - self.computed_at) < self.ttl

    def age_seconds(self) -> float:
        return time.monotonic() - self.computed_at


class BackgroundTaskCache:
    """
    TTL-based result cache with background recomputation.
    Heavy computations run in a thread pool (CPU-bound).
    """

    def __init__(self):
        self._cache:     Dict[str, CachedResult] = {}
        self._running:   Dict[str, bool]         = {}  # key → is_computing
        self._tasks:     Dict[str, Callable]     = {}  # registered task fns
        self._queue:     asyncio.Queue           = asyncio.Queue()
        self._worker_running = False

    def register(self, key: str, fn: Callable, ttl: int = None):
        """Register a computation function for a cache key."""
        self._tasks[key] = fn
        if ttl:
            DEFAULT_TTLS[key] = ttl
        logger.debug(f"BackgroundTaskCache: registered '{key}' (TTL={DEFAULT_TTLS.get(key, 3600)}s)")

    def get(self, key: str) -> Optional[Dict]:
        """
        Get cached result. Returns None if never computed.
        Returns result with stale=True if TTL expired but computation in progress.
        """
        cached = self._cache.get(key)
        if cached is None:
            return None
        result = dict(cached.value) if isinstance(cached.value, dict) else {"value": cached.value}
        result["_cached_at"]  = datetime.fromtimestamp(
            cached.computed_at, tz=timezone.utc
        ).isoformat()
        result["_age_seconds"] = round(cached.age_seconds(), 1)
        result["_stale"]      = not cached.is_fresh()
        result["_computing"]  = self._running.get(key, False)
        return result

    def trigger(self, key: str) -> bool:
        """
        Trigger background recomputation for a key.
        Returns True if queued, False if already running.
        """
        if self._running.get(key):
            return False
        if key not in self._tasks:
            logger.warning(f"BackgroundTaskCache: no task registered for '{key}'")
            return False
        try:
            self._queue.put_nowait(key)
            return True
        except asyncio.QueueFull:
            return False

    def is_fresh(self, key: str) -> bool:
        cached = self._cache.get(key)
        return cached is not None and cached.is_fresh()

    def is_computing(self, key: str) -> bool:
        return self._running.get(key, False)

    async def background_worker(self):
        """
        Background task that processes computation requests.
        Run as: asyncio.create_task(cache.background_worker())
        """
        self._worker_running = True
        logger.info("BackgroundTaskCache: worker started")
        while self._worker_running:
            try:
                key = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                await self._compute(key)
            except asyncio.TimeoutError:
                # Check for auto-refresh of stale entries
                await self._auto_refresh()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"BackgroundTaskCache worker error: {e}")

    def stop(self):
        self._worker_running = False

    async def _compute(self, key: str):
        """Run computation in thread pool (handles both sync and async fns)."""
        if self._running.get(key):
            return
        fn = self._tasks.get(key)
        if not fn:
            return

        self._running[key] = True
        t0 = time.monotonic()
        try:
            # Run in thread pool if sync (most heavy computations are CPU-bound)
            if asyncio.iscoroutinefunction(fn):
                result = await fn()
            else:
                result = await asyncio.to_thread(fn)

            ttl = DEFAULT_TTLS.get(key, 3600)
            self._cache[key] = CachedResult(
                key=key, value=result,
                computed_at=time.monotonic(), ttl=ttl,
            )
            elapsed = (time.monotonic() - t0) * 1000
            logger.info(f"BackgroundTaskCache: '{key}' computed in {elapsed:.0f}ms, cached {ttl}s")

        except Exception as e:
            logger.error(f"BackgroundTaskCache: '{key}' computation failed: {e}")
        finally:
            self._running[key] = False

    async def _auto_refresh(self):
        """Auto-refresh entries that are stale and have registered tasks."""
        for key, cached in list(self._cache.items()):
            if not cached.is_fresh() and key in self._tasks and not self._running.get(key):
                # Only auto-refresh if entry is not too old (2× TTL = give up)
                if cached.age_seconds() < cached.ttl * 2:
                    await self._queue.put(key)

    def status(self) -> Dict:
        return {
            key: {
                "fresh":      self.is_fresh(key),
                "computing":  self.is_computing(key),
                "age_seconds": round(self._cache[key].age_seconds(), 1) if key in self._cache else None,
                "ttl":        DEFAULT_TTLS.get(key),
            }
            for key in set(list(self._cache.keys()) + list(self._tasks.keys()))
        }


# ── Updated API endpoints using cache ──────────────────────────────────────

def make_cached_v5_router(cache: BackgroundTaskCache):
    """
    Factory that returns the v5 router with caching applied to heavy endpoints.
    Use this instead of the uncached v5_dashboard_api router.

    Usage in main.py:
        from app.services.core.background_task_cache import BackgroundTaskCache, make_cached_v5_router
        cache = BackgroundTaskCache()
        asyncio.create_task(cache.background_worker())
        # Register heavy tasks:
        cache.register("montecarlo",  lambda: _run_montecarlo(v5))
        cache.register("attribution", lambda: v5.attribution.summary_table())
        cache.register("full_status", lambda: v5.get_system_status())
        app.include_router(make_cached_v5_router(cache))
    """
    from fastapi import APIRouter
    router = APIRouter(prefix="/api/v5/cached", tags=["V5 Cached"])

    @router.get("/montecarlo")
    async def montecarlo_cached():
        result = cache.get("montecarlo")
        if result is None:
            triggered = cache.trigger("montecarlo")
            return {
                "status":       "computing",
                "triggered":    triggered,
                "retry_after":  30,
                "message":      "Monte Carlo not yet computed. Retry in 30s.",
            }
        return result

    @router.post("/montecarlo/run")
    async def montecarlo_trigger():
        triggered = cache.trigger("montecarlo")
        return {
            "status":      "queued" if triggered else "already_running",
            "retry_after": 30,
        }

    @router.get("/attribution")
    async def attribution_cached():
        result = cache.get("attribution")
        if result is None:
            cache.trigger("attribution")
            return {"status": "computing", "retry_after": 15}
        return result

    @router.get("/status")
    async def status_cached():
        result = cache.get("full_status")
        if result is None:
            cache.trigger("full_status")
            return {"status": "computing", "retry_after": 5}
        return result

    @router.get("/cache/status")
    async def cache_status():
        return cache.status()

    return router
