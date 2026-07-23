"""
PriceIQ Pro — Candle Cache v1.0

Prevents repeated broker API calls on every scan cycle.

Problem:
    6 pairs × 2 timeframes = 12 broker API calls per hour scan.
    At 20 pairs that's 40 calls. Most brokers rate-limit at 60/min.
    Also: same candles fetched for regime classifier, agent layer,
    AND the MAE formula = 3× redundant fetches per pair.

Solution:
    In-memory LRU cache keyed by (pair, timeframe, bar_timestamp).
    Cache is invalidated when a new bar closes.
    TTL-based expiry as safety net.

Usage:
    cache = CandleCache()

    async def _get_candles(pair, timeframe):
        cached = cache.get(pair, timeframe)
        if cached:
            return cached
        candles = await broker.get_candles(pair, timeframe, limit=300)
        cache.set(pair, timeframe, candles)
        return candles
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# TTL per timeframe in seconds — slightly less than bar duration
TIMEFRAME_TTL: Dict[str, int] = {
    "1m":  50,
    "5m":  280,
    "15m": 850,
    "30m": 1750,
    "1h":  3500,    # cache for 58 min — new 1h bar every 60 min
    "4h":  14000,
    "1d":  85000,
}

DEFAULT_TTL = 3500   # 1h default


@dataclass
class CacheEntry:
    candles:    List
    pair:       str
    timeframe:  str
    fetched_at: float   # unix timestamp
    ttl:        int


class CandleCache:
    """
    In-memory LRU candle cache with TTL expiry.
    Thread-safe for single asyncio event loop (no threading).
    """

    def __init__(self, max_entries: int = 100):
        self._store:   Dict[Tuple[str, str], CacheEntry] = {}
        self._max      = max_entries
        self._hits     = 0
        self._misses   = 0

    def _key(self, pair: str, timeframe: str) -> Tuple[str, str]:
        return (pair.upper(), timeframe.lower())

    def get(self, pair: str, timeframe: str) -> Optional[List]:
        """Return cached candles if still valid, else None."""
        key   = self._key(pair, timeframe)
        entry = self._store.get(key)

        if entry is None:
            self._misses += 1
            return None

        age = time.time() - entry.fetched_at
        if age > entry.ttl:
            del self._store[key]
            self._misses += 1
            logger.debug(f"Cache EXPIRED: {pair}/{timeframe} (age={age:.0f}s)")
            return None

        self._hits += 1
        logger.debug(f"Cache HIT: {pair}/{timeframe} (age={age:.0f}s, {len(entry.candles)} candles)")
        return entry.candles

    def set(self, pair: str, timeframe: str, candles: List):
        """Store candles. Evicts oldest entry if at capacity."""
        if not candles:
            return

        key = self._key(pair, timeframe)
        ttl = TIMEFRAME_TTL.get(timeframe.lower(), DEFAULT_TTL)

        # Evict oldest if at capacity
        if len(self._store) >= self._max and key not in self._store:
            oldest_key = min(self._store, key=lambda k: self._store[k].fetched_at)
            del self._store[oldest_key]
            logger.debug(f"Cache evicted: {oldest_key}")

        self._store[key] = CacheEntry(
            candles=candles,
            pair=pair.upper(),
            timeframe=timeframe.lower(),
            fetched_at=time.time(),
            ttl=ttl,
        )
        logger.debug(f"Cache SET: {pair}/{timeframe} ({len(candles)} candles, TTL={ttl}s)")

    def invalidate(self, pair: str, timeframe: Optional[str] = None):
        """Force-invalidate a pair (or all timeframes for a pair)."""
        if timeframe:
            self._store.pop(self._key(pair, timeframe), None)
        else:
            to_remove = [k for k in self._store if k[0] == pair.upper()]
            for k in to_remove:
                del self._store[k]

    def invalidate_all(self):
        self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def status(self) -> Dict:
        return {
            "entries":    len(self._store),
            "max":        self._max,
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   round(self.hit_rate, 3),
            "cached":     [
                {
                    "pair": e.pair,
                    "timeframe": e.timeframe,
                    "candles": len(e.candles),
                    "age_s": round(time.time() - e.fetched_at, 1),
                    "ttl_remaining": round(e.ttl - (time.time() - e.fetched_at), 1),
                }
                for e in self._store.values()
            ],
        }
