"""
PriceIQ Pro — Live Price Validator v3.2
Updated with configurable thresholds, caching, and bid/ask awareness.

Checks that a signal's entry price matches the current market price
before the alert is sent. Prevents alerts with stale/wrong prices.
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

# Configurable thresholds — override via settings
MAX_DEVIATION_PIPS = getattr(settings, 'MAX_DEVIATION_PIPS', {
    "EURUSD": 10, "GBPUSD": 12, "USDJPY": 12,
    "USDCHF": 12, "AUDUSD": 12, "NZDUSD": 15,
    "USDCAD": 15, "EURGBP": 10, "EURJPY": 15,
    "GBPJPY": 20, "XAUUSD": 300, "XAGUSD": 400,
    "BTCUSD": 500, "ETHUSD": 500,
})
DEFAULT_MAX_PIPS = getattr(settings, 'DEFAULT_MAX_PIPS', 15)

# Cache TTL for live prices (seconds)
PRICE_CACHE_TTL = getattr(settings, 'PRICE_CACHE_TTL', 5)


def _pip_size(pair: str) -> float:
    """Get pip size for a pair."""
    pair = pair.upper()
    if "JPY" in pair:
        return 0.01
    if "XAU" in pair or "XAG" in pair:
        return 0.1
    if "BTC" in pair or "ETH" in pair:
        return 1.0
    return 0.0001


class PriceValidationResult(BaseModel):
    is_valid:       bool
    signal_entry:   float
    live_price:     Optional[float]
    live_bid:       Optional[float] = None
    live_ask:       Optional[float] = None
    deviation_pips: Optional[float]
    max_pips:       float
    reason:         Optional[str]
    checked_at:     datetime
    used_cached:    bool = False


class PriceValidator:
    """
    Validates signal entry price against live market price.
    Uses Twelve Data quote endpoint with caching.
    """

    def __init__(self, td_client=None):
        self.td_client = td_client
        # Simple in-memory cache: pair -> (price_data, timestamp)
        self._price_cache: Dict[str, tuple] = {}
        self._cache_lock = asyncio.Lock()

    async def validate(
        self,
        pair:         str,
        signal_entry: float,
        direction:    str,
        use_cache:    bool = True,
    ) -> PriceValidationResult:
        """
        Validate that signal_entry is close to current live price.

        Args:
            pair         : e.g. "EURUSD"
            signal_entry : entry price from signal
            direction    : "buy" or "sell"
            use_cache    : Use cached price if available (default True)

        Returns:
            PriceValidationResult with is_valid=True if price is acceptable.
        """
        now = datetime.now(timezone.utc)
        pair = pair.upper()
        max_pips = MAX_DEVIATION_PIPS.get(pair, DEFAULT_MAX_PIPS)
        pip_size = _pip_size(pair)

        # No TD client — skip validation with warning
        if not self.td_client:
            logger.warning(
                f"PriceValidator: No Twelve Data client — skipping live price check for {pair}. "
                f"Set TWELVE_DATA_API_KEY to enable this safety check."
            )
            return PriceValidationResult(
                is_valid=True,
                signal_entry=signal_entry,
                live_price=None,
                deviation_pips=None,
                max_pips=max_pips,
                reason="Live price check skipped — TWELVE_DATA_API_KEY not configured",
                checked_at=now,
            )

        # Check cache
        cached = await self._get_cached_price(pair)
        if use_cache and cached:
            live_price, bid, ask, used_cache = cached["price"], cached["bid"], cached["ask"], True
        else:
            # Fetch fresh price
            quote = await self.td_client.get_quote(pair)
            if quote:
                live_price = quote.get("close")
                bid = quote.get("bid")
                ask = quote.get("ask")
                used_cache = False
                # Update cache
                await self._set_cached_price(pair, live_price, bid, ask)
            else:
                live_price = await self.td_client.get_live_price(pair)
                bid = ask = None
                used_cache = False
                if live_price:
                    await self._set_cached_price(pair, live_price, None, None)

        if live_price is None:
            logger.warning(f"PriceValidator: Could not fetch live price for {pair} — allowing signal")
            return PriceValidationResult(
                is_valid=True,
                signal_entry=signal_entry,
                live_price=None,
                deviation_pips=None,
                max_pips=max_pips,
                reason="Live price unavailable — signal allowed with caution",
                checked_at=now,
            )

        # Use bid/ask based on direction for more accurate validation
        reference_price = live_price
        if direction == "buy" and ask:
            reference_price = ask  # Buy entry should be near ask
        elif direction == "sell" and bid:
            reference_price = bid  # Sell entry should be near bid

        deviation = abs(signal_entry - reference_price)
        deviation_pips = deviation / pip_size

        if deviation_pips > max_pips:
            reason = (
                f"Signal entry {signal_entry:.5f} is {deviation_pips:.1f} pips away from "
                f"live {direction} price {reference_price:.5f} (max allowed: {max_pips} pips). "
                f"Signal generated on stale data — do NOT trade this entry."
            )
            logger.warning(f"❌ Price validation FAILED for {pair}: {reason}")
            return PriceValidationResult(
                is_valid=False,
                signal_entry=signal_entry,
                live_price=live_price,
                live_bid=bid,
                live_ask=ask,
                deviation_pips=round(deviation_pips, 1),
                max_pips=max_pips,
                reason=reason,
                checked_at=now,
                used_cached=used_cache,
            )

        logger.info(
            f"✅ Price validated: {pair} entry {signal_entry:.5f} "
            f"vs live {reference_price:.5f} — deviation {deviation_pips:.1f} pips (max {max_pips})"
        )
        return PriceValidationResult(
            is_valid=True,
            signal_entry=signal_entry,
            live_price=live_price,
            live_bid=bid,
            live_ask=ask,
            deviation_pips=round(deviation_pips, 1),
            max_pips=max_pips,
            reason=None,
            checked_at=now,
            used_cached=used_cache,
        )

    async def validate_signal(self, signal) -> PriceValidationResult:
        """
        Convenience wrapper — validates a TradeSignal object.
        Returns full PriceValidationResult.
        """
        return await self.validate(
            pair=signal.pair,
            signal_entry=signal.entry_price,
            direction=signal.direction.value,
        )

    async def _get_cached_price(self, pair: str) -> Optional[Dict]:
        """Get cached price if not expired."""
        async with self._cache_lock:
            cached = self._price_cache.get(pair)
            if cached:
                price_data, timestamp = cached
                if datetime.now(timezone.utc) - timestamp < timedelta(seconds=PRICE_CACHE_TTL):
                    return price_data
                else:
                    del self._price_cache[pair]
            return None

    async def _set_cached_price(self, pair: str, price: float, bid: Optional[float], ask: Optional[float]):
        """Cache price with timestamp."""
        async with self._cache_lock:
            self._price_cache[pair] = (
                {"price": price, "bid": bid, "ask": ask},
                datetime.now(timezone.utc)
            )

    def clear_cache(self):
        """Clear price cache."""
        self._price_cache.clear()
