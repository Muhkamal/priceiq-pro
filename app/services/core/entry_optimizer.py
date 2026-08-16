"""
Entry Optimizer — Suggests limit entry prices instead of market.
Better fill = wider effective SL = higher survival rate.
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class EntryOptimizer:
    """
    For signal bots that execute manually:
    - Suggests a limit price slightly better than current market
    - This improves R:R without changing the chart setup
    """

    def suggest_limit(self, candles: List[Any], direction: str, pair: str) -> Dict[str, Any]:
        current = candles[-1].close
        atr = self._quick_atr(candles)
        decimals = 2 if "XAU" in pair else (0 if "BTC" in pair else 5)

        if direction == "buy":
            # Suggest entry at current close minus small buffer (pullback zone)
            # or at the low of the signal candle, whichever is better
            signal_low = candles[-1].low
            limit_price = min(current - atr * 0.15, signal_low + atr * 0.05)
            limit_price = round(max(limit_price, current - atr * 0.4), decimals)
        else:
            signal_high = candles[-1].high
            limit_price = max(current + atr * 0.15, signal_high - atr * 0.05)
            limit_price = round(min(limit_price, current + atr * 0.4), decimals)

        return {
            "market_entry": round(current, decimals),
            "limit_entry": limit_price,
            "limit_vs_market": round(abs(current - limit_price), decimals),
            "instruction": f"Set BUY LIMIT @ {limit_price}" if direction == "buy" else f"Set SELL LIMIT @ {limit_price}",
        }

    def _quick_atr(self, candles: List[Any], period: int = 5) -> float:
        if len(candles) < period:
            return 0.0001
        import numpy as np
        highs = [c.high for c in candles[-period:]]
        lows = [c.low for c in candles[-period:]]
        return float(np.mean([h - l for h, l in zip(highs, lows)]))


entry_optimizer = EntryOptimizer()
