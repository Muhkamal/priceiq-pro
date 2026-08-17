"""
Order Block Detector — institutional level detection.
Finds the last strong momentum candle before a significant move.
These levels often act as hidden support/resistance.
"""
import numpy as np
from typing import List, Any, Dict
import logging

logger = logging.getLogger(__name__)


class OrderBlockDetector:
    """
    Order blocks: strong bullish/bearish candles before a reversal/extension.
    Institutions often defend these zones.
    """

    def find_blocks(self, candles: List[Any]) -> List[Dict]:
        if len(candles) < 10:
            return []
        results = []
        lookback = min(len(candles) - 1, 100)

        for i in range(2, lookback):
            c = candles[-(i + 1)]
            next_c = candles[-i]

            if c.range == 0:
                continue

            body_pct = abs(c.close - c.open) / c.range

            # Bearish order block (resistance): strong bearish candle + drop
            if c.close < c.open and body_pct > 0.7:
                move = (c.open - next_c.close) / c.range
                if move > 1.0:
                    level = (c.open + c.close) / 2
                    results.append({
                        "level": round(level, 6),
                        "type": "bearish_ob",
                        "strength": 0.65,
                        "was_resistance": True,
                        "was_support": False,
                    })

            # Bullish order block (support): strong bullish candle + rise
            if c.close > c.open and body_pct > 0.7:
                move = (next_c.close - c.open) / c.range
                if move > 1.0:
                    level = (c.open + c.close) / 2
                    results.append({
                        "level": round(level, 6),
                        "type": "bullish_ob",
                        "strength": 0.65,
                        "was_support": True,
                        "was_resistance": False,
                    })

        return results[:8]  # cap to avoid noise

    def find_psychological_levels(self, candles: List[Any], pair: str) -> List[Dict]:
        """Round numbers that markets respect."""
        if not candles:
            return []
        price = candles[-1].close
        levels = []

        if price > 100:
            interval = 50.0 if price > 500 else 1.0
        elif price > 1:
            interval = 0.01
        else:
            interval = 0.001

        margin = price * 0.05
        low_bound = price - margin
        high_bound = price + margin
        n = int(low_bound / interval)

        while n * interval <= high_bound:
            lvl = round(n * interval, 6)
            if lvl > 0:
                touches = self._count_touches(candles, lvl, 0.003)
                strength = 0.3 + min(0.5, touches / 8.0)
                levels.append({
                    "level": lvl,
                    "type": "psychological",
                    "strength": strength,
                    "was_support": True,
                    "was_resistance": True,
                })
            n += 1

        return levels

    def _count_touches(self, candles: List[Any], level: float, threshold: float) -> int:
        touches = 0
        for c in candles[-100:]:
            if abs(c.high - level) / level < threshold or abs(c.low - level) / level < threshold:
                touches += 1
        return touches


order_block_detector = OrderBlockDetector()
