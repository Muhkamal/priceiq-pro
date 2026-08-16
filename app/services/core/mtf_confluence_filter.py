"""
MTF Confluence Filter — Only take 1H signals that align with 4H trend.
Prevents buying into 4H resistance / selling into 4H support.
"""
import numpy as np
from typing import List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MTFConfluenceFilter:
    """
    Checks if 1H signal direction aligns with 4H trend.
    Uses 4H EMA(20) vs EMA(50) as trend proxy.
    """

    def __init__(self):
        self.min_bars = 55

    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        k = 2 / (period + 1)
        ema = [prices[0]]
        for p in prices[1:]:
            ema.append(p * k + ema[-1] * (1 - k))
        return np.array(ema)

    def _trend_direction(self, candles_4h: List[Any]) -> Optional[str]:
        if len(candles_4h) < 50:
            return None
        closes = np.array([c.close for c in candles_4h])
        ema20 = self._ema(closes, 20)[-1]
        ema50 = self._ema(closes, 50)[-1]
        if ema20 > ema50 * 1.002:
            return "buy"
        elif ema20 < ema50 * 0.998:
            return "sell"
        return "neutral"

    def check(
        self,
        direction_1h: str,
        candles_4h: List[Any],
        pair: str = "",
    ) -> tuple:
        """
        Returns: (allow: bool, confidence_boost: float, reason: str)
        confidence_boost: +0.05 if strong alignment, -0.10 if against trend
        """
        trend_4h = self._trend_direction(candles_4h)
        if trend_4h is None:
            return True, 0.0, "No 4H data"

        if trend_4h == "neutral":
            return True, 0.0, "4H neutral"

        if direction_1h == trend_4h:
            return True, 0.05, f"4H trend aligned ({trend_4h})"

        # Direction against 4H trend — block or heavily penalize
        return False, -0.15, f"1H {direction_1h} vs 4H {trend_4h} — blocked"


mtf_filter = MTFConfluenceFilter()
