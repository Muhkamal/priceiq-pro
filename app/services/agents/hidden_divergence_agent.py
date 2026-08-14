"""
Hidden Divergence Agent — Trend continuation.
Bullish hidden div: price higher low, RSI lower low → buy in uptrend.
Bearish hidden div: price lower high, RSI higher high → sell in downtrend.
"""
import numpy as np
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class HiddenDivergenceAgent:
    """Trend-continuation via RSI hidden divergence."""

    NAME = "HiddenDivergenceAgent"

    def __init__(self, rsi_period: int = 14, lookback: int = 30):
        self.rsi_period = rsi_period
        self.lookback = lookback

    def _rsi(self, closes: np.ndarray) -> np.ndarray:
        delta = np.diff(closes)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(self.rsi_period) / self.rsi_period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(self.rsi_period) / self.rsi_period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _find_swing_lows(self, prices: np.ndarray, window: int = 5):
        """Return indices of local minima."""
        lows = []
        for i in range(window, len(prices) - window):
            if prices[i] == np.min(prices[i - window:i + window + 1]):
                lows.append(i)
        return lows

    def _find_swing_highs(self, prices: np.ndarray, window: int = 5):
        """Return indices of local maxima."""
        highs = []
        for i in range(window, len(prices) - window):
            if prices[i] == np.max(prices[i - window:i + window + 1]):
                highs.append(i)
        return highs

    def generate(self, candles: List[Any], pair: str = "") -> Optional[Any]:
        if len(candles) < self.lookback + 20:
            return None

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])

        rsi = self._rsi(closes)
        if len(rsi) < self.lookback:
            return None

        # Need trend context — use EMA slope
        ema_fast = np.mean(closes[-10:])
        ema_slow = np.mean(closes[-30:])
        uptrend = ema_fast > ema_slow

        direction = None
        confidence = 0.5

        if uptrend:
            # Look for bullish hidden divergence
            price_lows = self._find_swing_lows(closes[-self.lookback:])
            if len(price_lows) >= 2:
                p1, p2 = price_lows[-2], price_lows[-1]
                # Price: higher low
                if closes[-self.lookback:][p2] > closes[-self.lookback:][p1]:
                    r1 = rsi[p1]
                    r2 = rsi[p2]
                    # RSI: lower low
                    if r2 < r1 and r2 < 40:
                        direction = "buy"
                        confidence = min(0.85, 0.55 + (r1 - r2) / 100)

        else:
            # Look for bearish hidden divergence
            price_highs = self._find_swing_highs(closes[-self.lookback:])
            if len(price_highs) >= 2:
                p1, p2 = price_highs[-2], price_highs[-1]
                # Price: lower high
                if closes[-self.lookback:][p2] < closes[-self.lookback:][p1]:
                    r1 = rsi[p1]
                    r2 = rsi[p2]
                    # RSI: higher high
                    if r2 > r1 and r2 > 60:
                        direction = "sell"
                        confidence = min(0.85, 0.55 + (r2 - r1) / 100)

        if not direction:
            return None

        atr = np.mean(highs[-14:] - lows[-14:])
        entry = closes[-1]
        if direction == "buy":
            sl = entry - atr * 1.8
            tp1 = entry + atr * 2.8
        else:
            sl = entry + atr * 1.8
            tp1 = entry - atr * 2.8

        return SimpleSignal(
            direction=direction,
            confidence=confidence,
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp1 + (tp1 - entry) if direction == "buy" else tp1 - (entry - tp1),
            agent=self.NAME,
            explanation=f"Hidden divergence {'bullish' if direction=='buy' else 'bearish'} | RSI {r2:.1f}",
        )


class SimpleSignal:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
