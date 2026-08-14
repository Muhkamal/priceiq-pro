"""
Fractal Breakout Agent — Structure-based breakout.
Buy above recent fractal high, sell below recent fractal low.
"""
import numpy as np
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FractalAgent:
    """Breakout agent using Williams Fractals."""

    NAME = "FractalAgent"

    def __init__(self, lookback: int = 5, confirmation_bars: int = 1):
        self.lookback = lookback
        self.confirmation_bars = confirmation_bars

    def _fractal_highs(self, highs: np.ndarray) -> List[int]:
        """Indices where high is highest in ±2 bars."""
        fractals = []
        window = 2
        for i in range(window, len(highs) - window):
            if highs[i] == np.max(highs[i - window:i + window + 1]):
                fractals.append(i)
        return fractals

    def _fractal_lows(self, lows: np.ndarray) -> List[int]:
        """Indices where low is lowest in ±2 bars."""
        fractals = []
        window = 2
        for i in range(window, len(lows) - window):
            if lows[i] == np.min(lows[i - window:i + window + 1]):
                fractals.append(i)
        return fractals

    def generate(self, candles: List[Any], pair: str = "") -> Optional[Any]:
        if len(candles) < 20:
            return None

        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])

        f_highs = self._fractal_highs(highs)
        f_lows = self._fractal_lows(lows)

        if not f_highs or not f_lows:
            return None

        recent_high = highs[f_highs[-1]]
        recent_low = lows[f_lows[-1]]
        current = closes[-1]

        # ADX filter — only trade breakouts when ADX > 20 (trend strength)
        adx = self._approx_adx(highs, lows, closes)
        if adx < 20:
            return None

        direction = None
        if current > recent_high:
            direction = "buy"
        elif current < recent_low:
            direction = "sell"

        if not direction:
            return None

        atr = np.mean(highs[-14:] - lows[-14:])
        entry = current
        if direction == "buy":
            sl = recent_low - atr * 0.5
            tp1 = entry + abs(entry - sl) * 2.0
        else:
            sl = recent_high + atr * 0.5
            tp1 = entry - abs(sl - entry) * 2.0

        return SimpleSignal(
            direction=direction,
            confidence=min(0.85, 0.50 + adx / 100),
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp1 * 1.5 if direction == "buy" else tp1 - (tp1 - entry) * 0.5,
            agent=self.NAME,
            explanation=f"Fractal {'high' if direction=='buy' else 'low'} breakout | ADX={adx:.1f}",
        )

    def _approx_adx(self, highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return 0
        tr1 = highs[-period:] - lows[-period:]
        tr2 = np.abs(highs[-period:] - np.roll(closes, 1)[-period:])
        tr3 = np.abs(lows[-period:] - np.roll(closes, 1)[-period:])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return np.mean(tr) / (np.mean(closes[-period:]) + 1e-9) * 100


class SimpleSignal:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
