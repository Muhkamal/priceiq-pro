"""
Williams %R Agent — Mean reversion on extremes.
Oversold (<-80) in ranging = buy. Overbought (>-20) in ranging = sell.
"""
import numpy as np
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class WilliamsRAgent:
    """Mean-reversion agent using Williams %R oscillator."""

    NAME = "WilliamsRAgent"

    def __init__(self, lookback: int = 14, oversold: float = -80, overbought: float = -20):
        self.lookback = lookback
        self.oversold = oversold
        self.overbought = overbought

    def _williams_r(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        if len(highs) < self.lookback:
            return -50.0
        highest_high = np.max(highs[-self.lookback:])
        lowest_low = np.min(lows[-self.lookback:])
        if highest_high == lowest_low:
            return -50.0
        return -100 * (highest_high - closes[-1]) / (highest_high - lowest_low)

    def generate(self, candles: List[Any], pair: str = "") -> Optional[Any]:
        if len(candles) < self.lookback + 5:
            return None

        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])

        wr = self._williams_r(highs, lows, closes)

        # Regime filter: only trade extremes
        atr = np.mean(highs[-14:] - lows[-14:])
        adx = self._approx_adx(highs, lows, closes)
        if adx > 30:
            return None  # Don't mean-revert in strong trends

        direction = None
        if wr < self.oversold:
            direction = "buy"
        elif wr > self.overbought:
            direction = "sell"

        if not direction:
            return None

        # Dynamic SL/TP based on ATR
        sl_mult = 1.5
        tp_mult = 2.5
        entry = closes[-1]
        if direction == "buy":
            sl = entry - atr * sl_mult
            tp1 = entry + atr * tp_mult
        else:
            sl = entry + atr * sl_mult
            tp1 = entry - atr * tp_mult

        # Simple object to match your result interface
        return SimpleSignal(
            direction=direction,
            confidence=min(0.95, max(0.45, abs(wr + 50) / 50)),
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp1 + (tp1 - entry) if direction == "buy" else tp1 - (entry - tp1),
            agent=self.NAME,
            explanation=f"Williams %R = {wr:.1f} (extreme {'oversold' if direction=='buy' else 'overbought'})",
        )

    def _approx_adx(self, highs, lows, closes, period=14):
        """Lightweight ADX approximation for regime filter."""
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
