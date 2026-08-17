"""
BreakoutPullbackAgent — Breakout → Pullback → Confirmation → Entry
Adds the 4-step discretionary structure as an 8th agent.
Only fires when price breaks a level, retests it, AND confirms with
a candlestick pattern at the retest zone.
"""
import numpy as np
from typing import List, Any, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BreakoutResult:
    direction: str = ""
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    reasoning: str = ""
    agent_used: str = "BreakoutPullbackAgent"
    regime_fit: float = 0.0
    fill_price: float = 0.0  # alias for entry_price


class BreakoutPullbackAgent:
    """
    1. BREAKOUT: Price closed above resistance / below support (last 5 bars)
    2. PULLBACK: Price retests the broken level within tolerance
    3. CONFIRMATION: Candlestick pattern at the retest (hammer/engulfing/pin)
    4. ENTRY: Signal fired on confirmation close
    """

    def __init__(self):
        self.lookback = 15          # how far back to look for breakout
        self.retest_bars = 5        # max bars allowed for pullback after breakout
        self.tolerance_pct = 0.0015  # 0.15% for FX, wider for XAU/BTC handled dynamically

    def analyze(self, candles: List[Any], pair: str = "", regime: str = "") -> Optional[BreakoutResult]:
        if len(candles) < 60:
            return None

        # Dynamic tolerance
        tol = self.tolerance_pct
        if "XAU" in pair:
            tol = 0.003  # 0.3% for gold
        elif "BTC" in pair:
            tol = 0.005  # 0.5% for crypto

        # Find swing levels (support/resistance)
        highs = [c.high for c in candles[-60:-1]]
        lows = [c.low for c in candles[-60:-1]]
        resistance = self._find_level(highs, mode="resistance")
        support = self._find_level(lows, mode="support")

        if not resistance or not support:
            return None

        current = candles[-1]
        prev = candles[-2] if len(candles) >= 2 else current

        # ── CHECK 1: BREAKOUT occurred recently ──
        breakout_dir = None
        breakout_level = None
        breakout_idx = None

        for i in range(-self.lookback, -1):
            c = candles[i]
            if c.close > resistance * (1 + tol * 0.3):  # broke above
                breakout_dir = "buy"
                breakout_level = resistance
                breakout_idx = i
                break
            elif c.close < support * (1 - tol * 0.3):   # broke below
                breakout_dir = "sell"
                breakout_level = support
                breakout_idx = i
                break

        if not breakout_dir or breakout_idx is None:
            return None

        # ── CHECK 2: PULLBACK to broken level ──
        # Price must have come back within tolerance of the broken level
        in_retest_zone = False
        for i in range(breakout_idx + 1, 0):
            c = candles[i]
            if breakout_dir == "buy":
                # Price came back down near resistance (now support)
                if abs(c.low - breakout_level) / breakout_level < tol or c.close < breakout_level * (1 + tol):
                    in_retest_zone = True
            else:
                # Price came back up near support (now resistance)
                if abs(c.high - breakout_level) / breakout_level < tol or c.close > breakout_level * (1 - tol):
                    in_retest_zone = True

        if not in_retest_zone:
            return None

        # ── CHECK 3: CONFIRMATION at retest ──
        # Current or previous candle must show rejection of the level
        confirmed = False
        pattern_name = ""

        for check_candle in [current, prev]:
            if breakout_dir == "buy":
                # Bullish confirmation at support: hammer, bullish engulfing, or strong close
                if self._is_bullish_confirmation(check_candle, breakout_level):
                    confirmed = True
                    pattern_name = "bullish_confirmation"
                    break
            else:
                if self._is_bearish_confirmation(check_candle, breakout_level):
                    confirmed = True
                    pattern_name = "bearish_confirmation"
                    break

        if not confirmed:
            return None

        # ── BUILD SIGNAL ──
        entry = current.close
        atr = self._atr(candles[-20:])

        if breakout_dir == "buy":
            sl = min(current.low, breakout_level - atr * 0.5)
            tp1 = entry + abs(entry - sl) * 2.0
            tp2 = entry + abs(entry - sl) * 3.5
            conf = 0.75
            if current.close > current.open and (current.close - current.low) / current.range > 0.7:
                conf = 0.88  # strong bullish close
            reasoning = (
                f"Breakout above {breakout_level:.5f} → "
                f"pullback retest → {pattern_name} @ support"
            )
        else:
            sl = max(current.high, breakout_level + atr * 0.5)
            tp1 = entry - abs(sl - entry) * 2.0
            tp2 = entry - abs(sl - entry) * 3.5
            conf = 0.75
            if current.close < current.open and (current.high - current.close) / current.range > 0.7:
                conf = 0.88
            reasoning = (
                f"Breakout below {breakout_level:.5f} → "
                f"pullback retest → {pattern_name} @ resistance"
            )

        # Regime fit
        regime_fit = 0.9 if regime in ("trending", "volatile") else 0.6

        return BreakoutResult(
            direction=breakout_dir,
            confidence=round(conf, 2),
            entry_price=round(entry, 5),
            fill_price=round(entry, 5),
            stop_loss=round(sl, 5),
            take_profit_1=round(tp1, 5),
            take_profit_2=round(tp2, 5),
            reasoning=reasoning,
            agent_used="BreakoutPullbackAgent",
            regime_fit=regime_fit,
        )

    def _find_level(self, prices: List[float], mode: str = "resistance") -> Optional[float]:
        """Find the most respected level via clustering."""
        if not prices:
            return None
        # Simple: use the mode/cluster of recent extremes
        if mode == "resistance":
            candidates = sorted(prices, reverse=True)[:10]
        else:
            candidates = sorted(prices)[:10]
        if not candidates:
            return None
        return float(np.median(candidates))

    def _is_bullish_confirmation(self, c, level: float) -> bool:
        """Hammer-like or engulfing at support."""
        if c.range == 0:
            return False
        # Close in upper half + lower wick (rejection of lower prices)
        close_pos = (c.close - c.low) / c.range
        lower_wick = (c.open if c.close > c.open else c.close) - c.low
        body = abs(c.close - c.open)
        has_lower_wick = lower_wick > body * 0.5 if body > 0 else lower_wick > c.range * 0.3
        return close_pos > 0.6 and has_lower_wick and c.close >= level * 0.998

    def _is_bearish_confirmation(self, c, level: float) -> bool:
        """Shooting-star-like or engulfing at resistance."""
        if c.range == 0:
            return False
        close_pos = (c.close - c.low) / c.range
        upper_wick = c.high - (c.open if c.close < c.open else c.close)
        body = abs(c.close - c.open)
        has_upper_wick = upper_wick > body * 0.5 if body > 0 else upper_wick > c.range * 0.3
        return close_pos < 0.4 and has_upper_wick and c.close <= level * 1.002

    def _atr(self, candles: List[Any], period: int = 14) -> float:
        if len(candles) < 2:
            return 0.0001
        trs = []
        for i in range(1, len(candles)):
            c = candles[i]
            p = candles[i - 1]
            tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
            trs.append(tr)
        return float(np.mean(trs[-period:])) if trs else 0.0001


breakout_pullback_agent = BreakoutPullbackAgent()
