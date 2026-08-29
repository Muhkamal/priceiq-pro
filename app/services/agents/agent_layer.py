"""
PriceIQ Pro — Multi-Agent Strategy Layer v1.3 (SAFE + 9 AGENTS)

Four competing strategy agents + COT + News with SAFETY GUARDS for live trading.

SAFETY CHANGES from v1.1:
    1. REMOVED fallback mode — if no valid signal, returns None (no trade)
    2. ADDED trend filter to MeanReversionAgent — blocks counter-trend signals
    3. ADDED regime-lock — only agents with regime_fit >= 0.5 can win
    4. RAISED minimum confidence to 0.50 (was 0.40)
    5. ADDED strong-trend detection to prevent catching falling knives
    
V1.3 CHANGES:
    6. Added `pair: str = None` to all evaluate() signatures for COT/News routing
    7. Integrated COTReportAgent and NewsSentimentAgent

V1.4 CHANGES:
    8. Added book + structure filters to MeanReversionAgent (A/B test mode)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ═══ V1.3: IMPORT EXTERNAL AGENTS ═══
from .cot_report_agent import COTReportAgent
from .news_sentiment_agent import NewsSentimentAgent

# ═══ V1.4: IMPORT BOOK FILTERS ═══
try:
    from app.services.core import mr_book_filters
except Exception:
    mr_book_filters = None

logger = logging.getLogger(__name__)


def _safe_div(a, b, default=0.0):
    try:
        return a / b if b != 0 and np.isfinite(b) else default
    except Exception:
        return default


def _safe_mean(vals, default=0.0):
    clean = [v for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(clean)) if clean else default


def _ema(values: List[float], period: int) -> List[float]:
    if not values or period < 1:
        return values or []
    k = 2 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = _safe_mean(gains[:period])
    avg_l = _safe_mean(losses[:period])
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    return 100 - _safe_div(100, 1 + _safe_div(avg_g, avg_l, 1.0))


def _atr(candles, period=14) -> float:
    trs = []
    for i in range(1, min(period + 2, len(candles))):
        c, p = candles[-i], candles[-i - 1]
        tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        trs.append(tr)
    return _safe_mean(trs, 0.001)


# ============================================================
# AGENT SIGNAL
# ============================================================

@dataclass
class AgentSignal:
    agent_name:     str
    direction:      Optional[str]
    confidence:     float
    win_probability: float
    expected_value: float
    stop_distance:  float
    tp1_distance:   float
    tp2_distance:   float
    regime_fit:     float
    reasoning:      str
    raw_features:   Dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        # SAFETY: raised minimum confidence to 0.50
        return (
            self.direction is not None
            and self.confidence >= 0.50
            and self.stop_distance > 0
            and self.tp1_distance > 0
        )

    def expected_value_calc(self, rr: float = None) -> float:
        rr = rr or _safe_div(self.tp1_distance, self.stop_distance, 1.5)
        wp = self.win_probability
        return round(wp * rr - (1 - wp) * 1.0, 4)


# ============================================================
# BASE AGENT
# ============================================================

class BaseAgent:
    NAME = "base"

    # ═══ V1.3 FIX: Added pair parameter ═══
    def evaluate(self, candles: list, regime: str, pair: str = None) -> AgentSignal:
        raise NotImplementedError

    def _null_signal(self, reason: str) -> AgentSignal:
        return AgentSignal(
            agent_name=self.NAME, direction=None, confidence=0.0,
            win_probability=0.0, expected_value=0.0,
            stop_distance=0.0, tp1_distance=0.0, tp2_distance=0.0,
            regime_fit=0.0, reasoning=reason,
        )

    def _closes(self, candles):
        return [c.close for c in candles if hasattr(c, "close") and np.isfinite(c.close)]


# ============================================================
# AGENT 1 — TREND AGENT
# ============================================================

class TrendAgent(BaseAgent):
    """
    EMA 20/50 crossover + MACD momentum.
    Best in trending regimes.
    """
    NAME = "TrendAgent"
    REGIME_FIT = {"trending": 1.0, "ranging": 0.2, "volatile": 0.4}

    def evaluate(self, candles: list, regime: str, pair: str = None) -> AgentSignal:
        if len(candles) < 55:
            return self._null_signal("Insufficient candles")

        closes = self._closes(candles)
        if len(closes) < 55:
            return self._null_signal("Insufficient valid closes")

        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)

        macd_line  = [f - s for f, s in zip(_ema(closes, 12), _ema(closes, 26))]
        macd_sig   = _ema(macd_line, 9)
        macd_hist  = macd_line[-1] - macd_sig[-1] if macd_sig else 0.0

        curr_ema20 = ema20[-1]
        curr_ema50 = ema50[-1]
        prev_ema20 = ema20[-2]
        prev_ema50 = ema50[-2]

        bullish_cross = (prev_ema20 <= prev_ema50) and (curr_ema20 > curr_ema50)
        bearish_cross = (prev_ema20 >= prev_ema50) and (curr_ema20 < curr_ema50)
        bullish_trend = curr_ema20 > curr_ema50 * 1.001
        bearish_trend = curr_ema20 < curr_ema50 * 0.999

        atr = _atr(candles)
        rsi = _rsi(closes)

        if bullish_cross or (bullish_trend and macd_hist > 0):
            direction   = "buy"
            confidence  = 0.80 if bullish_cross else 0.60
            win_prob    = 0.58 if bullish_cross else 0.52
            reasoning   = f"EMA20 > EMA50 {'(fresh cross)' if bullish_cross else '(continuation)'}. MACD hist={macd_hist:.6f}. RSI={rsi:.1f}."
        elif bearish_cross or (bearish_trend and macd_hist < 0):
            direction   = "sell"
            confidence  = 0.80 if bearish_cross else 0.60
            win_prob    = 0.58 if bearish_cross else 0.52
            reasoning   = f"EMA20 < EMA50 {'(fresh cross)' if bearish_cross else '(continuation)'}. MACD hist={macd_hist:.6f}. RSI={rsi:.1f}."
        else:
            return self._null_signal(f"No trend signal. EMA20={curr_ema20:.5f} EMA50={curr_ema50:.5f}")

        stop_dist = atr * 1.5
        tp1_dist  = atr * 2.0
        tp2_dist  = atr * 3.5

        ev = win_prob * _safe_div(tp1_dist, stop_dist) - (1 - win_prob)

        return AgentSignal(
            agent_name=self.NAME,
            direction=direction,
            confidence=confidence,
            win_probability=win_prob,
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.5),
            reasoning=reasoning,
            raw_features={"ema20": curr_ema20, "ema50": curr_ema50, "macd_hist": macd_hist, "rsi": rsi},
        )


# ============================================================
# AGENT 2 — MEAN REVERSION AGENT (WITH TREND FILTER)
# ============================================================

class MeanReversionAgent(BaseAgent):
    """
    RSI extremes + Bollinger Band touch.
    SAFETY: Blocks signals that fight strong trends.
    V1.4: Added book + structure filters (A/B test mode).
    """
    NAME = "MeanReversionAgent"
    REGIME_FIT = {"trending": 0.2, "ranging": 1.0, "volatile": 0.3}

    def evaluate(self, candles: list, regime: str, pair: str = None) -> AgentSignal:
        if len(candles) < 30:
            return self._null_signal("Insufficient candles")

        closes = self._closes(candles)
        if len(closes) < 20:
            return self._null_signal("Insufficient valid closes")

        rsi   = _rsi(closes)
        atr   = _atr(candles)
        sma20 = _safe_mean(closes[-20:])
        std20 = float(np.std(closes[-20:])) if len(closes) >= 20 else 0.001
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20
        curr     = closes[-1]

        # Compute trend filter (EMA 20 vs 50)
        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)
        strong_downtrend = ema20[-1] < ema50[-1] * 0.995
        strong_uptrend   = ema20[-1] > ema50[-1] * 1.005

        if rsi < 40 and curr <= lower_bb * 1.005:
            # SAFETY: Do not buy in strong downtrend (catching falling knife)
            if strong_downtrend:
                return self._null_signal(
                    f"MR buy blocked: strong downtrend EMA20={ema20[-1]:.2f} < EMA50={ema50[-1]:.2f}"
                )
            direction  = "buy"
            confidence = 0.70 if rsi < 30 else 0.55
            win_prob   = 0.58 if rsi < 30 else 0.52
            reasoning  = f"RSI={rsi:.1f} oversold, price at lower BB ({lower_bb:.5f}). Mean reversion BUY."

        elif rsi > 60 and curr >= upper_bb * 0.995:
            # SAFETY: Do not sell in strong uptrend
            if strong_uptrend:
                return self._null_signal(
                    f"MR sell blocked: strong uptrend EMA20={ema20[-1]:.2f} > EMA50={ema50[-1]:.2f}"
                )
            direction  = "sell"
            confidence = 0.70 if rsi > 70 else 0.55
            win_prob   = 0.58 if rsi > 70 else 0.52
            reasoning  = f"RSI={rsi:.1f} overbought, price at upper BB ({upper_bb:.5f}). Mean reversion SELL."
        else:
            return self._null_signal(f"No MR signal. RSI={rsi:.1f}, price={curr:.5f}, BB=[{lower_bb:.5f},{upper_bb:.5f}]")

        stop_dist = atr * 1.2
        tp1_dist  = abs(sma20 - curr) * 0.8
        tp2_dist  = atr * 2.5
        if tp1_dist < atr * 0.5:
            tp1_dist = atr * 1.0

        ev = win_prob * _safe_div(tp1_dist, stop_dist) - (1 - win_prob)

        # ═══ V1.4: Book + structure filters (Rayner Ch.5 + your zones) — A/B log ═══
        book = None
        if mr_book_filters is not None:
            book = mr_book_filters.check(candles, direction)
            would = []
            if not book["book_passed"]: would.append("book")
            if not book["struct_passed"]: would.append("structure")
            if would:
                logger.info(f"BOOK_MR {pair or self.NAME} {direction}: WOULD_BLOCK({'+'.join(would)}) | {book['summary']}")
            if not book["passed"]:
                return self._null_signal(f"MR filter: {book['summary']}")

        raw = {"rsi": rsi, "upper_bb": upper_bb, "lower_bb": lower_bb, "sma20": sma20}
        if book is not None:
            raw["book_filters"] = book

        return AgentSignal(
            agent_name=self.NAME,
            direction=direction,
            confidence=confidence,
            win_probability=win_prob,
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.5),
            reasoning=reasoning,
            raw_features=raw,
        )


# ============================================================
# AGENT 3 — BREAKOUT AGENT
# ============================================================

class BreakoutAgent(BaseAgent):
    """
    Volatility compression + range expansion.
    """
    NAME = "BreakoutAgent"
    REGIME_FIT = {"trending": 0.5, "ranging": 0.4, "volatile": 1.0}
    COMPRESSION_BARS = 10

    def evaluate(self, candles: list, regime: str, pair: str = None) -> AgentSignal:
        if len(candles) < 30:
            return self._null_signal("Insufficient candles")

        atr_current  = _atr(candles, 5)
        atr_baseline = _atr(candles, 20)
        compression  = _safe_div(atr_current, atr_baseline)

        recent = candles[-self.COMPRESSION_BARS:]
        highs  = [c.high  for c in recent if hasattr(c, "high")  and np.isfinite(c.high)]
        lows   = [c.low   for c in recent if hasattr(c, "low")   and np.isfinite(c.low)]

        if not highs or not lows:
            return self._null_signal("No valid OHLC in recent bars")

        range_high = max(highs)
        range_low  = min(lows)
        curr       = candles[-1].close
        atr        = _atr(candles)

        if compression > 0.98:
            return self._null_signal(f"No compression. ATR ratio={compression:.2f} (need < 0.98)")

        if curr > range_high:
            direction  = "buy"
            confidence = 0.68 if compression < 0.65 else 0.52
            win_prob   = 0.53
            reasoning  = f"Upside breakout above {range_high:.5f}. ATR compression={compression:.2f}."
        elif curr < range_low:
            direction  = "sell"
            confidence = 0.68 if compression < 0.65 else 0.52
            win_prob   = 0.53
            reasoning  = f"Downside breakout below {range_low:.5f}. ATR compression={compression:.2f}."
        else:
            return self._null_signal(f"Price inside range [{range_low:.5f}, {range_high:.5f}]. Awaiting breakout.")

        stop_dist = atr * 1.5
        tp1_dist  = atr * 2.5
        tp2_dist  = atr * 4.0

        ev = win_prob * _safe_div(tp1_dist, stop_dist) - (1 - win_prob)

        return AgentSignal(
            agent_name=self.NAME,
            direction=direction,
            confidence=confidence,
            win_probability=win_prob,
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.5),
            reasoning=reasoning,
            raw_features={"compression": compression, "range_high": range_high, "range_low": range_low},
        )


# ============================================================
# AGENT 4 — LIQUIDITY TRAP AGENT
# ============================================================

class LiquidityTrapAgent(BaseAgent):
    """
    Detects fake breakouts (stop hunts / wick traps).
    """
    NAME = "LiquidityTrapAgent"
    REGIME_FIT = {"trending": 0.3, "ranging": 0.8, "volatile": 0.7}
    WICK_RATIO_MIN = 1.8
    LOOKBACK = 20

    def evaluate(self, candles: list, regime: str, pair: str = None) -> AgentSignal:
        if len(candles) < self.LOOKBACK + 3:
            return self._null_signal("Insufficient candles")

        curr = candles[-1]
        if not all(hasattr(curr, a) for a in ["open", "high", "low", "close"]):
            return self._null_signal("Invalid candle data")

        body         = abs(curr.close - curr.open)
        upper_wick   = curr.high - max(curr.open, curr.close)
        lower_wick   = min(curr.open, curr.close) - curr.low
        atr          = _atr(candles)

        recent = candles[-self.LOOKBACK:-1]
        r_highs = [c.high for c in recent if hasattr(c, "high") and np.isfinite(c.high)]
        r_lows  = [c.low  for c in recent if hasattr(c, "low")  and np.isfinite(c.low)]

        if not r_highs or not r_lows:
            return self._null_signal("No recent highs/lows for trap detection")

        zone_high = max(r_highs)
        zone_low  = min(r_lows)

        if (curr.high > zone_high
                and curr.close < zone_high
                and body > 0
                and _safe_div(upper_wick, max(body, 0.0001)) >= self.WICK_RATIO_MIN):
            direction  = "sell"
            confidence = 0.72
            win_prob   = 0.56
            reasoning  = (
                f"Bearish liquidity trap: wick={upper_wick:.5f} spiked above {zone_high:.5f} "
                f"but closed at {curr.close:.5f}. Smart money fade SELL."
            )
        elif (curr.low < zone_low
                and curr.close > zone_low
                and body > 0
                and _safe_div(lower_wick, max(body, 0.0001)) >= self.WICK_RATIO_MIN):
            direction  = "buy"
            confidence = 0.72
            win_prob   = 0.56
            reasoning  = (
                f"Bullish liquidity trap: wick={lower_wick:.5f} spiked below {zone_low:.5f} "
                f"but closed at {curr.close:.5f}. Smart money fade BUY."
            )
        else:
            return self._null_signal("No liquidity trap detected")

        stop_dist = atr * 1.3
        tp1_dist  = atr * 2.0
        tp2_dist  = atr * 3.5

        ev = win_prob * _safe_div(tp1_dist, stop_dist) - (1 - win_prob)

        return AgentSignal(
            agent_name=self.NAME,
            direction=direction,
            confidence=confidence,
            win_probability=win_prob,
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.5),
            reasoning=reasoning,
            raw_features={"zone_high": zone_high, "zone_low": zone_low,
                          "upper_wick": upper_wick, "lower_wick": lower_wick, "body": body},
        )



# ============================================================
# AGENT 5 — WILLIAMS %R AGENT
# ============================================================

class WilliamsRAgent(BaseAgent):
    """
    Mean-reversion on Williams %R extremes.
    Only trades in ranging / weak-trend conditions.
    """
    NAME = "WilliamsRAgent"
    REGIME_FIT = {"trending": 0.2, "ranging": 1.0, "volatile": 0.3}

    def evaluate(self, candles: list, regime: str, pair: str = None) -> AgentSignal:
        if len(candles) < 20:
            return self._null_signal("Insufficient candles")

        closes = self._closes(candles)
        highs = [c.high for c in candles if hasattr(c, "high") and np.isfinite(c.high)]
        lows = [c.low for c in candles if hasattr(c, "low") and np.isfinite(c.low)]
        if len(closes) < 14 or len(highs) < 14 or len(lows) < 14:
            return self._null_signal("Insufficient data")

        highest_high = max(highs[-14:])
        lowest_low = min(lows[-14:])
        curr = closes[-1]

        if highest_high == lowest_low:
            return self._null_signal("Flat market")

        wr = -100 * (highest_high - curr) / (highest_high - lowest_low)

        # ADX approx — block if trending strongly
        adx = self._approx_adx(highs, lows, closes)
        if adx > 30:
            return self._null_signal(f"ADX={adx:.1f} too strong for mean reversion")

        direction = None
        if wr < -80:
            direction = "buy"
            confidence = min(0.90, 0.55 + abs(wr + 80) / 40)
        elif wr > -20:
            direction = "sell"
            confidence = min(0.90, 0.55 + abs(wr + 20) / 40)

        if not direction:
            return self._null_signal(f"Williams %R={wr:.1f} not extreme")

        atr_val = _atr(candles)
        stop_dist = atr_val * 1.5
        tp1_dist = atr_val * 2.5
        tp2_dist = atr_val * 3.5
        win_prob = 0.56 if confidence > 0.70 else 0.52

        ev = win_prob * _safe_div(tp1_dist, stop_dist, 1.5) - (1 - win_prob)

        return AgentSignal(
            agent_name=self.NAME,
            direction=direction,
            confidence=round(confidence, 2),
            win_probability=round(win_prob, 2),
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.5),
            reasoning=f"Williams %R={wr:.1f} ({'oversold' if direction=='buy' else 'overbought'}) ADX≈{adx:.1f}",
            raw_features={"williams_r": wr, "adx_approx": adx},
        )

    def _approx_adx(self, highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return 0.0
        trs = []
        for i in range(-period, 0):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return float(np.mean(trs)) / (float(np.mean(closes[-period:])) + 1e-9) * 100


# ============================================================
# AGENT 6 — HIDDEN DIVERGENCE AGENT
# ============================================================

class HiddenDivergenceAgent(BaseAgent):
    """
    Trend-continuation via RSI hidden divergence.
    Bullish: price higher low, RSI lower low (uptrend)
    Bearish: price lower high, RSI higher high (downtrend)
    """
    NAME = "HiddenDivergenceAgent"
    REGIME_FIT = {"trending": 1.0, "ranging": 0.3, "volatile": 0.5}

    def evaluate(self, candles: list, regime: str, pair: str = None) -> AgentSignal:
        if len(candles) < 40:
            return self._null_signal("Insufficient candles")

        closes = self._closes(candles)
        if len(closes) < 30:
            return self._null_signal("Insufficient closes")

        rsi_val = _rsi(closes)
        ema_fast = _ema(closes, 10)[-1]
        ema_slow = _ema(closes, 30)[-1]
        uptrend = ema_fast > ema_slow * 1.001

        direction = None
        confidence = 0.5

        # Find swing lows/highs in last 30 bars
        if uptrend:
            lows = [(i, closes[-30:][i]) for i in range(1, 29)]
            # Simplified: check last 2 significant lows
            if closes[-1] > closes[-5] and closes[-5] > closes[-15]:
                # Price higher low structure
                rsi_now = _rsi(closes[-15:])
                rsi_then = _rsi(closes[-30:-15])
                if rsi_now < rsi_then and rsi_now < 45:
                    direction = "buy"
                    confidence = min(0.82, 0.55 + (rsi_then - rsi_now) / 80)
        else:
            if closes[-1] < closes[-5] and closes[-5] < closes[-15]:
                rsi_now = _rsi(closes[-15:])
                rsi_then = _rsi(closes[-30:-15])
                if rsi_now > rsi_then and rsi_now > 55:
                    direction = "sell"
                    confidence = min(0.82, 0.55 + (rsi_now - rsi_then) / 80)

        if not direction:
            return self._null_signal("No hidden divergence")

        atr_val = _atr(candles)
        stop_dist = atr_val * 1.8
        tp1_dist = atr_val * 2.8
        tp2_dist = atr_val * 4.0
        win_prob = 0.58 if confidence > 0.70 else 0.53

        ev = win_prob * _safe_div(tp1_dist, stop_dist, 1.5) - (1 - win_prob)

        return AgentSignal(
            agent_name=self.NAME,
            direction=direction,
            confidence=round(confidence, 2),
            win_probability=round(win_prob, 2),
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.5),
            reasoning=f"Hidden divergence {'bullish' if direction=='buy' else 'bearish'} | RSI={rsi_val:.1f}",
            raw_features={"rsi": rsi_val, "ema_fast": ema_fast, "ema_slow": ema_slow},
        )


# ============================================================
# AGENT 7 — FRACTAL BREAKOUT AGENT
# ============================================================

class FractalAgent(BaseAgent):
    """
    Breakout above/below recent Williams Fractals.
    Only trades when ADX > 20 (trend strength present).
    """
    NAME = "FractalAgent"
    REGIME_FIT = {"trending": 0.8, "ranging": 0.3, "volatile": 1.0}

    def evaluate(self, candles: list, regime: str, pair: str = None) -> AgentSignal:
        if len(candles) < 20:
            return self._null_signal("Insufficient candles")

        closes = self._closes(candles)
        highs = [c.high for c in candles if hasattr(c, "high") and np.isfinite(c.high)]
        lows = [c.low for c in candles if hasattr(c, "low") and np.isfinite(c.low)]
        if len(highs) < 10 or len(lows) < 10:
            return self._null_signal("Insufficient OHLC")

        # Fractal highs/lows (2-bar each side)
        fractal_highs = []
        fractal_lows = []
        for i in range(2, len(highs) - 2):
            if highs[i] == max(highs[i-2:i+3]):
                fractal_highs.append(highs[i])
            if lows[i] == min(lows[i-2:i+3]):
                fractal_lows.append(lows[i])

        if not fractal_highs or not fractal_lows:
            return self._null_signal("No fractals detected")

        recent_high = fractal_highs[-1]
        recent_low = fractal_lows[-1]
        curr = closes[-1]

        adx = self._approx_adx(highs, lows, closes)
        if adx < 20:
            return self._null_signal(f"ADX={adx:.1f} too weak for breakout")

        direction = None
        if curr > recent_high * 1.0005:
            direction = "buy"
        elif curr < recent_low * 0.9995:
            direction = "sell"

        if not direction:
            return self._null_signal(f"Price inside fractal range [{recent_low:.5f}, {recent_high:.5f}]")

        atr_val = _atr(candles)
        stop_dist = atr_val * 1.5 if direction == "buy" else atr_val * 1.5
        # Stop beyond the fractal that was broken
        if direction == "buy":
            stop_dist = max(stop_dist, abs(curr - recent_low) * 1.2)
        else:
            stop_dist = max(stop_dist, abs(recent_high - curr) * 1.2)

        tp1_dist = stop_dist * 2.0
        tp2_dist = stop_dist * 3.5
        confidence = min(0.85, 0.50 + adx / 100)
        win_prob = 0.55

        ev = win_prob * _safe_div(tp1_dist, stop_dist, 1.5) - (1 - win_prob)

        return AgentSignal(
            agent_name=self.NAME,
            direction=direction,
            confidence=round(confidence, 2),
            win_probability=round(win_prob, 2),
            expected_value=round(ev, 4),
            stop_distance=stop_dist,
            tp1_distance=tp1_dist,
            tp2_distance=tp2_dist,
            regime_fit=self.REGIME_FIT.get(regime, 0.5),
            reasoning=f"Fractal {'high' if direction=='buy' else 'low'} breakout @ {recent_high if direction=='buy' else recent_low:.5f} ADX={adx:.1f}",
            raw_features={"fractal_high": recent_high, "fractal_low": recent_low, "adx": adx},
        )

    def _approx_adx(self, highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return 0.0
        trs = []
        for i in range(-period, 0):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return float(np.mean(trs)) / (float(np.mean(closes[-period:])) + 1e-9) * 100


# ============================================================
# AGENT ORCHESTRATOR — SAFE (no fallback, regime-locked)
# ============================================================

@dataclass
class OrchestratorResult:
    selected_agent:  str
    selected_signal: AgentSignal
    all_signals:     Dict[str, AgentSignal]
    selection_score: float
    regime:          str
    reasoning:       str


class AgentOrchestrator:
    """
    Runs all 9 agents and selects the best signal.
    """

    def __init__(self):
        # ═══ V1.3: Instantiate External Agents ═══
        self.cot_agent = COTReportAgent()
        self.news_agent = NewsSentimentAgent()

        self.agents = {
            "TrendAgent":             TrendAgent(),
            "MeanReversionAgent":     MeanReversionAgent(),
            "BreakoutAgent":          BreakoutAgent(),
            "LiquidityTrapAgent":     LiquidityTrapAgent(),
            "WilliamsRAgent":         WilliamsRAgent(),
            "HiddenDivergenceAgent":  HiddenDivergenceAgent(),
            "FractalAgent":           FractalAgent(),
            # ═══ V1.3: Add External Agents ═══
            "COTReportAgent":         self.cot_agent,
            "NewsSentimentAgent":     self.news_agent,
        }
        self.weights: Dict[str, float] = {name: 1.0 for name in self.agents}

    def update_weights(self, weights: Dict[str, float]):
        for name, w in weights.items():
            if name in self.weights:
                self.weights[name] = max(0.1, min(w, 3.0))
        logger.info(f"Agent weights updated: {self.weights}")

    # ═══ V1.3: Async Data Refresher ═══
    async def refresh_external_data(self):
        """Safely fetches COT and News data. Throttled internally by the agents."""
        try: 
            await self.cot_agent.refresh()
        except Exception as e: 
            logger.warning(f"COT refresh error: {e}")
        try: 
            await self.news_agent.refresh()
        except Exception as e: 
            logger.warning(f"News refresh error: {e}")

    # ═══ V1.3 FIX: Added pair parameter to pass to agents ═══
    def run(self, candles: list, regime: str, pair: str = "EURUSD") -> Optional[OrchestratorResult]:
        all_signals: Dict[str, AgentSignal] = {}

        for name, agent in self.agents.items():
            try:
                # Pass pair so COT/News evaluate the correct instrument
                sig = agent.evaluate(candles, regime, pair=pair)
                all_signals[name] = sig
            except Exception as e:
                logger.warning(f"Agent {name} error: {e}")
                all_signals[name] = agent._null_signal(f"Agent error: {e}")

        # Filter valid signals
        valid = {name: sig for name, sig in all_signals.items() if sig.is_valid}

        if not valid:
            logger.info(f"No valid signals in {regime} regime — standing aside")
            return None

        # SAFETY: Regime-lock — only agents suited to current regime
        regime_appropriate = {
            name: sig for name, sig in valid.items()
            if sig.regime_fit >= 0.5
        }

        if regime_appropriate:
            valid = regime_appropriate
            logger.info(f"Regime-locked to {regime}: candidates={list(valid.keys())}")
        else:
            logger.info(
                f"No regime-appropriate signals in {regime} — "
                f"valid agents had fits: "
                + ", ".join(f"{n}={s.regime_fit:.2f}" for n, s in valid.items())
            )
            return None

        def score(name: str, sig: AgentSignal) -> float:
            ev_factor = max(0.1, 1.0 + sig.expected_value)
            return sig.confidence * sig.regime_fit * ev_factor * self.weights.get(name, 1.0)

        best_name = max(valid, key=lambda n: score(n, valid[n]))
        best_sig  = valid[best_name]
        best_score = score(best_name, best_sig)

        reasoning = (
            f"Selected {best_name} (score={best_score:.3f}) from "
            f"{len(valid)} valid agent(s) in {regime} regime. "
            f"Direction: {best_sig.direction}. "
            f"Confidence: {best_sig.confidence:.2f}. EV: {best_sig.expected_value:.3f}R."
        )

        return OrchestratorResult(
            selected_agent=best_name,
            selected_signal=best_sig,
            all_signals=all_signals,
            selection_score=round(best_score, 4),
            regime=regime,
            reasoning=reasoning,
        )
