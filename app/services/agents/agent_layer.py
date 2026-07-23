"""
PriceIQ Pro — Multi-Agent Strategy Layer v1.0

Four competing strategy agents. Each produces an AgentSignal independently.
The AgentOrchestrator selects the best agent per regime using weighted scoring.

Agents:
    TrendAgent          — EMA crossover + MACD momentum
    MeanReversionAgent  — RSI extremes + Bollinger Band touch
    BreakoutAgent       — Volatility compression + range expansion
    LiquidityTrapAgent  — False breakout detection (wick traps)

All agents share a common interface: agent.evaluate(candles, regime) → AgentSignal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

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
# AGENT SIGNAL — standard output for all agents
# ============================================================

@dataclass
class AgentSignal:
    agent_name:     str
    direction:      Optional[str]    # "buy" | "sell" | None
    confidence:     float            # 0.0 – 1.0
    win_probability: float           # estimated win prob 0–1
    expected_value: float            # EV in R-multiples
    stop_distance:  float            # in price units
    tp1_distance:   float
    tp2_distance:   float
    regime_fit:     float            # how well agent fits current regime (0–1)
    reasoning:      str
    raw_features:   Dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return (
            self.direction is not None
            and self.confidence > 0.40
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

    def evaluate(self, candles: list, regime: str) -> AgentSignal:
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
    EMA 20/50 crossover + MACD histogram direction + momentum filter.
    Best in trending regimes.
    """
    NAME = "TrendAgent"

    REGIME_FIT = {"trending": 1.0, "ranging": 0.2, "volatile": 0.4}

    def evaluate(self, candles: list, regime: str) -> AgentSignal:
        if len(candles) < 55:
            return self._null_signal("Insufficient candles")

        closes = self._closes(candles)
        if len(closes) < 55:
            return self._null_signal("Insufficient valid closes")

        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)

        # MACD
        macd_line  = [f - s for f, s in zip(_ema(closes, 12), _ema(closes, 26))]
        macd_sig   = _ema(macd_line, 9)
        macd_hist  = macd_line[-1] - macd_sig[-1] if macd_sig else 0.0

        curr_ema20 = ema20[-1]
        curr_ema50 = ema50[-1]
        prev_ema20 = ema20[-2]
        prev_ema50 = ema50[-2]

        # Fresh crossover
        bullish_cross = (prev_ema20 <= prev_ema50) and (curr_ema20 > curr_ema50)
        bearish_cross = (prev_ema20 >= prev_ema50) and (curr_ema20 < curr_ema50)
        # Trend continuation
        bullish_trend = curr_ema20 > curr_ema50 * 1.001
        bearish_trend = curr_ema20 < curr_ema50 * 0.999

        atr = _atr(candles)
        rsi = _rsi(closes)

        if bullish_cross or (bullish_trend and macd_hist > 0 and rsi < 65):
            direction   = "buy"
            confidence  = 0.80 if bullish_cross else 0.65
            win_prob    = 0.58 if bullish_cross else 0.54
            reasoning   = f"EMA20 > EMA50 {'(fresh cross)' if bullish_cross else '(continuation)'}. MACD hist={macd_hist:.6f}. RSI={rsi:.1f}."
        elif bearish_cross or (bearish_trend and macd_hist < 0 and rsi > 35):
            direction   = "sell"
            confidence  = 0.80 if bearish_cross else 0.65
            win_prob    = 0.58 if bearish_cross else 0.54
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
# AGENT 2 — MEAN REVERSION AGENT
# ============================================================

class MeanReversionAgent(BaseAgent):
    """
    RSI extremes + Bollinger Band touch + close-to-mean confirmation.
    Best in ranging regimes.
    """
    NAME = "MeanReversionAgent"

    REGIME_FIT = {"trending": 0.2, "ranging": 1.0, "volatile": 0.3}

    def evaluate(self, candles: list, regime: str) -> AgentSignal:
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

        # Oversold + below lower BB → buy reversion
        if rsi < 32 and curr <= lower_bb * 1.001:
            direction  = "buy"
            confidence = 0.75 if rsi < 25 else 0.60
            win_prob   = 0.62 if rsi < 25 else 0.56
            reasoning  = f"RSI={rsi:.1f} oversold, price at lower BB ({lower_bb:.5f}). Mean reversion BUY."
        # Overbought + above upper BB → sell reversion
        elif rsi > 68 and curr >= upper_bb * 0.999:
            direction  = "sell"
            confidence = 0.75 if rsi > 75 else 0.60
            win_prob   = 0.62 if rsi > 75 else 0.56
            reasoning  = f"RSI={rsi:.1f} overbought, price at upper BB ({upper_bb:.5f}). Mean reversion SELL."
        else:
            return self._null_signal(f"No MR signal. RSI={rsi:.1f}, price={curr:.5f}, BB=[{lower_bb:.5f},{upper_bb:.5f}]")

        stop_dist = atr * 1.2
        tp1_dist  = abs(sma20 - curr) * 0.8   # TP near mean
        tp2_dist  = atr * 2.5
        if tp1_dist < atr * 0.5:
            tp1_dist = atr * 1.0   # minimum sensible TP

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
            raw_features={"rsi": rsi, "upper_bb": upper_bb, "lower_bb": lower_bb, "sma20": sma20},
        )


# ============================================================
# AGENT 3 — BREAKOUT AGENT
# ============================================================

class BreakoutAgent(BaseAgent):
    """
    Volatility compression (ATR squeeze) followed by range expansion.
    Best in volatile / transitional regimes.
    """
    NAME = "BreakoutAgent"

    REGIME_FIT = {"trending": 0.5, "ranging": 0.4, "volatile": 1.0}
    COMPRESSION_BARS = 10

    def evaluate(self, candles: list, regime: str) -> AgentSignal:
        if len(candles) < 30:
            return self._null_signal("Insufficient candles")

        atr_current  = _atr(candles, 5)
        atr_baseline = _atr(candles, 20)
        compression  = _safe_div(atr_current, atr_baseline)

        # Recent range highs / lows
        recent = candles[-self.COMPRESSION_BARS:]
        highs  = [c.high  for c in recent if hasattr(c, "high")  and np.isfinite(c.high)]
        lows   = [c.low   for c in recent if hasattr(c, "low")   and np.isfinite(c.low)]

        if not highs or not lows:
            return self._null_signal("No valid OHLC in recent bars")

        range_high = max(highs)
        range_low  = min(lows)
        curr       = candles[-1].close
        atr        = _atr(candles)

        # Compression check: ATR contracted
        if compression > 0.90:
            return self._null_signal(f"No compression. ATR ratio={compression:.2f} (need < 0.90)")

        # Breakout: current bar closes outside the compressed range
        if curr > range_high:
            direction  = "buy"
            confidence = 0.72 if compression < 0.65 else 0.58
            win_prob   = 0.55
            reasoning  = f"Upside breakout above {range_high:.5f}. ATR compression={compression:.2f}."
        elif curr < range_low:
            direction  = "sell"
            confidence = 0.72 if compression < 0.65 else 0.58
            win_prob   = 0.55
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
    Detects fake breakouts (stop hunts / wick traps) and fades them.

    Pattern: price spikes beyond S/R with a large wick but closes back
    inside the range — indicating smart money trapped retail.
    """
    NAME = "LiquidityTrapAgent"

    REGIME_FIT = {"trending": 0.3, "ranging": 0.8, "volatile": 0.7}
    WICK_RATIO_MIN = 2.5   # wick must be 2.5× body
    LOOKBACK = 20

    def evaluate(self, candles: list, regime: str) -> AgentSignal:
        if len(candles) < self.LOOKBACK + 3:
            return self._null_signal("Insufficient candles")

        curr = candles[-1]
        if not all(hasattr(curr, a) for a in ["open", "high", "low", "close"]):
            return self._null_signal("Invalid candle data")

        body         = abs(curr.close - curr.open)
        upper_wick   = curr.high - max(curr.open, curr.close)
        lower_wick   = min(curr.open, curr.close) - curr.low
        atr          = _atr(candles)

        # Recent high / low (liquidity zones)
        recent = candles[-self.LOOKBACK:-1]
        r_highs = [c.high for c in recent if hasattr(c, "high") and np.isfinite(c.high)]
        r_lows  = [c.low  for c in recent if hasattr(c, "low")  and np.isfinite(c.low)]

        if not r_highs or not r_lows:
            return self._null_signal("No recent highs/lows for trap detection")

        zone_high = max(r_highs)
        zone_low  = min(r_lows)

        # Bearish trap: wick punched through zone_high but closed back below
        if (curr.high > zone_high                         # spiked above
                and curr.close < zone_high               # closed back inside
                and body > 0
                and _safe_div(upper_wick, max(body, 0.0001)) >= self.WICK_RATIO_MIN):
            direction  = "sell"
            confidence = 0.78
            win_prob   = 0.60
            reasoning  = (
                f"Bearish liquidity trap: wick={upper_wick:.5f} spiked above {zone_high:.5f} "
                f"but closed at {curr.close:.5f}. Smart money fade SELL."
            )
        # Bullish trap: wick punched below zone_low but closed back above
        elif (curr.low < zone_low
                and curr.close > zone_low
                and body > 0
                and _safe_div(lower_wick, max(body, 0.0001)) >= self.WICK_RATIO_MIN):
            direction  = "buy"
            confidence = 0.78
            win_prob   = 0.60
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
# AGENT ORCHESTRATOR — selects best agent per regime
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
    Runs all four agents and selects the best signal using:

        score = confidence × regime_fit × (1 + expected_value)

    Agent weights evolve over time via the LearningSystem feedback loop.
    """

    def __init__(self):
        self.agents = {
            "TrendAgent":          TrendAgent(),
            "MeanReversionAgent":  MeanReversionAgent(),
            "BreakoutAgent":       BreakoutAgent(),
            "LiquidityTrapAgent":  LiquidityTrapAgent(),
        }
        # Starting weights — updated by LearningSystem
        self.weights: Dict[str, float] = {name: 1.0 for name in self.agents}

    def update_weights(self, weights: Dict[str, float]):
        """Called by LearningSystem after trade outcomes are recorded."""
        for name, w in weights.items():
            if name in self.weights:
                self.weights[name] = max(0.1, min(w, 3.0))   # clamp to sensible range
        logger.info(f"Agent weights updated: {self.weights}")

    def run(self, candles: list, regime: str) -> Optional[OrchestratorResult]:
        """
        Evaluate all agents and return the best signal, or None if no valid signal found.
        """
        all_signals: Dict[str, AgentSignal] = {}

        for name, agent in self.agents.items():
            try:
                sig = agent.evaluate(candles, regime)
                all_signals[name] = sig
            except Exception as e:
                logger.warning(f"Agent {name} error: {e}")
                all_signals[name] = agent._null_signal(f"Agent error: {e}")

        # Filter valid signals only
        valid = {name: sig for name, sig in all_signals.items() if sig.is_valid}

        if not valid:
            return None

        # Score = confidence × regime_fit × (1 + ev) × weight
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
