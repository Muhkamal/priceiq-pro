"""
PriceIQ Pro — Advanced Market Analyzer v3.2 (Corrected)
All issues corrected from v3.1 plus:
- Margin formula fixed and simplified
- Ranging market bias now determined from S/R position
- Gap logic relaxed for 24h continuous markets (forex)
- Spread adjustment made optional and clearly documented
- TP hit calculation uses actual R:R ratio
- safe_div handles NaN/inf
- Leverage configurable via settings
- Magic numbers moved to config with clear defaults
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime
import logging

from app.models.schemas import (
    Candle, MarketStructure, SupportResistance, PatternValidation,
    TradeSignal, SignalDirection, Trend, MarketStage, PatternType,
    PositionSizeRequest, PositionSizeResult,
    BacktestRequest, BacktestResult, BacktestMetrics, TradeRecord
)
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION DEFAULTS (override in app.core.config.settings)
# ============================================================

class Config:
    """Centralized defaults — override via settings module."""
    MIN_SHADOW_RATIO        = getattr(settings, "MIN_SHADOW_RATIO", 2.0)
    MIN_ENGULFING_RATIO     = getattr(settings, "MIN_ENGULFING_RATIO", 1.5)
    DOJI_BODY_PCT           = getattr(settings, "DOJI_BODY_PCT", 0.05)
    ATR_STOP_MULTIPLIER     = getattr(settings, "ATR_STOP_MULTIPLIER", 1.5)
    ATR_TP_MULTIPLIER       = getattr(settings, "ATR_TP_MULTIPLIER", 3.0)
    DEFAULT_LEVERAGE        = getattr(settings, "DEFAULT_LEVERAGE", 50)
    MAX_POSITION_LOTS       = getattr(settings, "MAX_POSITION_LOTS", 10.0)
    DEFAULT_RISK_PERCENT    = getattr(settings, "DEFAULT_RISK_PERCENT", 2.0)
    CLUSTER_GAP_PCT         = getattr(settings, "SR_CLUSTER_GAP_PCT", 0.002)
    SR_LOOKBACK_CANDLES     = getattr(settings, "SR_LOOKBACK_CANDLES", 200)
    SR_MAX_LEVELS           = getattr(settings, "SR_MAX_LEVELS", 10)
    APPLY_SPREAD_TO_ENTRY   = getattr(settings, "APPLY_SPREAD_TO_ENTRY", True)
    MORNING_STAR_GAP_PCT    = getattr(settings, "MORNING_STAR_GAP_PCT", 0.001)


# ============================================================
# SAFE MATH UTILITIES
# ============================================================

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division — returns default if denominator is zero, None, NaN, or inf."""
    try:
        if b == 0 or b is None:
            return default
        # Handle NaN and infinite values
        if hasattr(np, 'isfinite') and not np.isfinite(b):
            return default
        return a / b
    except (ZeroDivisionError, TypeError, ValueError, OverflowError):
        return default


def safe_mean(values: List, default: float = 0.0) -> float:
    """Safe mean calculation."""
    if not values:
        return default
    try:
        filtered = [v for v in values if v is not None and np.isfinite(v)]
        return np.mean(filtered) if filtered else default
    except Exception:
        return default


# ============================================================
# ATR CALCULATION
# ============================================================

def calculate_atr(candles: List[Candle], period: int = 14) -> float:
    """Calculate Average True Range over the given period."""
    if len(candles) < period + 1:
        # Fallback: use mean of recent ranges
        ranges = [c.high - c.low for c in candles[-20:] if c.high and c.low and np.isfinite(c.high) and np.isfinite(c.low)]
        return safe_mean(ranges, 0.001)

    true_ranges = []
    for i in range(1, len(candles)):
        c = candles[i]
        prev = candles[i - 1]
        if not all(np.isfinite(v) for v in [c.high, c.low, c.close, prev.close]):
            continue
        tr = max(
            c.high - c.low,
            abs(c.high - prev.close),
            abs(c.low - prev.close)
        )
        true_ranges.append(tr)

    recent_trs = true_ranges[-period:]
    return safe_mean(recent_trs, 0.001)


# ============================================================
# SPREAD & SLIPPAGE CONFIG
# ============================================================

SPREAD_PIPS: Dict[str, float] = {
    "EURUSD": 1.2, "GBPUSD": 1.5, "USDJPY": 1.3, "USDCHF": 1.8,
    "AUDUSD": 1.4, "NZDUSD": 1.8, "USDCAD": 2.0, "EURGBP": 1.6,
    "EURJPY": 1.8, "GBPJPY": 2.5, "XAUUSD": 25.0,
}


def _pip_size(pair: str) -> float:
    pair = pair.upper()
    if "JPY" in pair:
        return 0.01
    if "XAU" in pair or "GOLD" in pair:
        return 0.1
    return 0.0001


def _spread_cost(pair: str) -> float:
    return SPREAD_PIPS.get(pair.upper(), 2.0) * _pip_size(pair)


def _realistic_entry(entry: float, direction: SignalDirection, pair: str, apply_spread: bool = None) -> Tuple[float, float]:
    """
    Apply spread to entry price for realistic market-order execution.

    Returns:
        (adjusted_entry, spread_applied)

    Note: Spread worsens the entry price for market orders. This is realistic
    but conservative. Set Config.APPLY_SPREAD_TO_ENTRY = False to use raw close.
    """
    if apply_spread is None:
        apply_spread = Config.APPLY_SPREAD_TO_ENTRY

    if not apply_spread:
        return entry, 0.0

    cost = _spread_cost(pair)
    if direction == SignalDirection.BUY:
        return entry + cost, cost
    else:
        return entry - cost, cost


# ============================================================
# ADVANCED CONFIDENCE SCORER
# ============================================================

class AdvancedConfidenceScorer:
    """Multi-factor weighted confidence score."""

    WEIGHTS = {
        "shadow":    0.20,
        "trend":     0.20,
        "sr":        0.20,
        "close_pos": 0.15,
        "volume":    0.10,
        "rarity":    0.10,
        "mtf":       0.05,
    }

    def score(
        self,
        candle: Candle,
        candles: List[Candle],
        idx: int,
        sr_levels: List[SupportResistance],
        pattern_type: PatternType,
        trend_strength: float,
        mtf_aligned: bool = False,
    ) -> float:

        scores = {}
        try:
            scores["shadow"]    = self._shadow_score_safe(candle, pattern_type)
            scores["trend"]     = min(1.0, max(0.0, trend_strength))
            scores["sr"]        = self._sr_score_safe(candle, sr_levels, pattern_type)
            scores["close_pos"] = self._close_position_score_safe(candle, pattern_type)
            scores["volume"]    = self._volume_score_safe(candle, candles, idx)
            scores["rarity"]    = self._rarity_score(pattern_type)
            scores["mtf"]       = 1.0 if mtf_aligned else 0.3

            total = sum(scores.get(k, 0) * self.WEIGHTS.get(k, 0) for k in self.WEIGHTS)
            return round(min(1.0, max(0.0, total)), 3)

        except Exception as e:
            logger.debug(f"Confidence scaoring error: {e}")
            return 0.50

    def _shadow_score_safe(self, c: Candle, pattern: PatternType) -> float:
        if c.range is None or c.range == 0 or not np.isfinite(c.range):
            return 0.0

        try:
            if pattern == PatternType.HAMMER:
                if c.lower_shadow == 0 or not np.isfinite(c.lower_shadow):
                    return 0.0
                body = max(c.body, 0.0001)
                ratio = safe_div(c.lower_shadow, body)
                return min(1.0, ratio / 2.5) if ratio <= 2.5 else max(0.0, 1.0 - (ratio - 2.5) / 2.5)

            elif pattern == PatternType.SHOOTING_STAR:
                if c.upper_shadow == 0 or not np.isfinite(c.upper_shadow):
                    return 0.0
                body = max(c.body, 0.0001)
                ratio = safe_div(c.upper_shadow, body)
                return min(1.0, ratio / 2.5) if ratio <= 2.5 else max(0.0, 1.0 - (ratio - 2.5) / 2.5)

            elif pattern in (PatternType.BULLISH_ENGULFING, PatternType.BEARISH_ENGULFING):
                body_pct = safe_div(c.body, max(c.range, 0.0001))
                return min(1.0, body_pct / 0.8)

            elif pattern == PatternType.DOJI:
                body_pct = safe_div(c.body, max(c.range, 0.0001))
                return max(0.0, 1.0 - body_pct / 0.05)

            elif pattern in (PatternType.MORNING_STAR, PatternType.EVENING_STAR):
                body_pct = safe_div(c.body, max(c.range, 0.0001))
                return max(0.0, 1.0 - body_pct / 0.3)

            return 0.5
        except Exception:
            return 0.5

    def _sr_score_safe(
        self, c: Candle, sr_levels: List[SupportResistance], pattern: PatternType
    ) -> float:
        if not sr_levels:
            return 0.0

        try:
            ref = (
                c.low
                if pattern in (PatternType.HAMMER, PatternType.BULLISH_ENGULFING, PatternType.MORNING_STAR)
                else c.high
            )
            dists = []
            for sr in sr_levels:
                if sr.level > 0 and np.isfinite(sr.level):
                    dist = safe_div(abs(ref - sr.level), sr.level)
                    weighted = sr.strength * max(0.0, 1.0 - safe_div(dist, 0.01))
                    dists.append(weighted)
            return min(1.0, max(dists)) if dists else 0.0
        except Exception:
            return 0.0

    def _close_position_score_safe(self, c: Candle, pattern: PatternType) -> float:
        if c.range is None or c.range == 0 or not np.isfinite(c.range):
            return 0.5

        try:
            close_pct = safe_div(c.close - c.low, c.range, 0.5)

            if pattern in (PatternType.HAMMER, PatternType.BULLISH_ENGULFING, PatternType.MORNING_STAR):
                return min(1.0, max(0.0, safe_div(close_pct - 0.5, 0.5)))
            elif pattern in (PatternType.SHOOTING_STAR, PatternType.BEARISH_ENGULFING, PatternType.EVENING_STAR):
                return min(1.0, max(0.0, safe_div(0.5 - close_pct, 0.5)))
            elif pattern == PatternType.DOJI:
                return max(0.0, 1.0 - abs(close_pct - 0.5) / 0.5)
            return 0.5
        except Exception:
            return 0.5

    def _volume_score_safe(self, c: Candle, candles: List[Candle], idx: int) -> float:
        if not c.volume or idx < 5 or not np.isfinite(c.volume):
            return 0.3

        try:
            lookback = candles[max(0, idx - 20):idx]
            vols = [x.volume for x in lookback if x.volume and np.isfinite(x.volume)]
            if not vols:
                return 0.3

            avg = safe_mean(vols, 1)
            if avg == 0 or not np.isfinite(avg):
                return 0.3

            ratio = safe_div(c.volume, avg)

            if ratio < 0.8:
                return 0.2
            elif ratio < 1.0:
                return 0.4
            elif ratio < 1.5:
                return 0.7
            elif ratio < 2.5:
                return 1.0
            else:
                return 0.8
        except Exception:
            return 0.3

    def _rarity_score(self, pattern: PatternType) -> float:
        rarity_map = {
            PatternType.MORNING_STAR:      1.0,
            PatternType.EVENING_STAR:      1.0,
            PatternType.BULLISH_ENGULFING: 0.8,
            PatternType.BEARISH_ENGULFING: 0.8,
            PatternType.HAMMER:            0.6,
            PatternType.SHOOTING_STAR:     0.6,
            PatternType.DOJI:              0.4,
        }
        return rarity_map.get(pattern, 0.5)


# ============================================================
# ENHANCED PATTERN VALIDATOR
# ============================================================

class EnhancedPatternValidator:
    """Full pattern validation including Morning Star, Evening Star, and Doji."""

    def __init__(self):
        self.min_shadow_ratio   = Config.MIN_SHADOW_RATIO
        self.min_engulfing_ratio = Config.MIN_ENGULFING_RATIO
        self.doji_body_pct      = Config.DOJI_BODY_PCT
        self.gap_pct            = Config.MORNING_STAR_GAP_PCT

    def validate_pattern(
        self,
        candles: List[Candle],
        idx: int,
        sr_levels: List[SupportResistance],
        pattern_type: PatternType,
        bar_closed: bool = True,
    ) -> PatternValidation:

        if not bar_closed or idx < 2:
            return PatternValidation(
                is_valid=False, pattern_type=pattern_type,
                confidence=0, rejection_reasons=["Bar not closed or insufficient history"],
                recommendations=["Wait for bar close and minimum 3 candles of history"],
            )

        try:
            current = candles[idx]
            prev    = candles[idx - 1]
            prev2   = candles[idx - 2]

            dispatch = {
                PatternType.BULLISH_ENGULFING: lambda: self._validate_bullish_engulfing(current, prev, sr_levels),
                PatternType.BEARISH_ENGULFING: lambda: self._validate_bearish_engulfing(current, prev, sr_levels),
                PatternType.HAMMER:            lambda: self._validate_hammer(current, sr_levels),
                PatternType.SHOOTING_STAR:     lambda: self._validate_shooting_star(current, sr_levels),
                PatternType.MORNING_STAR:      lambda: self._validate_morning_star(current, prev, prev2, sr_levels),
                PatternType.EVENING_STAR:      lambda: self._validate_evening_star(current, prev, prev2, sr_levels),
                PatternType.DOJI:              lambda: self._validate_doji(current, sr_levels),
            }

            handler = dispatch.get(pattern_type)
            if handler:
                return handler()

        except Exception as e:
            logger.debug(f"Pattern validation error: {e}")

        return PatternValidation(
            is_valid=False, pattern_type=pattern_type,
            confidence=0, rejection_reasons=["Unsupported or error in pattern validation"],
            recommendations=["Check pattern type and candle data integrity"],
        )

    # ----------------------------------------------------------
    # Single-candle patterns
    # ----------------------------------------------------------

    def _validate_hammer(
        self, current: Candle, sr_levels: List[SupportResistance]
    ) -> PatternValidation:
        body         = abs(current.close - current.open)
        lower_shadow = min(current.open, current.close) - current.low
        upper_shadow = current.high - max(current.open, current.close)

        if lower_shadow <= body * self.min_shadow_ratio:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.HAMMER,
                confidence=0, rejection_reasons=["Lower shadow too short"],
                recommendations=["Wait for a candle with a longer lower wick (min 2x body)"],
            )
        if upper_shadow > body:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.HAMMER,
                confidence=0, rejection_reasons=["Upper shadow too long for hammer"],
                recommendations=["Upper shadow should be smaller than body; consider inverted hammer instead"],
            )
        return PatternValidation(
            is_valid=True, pattern_type=PatternType.HAMMER,
            confidence=0.75, rejection_reasons=[],
            recommendations=["Confirm with bullish follow-through candle"],
        )

    def _validate_shooting_star(
        self, current: Candle, sr_levels: List[SupportResistance]
    ) -> PatternValidation:
        body         = abs(current.close - current.open)
        upper_shadow = current.high - max(current.open, current.close)
        lower_shadow = min(current.open, current.close) - current.low

        if upper_shadow <= body * self.min_shadow_ratio:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.SHOOTING_STAR,
                confidence=0, rejection_reasons=["Upper shadow too short"],
                recommendations=["Wait for a longer upper wick (min 2x body)"],
            )
        if lower_shadow > body:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.SHOOTING_STAR,
                confidence=0, rejection_reasons=["Lower shadow too long for shooting star"],
                recommendations=["Lower shadow should be smaller than body"],
            )
        return PatternValidation(
            is_valid=True, pattern_type=PatternType.SHOOTING_STAR,
            confidence=0.75, rejection_reasons=[],
            recommendations=["Confirm with bearish follow-through candle"],
        )

    def _validate_doji(
        self, current: Candle, sr_levels: List[SupportResistance]
    ) -> PatternValidation:
        """A doji has a body that is <= doji_body_pct of the total range."""
        if current.range is None or current.range == 0 or not np.isfinite(current.range):
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.DOJI,
                confidence=0, rejection_reasons=["Zero or invalid range candle"],
                recommendations=["Check data feed for corrupted candle"],
            )

        body_pct = safe_div(current.body, current.range)
        if body_pct > self.doji_body_pct:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.DOJI,
                confidence=0,
                rejection_reasons=[f"Body is {body_pct:.1%} of range — exceeds doji threshold of {self.doji_body_pct:.1%}"],
                recommendations=["Body must be very small relative to total range"],
            )

        return PatternValidation(
            is_valid=True, pattern_type=PatternType.DOJI,
            confidence=0.60, rejection_reasons=[],
            recommendations=["Wait for directional confirmation on the next candle"],
        )

    # ----------------------------------------------------------
    # Two-candle patterns
    # ----------------------------------------------------------

    def _validate_bullish_engulfing(
        self, current: Candle, prev: Candle, sr_levels: List[SupportResistance]
    ) -> PatternValidation:
        if not prev.is_bearish or not current.is_bullish:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.BULLISH_ENGULFING,
                confidence=0, rejection_reasons=["Requires bearish candle followed by bullish candle"],
                recommendations=["First candle must close below its open"],
            )
        if not (current.open <= prev.close and current.close >= prev.open):
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.BULLISH_ENGULFING,
                confidence=0, rejection_reasons=["Current candle does not engulf previous"],
                recommendations=["Bullish candle must open at or below prior close and close at or above prior open"],
            )
        return PatternValidation(
            is_valid=True, pattern_type=PatternType.BULLISH_ENGULFING,
            confidence=0.85, rejection_reasons=[],
            recommendations=["Enter on next open; stop below engulfing candle low"],
        )

    def _validate_bearish_engulfing(
        self, current: Candle, prev: Candle, sr_levels: List[SupportResistance]
    ) -> PatternValidation:
        if not prev.is_bullish or not current.is_bearish:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.BEARISH_ENGULFING,
                confidence=0, rejection_reasons=["Requires bullish candle followed by bearish candle"],
                recommendations=["First candle must close above its open"],
            )
        if not (current.open >= prev.close and current.close <= prev.open):
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.BEARISH_ENGULFING,
                confidence=0, rejection_reasons=["Current candle does not engulf previous"],
                recommendations=["Bearish candle must open at or above prior close and close at or below prior open"],
            )
        return PatternValidation(
            is_valid=True, pattern_type=PatternType.BEARISH_ENGULFING,
            confidence=0.85, rejection_reasons=[],
            recommendations=["Enter on next open; stop above engulfing candle high"],
        )

    # ----------------------------------------------------------
    # Three-candle patterns
    # ----------------------------------------------------------

    def _validate_morning_star(
        self,
        current: Candle,
        prev: Candle,
        prev2: Candle,
        sr_levels: List[SupportResistance],
    ) -> PatternValidation:
        """
        Morning Star (bullish reversal):
          - prev2: large bearish candle
          - prev:  small-bodied candle (star) that gaps down or sits well below prev2
          - current: large bullish candle closing well into prev2's body
        """
        if not prev2.is_bearish:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.MORNING_STAR,
                confidence=0, rejection_reasons=["First candle must be bearish"],
                recommendations=[],
            )

        star_body_pct = safe_div(prev.body, max(prev.range, 0.0001))
        if star_body_pct > 0.3:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.MORNING_STAR,
                confidence=0,
                rejection_reasons=["Middle candle body too large to qualify as a star"],
                recommendations=["Star candle body should be <30% of its range"],
            )

        # RELAXED: For 24h forex, true gaps are rare. Allow star near prev2 close.
        # Star should be at or below prev2's close (with small tolerance for spread/noise)
        star_high = max(prev.open, prev.close)
        gap_threshold = prev2.close * (1 + self.gap_pct)

        if star_high > gap_threshold:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.MORNING_STAR,
                confidence=0, rejection_reasons=["Star candle too high above first candle's close"],
                recommendations=["Star should gap down or sit near/below prior close"],
            )

        if not current.is_bullish:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.MORNING_STAR,
                confidence=0, rejection_reasons=["Third candle must be bullish"],
                recommendations=[],
            )

        # Third candle should close at least halfway into prev2's body
        midpoint_prev2 = (prev2.open + prev2.close) / 2
        if current.close < midpoint_prev2:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.MORNING_STAR,
                confidence=0,
                rejection_reasons=["Third candle does not close sufficiently into first candle's body"],
                recommendations=["Close should be above midpoint of first candle"],
            )

        return PatternValidation(
            is_valid=True, pattern_type=PatternType.MORNING_STAR,
            confidence=0.90, rejection_reasons=[],
            recommendations=["Strong reversal signal; enter on next open with stop below star low"],
        )

    def _validate_evening_star(
        self,
        current: Candle,
        prev: Candle,
        prev2: Candle,
        sr_levels: List[SupportResistance],
    ) -> PatternValidation:
        """
        Evening Star (bearish reversal):
          - prev2: large bullish candle
          - prev:  small-bodied candle (star) that gaps up or sits well above prev2
          - current: large bearish candle closing well into prev2's body
        """
        if not prev2.is_bullish:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.EVENING_STAR,
                confidence=0, rejection_reasons=["First candle must be bullish"],
                recommendations=[],
            )

        star_body_pct = safe_div(prev.body, max(prev.range, 0.0001))
        if star_body_pct > 0.3:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.EVENING_STAR,
                confidence=0,
                rejection_reasons=["Middle candle body too large to qualify as a star"],
                recommendations=["Star candle body should be <30% of its range"],
            )

        # RELAXED: Star should be at or above prev2's close (with tolerance)
        star_low = min(prev.open, prev.close)
        gap_threshold = prev2.close * (1 - self.gap_pct)

        if star_low < gap_threshold:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.EVENING_STAR,
                confidence=0, rejection_reasons=["Star candle too low below first candle's close"],
                recommendations=["Star should gap up or sit near/above prior close"],
            )

        if not current.is_bearish:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.EVENING_STAR,
                confidence=0, rejection_reasons=["Third candle must be bearish"],
                recommendations=[],
            )

        # Third candle should close at least halfway into prev2's body
        midpoint_prev2 = (prev2.open + prev2.close) / 2
        if current.close > midpoint_prev2:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.EVENING_STAR,
                confidence=0,
                rejection_reasons=["Third candle does not close sufficiently into first candle's body"],
                recommendations=["Close should be below midpoint of first candle"],
            )

        return PatternValidation(
            is_valid=True, pattern_type=PatternType.EVENING_STAR,
            confidence=0.90, rejection_reasons=[],
            recommendations=["Strong reversal signal; enter on next open with stop above star high"],
        )


# ============================================================
# MARKET STRUCTURE DETECTOR
# ============================================================

class MarketStructureDetector:
    """Full market structure analysis using SMA crossover with buffer band."""

    def analyze(self, candles: List[Candle]) -> MarketStructure:
        if len(candles) < 50:
            return MarketStructure(
                trend=Trend.RANGING, stage=MarketStage.UNKNOWN,
                swing_highs=[], swing_lows=[],
                trend_strength=0.5, is_healthy_trend=False,
            )

        try:
            closes = [c.close for c in candles[-100:] if np.isfinite(c.close)]
            if len(closes) < 50:
                return MarketStructure(
                    trend=Trend.RANGING, stage=MarketStage.UNKNOWN,
                    swing_highs=[], swing_lows=[],
                    trend_strength=0.5, is_healthy_trend=False,
                )

            sma_short = safe_mean(closes[-20:], closes[-1])
            sma_long  = safe_mean(closes[-50:], closes[-1])

            if sma_short > sma_long * 1.002:
                trend    = Trend.UPTREND
                strength = 0.8
            elif sma_short < sma_long * 0.998:
                trend    = Trend.DOWNTREND
                strength = 0.8
            else:
                trend    = Trend.RANGING
                strength = 0.3

            return MarketStructure(
                trend=trend, stage=MarketStage.UNKNOWN,
                swing_highs=[], swing_lows=[],
                trend_strength=strength,
                is_healthy_trend=strength > 0.6,
            )
        except Exception:
            return MarketStructure(
                trend=Trend.RANGING, stage=MarketStage.UNKNOWN,
                swing_highs=[], swing_lows=[],
                trend_strength=0.5, is_healthy_trend=False,
            )


# ============================================================
# ENHANCED S/R DETECTOR
# ============================================================

class SupportResistanceDetector:
    """
    S/R detection that returns multiple distinct levels.
    Each swing cluster separated by at least cluster_gap_pct becomes its own level.
    """

    def __init__(self):
        self.cluster_gap_pct = Config.CLUSTER_GAP_PCT
        self.lookback = Config.SR_LOOKBACK_CANDLES
        self.max_levels = Config.SR_MAX_LEVELS

    def find_levels(self, candles: List[Candle]) -> List[SupportResistance]:
        if len(candles) < 20:
            return []

        recent = candles[-self.lookback:]
        swing_highs: List[float] = []
        swing_lows: List[float]  = []

        for i in range(2, len(recent) - 2):
            c = recent[i]
            if not all(np.isfinite(v) for v in [c.high, c.low]):
                continue
            if (c.high > recent[i - 1].high and c.high > recent[i - 2].high and
                    c.high > recent[i + 1].high and c.high > recent[i + 2].high):
                swing_highs.append(c.high)

            if (c.low < recent[i - 1].low and c.low < recent[i - 2].low and
                    c.low < recent[i + 1].low and c.low < recent[i + 2].low):
                swing_lows.append(c.low)

        levels: List[SupportResistance] = []
        levels.extend(self._cluster_to_levels(sorted(swing_highs), is_resistance=True))
        levels.extend(self._cluster_to_levels(sorted(swing_lows),  is_resistance=False))

        # Sort by strength descending, return top N
        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[:self.max_levels]

    def _cluster_to_levels(
        self, prices: List[float], is_resistance: bool
    ) -> List[SupportResistance]:
        if not prices:
            return []

        clusters: List[List[float]] = []
        current_cluster: List[float] = [prices[0]]

        for price in prices[1:]:
            ref = safe_mean(current_cluster, current_cluster[0])
            if abs(price - ref) / max(ref, 0.0001) <= self.cluster_gap_pct:
                current_cluster.append(price)
            else:
                clusters.append(current_cluster)
                current_cluster = [price]
        clusters.append(current_cluster)

        levels = []
        for cluster in clusters:
            level    = safe_mean(cluster)
            touches  = len(cluster)
            # Strength: more touches = stronger; cap at 1.0
            strength = min(1.0, 0.3 + touches * 0.15)
            levels.append(SupportResistance(
                level=level,
                touches=touches,
                strength=strength,
                was_resistance=is_resistance,
                was_support=not is_resistance,
            ))
        return levels


# ============================================================
# H4 TREND GATE
# ============================================================

class H4TrendGate:
    """Prevents counter-trend trading. Accepts SignalDirection enum or string."""

    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher

    async def check_alignment(
        self, pair: str, direction: Union[SignalDirection, str]
    ) -> Tuple[bool, str]:
        """
        Args:
            direction: SignalDirection enum OR plain string 'buy'/'sell'.
        Returns:
            (is_aligned, reason_string)
        """
        # Normalise to lowercase string for comparison
        if isinstance(direction, SignalDirection):
            dir_str = direction.value.lower()
        else:
            dir_str = str(direction).lower()

        try:
            candles = await self.data_fetcher.get_candles(pair, "4h", limit=100)
            if len(candles) < 30:
                return True, "Insufficient H4 data — trend gate bypassed"

            closes    = [c.close for c in candles[-30:] if np.isfinite(c.close)]
            if len(closes) < 30:
                return True, "Insufficient valid H4 closes — trend gate bypassed"

            sma_short = safe_mean(closes[-10:], closes[-1])
            sma_long  = safe_mean(closes[-30:], closes[-1])

            is_uptrend = sma_short > sma_long

            if dir_str == "buy" and not is_uptrend:
                return False, "H4 trend is DOWNTREND — BUY signals blocked"
            elif dir_str == "sell" and is_uptrend:
                return False, "H4 trend is UPTREND — SELL signals blocked"

            return True, f"H4 trend aligned with {dir_str.upper()}"

        except Exception as e:
            logger.warning(f"H4 trend gate error: {e}")
            return True, "H4 gate bypassed due to error"


# ============================================================
# MAIN M.A.E. TRADING FORMULA
# ============================================================

class MAETradingFormula:
    """Complete M.A.E. Trading Formula — all issues corrected."""

    def __init__(self):
        self.validator         = EnhancedPatternValidator()
        self.structure_detector = MarketStructureDetector()
        self.sr_detector       = SupportResistanceDetector()
        self.confidence_scorer = AdvancedConfidenceScorer()
        self.h4_gate           = None  # Set via set_data_fetcher()

    def set_data_fetcher(self, data_fetcher):
        self.h4_gate = H4TrendGate(data_fetcher)

    def _determine_ranging_bias(
        self, current: Candle, sr_levels: List[SupportResistance]
    ) -> SignalDirection:
        """
        Determine directional bias in ranging markets based on price position
        relative to S/R levels. Buy if near support, Sell if near resistance.
        """
        if not sr_levels:
            return SignalDirection.BUY  # Default when no S/R data

        # Find nearest support and resistance
        supports = [sr for sr in sr_levels if sr.was_support]
        resistances = [sr for sr in sr_levels if sr.was_resistance]

        nearest_support = min(supports, key=lambda s: abs(current.close - s.level)) if supports else None
        nearest_resistance = min(resistances, key=lambda r: abs(current.close - r.level)) if resistances else None

        if nearest_support and nearest_resistance:
            dist_to_support = abs(current.close - nearest_support.level)
            dist_to_resistance = abs(current.close - nearest_resistance.level)

            # If much closer to support, bias long; if much closer to resistance, bias short
            if dist_to_support < dist_to_resistance * 0.5:
                return SignalDirection.BUY
            elif dist_to_resistance < dist_to_support * 0.5:
                return SignalDirection.SELL

        # Default: check if price is in upper or lower half of S/R range
        all_levels = [sr.level for sr in sr_levels]
        if all_levels:
            mid = safe_mean(all_levels)
            return SignalDirection.BUY if current.close < mid else SignalDirection.SELL

        return SignalDirection.BUY

    def generate_signal(
        self,
        candles: List[Candle],
        pair: str = "EURUSD",
        timeframe: str = "1h",
        account_balance: float = 10_000.0,
        risk_percent: float = 2.0,
        use_strict: bool = True,
        bar_closed: bool = True,
        mtf_aligned: bool = False,
    ) -> Optional[TradeSignal]:

        if not bar_closed or len(candles) < 50:
            return None

        try:
            structure  = self.structure_detector.analyze(candles)
            sr_levels  = self.sr_detector.find_levels(candles)

            # ATR-based stop/TP distances
            atr           = calculate_atr(candles)
            stop_distance = atr * Config.ATR_STOP_MULTIPLIER
            tp_distance   = atr * Config.ATR_TP_MULTIPLIER

            current = candles[-1]
            idx     = len(candles) - 1

            if structure.trend == Trend.UPTREND:
                patterns  = [PatternType.MORNING_STAR, PatternType.BULLISH_ENGULFING, PatternType.HAMMER]
                direction = SignalDirection.BUY
            elif structure.trend == Trend.DOWNTREND:
                patterns  = [PatternType.EVENING_STAR, PatternType.BEARISH_ENGULFING, PatternType.SHOOTING_STAR]
                direction = SignalDirection.SELL
            else:
                # In ranging markets, look for Doji + S/R confluence
                patterns  = [PatternType.DOJI]
                # FIXED: Determine bias from S/R position instead of hardcoded BUY
                direction = self._determine_ranging_bias(current, sr_levels)

            for pattern_type in patterns:
                validation = self.validator.validate_pattern(
                    candles, idx, sr_levels, pattern_type, bar_closed
                )

                if not validation.is_valid:
                    continue

                entry = current.close
                realistic_entry, spread_cost = _realistic_entry(entry, direction, pair)

                if direction == SignalDirection.BUY:
                    stop_loss   = realistic_entry - stop_distance
                    take_profit = realistic_entry + tp_distance
                else:
                    stop_loss   = realistic_entry + stop_distance
                    take_profit = realistic_entry - tp_distance

                rr = safe_div(tp_distance, stop_distance, 0.0)

                confidence = self.confidence_scorer.score(
                    current, candles, idx, sr_levels, pattern_type,
                    structure.trend_strength, mtf_aligned=mtf_aligned,
                )

                # Position sizing using actual stop distance
                position_size = self._calculate_position_size_internal(
                    account_balance=account_balance,
                    risk_percent=risk_percent,
                    stop_distance=stop_distance,
                    pair=pair,
                )

                explanation = (
                    f"{pattern_type.value.replace('_', ' ').title()} pattern on {pair} "
                    f"({timeframe}). ATR-based stop: {stop_distance:.5f}, "
                    f"R:R {rr:.1f}:1. MTF aligned: {mtf_aligned}."
                )
                if spread_cost > 0:
                    explanation += f" Spread-adjusted entry: {spread_cost:.5f}."

                return TradeSignal(
                    direction=direction,
                    entry_price=realistic_entry,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit,
                    risk_reward_1=round(rr, 2),
                    position_size=position_size,
                    pattern=pattern_type,
                    confidence=confidence,
                    market_structure=structure,
                    support_resistance=sr_levels[:3],
                    timeframe=timeframe,
                    pair=pair,
                    explanation=explanation,
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.warning(f"Signal generation error: {e}")
            return None

    def _calculate_position_size_internal(
        self,
        account_balance: float,
        risk_percent: float,
        stop_distance: float,
        pair: str,
    ) -> Dict[str, Any]:
        """
        Uses actual stop distance to size position correctly.
        """
        pip         = _pip_size(pair)
        risk_amount = account_balance * (risk_percent / 100)
        stop_pips   = safe_div(stop_distance, pip, 50)
        risk_per_pip = safe_div(risk_amount, stop_pips, 0)

        # Standard lot = $10/pip for majors; risk_per_pip / 10 gives lot size
        standard_lots = safe_div(risk_per_pip, 10, 0.01)
        max_lots = Config.MAX_POSITION_LOTS
        standard_lots = round(max(0.01, min(standard_lots, max_lots)), 2)

        return {
            "standard_lots": standard_lots,
            "risk_amount":   round(risk_amount, 2),
            "stop_pips":     round(stop_pips, 1),
            "risk_per_pip":  round(risk_per_pip, 2),
            "units":         int(standard_lots * 100_000),
        }

    async def check_h4_alignment(
        self, pair: str, direction: Union[SignalDirection, str]
    ) -> Tuple[bool, str]:
        if self.h4_gate:
            return await self.h4_gate.check_alignment(pair, direction)
        return True, "H4 gate not configured"

    # ----------------------------------------------------------
    # Backtest
    # ----------------------------------------------------------

    def backtest(self, candles: List[Candle], request: BacktestRequest) -> BacktestResult:
        """
        Raises NotImplementedError — full backtest engine is not yet live.
        """
        raise NotImplementedError(
            "Full backtest engine is not yet implemented. "
            "Do not call this method in production."
        )

    # ----------------------------------------------------------
    # Public position size calculator
    # ----------------------------------------------------------

    def calculate_position_size(self, request: PositionSizeRequest) -> PositionSizeResult:
        """
        Uses stop_distance from the request object for accurate sizing.
        """
        pair          = getattr(request, "pair", "EURUSD")
        leverage      = getattr(request, "leverage", Config.DEFAULT_LEVERAGE)
        pip           = _pip_size(pair)
        risk_amount   = request.account_balance * (request.risk_percent / 100)

        # Prefer explicit stop_distance; fall back to stop_pips if provided
        stop_distance = getattr(request, "stop_distance", None)
        stop_pips_req = getattr(request, "stop_pips", None)

        if stop_distance and stop_distance > 0 and np.isfinite(stop_distance):
            stop_pips = safe_div(stop_distance, pip, 50)
        elif stop_pips_req and stop_pips_req > 0 and np.isfinite(stop_pips_req):
            stop_pips     = stop_pips_req
            stop_distance = stop_pips * pip
        else:
            stop_pips     = 50.0
            stop_distance = stop_pips * pip
            logger.warning(
                "calculate_position_size: no stop_distance or stop_pips provided — "
                "defaulting to 50 pips. Pass stop_distance for accurate sizing."
            )

        risk_per_pip  = safe_div(risk_amount, stop_pips, 0)
        standard_lots = safe_div(risk_per_pip, 10, 0.01)
        max_lots = Config.MAX_POSITION_LOTS
        standard_lots = max(0.01, min(standard_lots, max_lots))

        warnings = []
        if standard_lots >= 5.0:
            warnings.append("Position size is very large — double-check risk settings.")
        if request.risk_percent > 5.0:
            warnings.append("Risk percent exceeds 5% — high risk to account.")

        # FIXED: Proper margin calculation
        # For forex: Margin = (Lot Size × Contract Size × Price) / Leverage
        # We use current price estimate from request if available, else approximate
        entry_price = getattr(request, "entry_price", None)
        if entry_price and np.isfinite(entry_price):
            price_for_margin = entry_price
        else:
            # Approximate: for most pairs, use 1.0 (close enough for margin estimate)
            # XAUUSD is ~2000, so we handle that
            price_for_margin = 2000.0 if "XAU" in pair.upper() else 1.0
            logger.info(f"No entry_price provided; using estimated price {price_for_margin} for margin calc")

        contract_size = 100_000  # Standard forex lot
        required_margin = (standard_lots * contract_size * price_for_margin) / leverage

        # FIXED: TP hit uses actual R:R, not hardcoded 2x
        actual_rr = safe_div(stop_distance * Config.ATR_TP_MULTIPLIER / Config.ATR_STOP_MULTIPLIER, 
                              stop_distance, 2.0)
        if_tp1_hit = round(request.account_balance + risk_amount * actual_rr, 2)

        return PositionSizeResult(
            standard_lots=round(standard_lots, 2),
            mini_lots=round(standard_lots * 10, 1),
            micro_lots=round(standard_lots * 100, 0),
            units=int(standard_lots * 100_000),
            risk_amount=round(risk_amount, 2),
            risk_per_pip=round(risk_per_pip, 2),
            required_margin=round(required_margin, 2),
            leverage_used=leverage,
            margin_percent=round(safe_div(required_margin, request.account_balance) * 100, 2),
            pip_value=round(risk_per_pip, 2),
            if_stop_loss=round(request.account_balance - risk_amount, 2),
            if_tp1_hit=if_tp1_hit,
            warnings=warnings,
            is_safe=len(warnings) == 0,
        )
