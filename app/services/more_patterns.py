"""
PriceIQ Pro — Additional Candlestick Patterns v2.2
Covers: Doji, Morning Star, Evening Star

IMPROVEMENTS:
  - bar_closed guard on every validate_* method
  - Real multi-factor confidence via ConfidenceScorer
  - Tighter Morning Star / Evening Star penetration check
  - Doji dragonfly / gravestone subtypes detected
  - All patterns route through validate_pattern_extended()
  - Fixed divide-by-zero in trend slope calculation
  - Fixed Evening Star midpoint comparison direction
  - Deduplicated Morning/Evening Star validation logic
"""

from typing import List, Optional
import numpy as np

from app.models.schemas import (
    Candle, PatternValidation, PatternType,
    SignalDirection, SupportResistance
)
from app.services.market_analyzer import (
    StrictPatternValidator,
    MarketStructureDetector,
    ConfidenceScorer,
    confidence_scorer,
)


class ExtendedPatternValidator(StrictPatternValidator):
    """
    Adds Doji, Morning Star, and Evening Star to strict validation.
    All methods respect bar_closed and use real confidence scoring.
    """

    def __init__(self):
        super().__init__()
        self._structure_detector = MarketStructureDetector()

    # ──────────────────────────────────────────────────────────────
    # DOJI
    # ──────────────────────────────────────────────────────────────

    def validate_doji(
        self,
        candles: List[Candle],
        idx: int,
        sr_levels: List[SupportResistance],
        bar_closed: bool = True,
    ) -> PatternValidation:
        """
        Strict Doji validation — indecision candle at a key level.

        Sub-types detected:
        - Standard Doji  : body < 10% of range, balanced shadows
        - Dragonfly Doji : no upper shadow (bullish hint at support)
        - Gravestone Doji: no lower shadow (bearish hint at resistance)

        Rules:
        - Body < 10% of range
        - Must be at S/R
        - Prior trend must be clear (3+ directional bars in last 5)
        - Volume should be below or equal to average (genuine indecision)
        """
        guard = self._require_closed(bar_closed)
        if guard:
            guard.pattern_type = PatternType.DOJI
            return guard

        if idx < 5 or idx >= len(candles) - 1:
            return PatternValidation(
                is_valid=False, pattern_type=PatternType.DOJI,
                rejection_reasons=["Not enough candles for context"],
                recommendations=["Wait for more price history"]
            )

        current = candles[idx]
        rejection_reasons: List[str] = []
        recommendations: List[str] = []

        # 1. Body must be very small (< 10% of range)
        body_ratio = current.body / current.range if current.range > 0 else 1.0
        if body_ratio > 0.10:
            rejection_reasons.append(
                f"Body too large ({body_ratio:.1%} > 10%) — not a true doji"
            )

        # 2. Must be at a key S/R level (within 0.5%)
        at_key_level = any(
            abs(current.close - sr.level) / sr.level < 0.005
            for sr in sr_levels if sr.level > 0
        )
        if not at_key_level:
            rejection_reasons.append("Not at a key support/resistance level")
            recommendations.append("Wait for price to reach an S/R zone")

        # 3. Prior trend must be clear
        prior = candles[idx - 5:idx]
        bullish_count = sum(1 for c in prior if c.is_bullish)
        bearish_count = sum(1 for c in prior if c.is_bearish)
        if bullish_count < 3 and bearish_count < 3:
            rejection_reasons.append(
                "Prior trend unclear — need at least 3 directional candles in last 5"
            )

        # 4. Volume <= 120% of 10-bar average (indecision = quiet session)
        if current.volume and idx > 10:
            vols = [c.volume for c in candles[idx - 10:idx] if c.volume]
            if vols:
                avg_volume = np.mean(vols)
                if avg_volume > 0 and current.volume > avg_volume * 1.2:
                    recommendations.append(
                        "Volume above average — may be a news event, not pure indecision"
                    )

        # 5. Sub-type note (informational, not a rejection)
        if current.upper_shadow < current.range * 0.05 and current.lower_shadow > current.range * 0.4:
            recommendations.insert(0, "Sub-type: Dragonfly Doji — bullish hint at support")
        elif current.lower_shadow < current.range * 0.05 and current.upper_shadow > current.range * 0.4:
            recommendations.insert(0, "Sub-type: Gravestone Doji — bearish hint at resistance")

        is_valid = len(rejection_reasons) == 0
        trend_strength, _ = self._structure_detector.calculate_trend_strength(candles[:idx + 1])
        confidence = confidence_scorer.score(
            current, candles, idx, sr_levels, PatternType.DOJI, trend_strength
        ) if is_valid else 0.0

        return PatternValidation(
            is_valid=is_valid,
            pattern_type=PatternType.DOJI,
            confidence=confidence,
            rejection_reasons=rejection_reasons,
            recommendations=recommendations,
        )

    # ──────────────────────────────────────────────────────────────
    # MORNING STAR / EVENING STAR — shared core logic
    # ──────────────────────────────────────────────────────────────

    def _validate_star_pattern(
        self,
        candles: List[Candle],
        idx: int,
        sr_levels: List[SupportResistance],
        pattern_type: PatternType,
        is_bullish_reversal: bool,
    ) -> PatternValidation:
        """
        Shared validation logic for Morning Star and Evening Star.

        Args:
            is_bullish_reversal: True for Morning Star, False for Evening Star
        """
        if idx < 7:
            return PatternValidation(
                is_valid=False, pattern_type=pattern_type,
                rejection_reasons=["Need at least 8 candles for context"],
                recommendations=[]
            )

        first = candles[idx - 2]
        star = candles[idx - 1]
        third = candles[idx]
        rejection_reasons: List[str] = []
        recommendations: List[str] = []

        # Determine expected directions
        expected_first_dir = "bullish" if not is_bullish_reversal else "bearish"
        expected_third_dir = "bearish" if not is_bullish_reversal else "bullish"
        first_is_correct = first.is_bullish if not is_bullish_reversal else first.is_bearish
        third_is_correct = third.is_bearish if not is_bullish_reversal else third.is_bullish

        # 1. First candle: strong trend candle
        if not first_is_correct:
            rejection_reasons.append(f"Candle 1 must be {expected_first_dir}")
        if first.range > 0 and first.body / first.range < 0.60:
            rejection_reasons.append(
                f"Candle 1 body too small ({first.body / first.range:.1%}) — need strong {expected_first_dir} bar"
            )

        # 2. Star: small body (<= 30% of range)
        star_body_ratio = star.body / star.range if star.range > 0 else 1.0
        if star_body_ratio > 0.30:
            rejection_reasons.append(
                f"Star body too large ({star_body_ratio:.1%} > 30%)"
            )

        # 3. Third candle: strong reversal candle
        if not third_is_correct:
            rejection_reasons.append(f"Candle 3 must be {expected_third_dir}")
        if third.range > 0 and third.body / third.range < 0.60:
            rejection_reasons.append(
                f"Candle 3 body too small ({third.body / third.range:.1%}) — need strong {expected_third_dir} bar"
            )

        # 4. Third candle must close >= 50% into first candle's body
        first_body_top = max(first.open, first.close)
        first_body_bot = min(first.open, first.close)
        midpoint = first_body_bot + (first_body_top - first_body_bot) * 0.50

        if is_bullish_reversal:
            if third.close < midpoint:
                rejection_reasons.append(
                    f"Candle 3 closes at {third.close:.5f} — must close above {midpoint:.5f} (50% of candle 1 body)"
                )
        else:
            if third.close > midpoint:
                rejection_reasons.append(
                    f"Candle 3 closes at {third.close:.5f} — must close below {midpoint:.5f} (50% of candle 1 body)"
                )

        # 5. Must be at support (Morning Star) or resistance (Evening Star)
        if is_bullish_reversal:
            at_sr = any(
                abs(star.low - sr.level) / max(sr.level, 1e-10) < 0.005
                for sr in sr_levels if sr.was_support
            )
            if not at_sr:
                rejection_reasons.append("Star not at a valid support level")
                recommendations.append("Wait for price to reach key support")
        else:
            at_sr = any(
                abs(star.high - sr.level) / max(sr.level, 1e-10) < 0.005
                for sr in sr_levels if sr.was_resistance
            )
            if not at_sr:
                rejection_reasons.append("Star not at a valid resistance level")
                recommendations.append("Wait for price to reach key resistance")

        # 6. Prior trend verification
        prior = candles[max(0, idx - 10):idx - 2]
        if len(prior) >= 5:
            closes = [c.close for c in prior]
            # Use price change, not slope (avoids divide-by-zero)
            price_change = closes[-1] - closes[0]
            if is_bullish_reversal:
                if price_change >= 0:
                    rejection_reasons.append(
                        "No clear downtrend before pattern — Morning Star needs a prior decline"
                    )
            else:
                if price_change <= 0:
                    rejection_reasons.append(
                        "No clear uptrend before pattern — Evening Star needs a prior advance"
                    )

        is_valid = len(rejection_reasons) == 0
        trend_strength, _ = self._structure_detector.calculate_trend_strength(candles[:idx + 1])
        confidence = confidence_scorer.score(
            third, candles, idx, sr_levels, pattern_type, trend_strength
        ) if is_valid else 0.0

        return PatternValidation(
            is_valid=is_valid,
            pattern_type=pattern_type,
            confidence=confidence,
            rejection_reasons=rejection_reasons,
            recommendations=recommendations,
        )

    def validate_morning_star(
        self,
        candles: List[Candle],
        idx: int,
        sr_levels: List[SupportResistance],
        bar_closed: bool = True,
    ) -> PatternValidation:
        """
        Morning Star — 3-candle bullish reversal at support.

        Pattern (candles[idx-2], candles[idx-1], candles[idx]):
          1. Strong bearish candle  (body >= 60% of range)
          2. Star candle            (body <= 30% of range)
          3. Strong bullish candle  (body >= 60% of range, closes >= 50% into candle 1)
        """
        guard = self._require_closed(bar_closed)
        if guard:
            guard.pattern_type = PatternType.MORNING_STAR
            return guard

        return self._validate_star_pattern(
            candles, idx, sr_levels, PatternType.MORNING_STAR, is_bullish_reversal=True
        )

    def validate_evening_star(
        self,
        candles: List[Candle],
        idx: int,
        sr_levels: List[SupportResistance],
        bar_closed: bool = True,
    ) -> PatternValidation:
        """
        Evening Star — 3-candle bearish reversal at resistance.

        Pattern (candles[idx-2], candles[idx-1], candles[idx]):
          1. Strong bullish candle  (body >= 60% of range)
          2. Star candle            (body <= 30% of range)
          3. Strong bearish candle  (body >= 60% of range, closes <= 50% into candle 1)
        """
        guard = self._require_closed(bar_closed)
        if guard:
            guard.pattern_type = PatternType.EVENING_STAR
            return guard

        return self._validate_star_pattern(
            candles, idx, sr_levels, PatternType.EVENING_STAR, is_bullish_reversal=False
        )

    # ──────────────────────────────────────────────────────────────
    # ROUTER — covers all 7 pattern types
    # ──────────────────────────────────────────────────────────────

    def validate_pattern_extended(
        self,
        candles: List[Candle],
        idx: int,
        sr_levels: List[SupportResistance],
        pattern_type: PatternType,
        bar_closed: bool = True,
    ) -> PatternValidation:
        """
        Route to the appropriate validator.
        Extended patterns handled here; original 4 delegated to parent.
        """
        if pattern_type == PatternType.DOJI:
            return self.validate_doji(candles, idx, sr_levels, bar_closed)
        elif pattern_type == PatternType.MORNING_STAR:
            return self.validate_morning_star(candles, idx, sr_levels, bar_closed)
        elif pattern_type == PatternType.EVENING_STAR:
            return self.validate_evening_star(candles, idx, sr_levels, bar_closed)
        else:
            # Hammer, Shooting Star, Bullish/Bearish Engulfing
            return self.validate_pattern(candles, idx, sr_levels, pattern_type, bar_closed)
