"""
PriceIQ Pro — Multi-Timeframe Analysis v3.2
Updated for v3.2 corrected engine compatibility.

IMPROVEMENTS:
  - Correct import paths (market_analyzer_advanced)
  - Uses engine's native MTF confidence scoring (no manual boost)
  - Async support with H4 gate integration
  - Preserves engine explanation (appends MTF info)
  - Logging for all gate failures
  - Input validation
"""

import logging
from typing import List, Optional, Dict

from app.models.schemas import (
    Candle, TradeSignal, Trend, SignalDirection, MarketStructure
)
from app.services.market_analyzer_advanced import (
    MarketStructureDetector,
    MAETradingFormula,
)

logger = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyzes D1 → H4 → H1 to confirm signals with higher-timeframe alignment.

    Uses the v3.2 engine's native multi-timeframe support:
    - H4 gate is checked via mae_engine.check_h4_alignment()
    - Confidence scoring uses mtf_aligned parameter
    - No manual confidence overrides
    """

    def __init__(self):
        self.structure_detector = MarketStructureDetector()
        self.mae_engine = MAETradingFormula()

    async def analyze_all_timeframes(
        self,
        candles_h1: List[Candle],
        candles_h4: List[Candle],
        candles_d1: List[Candle],
        pair: str = "EURUSD",
        account_balance: float = 10_000.0,
        risk_percent: float = 2.0,
        bar_closed: bool = True,
    ) -> Optional[TradeSignal]:
        """
        Generate a signal only if D1 → H4 → H1 all agree.

        Args:
            candles_h1: 1-hour candles
            candles_h4: 4-hour candles
            candles_d1: Daily candles
            pair: Currency pair
            account_balance: Account balance
            risk_percent: Risk per trade
            bar_closed: Must be True for live trading

        Returns:
            TradeSignal if all timeframes align, None otherwise.
        """
        pair = pair.upper()

        # FIX: Input validation
        if not candles_h1 or len(candles_h1) < 50:
            logger.warning(f"MTF: Insufficient H1 candles for {pair}: {len(candles_h1) if candles_h1 else 0}")
            return None
        if not candles_h4 or len(candles_h4) < 30:
            logger.warning(f"MTF: Insufficient H4 candles for {pair}: {len(candles_h4) if candles_h4 else 0}")
            return None
        if not candles_d1 or len(candles_d1) < 20:
            logger.warning(f"MTF: Insufficient D1 candles for {pair}: {len(candles_d1) if candles_d1 else 0}")
            return None

        # FIX: refuse to process live unclosed bar
        if not bar_closed:
            logger.info(f"MTF: Bar not closed for {pair} — skipping")
            return None

        structure_d1 = self.structure_detector.analyze(candles_d1)
        structure_h4 = self.structure_detector.analyze(candles_h4)

        # --- Gate 1: D1 must have a clear direction ---
        if structure_d1.trend in (Trend.RANGING, Trend.UNKNOWN):
            logger.info(f"MTF: D1 ranging/unknown for {pair} — no trade")
            return None

        # --- Gate 2: H4 must match D1 ---
        if structure_h4.trend != structure_d1.trend:
            logger.info(
                f"MTF: H4 {structure_h4.trend.value} disagrees with D1 {structure_d1.trend.value} "
                f"for {pair} — no trade"
            )
            return None

        # --- Gate 3: H4 trend must be healthy ---
        if not structure_h4.is_healthy_trend:
            logger.info(f"MTF: H4 trend unhealthy for {pair} — no trade")
            return None

        # --- Gate 4: H4 alignment check (async) ---
        # Use v3.2 native H4 gate
        expected_direction = "buy" if structure_d1.trend == Trend.UPTREND else "sell"
        h4_allowed, h4_reason = await self.mae_engine.check_h4_alignment(pair, expected_direction)

        if not h4_allowed:
            logger.info(f"MTF: H4 gate blocked {pair} {expected_direction} — {h4_reason}")
            return None

        logger.info(f"MTF: H4 gate passed for {pair} {expected_direction} — {h4_reason}")

        # --- Gate 5: Generate H1 signal with MTF alignment ---
        # FIX: Pass mtf_aligned=True so engine scores confidence correctly
        # No manual confidence boost — engine handles it
        signal = self.mae_engine.generate_signal(
            candles=candles_h1,
            pair=pair,
            timeframe="1h",
            account_balance=account_balance,
            risk_percent=risk_percent,
            use_strict=True,
            bar_closed=bar_closed,
            mtf_aligned=True,  # Engine uses this for confidence scoring
        )

        if signal is None:
            logger.info(f"MTF: No H1 signal generated for {pair}")
            return None

        # --- Gate 6: H1 signal direction must match D1 ---
        if structure_d1.trend == Trend.UPTREND and signal.direction != SignalDirection.BUY:
            logger.info(
                f"MTF: H1 signal {signal.direction.value} contradicts D1 UPTREND for {pair}"
            )
            return None
        if structure_d1.trend == Trend.DOWNTREND and signal.direction != SignalDirection.SELL:
            logger.info(
                f"MTF: H1 signal {signal.direction.value} contradicts D1 DOWNTREND for {pair}"
            )
            return None

        # All gates passed — append MTF info to existing explanation
        # FIX: Don't overwrite engine's explanation
        mtf_explanation = self._generate_explanation(signal, structure_d1, structure_h4, h4_reason)
        if signal.explanation:
            signal.explanation = f"{signal.explanation}

{mtf_explanation}"
        else:
            signal.explanation = mtf_explanation

        logger.info(
            f"MTF: Signal confirmed for {pair} {signal.direction.value} "
            f"| Pattern: {signal.pattern.value} | Confidence: {signal.confidence:.0%}"
        )

        return signal

    def get_timeframe_alignment(
        self,
        candles_h1: List[Candle],
        candles_h4: List[Candle],
        candles_d1: List[Candle],
    ) -> Dict:
        """
        Return a detailed alignment status dict for dashboard display.
        """
        structure_d1 = self.structure_detector.analyze(candles_d1)
        structure_h4 = self.structure_detector.analyze(candles_h4)
        structure_h1 = self.structure_detector.analyze(candles_h1)

        breakdown: Dict[str, int] = {
            "d1_trending":    0,
            "h4_matches_d1":  0,
            "h1_matches_d1":  0,
            "h4_healthy":     0,
            "h1_healthy":     0,
            "h1_not_ranging": 0,  # NEW: Penalize ranging H1
        }

        if structure_d1.trend not in (Trend.RANGING, Trend.UNKNOWN):
            breakdown["d1_trending"] = 30
        if structure_h4.trend == structure_d1.trend:
            breakdown["h4_matches_d1"] = 25
        if structure_h1.trend == structure_d1.trend:
            breakdown["h1_matches_d1"] = 15  # Reduced from 20
        if structure_h4.is_healthy_trend:
            breakdown["h4_healthy"] = 15
        if structure_h1.is_healthy_trend:
            breakdown["h1_healthy"] = 10
        if structure_h1.trend not in (Trend.RANGING, Trend.UNKNOWN):
            breakdown["h1_not_ranging"] = 5  # NEW

        score = sum(breakdown.values())
        is_aligned = score >= 70

        if score >= 85:
            recommendation = "Strong alignment — look for entry signals now"
        elif score >= 70:
            recommendation = "Alignment confirmed — proceed with caution"
        elif score >= 50:
            recommendation = "Partial alignment — wait for H4 or D1 to clarify"
        else:
            recommendation = "No alignment — stay flat until trends agree"

        return {
            "d1_trend":           structure_d1.trend.value,
            "h4_trend":           structure_h4.trend.value,
            "h1_trend":           structure_h1.trend.value,
            "d1_trend_strength":  round(structure_d1.trend_strength, 3),
            "h4_trend_strength":  round(structure_h4.trend_strength, 3),
            "h1_trend_strength":  round(structure_h1.trend_strength, 3),
            "d1_healthy":         structure_d1.is_healthy_trend,
            "h4_healthy":         structure_h4.is_healthy_trend,
            "h1_healthy":         structure_h1.is_healthy_trend,
            "alignment_score":    score,
            "score_breakdown":    breakdown,
            "is_aligned":         is_aligned,
            "recommendation":     recommendation,
        }

    def _generate_explanation(
        self,
        signal: TradeSignal,
        structure_d1: MarketStructure,
        structure_h4: MarketStructure,
        h4_reason: str,
    ) -> str:
        """Generate MTF explanation appended to engine's existing explanation."""
        direction_emoji = "BUY" if signal.direction == SignalDirection.BUY else "SELL"
        pattern_name = signal.pattern.value.replace("_", " ").title()

        lines = [
            f"=== MULTI-TIMEFRAME CONFIRMATION ===",
            f"",
            f"Daily (D1): {structure_d1.trend.value.upper()} | Strength: {structure_d1.trend_strength:.0%}",
            f"4-Hour (H4): {structure_h4.trend.value.upper()} | Strength: {structure_h4.trend_strength:.0%} | {h4_reason}",
            f"1-Hour (H1): {pattern_name} | Confidence: {signal.confidence:.0%}",
            f"",
            f"All timeframes aligned — {direction_emoji} signal confirmed.",
        ]

        return "\n".join(lines)
