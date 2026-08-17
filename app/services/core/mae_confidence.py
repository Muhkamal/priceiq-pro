"""
M.A.E. Enhanced Confidence Scorer
Multi-factor weighted confidence (0-1) that actually varies.
Drop this in to override basic agent confidence before signals go out.
"""

import numpy as np
from typing import List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MAEConfidenceScorer:
    WEIGHTS = {
        "shadow":    0.20,
        "trend":     0.20,
        "sr":        0.20,
        "close_pos": 0.15,
        "volume":    0.10,
        "rarity":    0.10,
        "mtf":       0.05,
    }

    def score(self, candle: Any, candles: List[Any], idx: int,
              sr_levels: List[Any], pattern_type: str,
              trend_strength: float, mtf_aligned: bool = False) -> float:
        scores = {}
        scores["shadow"] = self._shadow_score(candle, pattern_type)
        scores["trend"] = min(1.0, max(0.0, trend_strength))
        scores["sr"] = self._sr_score(candle, sr_levels, pattern_type)
        scores["close_pos"] = self._close_position_score(candle, pattern_type)
        scores["volume"] = self._volume_score(candle, candles, idx)
        scores["rarity"] = self._rarity_score(pattern_type)
        scores["mtf"] = 1.0 if mtf_aligned else 0.5

        total = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        return round(min(1.0, max(0.0, total)), 3)

    def _shadow_score(self, c: Any, pattern: str) -> float:
        if c.range == 0:
            return 0.0
        body = abs(c.close - c.open)
        if pattern in ("hammer", "bullish_engulfing", "morning_star", "breakout_pullback_buy"):
            lower = min(c.open, c.close) - c.low
            if lower == 0:
                return 0.0
            ratio = lower / body if body > 0 else lower / c.range
            return min(1.0, ratio / 2.5) if ratio <= 2.5 else max(0.0, 1.0 - (ratio - 2.5) / 2.5)
        elif pattern in ("shooting_star", "bearish_engulfing", "evening_star", "breakout_pullback_sell"):
            upper = c.high - max(c.open, c.close)
            if upper == 0:
                return 0.0
            ratio = upper / body if body > 0 else upper / c.range
            return min(1.0, ratio / 2.5) if ratio <= 2.5 else max(0.0, 1.0 - (ratio - 2.5) / 2.5)
        elif "engulfing" in pattern:
            body_pct = body / c.range
            return min(1.0, body_pct / 0.8)
        return 0.5

    def _sr_score(self, c: Any, sr_levels: List[Any], pattern: str) -> float:
        if not sr_levels:
            return 0.0
        ref = c.low if pattern in ("hammer", "bullish_engulfing", "morning_star", "breakout_pullback_buy") else c.high
        dists = []
        for sr in sr_levels:
            level = sr.get("level", sr) if isinstance(sr, dict) else getattr(sr, "level", sr)
            strength = sr.get("strength", 0.5) if isinstance(sr, dict) else getattr(sr, "strength", 0.5)
            dist = abs(ref - level) / level
            weighted = strength * max(0.0, 1.0 - dist / 0.01)
            dists.append(weighted)
        return min(1.0, max(dists)) if dists else 0.0

    def _close_position_score(self, c: Any, pattern: str) -> float:
        if c.range == 0:
            return 0.5
        close_pct = (c.close - c.low) / c.range
        if pattern in ("hammer", "bullish_engulfing", "morning_star", "breakout_pullback_buy"):
            return min(1.0, max(0.0, (close_pct - 0.5) / 0.5))
        elif pattern in ("shooting_star", "bearish_engulfing", "evening_star", "breakout_pullback_sell"):
            return min(1.0, max(0.0, (0.5 - close_pct) / 0.5))
        return 0.5

    def _volume_score(self, c: Any, candles: List[Any], idx: int) -> float:
        if not getattr(c, "volume", None) or idx < 5:
            return 0.5
        lookback = candles[max(0, idx - 20):idx]
        vols = [x.volume for x in lookback if getattr(x, "volume", None)]
        if not vols:
            return 0.5
        avg = np.mean(vols)
        ratio = c.volume / avg
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

    def _rarity_score(self, pattern: str) -> float:
        rarity_map = {
            "morning_star": 1.0, "evening_star": 1.0,
            "bullish_engulfing": 0.8, "bearish_engulfing": 0.8,
            "hammer": 0.6, "shooting_star": 0.6,
            "doji": 0.4,
            "breakout_pullback_buy": 0.9, "breakout_pullback_sell": 0.9,
        }
        return rarity_map.get(pattern, 0.5)

mae_confidence = MAEConfidenceScorer()
