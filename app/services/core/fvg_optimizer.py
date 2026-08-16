"""
Fair Value Gap (FVG) Optimizer — Smart Money Concept entry refinement.
An FVG is a 3-candle pattern where candle 1 and 3 don't overlap.
Price often retraces to fill the gap before continuing = better entry.
"""
import numpy as np
from typing import List, Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FVGOptimizer:
    """
    Detects recent Fair Value Gaps and suggests entry at the gap fill level.
    This improves fill price vs market order = better R:R.
    """

    def find_fvg(self, candles: List[Any]) -> Optional[Dict]:
        if len(candles) < 5:
            return None

        # Look at last 3 completed candles (excluding current forming)
        c1 = candles[-4]  # candle 1
        c2 = candles[-3]  # candle 2 (impulse)
        c3 = candles[-2]  # candle 3

        # Bullish FVG: c2 low > c1 high (gap up), c3 low > c1 high (unfilled)
        if c2.low > c1.high and c3.low > c1.high:
            return {
                "type": "bullish",
                "top": c2.low,
                "bottom": c1.high,
                "mid": (c2.low + c1.high) / 2,
                "direction": "buy",
            }

        # Bearish FVG: c2 high < c1 low (gap down), c3 high < c1 low (unfilled)
        if c2.high < c1.low and c3.high < c1.low:
            return {
                "type": "bearish",
                "top": c1.low,
                "bottom": c2.high,
                "mid": (c1.low + c2.high) / 2,
                "direction": "sell",
            }

        return None

    def suggest_entry(self, candles: List[Any], signal_direction: str, pair: str) -> Dict:
        fvg = self.find_fvg(candles)
        current = candles[-1].close
        decimals = 2 if "XAU" in pair else (0 if "BTC" in pair else 5)

        if not fvg or fvg["direction"] != signal_direction:
            return {"use_fvg": False, "entry": round(current, decimals), "type": "market"}

        # Price hasn't filled the gap yet — enter at gap mid or near
        gap_mid = fvg["mid"]
        if signal_direction == "buy" and current > gap_mid:
            # Gap is below current price — set limit at gap mid
            entry = round(gap_mid, decimals)
            return {
                "use_fvg": True,
                "entry": entry,
                "type": "fvg_limit",
                "fvg_type": fvg["type"],
                "gap_top": round(fvg["top"], decimals),
                "gap_bottom": round(fvg["bottom"], decimals),
            }
        elif signal_direction == "sell" and current < gap_mid:
            entry = round(gap_mid, decimals)
            return {
                "use_fvg": True,
                "entry": entry,
                "type": "fvg_limit",
                "fvg_type": fvg["type"],
                "gap_top": round(fvg["top"], decimals),
                "gap_bottom": round(fvg["bottom"], decimals),
            }

        return {"use_fvg": False, "entry": round(current, decimals), "type": "market"}


fvg_optimizer = FVGOptimizer()
