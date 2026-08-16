"""
Smart Stop Calculator — Professional-grade SL/TP placement.
Structure FIRST (swing high/low), ATR SECOND (volatility buffer).
Prevents stop-hunts by placing stops where the trade thesis is actually invalidated.
"""
import numpy as np
from typing import List, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Pair-specific calibration from professional algo research
PAIR_CONFIG: Dict[str, Dict] = {
    "XAUUSD":  {"atr_mult": 2.0, "structure_buffer_atr": 0.3, "min_bars": 10, "spread_buffer": 0.05},
    "EURUSD":  {"atr_mult": 1.5, "structure_buffer_atr": 0.5, "min_bars": 10, "spread_buffer": 0.00005},
    "GBPUSD":  {"atr_mult": 1.8, "structure_buffer_atr": 0.5, "min_bars": 10, "spread_buffer": 0.00005},
    "USDCHF":  {"atr_mult": 1.5, "structure_buffer_atr": 0.5, "min_bars": 10, "spread_buffer": 0.00005},
    "AUDUSD":  {"atr_mult": 1.6, "structure_buffer_atr": 0.5, "min_bars": 10, "spread_buffer": 0.00005},
    "BTCUSD":  {"atr_mult": 3.0, "structure_buffer_atr": 1.0, "min_bars": 14, "spread_buffer": 5.0},
}

SESSION_VOL_BOOST: Dict[str, float] = {
    "london": 1.15,
    "london_newyork": 1.25,
    "newyork": 1.20,
    "newyork_asia": 1.05,
    "tokyo": 0.95,
    "other": 1.0,
}


class SmartStopCalculator:
    """
    Calculates SL/TP using:
    1. Structure (swing high/low) — where the setup is invalidated
    2. ATR buffer — volatility cushion beyond structure
    3. Pair calibration — EURUSD is not BTCUSD
    4. Session boost — wider during volatile sessions
    """

    def __init__(self):
        self.config = PAIR_CONFIG

    def _atr(self, candles: List[Any], period: int = 14) -> float:
        if len(candles) < period + 1:
            return 0.0001
        highs = np.array([c.high for c in candles[-period-1:]])
        lows = np.array([c.low for c in candles[-period-1:]])
        closes = np.array([c.close for c in candles[-period-1:]])
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return float(np.mean(tr))

    def _find_swing_low(self, candles: List[Any], lookback: int = 10) -> float:
        """Lowest low in lookback that is lower than neighbors (fractal-like)."""
        if len(candles) < lookback + 2:
            return min(c.low for c in candles[-lookback:]) if candles else 0.0
        lows = [c.low for c in candles[-lookback:]]
        # Simple: lowest low in last N bars, but ensure it's not the very last bar (wicks)
        # Look at bars 1..-1 (exclude current forming bar)
        candidate_lows = lows[:-1]
        if not candidate_lows:
            return lows[0] if lows else 0.0
        return min(candidate_lows)

    def _find_swing_high(self, candles: List[Any], lookback: int = 10) -> float:
        """Highest high in lookback that is higher than neighbors."""
        if len(candles) < lookback + 2:
            return max(c.high for c in candles[-lookback:]) if candles else 0.0
        highs = [c.high for c in candles[-lookback:]]
        candidate_highs = highs[:-1]
        if not candidate_highs:
            return highs[0] if highs else 0.0
        return max(candidate_highs)

    def calculate(
        self,
        candles: List[Any],
        direction: str,
        pair: str,
        session: str = "",
        entry_price: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Returns: {
            'entry': float,
            'sl': float,
            'tp1': float,
            'tp2': float,
            'sl_distance': float,
            'tp1_distance': float,
            'rr': float,
            'method': str,
            'structure_level': float,
        }
        """
        cfg = self.config.get(pair, self.config["EURUSD"])
        atr = self._atr(candles, 14)
        boost = SESSION_VOL_BOOST.get(session, 1.0)

        current = entry_price if entry_price else candles[-1].close
        decimals = 2 if "XAU" in pair else (0 if "BTC" in pair else 5)

        if direction == "buy":
            structure = self._find_swing_low(candles, cfg["min_bars"])
            # SL = below structure by ATR buffer, then apply pair mult + session boost
            buffer = atr * cfg["structure_buffer_atr"] * boost
            sl_raw = structure - buffer - cfg["spread_buffer"]
            # Also enforce minimum distance: at least atr * atr_mult * boost from entry
            min_sl = current - (atr * cfg["atr_mult"] * boost)
            sl = min(sl_raw, min_sl)  # whichever is lower (wider stop)
            sl = round(sl, decimals)

            sl_dist = abs(current - sl)
            tp1_dist = sl_dist * 2.0  # 1:2 risk/reward minimum
            tp2_dist = sl_dist * 3.5
            tp1 = round(current + tp1_dist, decimals)
            tp2 = round(current + tp2_dist, decimals)

        else:  # sell
            structure = self._find_swing_high(candles, cfg["min_bars"])
            buffer = atr * cfg["structure_buffer_atr"] * boost
            sl_raw = structure + buffer + cfg["spread_buffer"]
            min_sl = current + (atr * cfg["atr_mult"] * boost)
            sl = max(sl_raw, min_sl)  # whichever is higher (wider stop)
            sl = round(sl, decimals)

            sl_dist = abs(sl - current)
            tp1_dist = sl_dist * 2.0
            tp2_dist = sl_dist * 3.5
            tp1 = round(current - tp1_dist, decimals)
            tp2 = round(current - tp2_dist, decimals)

        rr = round(tp1_dist / max(sl_dist, 1e-9), 2)

        return {
            "entry": round(current, decimals),
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "sl_distance": round(sl_dist, decimals),
            "tp1_distance": round(tp1_dist, decimals),
            "rr": rr,
            "method": f"structure+ATR({cfg['atr_mult']}x)+{session}",
            "structure_level": round(structure, decimals),
        }


smart_stop = SmartStopCalculator()
