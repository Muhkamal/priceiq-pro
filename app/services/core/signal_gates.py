"""
Signal Gates — Macro, COT, News, MTF filters
Applied AFTER agents generate signals, BEFORE execution.
"""
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class MacroGate:
    """Blocks gold longs when real rates / DXY are spiking."""

    def __init__(self):
        self.enabled = bool(os.getenv("FRED_API_KEY"))
        self.dxy_threshold = float(os.getenv("DXY_RISE_THRESHOLD", "1.5"))
        self.real_rate_threshold = float(os.getenv("REAL_RATE_THRESHOLD", "0.5"))
        self._data: Dict = {}

    def set_data(self, data: Dict):
        self._data = data or {}

    def check(self, pair: str, direction: str) -> tuple:
        if not self.enabled or not self._data:
            return True, 0.0

        # Block XAUUSD longs when DXY is surging
        if pair == "XAUUSD" and direction == "buy":
            dxy_change = self._data.get("dxy_1w_change", 0)
            if dxy_change > self.dxy_threshold:
                logger.info(f"Macro gate: blocked XAUUSD buy (DXY +{dxy_change:.2f}%)")
                return False, -0.15
            real_rate = self._data.get("real_rate_delta", 0)
            if real_rate > self.real_rate_threshold:
                logger.info(f"Macro gate: blocked XAUUSD buy (real rates +{real_rate:.2f})")
                return False, -0.10

        return True, 0.0


class NewsGate:
    """Reduces confidence if news sentiment opposes signal direction."""

    def __init__(self):
        self.enabled = True
        self._data: Dict = {}

    def set_data(self, data: Dict):
        self._data = data or {}

    def check(self, pair: str, direction: str) -> tuple:
        if not self._data:
            return True, 0.0

        score = self._data.get("score", 0)  # -1 bearish to +1 bullish
        strength = abs(score)

        if pair == "XAUUSD":
            if direction == "buy" and score < -0.3:
                return True, -0.10 * strength
            if direction == "sell" and score > 0.3:
                return True, -0.10 * strength

        return True, 0.0


class COTGate:
    """Weekly weight bias from COT positioning."""

    def __init__(self):
        self.enabled = True
        self._data: Dict = {}

    def set_data(self, data: Dict):
        self._data = data or {}

    def get_bias(self, pair: str) -> Dict:
        if not self._data:
            return {"mean_reversion_boost": 0.0, "trend_boost": 0.0}

        commercial_net = self._data.get("commercial_net", 0)
        extreme = abs(commercial_net) > 0.7

        if pair == "XAUUSD" and extreme:
            if commercial_net > 0.7:
                return {"mean_reversion_boost": 0.0, "trend_boost": 0.05}
            elif commercial_net < -0.7:
                return {"mean_reversion_boost": 0.05, "trend_boost": 0.0}

        return {"mean_reversion_boost": 0.0, "trend_boost": 0.0}
