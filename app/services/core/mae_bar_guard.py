"""
M.A.E. Bar Closed Guard
Prevents signal generation on incomplete 1H candles.
"""

from datetime import datetime, timezone
from typing import Any, List
import logging

logger = logging.getLogger(__name__)

class BarClosedGuard:
    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe

    def is_bar_closed(self, candles: List[Any]) -> bool:
        if not candles:
            return False
        latest = candles[-1]
        ts = getattr(latest, "timestamp", None)
        if ts is None:
            ts = getattr(latest, "time", None)
        if ts is None:
            return True  # conservative: if no ts, assume closed

        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return True

        now = datetime.now(timezone.utc)
        age_seconds = (now - ts).total_seconds()
        if age_seconds > 3000:  # > 50 minutes old
            return True
        if age_seconds < 0:
            return False

        minute = now.minute
        if 0 <= minute <= 5 or 55 <= minute <= 59:
            return True
        return False

    def check(self, candles: List[Any]) -> bool:
        ok = self.is_bar_closed(candles)
        if not ok:
            logger.info("MAE BarClosedGuard: rejecting signal — candle not yet closed")
        return ok

mae_bar_guard = BarClosedGuard()
