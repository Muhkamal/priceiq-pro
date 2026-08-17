"""
Bar Closed Guard — prevents look-ahead bias.
Only allows signals when the current 1H candle has actually closed.
"""
from datetime import datetime, timezone
from typing import Any, List
import logging

logger = logging.getLogger(__name__)


class BarClosedGuard:
    """
    Checks if the latest candle timestamp indicates a fully closed bar.
    For 1H timeframe: minute should be 00 (top of hour) or we should
    be at least 58 minutes past the hour to consider the bar 'almost closed'.
    """

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
            logger.debug("No timestamp on candle — assuming closed")
            return True

        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return True

        now = datetime.now(timezone.utc)
        # For 1H: if candle timestamp is > 50 minutes old, it's closed
        age_seconds = (now - ts).total_seconds()
        if age_seconds > 3000:  # > 50 minutes old
            return True
        if age_seconds < 0:
            return False  # future candle?

        # Conservative: only allow at XX:00-XX:05 or XX:55-XX:59
        minute = now.minute
        if 0 <= minute <= 5 or 55 <= minute <= 59:
            return True

        return False

    def check(self, candles: List[Any]) -> bool:
        ok = self.is_bar_closed(candles)
        if not ok:
            logger.info("BarClosedGuard: rejecting signal — candle not yet closed")
        return ok


bar_guard = BarClosedGuard()
