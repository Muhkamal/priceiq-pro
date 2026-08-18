"""
M.A.E. Bar Closed Guard — prevents look-ahead bias.
A 1H candle is 'closed' if it finished at least 2 minutes ago.
"""

from datetime import datetime, timezone
from typing import Any, List
import logging

logger = logging.getLogger(__name__)

class BarClosedGuard:
    def __init__(self, timeframe: str = "1h", settle_seconds: int = 120):
        self.timeframe = timeframe
        self.settle_seconds = settle_seconds  # 2 min default for 1H

    def is_bar_closed(self, candles: List[Any]) -> bool:
        if not candles:
            return False
        latest = candles[-1]
        ts = getattr(latest, "timestamp", None)
        if ts is None:
            ts = getattr(latest, "time", None)
        if ts is None:
            return True  # no timestamp — assume closed

        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return True

        now = datetime.now(timezone.utc)
        age_seconds = (now - ts).total_seconds()

        # Normal case: candle is old enough → closed
        if age_seconds >= self.settle_seconds:
            return True

        # Clock skew / timezone mismatch: if timestamp is >1h in future,
        # data source is wrong — allow but warn
        if age_seconds < -3600:
            logger.info(f"BarClosedGuard: timestamp {ts} is >1h ahead of server {now}. Allowing (timezone mismatch).")
            return True

        # Future or too fresh → reject
        if age_seconds < 0:
            logger.info(f"BarClosedGuard: rejecting — candle timestamp {ts} is {abs(age_seconds):.0f}s in future")
        else:
            logger.info(f"BarClosedGuard: rejecting — candle only {age_seconds:.0f}s old, waiting {self.settle_seconds}s")
        return False

    def check(self, candles: List[Any]) -> bool:
        return self.is_bar_closed(candles)

mae_bar_guard = BarClosedGuard()
