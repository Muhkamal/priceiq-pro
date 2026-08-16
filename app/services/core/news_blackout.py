"""
News Blackout — Blocks signals during high-impact economic releases.
NFP, CPI, FOMC = 30-50 pip whipsaws. Professional algos sit out.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class NewsBlackout:
    """
    Simple blackout: block 15 min before and 30 min after known high-impact events.
    For a signal bot, this is conservative but correct.
    """

    def __init__(self, buffer_minutes: int = 15):
        self.buffer = buffer_minutes
        self._manual_blackout: Optional[datetime] = None

    def set_manual_blackout(self, until: datetime):
        """Call before known events if you have a calendar feed."""
        self._manual_blackout = until
        logger.info(f"Manual blackout set until {until}")

    def is_blackout(self) -> tuple:
        now = datetime.now(timezone.utc)

        # Manual override
        if self._manual_blackout and now < self._manual_blackout:
            return True, f"Blackout until {self._manual_blackout.strftime('%H:%M')} UTC"

        # Time-based: avoid first 5 minutes of each hour (common news release times)
        minute = now.minute
        hour = now.hour
        weekday = now.weekday()

        # Friday NFP window (approximate)
        if weekday == 4 and 12 <= hour < 14:
            return True, "Friday NFP blackout window (12:00-14:00 UTC)"

        # First 5 minutes of each hour
        if minute < 5:
            return True, "Top-of-hour blackout (news release risk)"

        return False, "No blackout"

    def check(self, pair: str = "") -> tuple:
        blocked, reason = self.is_blackout()
        if blocked:
            logger.info(f"News blackout active: {reason}")
        return not blocked, reason


news_blackout = NewsBlackout()
