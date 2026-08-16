"""
Correlation Filter — Prevents taking multiple signals on highly correlated pairs.
EURUSD/GBPUSD ~85% correlated. Taking both = 2× risk on same directional bet.
"""
import logging
from typing import List, Dict, Set
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# 90-day rolling correlation matrix (approximate, updated periodically)
# Values > 0.7 = highly correlated, > 0.5 = moderately correlated
CORRELATION_PAIRS: Dict[str, List[str]] = {
    "EURUSD": ["GBPUSD", "AUDUSD", "NZDUSD"],
    "GBPUSD": ["EURUSD", "AUDUSD", "NZDUSD"],
    "AUDUSD": ["EURUSD", "GBPUSD", "NZDUSD"],
    "USDCHF": ["EURUSD", "GBPUSD"],  # Inverse correlation
    "XAUUSD": ["EURUSD", "GBPUSD"],  # Risk-on correlation
    "BTCUSD": ["ETHUSD", "XAUUSD"],  # Crypto/risk correlation
}

class CorrelationFilter:
    """
    Tracks recently signaled pairs and blocks new signals on correlated pairs
    within a cooldown window (default 4 hours = 4 candles on 1H).
    """

    def __init__(self, cooldown_hours: int = 4):
        self._recent_signals: Dict[str, datetime] = {}  # pair -> last signal time
        self.cooldown = cooldown_hours

    def record(self, pair: str):
        self._recent_signals[pair] = datetime.now(timezone.utc)

    def check(self, pair: str) -> tuple:
        """
        Returns: (allow: bool, blocked_by: str or None, reason: str)
        """
        now = datetime.now(timezone.utc)
        correlated = CORRELATION_PAIRS.get(pair, [])

        for other, last_time in list(self._recent_signals.items()):
            # Clean old entries
            if (now - last_time).total_seconds() > self.cooldown * 3600:
                del self._recent_signals[other]
                continue

            if other in correlated:
                return False, other, f"Blocked: {pair} correlated with {other} (active)"

        return True, None, "No correlation conflict"

    def get_active(self) -> List[str]:
        now = datetime.now(timezone.utc)
        return [p for p, t in self._recent_signals.items()
                if (now - t).total_seconds() <= self.cooldown * 3600]


corr_filter = CorrelationFilter()
