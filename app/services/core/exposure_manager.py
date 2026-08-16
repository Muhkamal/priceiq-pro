"""
Exposure Manager — Portfolio-level risk control.
Max 2 correlated pairs, max 3 total signals per 4H window.
Prevents over-concentration.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict

logger = logging.getLogger(__name__)

# Correlation groups — taking 2 from same group = allowed, 3 = blocked
CORR_GROUPS = {
    "usd_majors_eur_gbp": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
    "usd_safe": ["USDCHF", "USDJPY"],
    "commodity": ["XAUUSD", "XAGUSD"],
    "crypto": ["BTCUSD", "ETHUSD"],
}


class ExposureManager:
    """
    Tracks open signals and enforces portfolio-level limits.
    """

    def __init__(self):
        self._signals: List[Dict] = []  # {pair, direction, time}

    def record(self, pair: str, direction: str):
        self._signals.append({
            "pair": pair,
            "direction": direction,
            "time": datetime.now(timezone.utc),
        })
        self._clean_old()

    def _clean_old(self, max_hours: int = 4):
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_hours)
        self._signals = [s for s in self._signals if s["time"] > cutoff]

    def can_add(self, pair: str) -> tuple:
        self._clean_old()
        current_pairs = [s["pair"] for s in self._signals]

        # Max 3 signals in 4H window
        if len(current_pairs) >= 3:
            return False, f"Max 3 signals per 4H reached ({len(current_pairs)} active)"

        # Max 1 per correlation group (already have one in same group)
        for group_name, members in CORR_GROUPS.items():
            if pair in members:
                group_count = sum(1 for p in current_pairs if p in members)
                if group_count >= 1:
                    return False, f"Max 1 signal per {group_name} group ({pair} blocked)"

        return True, "Exposure OK"

    def get_summary(self) -> str:
        self._clean_old()
        if not self._signals:
            return "No active exposure"
        lines = [f"{s['pair']} {s['direction'].upper()}" for s in self._signals]
        return " | ".join(lines)


exposure_manager = ExposureManager()
