"""
Weekend Gap Handler — Prevents false SL/TP resolution on market re-open gaps.
Forex closes Fri ~22:00 UTC, opens Sun ~22:00 UTC.
"""
from datetime import datetime, timezone, timedelta
from typing import List, Any, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Forex market hours (approximate UTC)
FRIDAY_CLOSE_HOUR = 22
SUNDAY_OPEN_HOUR = 22


def is_weekend_gap_candle(candle_time: datetime, pair: str) -> bool:
    """
    Detects if a candle is the Sunday re-open gap candle.
    """
    if "BTC" in pair:
        return False  # Crypto never closes

    weekday = candle_time.weekday()
    hour = candle_time.hour

    # Sunday 22:00 - 23:59 (first trading hours after weekend)
    if weekday == 6 and hour >= SUNDAY_OPEN_HOUR:
        return True

    # Also catch Monday 00:00 - 01:00 (some brokers open slightly different)
    if weekday == 0 and hour <= 1:
        return True

    return False


def filter_weekend_gaps(candles: List[Any], pair: str) -> List[Any]:
    """
    Removes Sunday gap candles from resolution check.
    """
    if "BTC" in pair:
        return candles

    filtered = []
    for c in candles:
        ts = getattr(c, "timestamp", None)
        if ts is None:
            # Try common attributes
            ts = getattr(c, "time", None)
        if ts is None:
            filtered.append(c)
            continue

        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                filtered.append(c)
                continue

        if not is_weekend_gap_candle(ts, pair):
            filtered.append(c)
        else:
            logger.info(f"Weekend gap candle skipped for {pair}: {ts}")

    return filtered


def get_market_status(pair: str) -> Dict[str, Any]:
    """
    Returns market open/close status for a pair.
    """
    now = datetime.now(timezone.utc)
    weekday = now.weekday()
    hour = now.hour

    if "BTC" in pair:
        return {"open": True, "session": "crypto_24_7", "next_close": None}

    # Forex: closed Sat 00:00 - Sun 22:00 UTC
    if weekday == 5:  # Saturday
        return {"open": False, "session": "closed_weekend", "next_open": "Sunday 22:00 UTC"}
    if weekday == 6 and hour < SUNDAY_OPEN_HOUR:  # Sunday before open
        return {"open": False, "session": "closed_weekend", "next_open": "Sunday 22:00 UTC"}

    return {"open": True, "session": "active", "next_close": "Friday 22:00 UTC"}
