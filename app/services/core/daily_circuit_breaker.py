"""
Daily Loss Circuit Breaker — Prop-firm style hard stop.
Tracks realized P&L per day. If daily loss >= limit, blocks ALL new signals.
"""
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

DB_PATH = Path("daily_pnl_tracker.json")
DAILY_LOSS_LIMIT_PCT = 0.02  # 2% of account


class DailyCircuitBreaker:
    """
    Prevents death spirals. Once daily loss hits 2%, no more signals until next day.
    """

    def __init__(self, account_balance: float = 100.0):
        self.balance = account_balance
        self.limit = account_balance * DAILY_LOSS_LIMIT_PCT
        self._data = self._load()
        self._reset_if_new_day()

    def _load(self) -> Dict:
        if DB_PATH.exists():
            try:
                with open(DB_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "pnl": 0.0}

    def _save(self):
        try:
            with open(DB_PATH, "w") as f:
                json.dump(self._data, f)
        except Exception as e:
            logger.warning(f"Circuit breaker save failed: {e}")

    def _reset_if_new_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._data.get("date") != today:
            self._data = {"date": today, "pnl": 0.0}
            self._save()
            logger.info("Daily circuit breaker reset for new day")

    def record_outcome(self, pnl_r: float, risk_amount: float):
        """
        pnl_r: +1.5 for win, -1.0 for loss
        risk_amount: dollar risk per trade (e.g., $2 for 2% of $100)
        """
        dollar_pnl = pnl_r * risk_amount
        self._data["pnl"] += dollar_pnl
        self._save()
        logger.info(f"Daily P&L: ${self._data['pnl']:+.2f} / limit -${self.limit:.2f}")

    def can_trade(self) -> tuple:
        self._reset_if_new_day()
        remaining = self.limit + self._data["pnl"]  # pnl is negative for losses
        if remaining <= 0:
            return False, f"Circuit breaker: daily loss ${abs(self._data['pnl']):.2f} >= limit ${self.limit:.2f}"
        return True, f"Daily headroom: ${remaining:.2f}"

    def get_status(self) -> Dict:
        self._reset_if_new_day()
        return {
            "date": self._data["date"],
            "daily_pnl": round(self._data["pnl"], 2),
            "limit": round(self.limit, 2),
            "can_trade": self._data["pnl"] > -self.limit,
        }


circuit_breaker = DailyCircuitBreaker()
