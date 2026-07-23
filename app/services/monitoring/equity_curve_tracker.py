"""
PriceIQ Pro — Equity Curve Tracker v1.0

Persistent time-series of account balance snapshots.
Lets you plot growth, identify when drawdown started,
compare this week vs last week, and compute rolling Sharpe.

Snapshots recorded at:
    - Every trade close (exact balance after PnL)
    - Every hourly heartbeat (running balance)
    - Every day end (daily mark)

Computes:
    - Rolling Sharpe (7d, 30d, all-time)
    - Rolling drawdown curve
    - Daily / weekly / monthly returns
    - Best and worst days
    - Drawdown start date and recovery time

Usage:
    tracker = EquityCurveTracker(starting_balance=10_000)
    tracker.record(balance=10_150.0, event="trade_close", note="XAUUSD win +1.5R")
    tracker.record(balance=10_080.0, event="trade_close", note="EURUSD loss -1R")
    report = tracker.full_report()
    print(tracker.ascii_chart(width=60))
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PERSIST_PATH = "equity_curve.json"
MAX_SNAPSHOTS = 10_000   # keep last 10K snapshots in memory


@dataclass
class EquitySnapshot:
    timestamp:  str
    balance:    float
    event:      str    # "trade_close" | "heartbeat" | "day_end" | "startup"
    note:       str    = ""
    drawdown:   float  = 0.0    # from peak at this point


class EquityCurveTracker:
    """
    Persistent equity curve with analytics.
    """

    def __init__(
        self,
        starting_balance: float = 10_000.0,
        path: str = PERSIST_PATH,
    ):
        self.starting_balance = starting_balance
        self._path     = path
        self._snapshots: List[EquitySnapshot] = []
        self._peak     = starting_balance
        self._load()

        # Seed with starting balance if empty
        if not self._snapshots:
            self.record(starting_balance, "startup", "System initialised")

    # ── Record ───────────────────────────────────────────────

    def record(self, balance: float, event: str = "heartbeat", note: str = ""):
        """Add a balance snapshot."""
        if balance > self._peak:
            self._peak = balance
        dd = (self._peak - balance) / self._peak if self._peak > 0 else 0.0

        snap = EquitySnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            balance=round(balance, 2),
            event=event,
            note=note,
            drawdown=round(dd, 6),
        )
        self._snapshots.append(snap)
        # Trim to max
        if len(self._snapshots) > MAX_SNAPSHOTS:
            self._snapshots = self._snapshots[-MAX_SNAPSHOTS:]
        self._save()

    # ── Analytics ─────────────────────────────────────────────

    def current_balance(self) -> float:
        return self._snapshots[-1].balance if self._snapshots else self.starting_balance

    def current_drawdown(self) -> float:
        return self._snapshots[-1].drawdown if self._snapshots else 0.0

    def total_return_pct(self) -> float:
        if not self._snapshots:
            return 0.0
        return (self._snapshots[-1].balance - self.starting_balance) / self.starting_balance

    def max_drawdown(self) -> float:
        if not self._snapshots:
            return 0.0
        return max(s.drawdown for s in self._snapshots)

    def drawdown_start(self) -> Optional[str]:
        """Timestamp when current drawdown began."""
        if not self._snapshots:
            return None
        # Walk backwards to find last peak
        peak_bal = self._snapshots[-1].balance
        for s in reversed(self._snapshots):
            if s.balance >= peak_bal:
                return s.timestamp
            peak_bal = max(peak_bal, s.balance)
        return self._snapshots[0].timestamp

    def daily_returns(self, n_days: int = 30) -> List[float]:
        """Daily percentage returns for last N days."""
        if len(self._snapshots) < 2:
            return []
        # Group by day
        from collections import defaultdict
        day_balances: Dict[str, List[float]] = defaultdict(list)
        for s in self._snapshots:
            day = s.timestamp[:10]   # YYYY-MM-DD
            day_balances[day].append(s.balance)

        days    = sorted(day_balances.keys())[-n_days - 1:]
        returns = []
        for i in range(1, len(days)):
            prev = day_balances[days[i-1]][-1]
            curr = day_balances[days[i]][-1]
            if prev > 0:
                returns.append((curr - prev) / prev)
        return returns

    def rolling_sharpe(self, n_days: int = 30) -> Optional[float]:
        """Rolling Sharpe ratio over last N days."""
        rets = self.daily_returns(n_days)
        if len(rets) < 5:
            return None
        mu  = np.mean(rets)
        std = np.std(rets)
        if std == 0:
            return None
        ann = np.sqrt(252)   # annualise daily Sharpe
        return round(float(mu / std * ann), 3)

    def weekly_return(self) -> Optional[float]:
        """Return over last 7 calendar days."""
        rets = self.daily_returns(7)
        if not rets:
            return None
        return round(sum(rets), 4)

    def monthly_return(self) -> Optional[float]:
        rets = self.daily_returns(30)
        if not rets:
            return None
        return round(sum(rets), 4)

    def best_day(self) -> Optional[Dict]:
        rets = self.daily_returns(365)
        if not rets:
            return None
        idx = int(np.argmax(rets))
        return {"return": round(rets[idx], 4), "index": idx}

    def worst_day(self) -> Optional[Dict]:
        rets = self.daily_returns(365)
        if not rets:
            return None
        idx = int(np.argmin(rets))
        return {"return": round(rets[idx], 4), "index": idx}

    def full_report(self) -> Dict:
        return {
            "starting_balance":  self.starting_balance,
            "current_balance":   self.current_balance(),
            "total_return_pct":  round(self.total_return_pct(), 4),
            "max_drawdown":      round(self.max_drawdown(), 4),
            "current_drawdown":  round(self.current_drawdown(), 4),
            "drawdown_start":    self.drawdown_start(),
            "sharpe_7d":         self.rolling_sharpe(7),
            "sharpe_30d":        self.rolling_sharpe(30),
            "weekly_return":     self.weekly_return(),
            "monthly_return":    self.monthly_return(),
            "best_day":          self.best_day(),
            "worst_day":         self.worst_day(),
            "n_snapshots":       len(self._snapshots),
        }

    def recent_curve(self, n: int = 100) -> List[Dict]:
        """Last N snapshots for charting."""
        return [
            {"t": s.timestamp, "b": s.balance, "dd": s.drawdown, "e": s.event}
            for s in self._snapshots[-n:]
        ]

    def ascii_chart(self, width: int = 60, height: int = 12) -> str:
        """Simple ASCII equity curve for terminal / Telegram."""
        if len(self._snapshots) < 2:
            return "Insufficient data for chart."
        balances = [s.balance for s in self._snapshots[-width:]]
        min_b    = min(balances)
        max_b    = max(balances)
        rang     = max_b - min_b or 1.0

        lines = [
            f"Equity Curve (last {len(balances)} snapshots)",
            f"High: ${max_b:,.2f}  Low: ${min_b:,.2f}  Now: ${balances[-1]:,.2f}",
        ]
        for row in range(height, 0, -1):
            threshold = min_b + (row / height) * rang
            line      = ""
            for b in balances:
                line += "█" if b >= threshold else " "
            label = f"${threshold:>8,.0f} |"
            lines.append(label + line)
        lines.append(" " * 10 + "─" * len(balances))
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump([asdict(s) for s in self._snapshots[-1000:]], f)
        except Exception as e:
            logger.debug(f"EquityCurve save failed: {e}")

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._snapshots = [EquitySnapshot(**d) for d in data]
            if self._snapshots:
                self._peak = max(s.balance for s in self._snapshots)
            logger.info(f"EquityCurve loaded: {len(self._snapshots)} snapshots")
        except Exception as e:
            logger.warning(f"EquityCurve load failed: {e}")
