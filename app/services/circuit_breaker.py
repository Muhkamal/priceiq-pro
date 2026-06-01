"""
PriceIQ Pro — Drawdown Circuit Breaker v1.0

Automatically pauses trading when risk thresholds are breached:
  - Daily loss limit      : stop if account drops X% in one day
  - Weekly loss limit     : stop if account drops Y% in one week
  - Max consecutive losses: stop after N losses in a row
  - Total drawdown limit  : stop if peak-to-trough exceeds Z%
  - Manual kill switch    : hard stop that requires manual reset

State is persisted in-memory (swap for Redis/DB in production).
"""

from datetime import datetime, timezone, timedelta, date
from typing import Optional, List, Dict
from enum import Enum
from pydantic import BaseModel, Field


class BreakerState(str, Enum):
    ACTIVE    = "active"       # trading allowed
    DAILY     = "daily_halt"   # daily loss limit hit — resumes tomorrow
    WEEKLY    = "weekly_halt"  # weekly loss limit — resumes next Monday
    DRAWDOWN  = "drawdown_halt"# max drawdown hit — requires manual reset
    CONSEC    = "consecutive_halt" # too many losses in a row
    KILLED    = "killed"       # manual kill switch — requires manual reset


class BreakerCheckResult(BaseModel):
    is_allowed:    bool
    state:         BreakerState
    reason:        Optional[str] = None
    resume_at:     Optional[datetime] = None
    stats:         Dict = Field(default_factory=dict)
    checked_at:    datetime


class TradeOutcome(BaseModel):
    pair:       str
    pnl:        float          # absolute dollar P&L
    pnl_pct:    float          # % of account
    timestamp:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result:     str            # "win" | "loss"


class CircuitBreaker:
    """
    Stateful risk gate that blocks new signals when thresholds are breached.

    Thresholds (all configurable):
      daily_loss_limit_pct    : default 5%  — halt for remainder of day
      weekly_loss_limit_pct   : default 10% — halt until next Monday
      max_consecutive_losses  : default 5   — halt for 4 hours then reset
      max_drawdown_pct        : default 15% — halt until manual reset
    """

    def __init__(
        self,
        daily_loss_limit_pct:   float = 5.0,
        weekly_loss_limit_pct:  float = 10.0,
        max_consecutive_losses: int   = 5,
        max_drawdown_pct:       float = 15.0,
        starting_balance:       float = 10_000.0,
    ):
        self.daily_loss_limit   = daily_loss_limit_pct / 100
        self.weekly_loss_limit  = weekly_loss_limit_pct / 100
        self.max_consec_losses  = max_consecutive_losses
        self.max_drawdown_pct   = max_drawdown_pct / 100

        # Account tracking
        self.starting_balance   = starting_balance
        self.peak_balance       = starting_balance
        self.current_balance    = starting_balance

        # State
        self._state             = BreakerState.ACTIVE
        self._halt_until:       Optional[datetime] = None
        self._halt_reason:      Optional[str]      = None

        # Trade history
        self._trades:           List[TradeOutcome] = []
        self._consec_losses:    int                = 0

        # Daily/weekly tracking
        self._day_start_balance:    float = starting_balance
        self._week_start_balance:   float = starting_balance
        self._last_day_reset:       date  = datetime.now(timezone.utc).date()
        self._last_week_reset:      date  = datetime.now(timezone.utc).date()

        # Manual kill switch
        self._killed:           bool = False

    # ──────────────────────────────────────────────────────────
    # PUBLIC: Check before each trade
    # ──────────────────────────────────────────────────────────

    def check(self, now: Optional[datetime] = None) -> BreakerCheckResult:
        """
        Check whether new trades are allowed right now.
        Call this before every signal is generated or executed.
        """
        now = now or datetime.now(timezone.utc)
        self._maybe_reset_daily(now)
        self._maybe_reset_weekly(now)

        # Manual kill — hardest stop
        if self._killed:
            return BreakerCheckResult(
                is_allowed=False,
                state=BreakerState.KILLED,
                reason="Manual kill switch is active. Call reset_kill() to resume.",
                stats=self._stats(now),
                checked_at=now,
            )

        # Time-based halt still active?
        if self._state != BreakerState.ACTIVE and self._halt_until:
            if now < self._halt_until:
                return BreakerCheckResult(
                    is_allowed=False,
                    state=self._state,
                    reason=self._halt_reason,
                    resume_at=self._halt_until,
                    stats=self._stats(now),
                    checked_at=now,
                )
            else:
                # Halt expired — auto-resume (except DRAWDOWN and KILLED)
                if self._state not in (BreakerState.DRAWDOWN, BreakerState.KILLED):
                    self._state = BreakerState.ACTIVE
                    self._halt_until = None
                    self._halt_reason = None

        # Re-check all thresholds live
        triggered = self._check_thresholds(now)
        if triggered:
            return triggered

        return BreakerCheckResult(
            is_allowed=True,
            state=BreakerState.ACTIVE,
            stats=self._stats(now),
            checked_at=now,
        )

    def record_trade(self, outcome: TradeOutcome):
        """
        Record a completed trade outcome.
        Must be called after every trade closes.
        """
        self._trades.append(outcome)
        self.current_balance += outcome.pnl

        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        if outcome.result == "loss":
            self._consec_losses += 1
        else:
            self._consec_losses = 0

    def get_recent_trades(self, limit: int = 100) -> List[TradeOutcome]:
        """Get recent trades."""
        return self._trades[-limit:] if self._trades else []

    def get_total_trade_count(self) -> int:
        """Get total number of recorded trades."""
        return len(self._trades)

    def kill(self, reason: str = "Manual kill switch activated"):
        """Hard stop — requires manual reset_kill() to resume."""
        self._killed = True
        self._state  = BreakerState.KILLED
        self._halt_reason = reason

    def reset_kill(self):
        """Manually re-enable trading after a kill switch."""
        self._killed     = False
        self._state      = BreakerState.ACTIVE
        self._halt_until = None
        self._halt_reason = None

    def reset_drawdown_halt(self):
        """Manually re-enable after a drawdown halt (requires review)."""
        if self._state == BreakerState.DRAWDOWN:
            self._state       = BreakerState.ACTIVE
            self._halt_until  = None
            self._halt_reason = None

    def update_balance(self, new_balance: float):
        """Sync balance from broker (call periodically)."""
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    def get_stats(self) -> Dict:
        return self._stats(datetime.now(timezone.utc))

    # ──────────────────────────────────────────────────────────
    # Internal threshold checks
    # ──────────────────────────────────────────────────────────

    def _check_thresholds(self, now: datetime) -> Optional[BreakerCheckResult]:
        """Check all thresholds. Return a blocked result if any triggered."""

        # 1. Daily loss limit
        daily_loss_pct = (self._day_start_balance - self.current_balance) / max(self._day_start_balance, 1)
        if daily_loss_pct >= self.daily_loss_limit:
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            self._state       = BreakerState.DAILY
            self._halt_until  = tomorrow
            self._halt_reason = (
                f"Daily loss limit hit: {daily_loss_pct:.1%} loss today "
                f"(limit: {self.daily_loss_limit:.0%}). Resumes tomorrow."
            )
            return BreakerCheckResult(
                is_allowed=False,
                state=self._state,
                reason=self._halt_reason,
                resume_at=self._halt_until,
                stats=self._stats(now),
                checked_at=now,
            )

        # 2. Weekly loss limit
        weekly_loss_pct = (self._week_start_balance - self.current_balance) / max(self._week_start_balance, 1)
        if weekly_loss_pct >= self.weekly_loss_limit:
            days_until_monday = (7 - now.weekday()) % 7 or 7
            next_monday = (now + timedelta(days=days_until_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self._state       = BreakerState.WEEKLY
            self._halt_until  = next_monday
            self._halt_reason = (
                f"Weekly loss limit hit: {weekly_loss_pct:.1%} loss this week "
                f"(limit: {self.weekly_loss_limit:.0%}). Resumes Monday."
            )
            return BreakerCheckResult(
                is_allowed=False,
                state=self._state,
                reason=self._halt_reason,
                resume_at=self._halt_until,
                stats=self._stats(now),
                checked_at=now,
            )

        # 3. Max consecutive losses
        if self._consec_losses >= self.max_consec_losses:
            resume = now + timedelta(hours=4)
            self._state       = BreakerState.CONSEC
            self._halt_until  = resume
            self._halt_reason = (
                f"{self._consec_losses} consecutive losses. 4-hour cooling off period. "
                f"Review your setups before resuming."
            )
            return BreakerCheckResult(
                is_allowed=False,
                state=self._state,
                reason=self._halt_reason,
                resume_at=resume,
                stats=self._stats(now),
                checked_at=now,
            )

        # 4. Total drawdown from peak
        drawdown = (self.peak_balance - self.current_balance) / max(self.peak_balance, 1)
        if drawdown >= self.max_drawdown_pct:
            self._state       = BreakerState.DRAWDOWN
            self._halt_until  = None  # requires manual reset
            self._halt_reason = (
                f"Maximum drawdown breached: {drawdown:.1%} from peak "
                f"(limit: {self.max_drawdown_pct:.0%}). Manual reset required."
            )
            return BreakerCheckResult(
                is_allowed=False,
                state=self._state,
                reason=self._halt_reason,
                stats=self._stats(now),
                checked_at=now,
            )

        return None

    def _maybe_reset_daily(self, now: datetime):
        today = now.date()
        if today > self._last_day_reset:
            self._day_start_balance = self.current_balance
            self._last_day_reset    = today
            if self._state == BreakerState.DAILY:
                self._state       = BreakerState.ACTIVE
                self._halt_until  = None
                self._halt_reason = None

    def _maybe_reset_weekly(self, now: datetime):
        today = now.date()
        if now.weekday() == 0 and today > self._last_week_reset:
            self._week_start_balance = self.current_balance
            self._last_week_reset    = today
            if self._state == BreakerState.WEEKLY:
                self._state       = BreakerState.ACTIVE
                self._halt_until  = None
                self._halt_reason = None

    def _stats(self, now: datetime) -> Dict:
        daily_loss = (self._day_start_balance - self.current_balance) / max(self._day_start_balance, 1)
        weekly_loss = (self._week_start_balance - self.current_balance) / max(self._week_start_balance, 1)
        drawdown = (self.peak_balance - self.current_balance) / max(self.peak_balance, 1)

        return {
            "current_balance":       round(self.current_balance, 2),
            "peak_balance":          round(self.peak_balance, 2),
            "starting_balance":      round(self.starting_balance, 2),
            "daily_loss_pct":        round(daily_loss * 100, 2),
            "daily_loss_limit_pct":  round(self.daily_loss_limit * 100, 1),
            "weekly_loss_pct":       round(weekly_loss * 100, 2),
            "weekly_loss_limit_pct": round(self.weekly_loss_limit * 100, 1),
            "current_drawdown_pct":  round(drawdown * 100, 2),
            "max_drawdown_limit_pct":round(self.max_drawdown_pct * 100, 1),
            "consecutive_losses":    self._consec_losses,
            "max_consecutive_limit": self.max_consec_losses,
            "total_trades":          len(self._trades),
            "state":                 self._state.value,
        }
