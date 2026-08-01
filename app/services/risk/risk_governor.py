"""
PriceIQ Pro — Portfolio Risk Governor v1.1 (FIXED)

Fixes:
    1. ADDED per-trade risk limit enforcement (was declared but never checked)
    2. FIXED gold exposure calculation (was 1000× too high for XAUUSD)
    3. ADDED pair-aware pip value helper

This is NOT a per-trade risk calculator.
This is a GOVERNOR — it has veto power over all trade decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Correlation matrix (approximate, static) ────────────────
PAIR_CORRELATIONS: Dict[str, List[str]] = {
    "EURUSD": ["GBPUSD", "AUDUSD", "NZDUSD", "EURCAD"],
    "GBPUSD": ["EURUSD", "AUDUSD", "GBPCAD", "GBPJPY"],
    "USDJPY": ["EURJPY", "GBPJPY", "AUDJPY", "CADJPY"],
    "XAUUSD": ["XAGUSD"],
    "AUDUSD": ["EURUSD", "NZDUSD", "AUDJPY"],
    "USDCAD": ["CADJPY"],
}

CORR_THRESHOLD = 0.70


@dataclass
class OpenPosition:
    pair:       str
    direction:  str
    lots:       float
    entry:      float
    stop_loss:  float
    take_profit: float
    opened_at:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    unrealised_pnl: float = 0.0


@dataclass
class RiskDecision:
    allowed:        bool
    reason:         str
    adjusted_lots:  float
    risk_score:     float


class RiskGovernor:
    """
    Portfolio-level risk governor with hard stops.
    """

    def __init__(
        self,
        starting_balance: float = 10_000.0,
        max_drawdown_pct:      float = 0.10,
        soft_drawdown_pct:     float = 0.06,
        max_risk_per_trade:    float = 0.02,
        max_total_exposure:    float = 0.08,
        max_correlated_pos:    int   = 2,
        max_consecutive_losses: int  = 4,
        soft_loss_count:        int  = 3,
    ):
        self.starting_balance       = starting_balance
        self.current_balance        = starting_balance
        self.peak_balance           = starting_balance
        self.max_drawdown_pct       = max_drawdown_pct
        self.soft_drawdown_pct      = soft_drawdown_pct
        self.max_risk_per_trade     = max_risk_per_trade
        self.max_total_exposure     = max_total_exposure
        self.max_correlated_pos     = max_correlated_pos
        self.max_consecutive_losses = max_consecutive_losses
        self.soft_loss_count        = soft_loss_count

        self._open_positions: Dict[str, OpenPosition] = {}
        self._consecutive_losses: int = 0
        self._trade_history: List[Dict] = []

    # ── Public API ───────────────────────────────────────────

    @property
    def drawdown(self) -> float:
        return max(0.0, (self.peak_balance - self.current_balance) / self.peak_balance)

    @property
    def total_open_exposure(self) -> float:
        """Total open risk as fraction of current balance. FIXED: pair-aware pip values."""
        total_risk = 0.0
        for pos in self._open_positions.values():
            sl_dist = abs(pos.entry - pos.stop_loss)
            pip_val = self._pip_value(pos.pair)
            trade_risk = sl_dist * pos.lots * pip_val
            total_risk += trade_risk
        return total_risk / max(self.current_balance, 1.0)

    @property
    def risk_score(self) -> float:
        dd_score   = min(1.0, self.drawdown / self.max_drawdown_pct)
        exp_score  = min(1.0, self.total_open_exposure / self.max_total_exposure)
        loss_score = min(1.0, self._consecutive_losses / self.max_consecutive_losses)
        return round(max(dd_score, exp_score, loss_score), 4)

    def evaluate(
        self,
        pair: str,
        direction: str,
        proposed_lots: float,
        stop_distance: float,
        entry_price: float,
    ) -> RiskDecision:
        """Evaluate whether a proposed trade is allowed."""

        # ── Hard gate 1: drawdown ────────────────────────────
        if self.drawdown >= self.max_drawdown_pct:
            return RiskDecision(
                allowed=False,
                reason=f"HARD STOP: drawdown {self.drawdown:.1%} ≥ {self.max_drawdown_pct:.0%} limit. All trading paused.",
                adjusted_lots=0.0,
                risk_score=self.risk_score,
            )

        # ── Hard gate 2: consecutive losses ──────────────────
        if self._consecutive_losses >= self.max_consecutive_losses:
            return RiskDecision(
                allowed=False,
                reason=f"HARD STOP: {self._consecutive_losses} consecutive losses ≥ {self.max_consecutive_losses} limit. Cooling off.",
                adjusted_lots=0.0,
                risk_score=self.risk_score,
            )

        # ── Hard gate 3: total exposure ───────────────────────
        if self.total_open_exposure >= self.max_total_exposure:
            return RiskDecision(
                allowed=False,
                reason=f"HARD STOP: total open exposure {self.total_open_exposure:.1%} ≥ {self.max_total_exposure:.0%} limit.",
                adjusted_lots=0.0,
                risk_score=self.risk_score,
            )

        # ── Hard gate 4: correlation ──────────────────────────
        corr_count = self._count_correlated_positions(pair, direction)
        if corr_count >= self.max_correlated_pos:
            return RiskDecision(
                allowed=False,
                reason=f"HARD STOP: {corr_count} correlated positions already open (max {self.max_correlated_pos}).",
                adjusted_lots=0.0,
                risk_score=self.risk_score,
            )

        # ── Hard gate 5: per-trade risk limit (NEW) ──────────
        pip_val = self._pip_value(pair)
        trade_risk_usd = stop_distance * proposed_lots * pip_val
        trade_risk_pct = trade_risk_usd / max(self.current_balance, 1.0)
        if trade_risk_pct > self.max_risk_per_trade:
            return RiskDecision(
                allowed=False,
                reason=f"HARD STOP: trade risk {trade_risk_pct:.2%} > {self.max_risk_per_trade:.0%} limit ({pair} {proposed_lots}L SL={stop_distance})",
                adjusted_lots=0.0,
                risk_score=self.risk_score,
            )

        # ── Soft adjustment: soft drawdown ───────────────────
        size_multiplier = 1.0
        reasons = []
        if self.drawdown >= self.soft_drawdown_pct:
            size_multiplier = min(size_multiplier, 0.50)
            reasons.append(f"soft drawdown {self.drawdown:.1%} → 50% size reduction")

        # ── Soft adjustment: consecutive soft losses ──────────
        if self._consecutive_losses >= self.soft_loss_count:
            size_multiplier = min(size_multiplier, 0.50)
            reasons.append(f"{self._consecutive_losses} consecutive losses → 50% size reduction")

        adjusted_lots = round(max(0.01, proposed_lots * size_multiplier), 2)
        reason = "ALLOWED" + (f" with adjustments: {'; '.join(reasons)}" if reasons else "")

        return RiskDecision(
            allowed=True,
            reason=reason,
            adjusted_lots=adjusted_lots,
            risk_score=self.risk_score,
        )

    def open_position(self, position: OpenPosition):
        self._open_positions[position.pair.upper()] = position
        logger.info(
            f"RiskGovernor: opened {position.pair} {position.direction} "
            f"{position.lots}L @ {position.entry}"
        )

    def close_position(self, pair: str, realised_pnl: float):
        pair = pair.upper()
        self._open_positions.pop(pair, None)
        self.current_balance += realised_pnl

        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        if realised_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._trade_history.append({
            "pair": pair,
            "pnl":  realised_pnl,
            "balance_after": self.current_balance,
            "drawdown": self.drawdown,
            "consecutive_losses": self._consecutive_losses,
        })

        logger.info(
            f"RiskGovernor: closed {pair} PnL={realised_pnl:+.2f} "
            f"balance={self.current_balance:.2f} DD={self.drawdown:.2%} "
            f"streak={self._consecutive_losses}"
        )

    def update_unrealised(self, pair: str, current_price: float):
        pos = self._open_positions.get(pair.upper())
        if not pos:
            return
        pnl_pts = (current_price - pos.entry) if pos.direction == "buy" else (pos.entry - current_price)
        pip_val = self._pip_value(pos.pair)
        pos.unrealised_pnl = pnl_pts * pos.lots * pip_val

    def reset_loss_streak(self):
        logger.info(f"RiskGovernor: loss streak manually reset from {self._consecutive_losses}")
        self._consecutive_losses = 0

    def get_portfolio_summary(self) -> Dict:
        return {
            "current_balance":      round(self.current_balance, 2),
            "peak_balance":         round(self.peak_balance, 2),
            "drawdown":             round(self.drawdown, 4),
            "risk_score":           self.risk_score,
            "total_open_exposure":  round(self.total_open_exposure, 4),
            "consecutive_losses":   self._consecutive_losses,
            "open_positions":       len(self._open_positions),
            "open_pairs":           list(self._open_positions.keys()),
            "trading_allowed":      self.drawdown < self.max_drawdown_pct and
                                    self._consecutive_losses < self.max_consecutive_losses,
        }

    # ── Helpers ──────────────────────────────────────────────

    def _count_correlated_positions(self, pair: str, direction: str) -> int:
        correlated = PAIR_CORRELATIONS.get(pair.upper(), [])
        count = 0
        for open_pair, pos in self._open_positions.items():
            if open_pair == pair.upper():
                count += 1
            elif open_pair in correlated and pos.direction == direction:
                count += 1
        return count

    def _pip_value(self, pair: str) -> float:
        """
        Pair-aware pip value multiplier.
        
        XAUUSD:  1 lot = 100 oz, $1 per $1 move  → 100
        JPY pairs: pip = 0.01                      → 1000
        Standard: pip = 0.0001                     → 100_000
        """
        pair = pair.upper()
        if "XAU" in pair or "XAG" in pair:
            return 100.0
        if "JPY" in pair:
            return 1000.0
        return 100_000.0
