"""
PriceIQ Pro — Paper Trading Dry-Run Mode v1.0

Runs the complete V5 pipeline on live market data
with zero real orders sent to the broker.

This is different from OANDA paper trading:
    OANDA paper: real signals → OANDA demo account
    Dry-run:     real signals → simulated fills → learning loop updates
                 → Telegram alerts labeled [PAPER] → NO broker calls

Why this matters:
    - Test the full 10-gate pipeline with real live data
    - Verify learning loop updates work correctly
    - Verify Telegram alerts fire correctly
    - Verify trade manager manages simulated positions
    - Validate regime classifier on current market conditions
    - Build win probability calibration data risk-free
    - 2-week paper trading → confident live deployment

Dry-run vs Live differences:
    ✅ Same: regime classification, agent selection, all 10 gates
    ✅ Same: Telegram alerts (labeled [PAPER])
    ✅ Same: learning loop updates, weight adjustments
    ✅ Same: trade manager TP/SL management (simulated)
    ✅ Same: journal entries (tagged "paper")
    ❌ Different: no real broker orders sent
    ❌ Different: fill price = simulated (bid/ask estimate)
    ❌ Different: equity curve is virtual (does not affect real balance)

Usage:
    # Enable in environment:
    DRY_RUN_MODE=true

    # Or programmatically:
    v5.dry_run = True

    # Paper trading stats:
    stats = v5.paper_trader.get_stats()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Record of a simulated paper trade."""
    trade_id:       str
    pair:           str
    direction:      str
    entry:          float
    stop_loss:      float
    take_profit_1:  float
    take_profit_2:  float
    lots:           float
    agent:          str
    regime:         str
    confidence:     float
    win_probability: float
    opened_at:      str
    closed_at:      Optional[str] = None
    exit_price:     Optional[float] = None
    outcome:        Optional[str] = None
    r_multiple:     Optional[float] = None
    virtual_pnl:    Optional[float] = None
    management_events: List[str] = field(default_factory=list)
    tags:           List[str] = field(default_factory=lambda: ["paper"])


@dataclass
class PaperStats:
    total_trades:     int
    open_trades:      int
    closed_trades:    int
    win_rate:         Optional[float]
    total_virtual_pnl: float
    expectancy:       Optional[float]
    avg_r:            Optional[float]
    best_trade_r:     Optional[float]
    worst_trade_r:    Optional[float]
    by_agent:         Dict[str, Dict]
    by_regime:        Dict[str, Dict]


class PaperTradingEngine:
    """
    Simulates trade execution without sending real orders.
    Tracks virtual P&L, updates learning loop, fires Telegram alerts.
    """

    def __init__(
        self,
        starting_virtual_balance: float = 10_000.0,
        risk_pct: float = 2.0,
        telegram=None,
        learning_loop=None,
        journal=None,
    ):
        self.virtual_balance = starting_virtual_balance
        self.starting_balance = starting_virtual_balance
        self.risk_pct        = risk_pct
        self.telegram        = telegram
        self.learning        = learning_loop
        self.journal         = journal
        self._open_trades:   Dict[str, PaperTrade] = {}
        self._closed_trades: List[PaperTrade]      = []

    async def open_paper_trade(self, cycle_result) -> Optional[PaperTrade]:
        """
        Record a simulated trade open.
        cycle_result is a V5SignalResult.
        """
        if not cycle_result.signal_fired:
            return None

        pair  = cycle_result.pair
        now   = datetime.now(timezone.utc).isoformat()
        tid   = f"PAPER_{pair}_{now[:16].replace(':', '')}"

        trade = PaperTrade(
            trade_id=tid,
            pair=pair,
            direction=cycle_result.direction,
            entry=cycle_result.fill_price or 0,
            stop_loss=cycle_result.stop_loss or 0,
            take_profit_1=cycle_result.take_profit_1 or 0,
            take_profit_2=cycle_result.take_profit_2 or 0,
            lots=cycle_result.adjusted_lots,
            agent=cycle_result.agent_used or "",
            regime=cycle_result.regime,
            confidence=cycle_result.confidence,
            win_probability=cycle_result.win_probability,
            opened_at=now,
        )
        self._open_trades[pair] = trade

        msg = (
            f"📝 <b>[PAPER] {pair} {cycle_result.direction.upper()}</b>\n"
            f"Entry: {trade.entry:.5f} | SL: {trade.stop_loss:.5f}\n"
            f"TP1: {trade.take_profit_1:.5f} | Lots: {trade.lots}\n"
            f"Agent: {trade.agent} | Regime: {trade.regime}\n"
            f"Confidence: {trade.confidence:.0%} | WinProb: {trade.win_probability:.0%}\n"
            f"<i>This is a paper trade — no real order sent</i>"
        )
        await self._send(msg)
        logger.info(f"Paper trade opened: {pair} {cycle_result.direction}")
        return trade

    async def update_paper_trades(self, current_prices: Dict[str, float]):
        """
        Simulate price action on open paper trades.
        Call every bar (same as trade_manager.update_all).
        """
        to_close = []
        for pair, trade in self._open_trades.items():
            price = current_prices.get(pair)
            if not price:
                continue

            direction = trade.direction
            sl_hit  = (direction == "buy"  and price <= trade.stop_loss) or \
                      (direction == "sell" and price >= trade.stop_loss)
            tp1_hit = (direction == "buy"  and price >= trade.take_profit_1) or \
                      (direction == "sell" and price <= trade.take_profit_1)

            if sl_hit:
                await self._close_paper_trade(trade, trade.stop_loss, "loss")
                to_close.append(pair)
            elif tp1_hit:
                await self._close_paper_trade(trade, trade.take_profit_1, "win")
                to_close.append(pair)

        for pair in to_close:
            trade = self._open_trades.pop(pair, None)
            if trade:
                self._closed_trades.append(trade)

    async def _close_paper_trade(self, trade: PaperTrade, exit_price: float, outcome: str):
        """Close a paper trade and update all systems."""
        now       = datetime.now(timezone.utc).isoformat()
        risk_dist = abs(trade.entry - trade.stop_loss)
        pts       = (exit_price - trade.entry) if trade.direction == "buy" \
                    else (trade.entry - exit_price)
        r_mult    = round(pts / max(risk_dist, 1e-9), 3)
        virt_pnl  = round(pts * trade.lots * 100_000 / 10, 2)

        trade.closed_at   = now
        trade.exit_price  = exit_price
        trade.outcome     = outcome
        trade.r_multiple  = r_mult
        trade.virtual_pnl = virt_pnl

        # Update virtual balance
        self.virtual_balance += virt_pnl

        # Update learning loop (same as real trade)
        if self.learning:
            try:
                self.learning.update(
                    pair=trade.pair, agent_name=trade.agent,
                    direction=trade.direction, entry=trade.entry,
                    exit_price=exit_price, stop=trade.stop_loss,
                    tp1=trade.take_profit_1, outcome=outcome,
                    r_multiple=r_mult, regime=trade.regime,
                    confidence=trade.confidence, session="paper",
                )
            except Exception as e:
                logger.warning(f"Paper trade learning update failed: {e}")

        emoji = "✅" if outcome == "win" else "❌"
        msg   = (
            f"{emoji} <b>[PAPER] {trade.pair} CLOSED</b>\n"
            f"Outcome: {outcome.upper()} | {r_mult:+.2f}R\n"
            f"Virtual PnL: ${virt_pnl:+.2f}\n"
            f"Virtual Balance: ${self.virtual_balance:,.2f}\n"
            f"Agent: {trade.agent} | Regime: {trade.regime}"
        )
        await self._send(msg)
        logger.info(f"Paper trade closed: {trade.pair} {outcome} {r_mult:+.2f}R ${virt_pnl:+.2f}")

    def get_stats(self) -> PaperStats:
        """Full paper trading statistics."""
        from collections import defaultdict

        closed = self._closed_trades
        total  = len(closed) + len(self._open_trades)
        wins   = [t for t in closed if t.outcome == "win"]
        r_mults = [t.r_multiple for t in closed if t.r_multiple is not None]

        by_agent: Dict = defaultdict(lambda: {"n": 0, "wins": 0, "r_mults": []})
        by_regime: Dict = defaultdict(lambda: {"n": 0, "wins": 0, "r_mults": []})

        for t in closed:
            by_agent[t.agent]["n"]  += 1
            by_regime[t.regime]["n"] += 1
            if t.outcome == "win":
                by_agent[t.agent]["wins"]   += 1
                by_regime[t.regime]["wins"] += 1
            if t.r_multiple is not None:
                by_agent[t.agent]["r_mults"].append(t.r_multiple)
                by_regime[t.regime]["r_mults"].append(t.r_multiple)

        def _summarise(d):
            return {
                k: {
                    "n":        v["n"],
                    "win_rate": round(v["wins"] / v["n"], 3) if v["n"] else None,
                    "avg_r":    round(float(np.mean(v["r_mults"])), 3) if v["r_mults"] else None,
                }
                for k, v in d.items()
            }

        return PaperStats(
            total_trades=total,
            open_trades=len(self._open_trades),
            closed_trades=len(closed),
            win_rate=round(len(wins) / len(closed), 3) if closed else None,
            total_virtual_pnl=round(sum(t.virtual_pnl or 0 for t in closed), 2),
            expectancy=round(float(np.mean(r_mults)), 3) if r_mults else None,
            avg_r=round(float(np.mean(r_mults)), 3) if r_mults else None,
            best_trade_r=round(max(r_mults), 3) if r_mults else None,
            worst_trade_r=round(min(r_mults), 3) if r_mults else None,
            by_agent=_summarise(by_agent),
            by_regime=_summarise(by_regime),
        )

    async def send_daily_paper_summary(self):
        """Send daily paper trading summary via Telegram."""
        stats = self.get_stats()
        msg   = (
            f"📝 <b>Paper Trading Daily Summary</b>\n\n"
            f"Virtual Balance: ${self.virtual_balance:,.2f} "
            f"(start: ${self.starting_balance:,.2f})\n"
            f"Total trades: {stats.total_trades} "
            f"(open: {stats.open_trades})\n"
            f"Win rate: {stats.win_rate:.0%}\n" if stats.win_rate else "Win rate: N/A\n"
            f"Expectancy: {stats.expectancy:+.2f}R\n" if stats.expectancy else ""
            f"\n<i>Paper mode active — no real orders sent</i>"
        )
        await self._send(msg)

    async def _send(self, message: str):
        if self.telegram:
            try:
                await self.telegram.send_message(message)
            except Exception as e:
                logger.warning(f"PaperTrader Telegram failed: {e}")
